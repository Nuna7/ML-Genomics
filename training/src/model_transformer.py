"""
model_transformer.py
---------------------
Architecture 3: Transformer encoder over the 1D bin sequence (with a
relative-position attention bias) -> same outer-product expansion as the
other two models -> light 2D refinement.

Why a Transformer here specifically: convolutions (dilated or not) build
long-range context indirectly, by stacking layers so information from
distant bins eventually reaches a shared receptive field. Self-attention
lets every bin attend directly to every other bin in a single layer,
which is a more direct mechanism for capturing long-range dependencies --
and Hi-C loops are exactly that: two bins far apart in linear genomic
distance (tens to hundreds of kb) that are functionally coupled. The
hypothesis this architecture tests is whether direct pairwise attention
over the 1D tracks captures loop-relevant long-range dependencies more
efficiently (in terms of parameters/data) than a dilated CNN does.

One adaptation that matters for genomics specifically: a vanilla
Transformer's learned/sinusoidal positional embeddings only tell the
model "this is position 37," not "these two positions are 200kb apart
along the genome." Hi-C contact frequency depends strongly and smoothly
on genomic DISTANCE (closer bins almost always contact more, with rapid
decay), so the relative distance between bins is itself a primary signal,
not just an addressing mechanism. This implementation adds a learned bias
term to attention scores based on |i - j| (relative bin distance),
similar in spirit to T5-style relative position bias, instead of relying
purely on absolute positional embeddings to let the model rediscover that
"genomic distance matters" from scratch with limited training data.

Because of the O(L^2) cost of attention and the O(L^2) memory of the
output matrix itself, this is kept to a SINGLE transformer encoder
(few layers, modest dim) appropriate for the "no large model" /
CPU-or-single-GPU constraint -- this is intentionally not a
genomics foundation model.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RelativePositionBias(nn.Module):
    """
    Learned scalar bias added to attention logits, indexed by relative
    bin distance |i - j|, shared across all heads of a layer for
    simplicity (a per-head version is a straightforward extension if
    you want one head to specialize in short-range and another in
    long-range, but isn't necessary for a first comparison).
    """

    def __init__(self, n_bins: int):
        super().__init__()
        # one learnable scalar for every possible relative distance, 0..n_bins-1
        self.bias = nn.Parameter(torch.zeros(n_bins))

    def forward(self, n_bins: int) -> torch.Tensor:
        idx = torch.arange(n_bins, device=self.bias.device)
        dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()  # (L, L)
        return self.bias[dist]  # (L, L)


class RelPosTransformerLayer(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_bins: int, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.rel_bias = RelativePositionBias(n_bins)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)              # each (B, L, n_heads, head_dim)
        q = q.transpose(1, 2)                     # (B, n_heads, L, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, n_heads, L, L)
        attn = attn + self.rel_bias(L).unsqueeze(0).unsqueeze(0)  # broadcast over batch, heads
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = attn @ v                              # (B, n_heads, L, head_dim)
        out = out.transpose(1, 2).reshape(B, L, D)   # (B, L, D)
        out = self.out_proj(out)
        x = x + self.dropout(out)

        h2 = self.norm2(x)
        x = x + self.dropout(self.ff(h2))
        return x


class TransformerHiCModel(nn.Module):
    """
    tracks: (B, n_tracks, n_bins) -> matrix: (B, n_bins, n_bins)
    """

    def __init__(
        self,
        n_tracks: int = 3,
        n_bins: int = 100,
        dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.1,
        refine_hidden: int = 48,
    ):
        super().__init__()
        self.n_bins = n_bins

        self.input_proj = nn.Linear(n_tracks, dim)
        self.layers = nn.ModuleList(
            [
                RelPosTransformerLayer(dim, n_heads, n_bins, ff_mult, dropout)
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(dim)

        # same outer-product-with-interactions expansion as the other two models
        pair_dim = dim * 4
        self.pair_refine = nn.Sequential(
            nn.Conv2d(pair_dim, refine_hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(refine_hidden, refine_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(refine_hidden),
            nn.GELU(),
            nn.Conv2d(refine_hidden, 1, kernel_size=1),
        )
        nn.init.normal_(self.pair_refine[-1].weight, mean=0.0, std=0.5)
        nn.init.zeros_(self.pair_refine[-1].bias)

    def forward(self, tracks: torch.Tensor) -> torch.Tensor:
        B, n_tracks, L = tracks.shape
        assert L == self.n_bins, (
            f"Model was built for n_bins={self.n_bins} but got input length {L}. "
            f"The RelativePositionBias table is sized for a fixed n_bins; "
            f"changing window-size/bin-size requires rebuilding the model."
        )

        x = tracks.transpose(1, 2)        # (B, L, n_tracks)
        x = self.input_proj(x)            # (B, L, dim)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)            # (B, L, dim)

        emb_i = x.unsqueeze(2).expand(-1, -1, L, -1)   # (B, L, L, dim) -- i varies along dim1
        emb_j = x.unsqueeze(1).expand(-1, L, -1, -1)   # (B, L, L, dim) -- j varies along dim2
        pair = torch.cat(
            [emb_i, emb_j, emb_i * emb_j, torch.abs(emb_i - emb_j)], dim=-1
        )  # (B, L, L, 4*dim)
        pair = pair.permute(0, 3, 1, 2)   # (B, 4*dim, L, L) for Conv2d

        out = self.pair_refine(pair).squeeze(1)   # (B, L, L)
        # Linear output (no activation): predicts O/E deviations (signed).
        # History: tried sigmoid*7 then softplus; both removed when switching
        # to O/E targets which require signed outputs (mean ~0 per distance band).
        out = 0.5 * (out + out.transpose(1, 2))
        return out