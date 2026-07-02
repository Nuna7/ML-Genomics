"""
model_transformer.py
---------------------
Architecture 3: Transformer encoder over the 1D bin sequence (with a
relative-position attention bias) -> same outer-product expansion as the
other two models -> light 2D refinement.
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
        self.pos_embed = nn.Parameter(
            torch.randn(1, n_bins, dim) * 0.02
        )
        self.layers = nn.ModuleList(
            [
                RelPosTransformerLayer(dim, n_heads, n_bins, ff_mult, dropout)
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(dim)
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
        x = x + self.pos_embed
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
        out = torch.nn.functional.softplus(out)
        out = 0.5 * (out + out.transpose(1, 2))
        return out