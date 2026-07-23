"""
model_cnn.py
------------
Architecture 1: Dilated 1D CNN encoder -> outer-product expansion -> 2D conv refinement.

This is the architecture family used by Akita (Fudenberg et al.) and is
the most established approach for this exact task, so it's a sensible
baseline to measure the other two against.

The core idea, in three steps:

1. ENCODE the 1D tracks with a stack of dilated 1D convolutions. Dilation
   (not just stride/pooling) is used specifically because Hi-C structure
   (TADs, loops) operates at scales from ~50kb to ~1Mb, i.e. spanning 2+
   orders of magnitude in the same window. A plain CNN with small fixed
   receptive fields would need an impractically deep stack to "see" a
   1Mb-scale TAD; exponentially increasing dilation gets a wide effective
   receptive field with a shallow network.

2. EXPAND from 1D (length L) to 2D (L x L) via an outer-product-style
   construction: for every pair of bin embeddings (i, j), concatenate
   them (and add their elementwise product and absolute difference as
   extra features) to form a per-pixel feature vector. This is the
   standard trick for turning a sequence encoder into a pairwise/matrix
   predictor -- it's essentially what Akita and most "1D track -> 2D
   contact map" models do, instead of trying to learn a fully connected
   layer from L features to L*L outputs (which would have a parameter
   count that's quadratic in sequence length and wouldn't share weights
   across genomic distance the way Hi-C structure does).

3. REFINE with a small stack of 2D convolutions on the expanded grid, to
   let the model smooth/sharpen local pixel neighborhoods using context
   from nearby pixels, not just the two original 1D positions.

4. SYMMETRIZE the output (0.5 * (M + M^T)). Hi-C contact matrices are
   physically symmetric (contact frequency between bin i and bin j is the
   same as between bin j and bin i), and explicitly enforcing this is a
   useful inductive bias rather than hoping the network learns it from
   data alone with limited training examples.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DilatedConvBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, kernel_size: int = 3):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.norm = nn.BatchNorm1d(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.norm(self.conv(x)))  # residual


class CNNHiCModel(nn.Module):
    """
    tracks: (B, n_tracks, n_bins) -> matrix: (B, n_bins, n_bins)
    """

    def __init__(
        self,
        n_tracks: int = 3,
        n_bins: int = 100,
        hidden_dim: int = 64,
        n_dilated_blocks: int = 6,
        max_dilation: int = 32,
    ):
        super().__init__()
        self.n_bins = n_bins

        self.stem = nn.Sequential(
            nn.Conv1d(n_tracks, hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )

        dilations = [2 ** i for i in range(n_dilated_blocks)]
        dilations = [min(d, max_dilation) for d in dilations]
        self.dilated_blocks = nn.ModuleList(
            [DilatedConvBlock(hidden_dim, d) for d in dilations]
        )

        # pairwise feature: [emb_i, emb_j, emb_i * emb_j, |emb_i - emb_j|]
        pair_dim = hidden_dim * 4
        self.pair_refine = nn.Sequential(
            nn.Conv2d(pair_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            # Removed: second BatchNorm that was compressing variance to std~0.02
            # before the final conv, making sigmoid*7 initialization collapse to
            # near-constant output. One BN in a 3-layer refinement stack is enough.
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )
        nn.init.normal_(self.pair_refine[-1].weight, mean=0.0, std=0.5)
        nn.init.zeros_(self.pair_refine[-1].bias)

    def forward(self, tracks: torch.Tensor) -> torch.Tensor:
        B, _, L = tracks.shape
        assert L == self.n_bins, (
            f"Model was built for n_bins={self.n_bins} but got input length {L}. "
            f"This model's pair-expansion step assumes a fixed window size; "
            f"if you change resolution/window-size you must rebuild the model."
        )

        x = self.stem(tracks)                      # (B, H, L)
        for block in self.dilated_blocks:
            x = block(x)                            # (B, H, L)

        emb_i = x.unsqueeze(3).expand(-1, -1, -1, L)   # (B, H, L, L) -- broadcast along j
        emb_j = x.unsqueeze(2).expand(-1, -1, L, -1)   # (B, H, L, L) -- broadcast along i
        pair = torch.cat(
            [emb_i, emb_j, emb_i * emb_j, torch.abs(emb_i - emb_j)], dim=1
        )  # (B, 4H, L, L)

        out = self.pair_refine(pair).squeeze(1)     # (B, L, L)
        # Linear output (no activation): targets are O/E deviations (signed,
        # mean ~0 per distance band, range ~[-1.5, +2.5]). A non-negative
        # activation like softplus would be wrong here since O/E values can
        # and should be negative (below-average contact at a given distance).
        out = 0.5 * (out + out.transpose(1, 2))      # enforce symmetry
        return out