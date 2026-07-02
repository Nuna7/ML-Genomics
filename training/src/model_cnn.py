"""
model_cnn.py
------------
Architecture 1: Dilated 1D CNN encoder -> outer-product expansion -> 2D conv refinement.
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
        out = torch.nn.functional.softplus(out)
        out = 0.5 * (out + out.transpose(1, 2))      # enforce symmetry
        return out