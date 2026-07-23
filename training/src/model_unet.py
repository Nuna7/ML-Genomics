from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=pad),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DilatedConvBlock1D(nn.Module):
    """Same residual dilated block as in model_cnn.py, duplicated here
    intentionally (not imported) so this file has no cross-dependency on
    model_cnn.py -- the two models should be swappable/deletable
    independently when comparing architectures."""

    def __init__(self, channels: int, dilation: int, kernel_size: int = 3):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.norm = nn.BatchNorm1d(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.norm(self.conv(x)))


class UNetHiCModel(nn.Module):
    """
    tracks: (B, n_tracks, n_bins) -> matrix: (B, n_bins, n_bins)

    n_bins should ideally be divisible by 4 (two downsampling steps) for
    clean up/downsampling without needing crop/pad reconciliation. 100 is
    divisible by 4, so the default window/bin-size choice (1Mb / 10kb)
    works without modification.
    """

    def __init__(
        self,
        n_tracks: int = 3,
        n_bins: int = 100,
        hidden_dim: int = 48,
        n_dilated_blocks: int = 5,
        max_dilation: int = 16,
    ):
        super().__init__()
        self.n_bins = n_bins
        if n_bins % 4 != 0:
            raise ValueError(
                f"UNetHiCModel uses 2 downsampling steps (factor 4 total) and "
                f"got n_bins={n_bins}, which is not divisible by 4. Either pick "
                f"a window-size/bin-size combination where n_bins % 4 == 0, "
                f"or modify the up/downsampling depth in this model."
            )

        # 1D encoder (same structure as CNN model's encoder)
        self.stem = nn.Sequential(
            nn.Conv1d(n_tracks, hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        dilations = [min(2 ** i, max_dilation) for i in range(n_dilated_blocks)]
        self.dilated_blocks = nn.ModuleList(
            [DilatedConvBlock1D(hidden_dim, d) for d in dilations]
        )

        pair_dim = hidden_dim * 4  # [emb_i, emb_j, emb_i*emb_j, |emb_i-emb_j|]
        base = hidden_dim

        # Encoder path
        self.enc0 = ConvBNAct(pair_dim, base)
        self.enc1 = ConvBNAct(base, base * 2)
        self.enc2 = ConvBNAct(base * 2, base * 4)

        self.pool = nn.MaxPool2d(2)

        # Decoder path
        self.up1 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec1 = ConvBNAct(base * 4, base * 2)  # base*2 (skip) + base*2 (up) -> base*2

        self.up0 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec0 = ConvBNAct(base * 2, base)      # base (skip) + base (up) -> base

        self.head = nn.Conv2d(base, 1, kernel_size=1)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.5)
        nn.init.zeros_(self.head.bias)

    def forward(self, tracks: torch.Tensor) -> torch.Tensor:
        B, _, L = tracks.shape
        assert L == self.n_bins, (
            f"Model was built for n_bins={self.n_bins} but got input length {L}."
        )

        x = self.stem(tracks)
        for block in self.dilated_blocks:
            x = block(x)

        emb_i = x.unsqueeze(3).expand(-1, -1, -1, L)
        emb_j = x.unsqueeze(2).expand(-1, -1, L, -1)
        pair = torch.cat(
            [emb_i, emb_j, emb_i * emb_j, torch.abs(emb_i - emb_j)], dim=1
        )  # (B, 4H, L, L)

        e0 = self.enc0(pair)              # (B, base,   L,    L)
        e1 = self.enc1(self.pool(e0))     # (B, base*2, L/2,  L/2)
        e2 = self.enc2(self.pool(e1))     # (B, base*4, L/4,  L/4)

        d1 = self.up1(e2)                 # (B, base*2, L/2, L/2)
        d1 = F.interpolate(d1, size=e1.shape[-2:], mode="nearest") if d1.shape[-2:] != e1.shape[-2:] else d1
        d1 = self.dec1(torch.cat([d1, e1], dim=1))  # (B, base*2, L/2, L/2)

        d0 = self.up0(d1)                 # (B, base, L, L)
        d0 = F.interpolate(d0, size=e0.shape[-2:], mode="nearest") if d0.shape[-2:] != e0.shape[-2:] else d0
        d0 = self.dec0(torch.cat([d0, e0], dim=1))  # (B, base, L, L)

        out = self.head(d0).squeeze(1)    # (B, L, L)
        out = 0.5 * (out + out.transpose(1, 2))
        return out