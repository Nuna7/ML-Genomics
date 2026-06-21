"""
dataset.py
----------
torch.utils.data.Dataset that reads a manifest CSV (produced by
make_windows.py) and, for each row, fetches:
  - 3 x 1D signal tracks (CTCF, H3K27ac, DNase) over the window -> input
  - 1 x 2D Hi-C contact matrix over the same window -> target

Design choices and why:

1. Targets are log1p-transformed, not raw counts. Hi-C contact counts are
   extremely heavy-tailed (the diagonal can be 100-1000x the off-diagonal
   background), so training directly on raw counts means the loss is
   dominated by a handful of diagonal pixels and the model never learns
   useful structure further out. log1p is the standard transform used in
   this literature (Akita etc).

2. Targets are NOT separately rescaled to [0, 1] after log1p. log1p output
   is already a reasonably bounded, well-behaved range for these inputs
   (the visualization script's own colorbar uses log1p with vmax around
   2-3), so an extra min-max step per-window would actually make windows
   numerically incomparable to each other and is intentionally skipped.

3. Inputs (the 1D tracks) ARE z-score normalized per-track using
   normalization STATISTICS COMPUTED ONLY FROM THE TRAINING SPLIT. This
   matters: if you compute mean/std from train+val+test combined and then
   apply it everywhere, information about the val/test distribution leaks
   into the normalization the model is trained under. The Normalizer
   class below enforces this by requiring you to .fit() on a train-only
   dataset and then .apply() to all splits.

4. This Dataset does NOT cache fetched arrays to disk by default --
   re-fetching from local bigWig/.hic files each epoch is the bottleneck
   in practice for CPU training, see CachingHiCDataset below for an
   opt-in on-disk cache that trades disk space for epoch speed.

5. Failure mode handling: if a window's coordinates are invalid for a
   given file (e.g. too close to a chromosome end, or the .hic file truly
   has no data there), this raises rather than silently returning zeros.
   A model trained on silently-zeroed windows will look like it's learning
   "Hi-C is mostly zero" which is true but useless. If you hit this in
   practice, the right fix is to filter bad windows out of the manifest
   *before* training, not to swallow the error here.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).parent))
from genomic_io import hic_matrix, read_bigwig  # noqa: E402


TRACK_NAMES = ["CTCF", "H3K27ac", "DNase"]


@dataclass
class TrackPaths:
    ctcf: Path
    h3k27ac: Path
    dnase: Path
    hic: str  # path or URL, passed straight to hicstraw.HiCFile


class TrackNormalizer:
    """
    Per-track z-score normalizer. MUST be fit on a train-only Dataset and
    then reused (not refit) on val/test Datasets, or you leak distribution
    information across the chromosome split that make_windows.py worked
    hard to keep clean.
    """

    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, dataset: "HiCWindowDataset", max_windows: int = 200) -> "TrackNormalizer":
        if dataset.manifest["split"].nunique() > 1 or (dataset.manifest["split"] != "train").any():
            raise ValueError(
                "TrackNormalizer.fit() was called on a dataset that is not "
                "exclusively 'train' split. Refusing to proceed: fitting "
                "normalization stats on val/test data leaks distribution "
                "information across your chromosome holdout. Pass a "
                "train-only HiCWindowDataset."
            )
        n = min(max_windows, len(dataset))
        idx = np.linspace(0, len(dataset) - 1, n).astype(int)
        sums = np.zeros(len(TRACK_NAMES), dtype=np.float64)
        sq_sums = np.zeros(len(TRACK_NAMES), dtype=np.float64)
        count = 0
        for i in idx:
            #sample = dataset[i]
            #tracks = sample["tracks"].numpy()  # (3, n_bins)
            tracks = dataset.get_tracks(i)
            sums += tracks.sum(axis=1)
            sq_sums += (tracks ** 2).sum(axis=1)
            count += tracks.shape[1]
        mean = sums / count
        var = sq_sums / count - mean ** 2
        std = np.sqrt(np.clip(var, 1e-8, None))
        self.mean_ = mean.astype(np.float32)
        self.std_ = std.astype(np.float32)
        return self

    def apply(self, tracks: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("TrackNormalizer.apply() called before .fit()")
        return (tracks - self.mean_[:, None]) / self.std_[:, None]

    def state_dict(self) -> dict:
        return {"mean": self.mean_.tolist(), "std": self.std_.tolist()}

    @classmethod
    def from_state_dict(cls, d: dict) -> "TrackNormalizer":
        obj = cls()
        obj.mean_ = np.array(d["mean"], dtype=np.float32)
        obj.std_ = np.array(d["std"], dtype=np.float32)
        return obj


class HiCWindowDataset(Dataset):
    def __init__(
        self,
        manifest_csv: str | Path,
        split: str,
        tracks: TrackPaths,
        normalizer: Optional[TrackNormalizer] = None,
        clip_pct: float = 99.0,
        cache_dir: str | Path | None = None,
    ):
        full = pd.read_csv(manifest_csv)
        if split not in set(full["split"]):
            raise ValueError(
                f"split='{split}' not found in manifest. Available splits: "
                f"{sorted(full['split'].unique())}"
            )
        self.manifest = full[full["split"] == split].reset_index(drop=True)
        self.tracks = tracks
        self.normalizer = normalizer
        self.clip_pct = clip_pct

        bin_sizes = self.manifest["bin_size"].unique()
        n_bins_vals = self.manifest["n_bins"].unique()
        if len(bin_sizes) > 1 or len(n_bins_vals) > 1:
            raise ValueError(
                "Manifest contains rows with inconsistent bin_size/n_bins "
                "within a single split. This Dataset assumes one fixed "
                "resolution per split (and ideally across all splits, so "
                "model output shape doesn't change between train/val/test)."
            )

        self.cache_dir = None
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int):
        if self.cache_dir is not None:
            cache_file = self._cache_file(idx)

            if cache_file.exists():
                data = np.load(cache_file)

                tracks = data["tracks"]
                target = data["target"]

                if self.normalizer is not None:
                    tracks = self.normalizer.apply(tracks)

                row = self.manifest.iloc[idx]

                return {
                    "tracks": torch.from_numpy(tracks).float(),
                    "target": torch.from_numpy(target).float(),
                    "chrom": row["chrom"],
                    "start": int(row["start"]),
                    "end": int(row["end"]),
                }

        row = self.manifest.iloc[idx]

        chrom = row["chrom"]
        start = int(row["start"])
        end = int(row["end"])
        bin_size = int(row["bin_size"])

        tracks = self.get_tracks(idx)

        mat, actual_bin = hic_matrix(
            self.tracks.hic,
            chrom,
            start,
            end,
            bin_size,
        )

        if actual_bin != bin_size:
            raise RuntimeError(
                f"Requested bin_size={bin_size}, got {actual_bin}"
            )

        target = np.log1p(mat).astype(np.float32)

        if self.cache_dir is not None:
            np.savez_compressed(
                self._cache_file(idx),
                tracks=tracks.astype(np.float32),
                target=target,
            )

        if self.normalizer is not None:
            tracks = self.normalizer.apply(tracks)

        return {
            "tracks": torch.from_numpy(tracks).float(),
            "target": torch.from_numpy(target).float(),
            "chrom": chrom,
            "start": start,
            "end": end,
        }
    
    def _cache_file(self, idx: int) -> Path:
        row = self.manifest.iloc[idx]

        chrom = row["chrom"]
        start = int(row["start"])
        end = int(row["end"])

        return self.cache_dir / f"{chrom}_{start}_{end}.npz"

    def get_tracks(self, idx):
        row = self.manifest.iloc[idx]

        chrom = row["chrom"]
        start = int(row["start"])
        end = int(row["end"])
        n_bins = int(row["n_bins"])

        ctcf = read_bigwig(
            self.tracks.ctcf,
            chrom,
            start,
            end,
            n_bins,
            self.clip_pct,
        )
        h3k = read_bigwig(
            self.tracks.h3k27ac,
            chrom,
            start,
            end,
            n_bins,
            self.clip_pct,
        )
        dnase = read_bigwig(
            self.tracks.dnase,
            chrom,
            start,
            end,
            n_bins,
            self.clip_pct,
        )

        tracks = np.stack([ctcf, h3k, dnase], axis=0)
        return tracks