"""
make_windows.py
----------------
Generates the list of (chrom, start, end) windows that will become
training/validation/test examples, and writes them to a manifest CSV.

Hi-C contact frequency and 1D signal tracks are spatially autocorrelated --
a window starting at chr8:1,000,000 is highly similar to one starting at
chr8:1,010,000. If you split *windows* randomly into train/val/test, the
model can partially memorize neighboring training windows when it sees a
"held out" window next door, and your validation/test metrics will look
much better than the model actually generalizes.

Usage:
    python make_windows.py \
        --train-chroms chr1 chr2 chr3 chr8 \
        --val-chroms chr10 \
        --test-chroms chr17 \
        --window-size 1000000 \
        --stride 1000000 \
        --bin-size 10000 \
        --out manifest.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from genome_constants import chrom_sizes_hg38  # noqa: E402


def build_windows(chroms: list[str], window_size: int, stride: int) -> list[tuple[str, int, int]]:
    sizes = chrom_sizes_hg38()
    windows = []
    for chrom in chroms:
        if chrom not in sizes:
            raise ValueError(f"Unknown chromosome '{chrom}'. Known: {sorted(sizes)}")
        chrom_len = sizes[chrom]
        start = 0
        while start + window_size <= chrom_len:
            windows.append((chrom, start, start + window_size))
            start += stride
    return windows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-chroms", nargs="+", default=["chr1", "chr2", "chr3", "chr8"],
                    help="Chromosomes used ONLY for training.")
    p.add_argument("--val-chroms", nargs="+", default=["chr10"],
                    help="Chromosomes used ONLY for validation (model selection / early stopping).")
    p.add_argument("--test-chroms", nargs="+", default=["chr17"],
                    help="Chromosomes used ONLY for final test reporting. Touch this as little as possible.")
    p.add_argument("--window-size", type=int, default=1_000_000,
                    help="Input window size in bp. Default 1Mb.")
    p.add_argument("--stride", type=int, default=1_000_000,
                    help="Stride between window starts in bp. Equal to window-size means "
                         "non-overlapping windows (recommended to avoid near-duplicate "
                         "examples within a split, which inflates apparent dataset size "
                         "without adding real information).")
    p.add_argument("--bin-size", type=int, default=10_000,
                    help="Hi-C / signal bin size in bp. Must evenly divide window-size.")
    p.add_argument("--out", default="manifest.csv")
    args = p.parse_args()

    if args.window_size % args.bin_size != 0:
        print(
            f"ERROR: window-size ({args.window_size}) must be evenly divisible by "
            f"bin-size ({args.bin_size}), or the output matrix won't be square with "
            f"a clean number of bins.",
            file=sys.stderr,
        )
        return 1

    all_chrom_sets = {
        "train": set(args.train_chroms),
        "val": set(args.val_chroms),
        "test": set(args.test_chroms),
    }

    seen = {}
    for split_name, chrom_set in all_chrom_sets.items():
        for c in chrom_set:
            if c in seen:
                print(
                    f"ERROR: chromosome '{c}' appears in both '{seen[c]}' and "
                    f"'{split_name}' splits. A chromosome may only belong to ONE "
                    f"split. Fix --train-chroms / --val-chroms / --test-chroms.",
                    file=sys.stderr,
                )
                return 1
            seen[c] = split_name

    rows = []
    n_bins = args.window_size // args.bin_size
    for split_name, chrom_list in (
        ("train", args.train_chroms),
        ("val", args.val_chroms),
        ("test", args.test_chroms),
    ):
        windows = build_windows(chrom_list, args.window_size, args.stride)
        for chrom, start, end in windows:
            rows.append({
                "split": split_name,
                "chrom": chrom,
                "start": start,
                "end": end,
                "bin_size": args.bin_size,
                "n_bins": n_bins,
            })

    out_path = Path(args.out)
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["split", "chrom", "start", "end", "bin_size", "n_bins"])
        writer.writeheader()
        writer.writerows(rows)

    counts = {}
    for r in rows:
        counts[r["split"]] = counts.get(r["split"], 0) + 1
    print(f"Wrote {len(rows)} windows to {out_path}")
    for split_name in ("train", "val", "test"):
        chroms_str = ",".join(all_chrom_sets[split_name]) or "(none)"
        print(f"  {split_name:5s}: {counts.get(split_name, 0):5d} windows  "
              f"(chroms: {chroms_str})")
    if counts.get("val", 0) == 0 or counts.get("test", 0) == 0:
        print(
            "WARNING: val or test split is empty. You will not be able to do "
            "early stopping or final evaluation.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())