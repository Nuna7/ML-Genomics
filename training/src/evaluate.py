"""
evaluate.py
-----------

Usage:
    python evaluate.py \
        --checkpoint runs/cnn_run1/best_model.pt \
        --normalizer runs/cnn_run1/normalizer.json \
        --manifest manifest.csv \
        --ctcf-bw /path/to/ctcf.bigWig \
        --h3k27ac-bw /path/to/h3k27ac.bigWig \
        --dnase-bw /path/to/dnase.bigWig \
        --hic-source /path/to/file.hic \
        --out runs/cnn_run1/test_results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from dataset import HiCWindowDataset, TrackNormalizer, TrackPaths  # noqa: E402
from training.src.metrics import evaluate_batch, mse_loss  # noqa: E402
from train import build_model  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--normalizer", required=True, type=Path,
                    help="normalizer.json written by train.py for THIS SAME model run "
                         "-- using a normalizer fit during a different run risks subtly "
                         "different train statistics being applied at test time.")
    p.add_argument("--manifest", required=True)
    p.add_argument("--ctcf-bw", required=True, type=Path)
    p.add_argument("--h3k27ac-bw", required=True, type=Path)
    p.add_argument("--dnase-bw", required=True, type=Path)
    p.add_argument("--hic-source", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model_name = ckpt["model_name"]
    n_bins = ckpt["n_bins"]

    print(f"Loaded checkpoint: model={model_name}  n_bins={n_bins}  "
          f"trained_epoch={ckpt['epoch']}  val_loss_at_save={ckpt['val_loss']:.4f}")

    with open(args.normalizer) as fh:
        normalizer = TrackNormalizer.from_state_dict(json.load(fh))

    tracks = TrackPaths(
        ctcf=args.ctcf_bw, h3k27ac=args.h3k27ac_bw, dnase=args.dnase_bw,
        hic=args.hic_source,
    )

    test_ds = HiCWindowDataset(args.manifest, split="test", tracks=tracks, normalizer=normalizer)
    if len(test_ds) == 0:
        print("ERROR: test split is empty in this manifest.", file=sys.stderr)
        return 1
    print(f"Test set: {len(test_ds)} windows from chromosomes "
          f"{sorted(test_ds.manifest['chrom'].unique())}")

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = build_model(model_name, n_bins=n_bins).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    total_loss = 0.0
    n_batches = 0
    agg = {"mse": 0.0, "distance_stratified_mse": 0.0, "stratum_adjusted_corr": 0.0}
    with torch.no_grad():
        for batch in test_loader:
            t = batch["tracks"].to(args.device)
            y = batch["target"].to(args.device)
            pred = model(t)
            loss = mse_loss(pred, y)
            total_loss += loss.item()
            n_batches += 1
            m = evaluate_batch(pred, y)
            for k in agg:
                agg[k] += m[k]

    for k in agg:
        agg[k] /= max(1, n_batches)
    test_loss = total_loss / max(1, n_batches)

    results = {
        "model_name": model_name,
        "checkpoint_path": str(args.checkpoint),
        "checkpoint_trained_epoch": ckpt["epoch"],
        "n_test_windows": len(test_ds),
        "test_chroms": sorted(test_ds.manifest["chrom"].unique().tolist()),
        "test_loss": test_loss,
        "test_mse": agg["mse"],
        "test_distance_stratified_mse": agg["distance_stratified_mse"],
        "test_stratum_adjusted_corr": agg["stratum_adjusted_corr"],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)

    print()
    print(json.dumps(results, indent=2))
    print()
    print(f"Wrote final test results to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())