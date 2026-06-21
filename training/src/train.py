# """
# train.py
# --------
# Trains a single model (CNN, UNet, or Transformer) on the windows defined
# in a manifest CSV, with:
#   - normalizer fit ONLY on the train split (see dataset.py for why)
#   - per-epoch evaluation on the val split (used for model selection /
#     early stopping -- the test split is NOT touched by this script at all,
#     see evaluate.py for final test-set reporting)
#   - checkpointing the best val-loss model
#   - a training log (CSV) of every epoch's train/val metrics, which
#     compare_models.py later reads to build the comparison report

# IMPORTANT: different architectures may need different epoch budgets to
# reach a comparable point in training (see model_transformer.py's
# docstring and the empirical convergence-speed check described in
# project notes -- the Transformer's near-uniform attention at
# initialization means it converges more slowly per-step than the CNN or
# UNet). This script does NOT silently equalize epoch counts across models;
# you explicitly pass --epochs per run, and the comparison report
# (compare_models.py) plots full per-epoch curves specifically so you can
# see whether each model has actually plateaued, rather than comparing
# models at an arbitrary fixed epoch count that may be unfair to whichever
# architecture converges slower.

# Usage:
#     python train.py \
#         --manifest manifest.csv \
#         --model cnn \
#         --ctcf-bw /path/to/ctcf.bigWig \
#         --h3k27ac-bw /path/to/h3k27ac.bigWig \
#         --dnase-bw /path/to/dnase.bigWig \
#         --hic-source /path/to/file.hic \
#         --epochs 40 \
#         --batch-size 8 \
#         --lr 1e-3 \
#         --out-dir runs/cnn_run1
# """
# from __future__ import annotations

# import argparse
# import csv
# import json
# import sys
# import time
# from pathlib import Path

# import torch
# from torch.utils.data import DataLoader

# sys.path.insert(0, str(Path(__file__).parent))
# from dataset import HiCWindowDataset, TrackNormalizer, TrackPaths  # noqa: E402
# from metrics import mse_loss, evaluate_batch  # noqa: E402
# from model_cnn import CNNHiCModel  # noqa: E402
# from model_unet import UNetHiCModel  # noqa: E402
# from model_transformer import TransformerHiCModel  # noqa: E402


# MODEL_REGISTRY = {
#     "cnn": CNNHiCModel,
#     "unet": UNetHiCModel,
#     "transformer": TransformerHiCModel,
# }


# def build_model(name: str, n_bins: int, n_tracks: int = 3) -> torch.nn.Module:
#     if name not in MODEL_REGISTRY:
#         raise ValueError(f"Unknown model '{name}'. Choose from {list(MODEL_REGISTRY)}")
#     return MODEL_REGISTRY[name](n_tracks=n_tracks, n_bins=n_bins)


# def run_epoch(model, loader, optimizer, device, train: bool) -> dict:
#     model.train(mode=train)
#     total_loss = 0.0
#     n_batches = 0
#     agg_metrics = {"mse": 0.0, "distance_stratified_mse": 0.0, "stratum_adjusted_corr": 0.0}

#     ctx = torch.enable_grad() if train else torch.no_grad()
#     with ctx:
#         for batch in loader:
#             tracks = batch["tracks"].to(device)
#             target = batch["target"].to(device)

#             if train:
#                 optimizer.zero_grad()
#             pred = model(tracks)
#             loss = mse_loss(pred, target)
#             if train:
#                 loss.backward()
#                 optimizer.step()

#             total_loss += loss.item()
#             n_batches += 1
#             batch_metrics = evaluate_batch(pred, target)
#             for k in agg_metrics:
#                 agg_metrics[k] += batch_metrics[k]

#     for k in agg_metrics:
#         agg_metrics[k] /= max(1, n_batches)
#     agg_metrics["loss"] = total_loss / max(1, n_batches)
#     return agg_metrics


# def main() -> int:
#     p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
#     p.add_argument("--manifest", required=True)
#     p.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
#     p.add_argument("--ctcf-bw", required=True, type=Path)
#     p.add_argument("--h3k27ac-bw", required=True, type=Path)
#     p.add_argument("--dnase-bw", required=True, type=Path)
#     p.add_argument("--hic-source", required=True,
#                     help="Local path or URL to a .hic file, passed to hicstraw.HiCFile")
#     p.add_argument("--epochs", type=int, default=40)
#     p.add_argument("--batch-size", type=int, default=8)
#     p.add_argument("--lr", type=float, default=1e-3)
#     p.add_argument("--weight-decay", type=float, default=1e-5)
#     p.add_argument("--num-workers", type=int, default=0,
#                     help="DataLoader workers. NOTE: each worker opens its own "
#                          "bigWig/.hic file handles; if you hit file-handle or "
#                          "memory issues on a constrained machine, set this to 0 "
#                          "(main-process loading, slower but safest) rather than "
#                          "increasing it blindly.")
#     p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
#     p.add_argument("--out-dir", required=True, type=Path)
#     p.add_argument("--seed", type=int, default=42)
#     args = p.parse_args()

#     if args.device == "cuda" and not torch.cuda.is_available():
#         print("ERROR: --device cuda requested but torch.cuda.is_available() is False. "
#               "Falling back to cpu would silently give you a much slower run than you "
#               "expect, so refusing instead -- fix your CUDA setup or pass --device cpu "
#               "explicitly.", file=sys.stderr)
#         return 1

#     torch.manual_seed(args.seed)
#     args.out_dir.mkdir(parents=True, exist_ok=True)

#     tracks = TrackPaths(
#         ctcf=args.ctcf_bw, h3k27ac=args.h3k27ac_bw, dnase=args.dnase_bw,
#         hic=args.hic_source,
#     )

#     print(f"[1/5] Building train dataset...")
#     #train_ds_raw = HiCWindowDataset(args.manifest, split="train", tracks=tracks, normalizer=None)
#     train_ds_raw = HiCWindowDataset(
#         args.manifest,
#         split="train",
#         tracks=tracks,
#         normalizer=None,
#         cache_dir=args.out_dir / "cache/train",
#     )
#     if len(train_ds_raw) == 0:
#         print("ERROR: train split is empty. Check your manifest.", file=sys.stderr)
#         return 1

#     print(f"[2/5] Fitting track normalizer on train split ONLY...")
#     normalizer = TrackNormalizer().fit(train_ds_raw)
#     with open(args.out_dir / "normalizer.json", "w") as fh:
#         json.dump(normalizer.state_dict(), fh, indent=2)

#     print(f"[3/5] Building normalized train/val datasets...")
#     #train_ds = HiCWindowDataset(args.manifest, split="train", tracks=tracks, normalizer=normalizer)
#     #val_ds = HiCWindowDataset(args.manifest, split="val", tracks=tracks, normalizer=normalizer)

#     train_ds = HiCWindowDataset(
#         args.manifest,
#         split="train",
#         tracks=tracks,
#         normalizer=normalizer,
#         cache_dir=args.out_dir / "cache/train",
#     )

#     val_ds = HiCWindowDataset(
#         args.manifest,
#         split="val",
#         tracks=tracks,
#         normalizer=normalizer,
#         cache_dir=args.out_dir / "cache/val",
#     )
#     print(f"      train: {len(train_ds)} windows   val: {len(val_ds)} windows")
#     if len(val_ds) == 0:
#         print("WARNING: val split is empty -- you will have no signal for model "
#               "selection / early stopping. Check your manifest.", file=sys.stderr)

#     n_bins = int(train_ds.manifest["n_bins"].iloc[0])

#     train_loader = DataLoader(
#         train_ds, batch_size=args.batch_size, shuffle=True,
#         num_workers=args.num_workers,
#     )
#     val_loader = DataLoader(
#         val_ds, batch_size=args.batch_size, shuffle=False,
#         num_workers=args.num_workers,
#     )

#     print(f"[4/5] Building model '{args.model}' (n_bins={n_bins})...")
#     model = build_model(args.model, n_bins=n_bins).to(args.device)
#     n_params = sum(pp.numel() for pp in model.parameters())
#     print(f"      {n_params:,} parameters")

#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

#     log_path = args.out_dir / "training_log.csv"
#     best_val_loss = float("inf")
#     best_ckpt_path = args.out_dir / "best_model.pt"

#     print(f"[5/5] Training for {args.epochs} epochs...")
#     with open(log_path, "w", newline="") as log_fh:
#         fieldnames = [
#             "epoch", "train_loss", "train_mse", "train_distance_stratified_mse",
#             "train_stratum_adjusted_corr", "val_loss", "val_mse",
#             "val_distance_stratified_mse", "val_stratum_adjusted_corr",
#             "epoch_seconds",
#         ]
#         writer = csv.DictWriter(log_fh, fieldnames=fieldnames)
#         writer.writeheader()

#         for epoch in range(1, args.epochs + 1):
#             t0 = time.time()
#             train_metrics = run_epoch(model, train_loader, optimizer, args.device, train=True)
#             val_metrics = run_epoch(model, val_loader, optimizer, args.device, train=False) if len(val_ds) > 0 else {
#                 "loss": float("nan"), "mse": float("nan"),
#                 "distance_stratified_mse": float("nan"), "stratum_adjusted_corr": float("nan"),
#             }
#             dt = time.time() - t0

#             row = {
#                 "epoch": epoch,
#                 "train_loss": train_metrics["loss"],
#                 "train_mse": train_metrics["mse"],
#                 "train_distance_stratified_mse": train_metrics["distance_stratified_mse"],
#                 "train_stratum_adjusted_corr": train_metrics["stratum_adjusted_corr"],
#                 "val_loss": val_metrics["loss"],
#                 "val_mse": val_metrics["mse"],
#                 "val_distance_stratified_mse": val_metrics["distance_stratified_mse"],
#                 "val_stratum_adjusted_corr": val_metrics["stratum_adjusted_corr"],
#                 "epoch_seconds": dt,
#             }
#             writer.writerow(row)
#             log_fh.flush()

#             print(
#                 f"  epoch {epoch:3d}/{args.epochs}  "
#                 f"train_loss={train_metrics['loss']:.4f}  "
#                 f"val_loss={val_metrics['loss']:.4f}  "
#                 f"val_corr={val_metrics['stratum_adjusted_corr']:.3f}  "
#                 f"({dt:.1f}s)"
#             )

#             if len(val_ds) > 0 and val_metrics["loss"] < best_val_loss:
#                 best_val_loss = val_metrics["loss"]
#                 torch.save({
#                     "model_state_dict": model.state_dict(),
#                     "model_name": args.model,
#                     "n_bins": n_bins,
#                     "epoch": epoch,
#                     "val_loss": best_val_loss,
#                 }, best_ckpt_path)

#     print(f"\nDone. Best val_loss={best_val_loss:.4f}, checkpoint at {best_ckpt_path}")
#     print(f"Training log at {log_path}")
#     return 0


# if __name__ == "__main__":
#     raise SystemExit(main())

"""
train.py
--------
Trains a single model (CNN, UNet, or Transformer) on the windows defined
in a manifest CSV, with:
  - normalizer fit ONLY on the train split (see dataset.py for why)
  - per-epoch evaluation on the val split (used for model selection /
    early stopping -- the test split is NOT touched by this script at all,
    see evaluate.py for final test-set reporting)
  - checkpointing the best val-loss model
  - a training log (CSV) of every epoch's train/val metrics, which
    compare_models.py later reads to build the comparison report
  - OPTIONAL gradient clipping (--grad-clip-norm) and linear LR warmup
    (--warmup-steps), both off by default so existing CNN/UNet results
    stay comparable unless you explicitly turn them on. See the bottom of
    this docstring for why these two specifically.

IMPORTANT: different architectures may need different epoch budgets to
reach a comparable point in training (see model_transformer.py's
docstring and the empirical convergence-speed check described in
project notes -- the Transformer's near-uniform attention at
initialization means it converges more slowly per-step than the CNN or
UNet). This script does NOT silently equalize epoch counts across models;
you explicitly pass --epochs per run, and the comparison report
(compare_models.py) plots full per-epoch curves specifically so you can
see whether each model has actually plateaued, rather than comparing
models at an arbitrary fixed epoch count that may be unfair to whichever
architecture converges slower.

Usage:
    python train.py \
        --manifest manifest.csv \
        --model cnn \
        --ctcf-bw /path/to/ctcf.bigWig \
        --h3k27ac-bw /path/to/h3k27ac.bigWig \
        --dnase-bw /path/to/dnase.bigWig \
        --hic-source /path/to/file.hic \
        --epochs 40 \
        --batch-size 8 \
        --lr 1e-3 \
        --out-dir runs/cnn_run1

With stabilization tricks (recommended for the Transformer, optional for
CNN/UNet):
    python train.py \
        ... \
        --grad-clip-norm 1.0 \
        --warmup-steps 200 \
        --out-dir runs/transformer_run2

WHY THESE TWO TRICKS SPECIFICALLY (not a generic "try some tricks" list):

1. --grad-clip-norm caps the L2 norm of the gradient before the optimizer
   step. This directly targets a pattern observed in real runs: a
   transformer training log showed val_loss spike to ~9x its neighboring
   epochs' values for exactly one epoch (e.g. 0.30 -> 1.77 -> 0.30) while
   train_loss that same epoch stayed perfectly smooth. The smooth
   train_loss points to an unlucky val batch as the more likely cause
   rather than a genuine weight blowup -- but gradient clipping is cheap
   insurance against the closely related failure mode (an occasional
   large gradient from a single extreme training batch knocking weights
   into a bad region) regardless of which one it actually was, and is
   standard practice for transformer training in general.

2. --warmup-steps linearly ramps the learning rate from 0 up to --lr over
   the given number of OPTIMIZER STEPS (not epochs -- this matters at
   small dataset sizes where one epoch might only be ~100 steps).
   This directly targets a verified property of this codebase's
   TransformerHiCModel: at initialization, its relative-position bias is
   exactly zero and content-based attention is consequently near-uniform
   (measured: max attention weight per row landed around 0.016 vs a
   uniform baseline of 0.01 for 100 positions). Early gradients in this
   regime are less informative than they will be once attention has
   sharpened, so taking large steps on them risks moving the model in a
   bad direction before it has differentiated genomic positions at all.
   Warmup is the standard fix for this specific transformer cold-start
   problem.

Both flags are no-ops if left at their defaults (grad_clip_norm=None,
warmup_steps=0), so CNN/UNet baselines are unaffected unless you opt in.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from dataset import HiCWindowDataset, TrackNormalizer, TrackPaths  # noqa: E402
from metrics import mse_loss, evaluate_batch  # noqa: E402
from model_cnn import CNNHiCModel  # noqa: E402
from model_unet import UNetHiCModel  # noqa: E402
from model_transformer import TransformerHiCModel  # noqa: E402


MODEL_REGISTRY = {
    "cnn": CNNHiCModel,
    "unet": UNetHiCModel,
    "transformer": TransformerHiCModel,
}


def build_model(name: str, n_bins: int, n_tracks: int = 3) -> torch.nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](n_tracks=n_tracks, n_bins=n_bins)


def make_warmup_lr_lambda(warmup_steps: int):
    """
    Returns a function step -> lr_multiplier suitable for
    torch.optim.lr_scheduler.LambdaLR. Linearly ramps the multiplier from
    ~0 to 1.0 over the first `warmup_steps` optimizer steps, then holds at
    1.0 (no decay after warmup -- this is intentionally just warmup, not a
    full schedule, to keep the comparison to the no-warmup baseline as
    close to apples-to-apples as possible: only the first warmup_steps
    steps differ).

    warmup_steps=0 returns a constant multiplier of 1.0 (i.e. a no-op,
    equivalent to not using a scheduler at all).

    NOTE on exact values logged in lr_at_epoch_end: torch's LambdaLR
    applies lr_lambda(0) at scheduler CONSTRUCTION time, before the first
    .step() call -- so after N calls to scheduler.step(), the effective
    multiplier is lr_lambda(N), not lr_lambda(N-1). Concretely, with
    warmup_steps=8 and 4 optimizer steps/epoch, the multiplier at the end
    of epoch 1 is (4+1)/8 = 0.625, not 4/8 = 0.5 as naive hand-arithmetic
    might suggest. This is standard PyTorch behavior, not a bug here --
    documented so the lr_at_epoch_end column in training_log.csv doesn't
    look "off" if you sanity-check it by hand.
    """
    if warmup_steps <= 0:
        return lambda step: 1.0

    def lr_lambda(step: int) -> float:
        if step >= warmup_steps:
            return 1.0
        return (step + 1) / warmup_steps

    return lr_lambda


def run_epoch(model, loader, optimizer, device, train: bool,
               scheduler=None, grad_clip_norm: float | None = None) -> dict:
    """
    scheduler, if provided, is stepped ONCE PER OPTIMIZER STEP (i.e. once
    per training batch), not once per epoch -- warmup needs finer
    granularity than epoch-level stepping, especially since a single
    epoch here is often well under 1000 steps. scheduler is ignored
    entirely when train=False (no LR stepping during validation).

    grad_clip_norm, if provided, clips the L2 norm of all gradients to
    this value via torch.nn.utils.clip_grad_norm_, applied AFTER
    loss.backward() and BEFORE optimizer.step() -- this is the standard
    placement, since clipping needs the actual computed gradients to act
    on, and must happen before the optimizer consumes them.
    """
    model.train(mode=train)
    total_loss = 0.0
    n_batches = 0
    agg_metrics = {"mse": 0.0, "distance_stratified_mse": 0.0, "stratum_adjusted_corr": 0.0}
    last_lr = None

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            tracks = batch["tracks"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)

            if train:
                optimizer.zero_grad()
            pred = model(tracks)
            loss = mse_loss(pred, target)
            if train:
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                    last_lr = scheduler.get_last_lr()[0]

            total_loss += loss.item()
            n_batches += 1
            batch_metrics = evaluate_batch(pred, target)
            for k in agg_metrics:
                agg_metrics[k] += batch_metrics[k]

    for k in agg_metrics:
        agg_metrics[k] /= max(1, n_batches)
    agg_metrics["loss"] = total_loss / max(1, n_batches)
    agg_metrics["last_lr"] = last_lr
    return agg_metrics


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    p.add_argument("--ctcf-bw", required=True, type=Path)
    p.add_argument("--h3k27ac-bw", required=True, type=Path)
    p.add_argument("--dnase-bw", required=True, type=Path)
    p.add_argument("--hic-source", required=True,
                    help="Local path or URL to a .hic file, passed to hicstraw.HiCFile")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--num-workers", type=int, default=0,
                    help="DataLoader workers. NOTE: each worker opens its own "
                         "bigWig/.hic file handles; if you hit file-handle or "
                         "memory issues on a constrained machine, set this to 0 "
                         "(main-process loading, slower but safest) rather than "
                         "increasing it blindly.")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grad-clip-norm", type=float, default=None,
                    help="If set, clips gradient L2 norm to this value each step. "
                         "Off by default. Recommended for the Transformer (try 1.0); "
                         "optional for CNN/UNet.")
    p.add_argument("--warmup-steps", type=int, default=0,
                    help="Number of OPTIMIZER STEPS (not epochs) to linearly ramp "
                         "the LR from ~0 to --lr. 0 (default) disables warmup. "
                         "Recommended for the Transformer (try ~5-10%% of total "
                         "steps, e.g. 200 for a few thousand-step run); CNN/UNet "
                         "generally don't need this.")
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("ERROR: --device cuda requested but torch.cuda.is_available() is False. "
              "Falling back to cpu would silently give you a much slower run than you "
              "expect, so refusing instead -- fix your CUDA setup or pass --device cpu "
              "explicitly.", file=sys.stderr)
        return 1

    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.out_dir / "config.json", "w") as fh:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, fh, indent=2)

    tracks = TrackPaths(
        ctcf=args.ctcf_bw, h3k27ac=args.h3k27ac_bw, dnase=args.dnase_bw,
        hic=args.hic_source,
    )

    print(f"[1/5] Building train dataset...")
    train_ds_raw = HiCWindowDataset(args.manifest, split="train", tracks=tracks, normalizer=None)
    if len(train_ds_raw) == 0:
        print("ERROR: train split is empty. Check your manifest.", file=sys.stderr)
        return 1

    print(f"[2/5] Fitting track normalizer on train split ONLY...")
    normalizer = TrackNormalizer().fit(train_ds_raw)
    with open(args.out_dir / "normalizer.json", "w") as fh:
        json.dump(normalizer.state_dict(), fh, indent=2)

    print(f"[3/5] Building normalized train/val datasets...")
    train_ds = HiCWindowDataset(
        args.manifest,
        split="train",
        tracks=tracks,
        normalizer=normalizer,
        cache_dir=args.out_dir / "cache/train", # KEEP THIS
    )
    val_ds = HiCWindowDataset(
        args.manifest,
        split="val",
        tracks=tracks,
        normalizer=normalizer,
        cache_dir=args.out_dir / "cache/val", # KEEP THIS
    )

    # train_ds = HiCWindowDataset(args.manifest, split="train", tracks=tracks, normalizer=normalizer)
    # val_ds = HiCWindowDataset(args.manifest, split="val", tracks=tracks, normalizer=normalizer)
    print(f"      train: {len(train_ds)} windows   val: {len(val_ds)} windows")
    if len(val_ds) == 0:
        print("WARNING: val split is empty -- you will have no signal for model "
              "selection / early stopping. Check your manifest.", file=sys.stderr)

    n_bins = int(train_ds.manifest["n_bins"].iloc[0])

    pin_memory = args.device == "cuda"

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory
    )

    print(f"[4/5] Building model '{args.model}' (n_bins={n_bins})...")
    model = build_model(args.model, n_bins=n_bins).to(args.device)
    n_params = sum(pp.numel() for pp in model.parameters())
    print(f"      {n_params:,} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    if args.warmup_steps > 0:
        if args.warmup_steps >= total_steps:
            print(
                f"WARNING: --warmup-steps ({args.warmup_steps}) is >= total planned "
                f"optimizer steps for this run ({total_steps} = {steps_per_epoch} "
                f"steps/epoch x {args.epochs} epochs). The LR will still be "
                f"ramping at the end of training and will never reach --lr. "
                f"Consider a smaller --warmup-steps (e.g. {max(1, total_steps // 10)}, "
                f"about 10% of total steps) or more --epochs.",
                file=sys.stderr,
            )
        print(f"      LR warmup: {args.warmup_steps} steps "
              f"({steps_per_epoch} steps/epoch, {total_steps} total planned steps)")
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=make_warmup_lr_lambda(args.warmup_steps)
    )
    if args.grad_clip_norm is not None:
        print(f"      Gradient clipping: L2 norm <= {args.grad_clip_norm}")

    log_path = args.out_dir / "training_log.csv"
    best_val_loss = float("inf")
    best_ckpt_path = args.out_dir / "best_model.pt"

    print(f"[5/5] Training for {args.epochs} epochs...")
    with open(log_path, "w", newline="") as log_fh:
        fieldnames = [
            "epoch", "train_loss", "train_mse", "train_distance_stratified_mse",
            "train_stratum_adjusted_corr", "val_loss", "val_mse",
            "val_distance_stratified_mse", "val_stratum_adjusted_corr",
            "epoch_seconds", "lr_at_epoch_end",
        ]
        writer = csv.DictWriter(log_fh, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_metrics = run_epoch(
                model, train_loader, optimizer, args.device, train=True,
                scheduler=scheduler, grad_clip_norm=args.grad_clip_norm,
            )
            val_metrics = run_epoch(model, val_loader, optimizer, args.device, train=False) if len(val_ds) > 0 else {
                "loss": float("nan"), "mse": float("nan"),
                "distance_stratified_mse": float("nan"), "stratum_adjusted_corr": float("nan"),
            }
            dt = time.time() - t0

            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_mse": train_metrics["mse"],
                "train_distance_stratified_mse": train_metrics["distance_stratified_mse"],
                "train_stratum_adjusted_corr": train_metrics["stratum_adjusted_corr"],
                "val_loss": val_metrics["loss"],
                "val_mse": val_metrics["mse"],
                "val_distance_stratified_mse": val_metrics["distance_stratified_mse"],
                "val_stratum_adjusted_corr": val_metrics["stratum_adjusted_corr"],
                "epoch_seconds": dt,
                "lr_at_epoch_end": train_metrics.get("last_lr"),
            }
            writer.writerow(row)
            log_fh.flush()

            lr_str = f"{train_metrics.get('last_lr'):.2e}" if train_metrics.get("last_lr") is not None else "n/a"
            print(
                f"  epoch {epoch:3d}/{args.epochs}  "
                f"train_loss={train_metrics['loss']:.4f}  "
                f"val_loss={val_metrics['loss']:.4f}  "
                f"val_corr={val_metrics['stratum_adjusted_corr']:.3f}  "
                f"lr={lr_str}  "
                f"({dt:.1f}s)"
            )

            if len(val_ds) > 0 and val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "model_name": args.model,
                    "n_bins": n_bins,
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                    "normalizer_state": normalizer.state_dict(),
                }, best_ckpt_path)

    print(f"\nDone. Best val_loss={best_val_loss:.4f}, checkpoint at {best_ckpt_path}")
    print(f"Training log at {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())