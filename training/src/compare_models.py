"""
compare_models.py
------------------
Reads the training_log.csv from each of the three model runs (produced by
train.py) and produces:
  - a comparison plot: train/val loss curves for all three models, side by side
  - a comparison plot: val stratum-adjusted correlation over epochs (the
    metric that actually reflects whether the model captures real,
    distance-corrected structure, not just the easy distance trend)
  - a markdown summary table of best val metrics per model

Usage:
    python compare_models.py \
        --cnn-dir runs/cnn_run1 \
        --unet-dir runs/unet_run1 \
        --transformer-dir runs/transformer_run1 \
        --out-dir comparison_report
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


MODEL_COLORS = {"CNN": "#1f77b4", "UNet": "#2ca02c", "Transformer": "#d62728"}


def load_log(run_dir: Path) -> pd.DataFrame:
    log_path = run_dir / "training_log.csv"
    if not log_path.exists():
        raise FileNotFoundError(
            f"No training_log.csv found in {run_dir}. Did train.py finish "
            f"successfully for this run?"
        )
    return pd.read_csv(log_path)


def plot_curves(logs: dict[str, pd.DataFrame], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for name, df in logs.items():
        color = MODEL_COLORS.get(name, None)
        axes[0].plot(df["epoch"], df["train_loss"], color=color, ls="--", alpha=0.6, label=f"{name} train")
        axes[0].plot(df["epoch"], df["val_loss"], color=color, ls="-", label=f"{name} val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("MSE loss (log1p contact space)")
    axes[0].set_title("Train/val loss")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    for name, df in logs.items():
        color = MODEL_COLORS.get(name, None)
        axes[1].plot(df["epoch"], df["val_stratum_adjusted_corr"], color=color, label=name)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("stratum-adjusted correlation (val)")
    axes[1].set_title("Distance-corrected correlation\n(higher = better captures specific loops/TADs)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    axes[1].axhline(0, color="gray", lw=0.8)

    for name, df in logs.items():
        color = MODEL_COLORS.get(name, None)
        cumulative_time = df["epoch_seconds"].cumsum() / 60.0
        axes[2].plot(cumulative_time, df["val_loss"], color=color, label=name)
    axes[2].set_xlabel("cumulative training time (minutes)")
    axes[2].set_ylabel("val loss")
    axes[2].set_title("Val loss vs wall-clock time\n(controls for different per-epoch cost)")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_summary_table(logs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in logs.items():
        best_idx = df["val_loss"].idxmin()
        best_row = df.loc[best_idx]
        total_time_min = df["epoch_seconds"].sum() / 60.0
        rows.append({
            "model": name,
            "best_epoch": int(best_row["epoch"]),
            "best_val_loss": best_row["val_loss"],
            "val_mse": best_row["val_mse"],
            "val_distance_stratified_mse": best_row["val_distance_stratified_mse"],
            "val_stratum_adjusted_corr": best_row["val_stratum_adjusted_corr"],
            "total_epochs_trained": int(df["epoch"].max()),
            "total_train_time_min": round(total_time_min, 1),
        })
    return pd.DataFrame(rows).sort_values("best_val_loss")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cnn-dir", type=Path, required=True)
    p.add_argument("--unet-dir", type=Path, required=True)
    p.add_argument("--transformer-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = {"CNN": args.cnn_dir, "UNet": args.unet_dir, "Transformer": args.transformer_dir}
    logs = {}
    for name, run_dir in run_dirs.items():
        try:
            logs[name] = load_log(run_dir)
        except FileNotFoundError as e:
            print(f"WARNING: skipping {name} -- {e}")

    if len(logs) == 0:
        print("ERROR: no training logs found for any model. Nothing to compare.")
        return 1

    plot_path = args.out_dir / "comparison_curves.png"
    plot_curves(logs, plot_path)
    print(f"Wrote comparison plot to {plot_path}")

    summary = build_summary_table(logs)
    summary_csv = args.out_dir / "summary_table.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"Wrote summary table to {summary_csv}")
    print()
    print(summary.to_string(index=False))

    summary_md = args.out_dir / "summary_table.md"
    with open(summary_md, "w") as fh:
        fh.write("# Model comparison summary\n\n")
        fh.write(summary.to_markdown(index=False))
        fh.write("\n\n")
        fh.write(
            "**Note on fairness of this comparison**: best_epoch is chosen "
            "independently per model (lowest val_loss for that model's own "
            "training run), not a shared fixed epoch count, because the "
            "three architectures converge at different speeds (in "
            "particular, the Transformer's attention starts near-uniform "
            "and needs more optimization steps to sharpen -- see "
            "model_transformer.py). If one model was trained for far fewer "
            "total epochs than the others and its val_loss curve in "
            "comparison_curves.png has clearly not plateaued yet, its "
            "result here is not a fair final comparison -- train it "
            "further before concluding it is worse. Compare the "
            "val-loss-vs-wall-clock-time panel as well, since epoch count "
            "alone does not account for the fact that the Transformer and "
            "UNet have different per-epoch costs than the CNN.\n"
        )
    print(f"\nWrote markdown summary to {summary_md}")

    if len(logs) >= 2:
        max_epochs = {name: int(df["epoch"].max()) for name, df in logs.items()}
        if max(max_epochs.values()) > 2 * min(max_epochs.values()):
            print(
                "\nCAUTION: training run lengths differ by more than 2x across "
                f"models ({max_epochs}). Double-check whether the "
                "shorter run(s) had actually plateaued (flattened val_loss "
                "curve) before concluding they perform worse -- they may "
                "simply need more epochs, particularly if one of them is "
                "the Transformer."
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())