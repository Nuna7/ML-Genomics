# """
# visualize.py
# ------------
# Generates prediction vs. ground truth contact map figures for all three
# trained models, using windows from the test set (chr17).

# Produces two types of figures:
#   1. Raw log1p matrices: what the model actually predicts
#   2. O/E matrices: observed/expected, dividing out the distance-decay
#      trend so only loop/TAD structure remains

# We train on raw log1p (not O/E), so the model must implicitly learn to reproduce the distance-decay
# trend in addition to loop structure. O/E prediction would be a cleaner
# formulation -- the figures here show both so the difference is visible.

# Run:
#     modal run visualize.py
#     modal run visualize.py --model unet --n-windows 6

# Outputs saved to hic-runs volume at /figures/, then downloaded locally:
#     modal volume get hic-runs /figures ./figures_local/
# """
# from __future__ import annotations
# import sys
# from pathlib import Path
# import modal

# # Reuse the same volumes and image as the training pipeline
# app = modal.App("hic-visualize")

# vol_data  = modal.Volume.from_name("hic-data",  create_if_missing=False)
# vol_cache = modal.Volume.from_name("hic-cache", create_if_missing=False)
# vol_runs  = modal.Volume.from_name("hic-runs",  create_if_missing=False)

# VOLUME_DATA  = Path("/mnt/data")
# VOLUME_CACHE = Path("/mnt/cache")
# VOLUME_RUNS  = Path("/mnt/runs")

# image = (
#     modal.Image.debian_slim(python_version="3.11")
#     .apt_install("libcurl4-openssl-dev", "build-essential", "zlib1g-dev")
#     .pip_install(
#         "torch==2.3.1", "numpy", "pandas",
#         "pyBigWig", "hic-straw", "pyyaml",
#         "matplotlib", "scipy", "requests"
#     )
#     .add_local_dir(Path(__file__).parent / "src", remote_path="/src")
#     .add_local_file(
#         Path(__file__).parent / "configs" / "encode_sources.yaml",
#         remote_path="/configs/encode_sources.yaml")
# )


# def _cfg():
#     import yaml
#     with open("/configs/encode_sources.yaml") as f:
#         return yaml.safe_load(f)


# def compute_oe(matrix: "np.ndarray") -> "np.ndarray":
#     """
#     Compute O/E (observed / expected) matrix.

#     Expected contact at distance d = mean of all pixels at diagonal
#     offset d in the matrix. Dividing by the expected removes the
#     distance-decay trend, leaving only loop/TAD structure.

#     We work in log1p space, so O/E becomes:
#         log1p(obs) - mean_at_distance_d(log1p(obs))
#     i.e. a signed deviation from the mean at each distance, rather than
#     a ratio. This is sometimes called "balanced" or "distance-corrected"
#     representation and is what stratum_adjusted_corr computes over.
#     """
#     import numpy as np
#     n = matrix.shape[0]
#     oe = matrix.copy()
#     for d in range(n):
#         # All pixels at distance d from the diagonal
#         if d == 0:
#             diag_vals = np.diag(matrix, 0)
#         else:
#             diag_vals = np.concatenate([np.diag(matrix, d), np.diag(matrix, -d)])
#         mean_d = diag_vals.mean()
#         if d == 0:
#             np.fill_diagonal(oe, matrix.diagonal() - mean_d)
#         else:
#             for i in range(n - d):
#                 oe[i, i+d] = matrix[i, i+d] - mean_d
#                 oe[i+d, i] = matrix[i+d, i] - mean_d
#     return oe


# @app.function(image=image,
#               volumes={VOLUME_DATA: vol_data, VOLUME_CACHE: vol_cache,
#                        VOLUME_RUNS: vol_runs},
#               timeout=60*60*1, cpu=4, memory=8192)
# def make_figures(model_names: list[str], n_windows: int = 4):
#     """
#     Load checkpoints, run inference on test windows, save figures.
#     Figures go to /mnt/runs/figures/ on the hic-runs volume.
#     """
#     import json
#     import numpy as np
#     import pandas as pd
#     import torch
#     import matplotlib
#     matplotlib.use("Agg")
#     import matplotlib.pyplot as plt
#     import matplotlib.gridspec as gridspec
#     sys.path.insert(0, "/src")
#     from genomic_io import hic_matrix, read_bigwig

#     cfg    = _cfg()
#     raw    = VOLUME_DATA / "raw"
#     fig_dir = VOLUME_RUNS / "figures"
#     fig_dir.mkdir(parents=True, exist_ok=True)

#     # Load normalizer
#     norm_path = VOLUME_RUNS / "normalizer.json"
#     if not norm_path.exists():
#         raise FileNotFoundError(
#             "normalizer.json not found in hic-runs volume. "
#             "Run modal_pipeline.py first to train the models.")
#     with open(norm_path) as f:
#         norm_raw = json.load(f)
#     normalizer = {cl: {
#         "mean": np.array(v["mean"], dtype=np.float32),
#         "std":  np.array(v["std"],  dtype=np.float32),
#     } for cl, v in norm_raw.items()}

#     # Load test manifest
#     manifest_path = VOLUME_DATA / "loop_manifest.csv"
#     manifest = pd.read_csv(manifest_path)
#     test_manifest = manifest[manifest["split"] == "test"].reset_index(drop=True)
#     print(f"Test windows: {len(test_manifest)} total")
#     print(test_manifest[["cell_line","chrom","start","end"]].value_counts("cell_line").to_string())

#     # Sample windows: try to get n_windows/2 per cell line for balance
#     sampled = []
#     for cl in test_manifest["cell_line"].unique():
#         cl_rows = test_manifest[test_manifest["cell_line"] == cl]
#         n = min(n_windows // 2, len(cl_rows))
#         sampled.append(cl_rows.sample(n=n, random_state=42))
#     sampled_manifest = pd.concat(sampled).reset_index(drop=True)
#     print(f"\nSampling {len(sampled_manifest)} windows for visualization")

#     def fetch_window(row):
#         """Fetch tracks + Hi-C for one window, return normalized tracks and target."""
#         cl = row["cell_line"]
#         ch, s, e = row["chrom"], int(row["start"]), int(row["end"])
#         n  = int(row["n_bins"])
#         bs = int(row["bin_size"])

#         # Check cache first
#         cache_dir = VOLUME_CACHE / "windows"
#         cache_f   = cache_dir / f"{cl}_{ch}_{s}_{e}.npz"
#         if cache_f.exists():
#             data   = np.load(cache_f)
#             tracks = data["tracks"]
#             target = data["target"]
#         else:
#             ctcf  = read_bigwig(raw / f"{cl}_ctcf.bigWig",    ch, s, e, n)
#             h3k   = read_bigwig(raw / f"{cl}_h3k27ac.bigWig", ch, s, e, n)
#             dnase = read_bigwig(raw / f"{cl}_dnase.bigWig",   ch, s, e, n)
#             tracks = np.stack([ctcf, h3k, dnase], axis=0)
#             mat, _ = hic_matrix(cfg["cell_lines"][cl]["hic_url"], ch, s, e, bs)
#             target = np.log1p(mat).astype(np.float32)

#         norm  = normalizer[cl]
#         t_norm = (tracks - norm["mean"][:, None]) / norm["std"][:, None]
#         return torch.from_numpy(t_norm).float().unsqueeze(0), target

#     def load_model(model_name):
#         ckpt_path = VOLUME_RUNS / model_name / "best_model.pt"
#         if not ckpt_path.exists():
#             raise FileNotFoundError(
#                 f"No checkpoint found at {ckpt_path}. "
#                 f"Run modal_pipeline.py --model {model_name} first.")
#         ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
#         n_bins = ckpt["n_bins"]
#         name   = ckpt["model_name"]
#         if name == "cnn":
#             from model_cnn import CNNHiCModel
#             model = CNNHiCModel(n_tracks=3, n_bins=n_bins)
#         elif name == "unet":
#             from model_unet import UNetHiCModel
#             model = UNetHiCModel(n_tracks=3, n_bins=n_bins)
#         elif name == "transformer":
#             from model_transformer import TransformerHiCModel
#             model = TransformerHiCModel(n_tracks=3, n_bins=n_bins, dropout=0.0)
#         model.load_state_dict(ckpt["model_state_dict"])
#         model.eval()
#         print(f"  Loaded {name} checkpoint from epoch {ckpt['epoch']} "
#               f"(test_mse={ckpt.get('test_mse',float('nan')):.4f}  "
#               f"test_corr={ckpt.get('test_corr',float('nan')):.4f})")
#         return model, ckpt

#     # Load all requested models
#     models = {}
#     ckpts  = {}
#     for mn in model_names:
#         print(f"\nLoading {mn}...")
#         try:
#             models[mn], ckpts[mn] = load_model(mn)
#         except FileNotFoundError as e:
#             print(f"  SKIP: {e}")

#     if not models:
#         raise RuntimeError("No model checkpoints found. Train models first.")

#     # ── Figure 1: per-window panels (raw log1p) ───────────────────────────
#     # Each row = one genomic window
#     # Each column = Ground Truth | CNN pred | UNet pred | Transformer pred
#     n_win  = len(sampled_manifest)
#     n_mods = len(models)
#     n_cols  = 1 + n_mods  # GT + one per model

#     vmax_raw = 3.0   # log1p(20) ≈ 3.0, covers most loop pixels at 1kb
#     vmax_oe  = 1.5   # O/E deviations beyond ±1.5 are extreme outliers

#     fig_raw, axes_raw = plt.subplots(
#         n_win, n_cols, figsize=(4 * n_cols, 4 * n_win),
#         squeeze=False)
#     fig_oe, axes_oe = plt.subplots(
#         n_win, n_cols, figsize=(4 * n_cols, 4 * n_win),
#         squeeze=False)

#     col_labels = ["Ground\nTruth"] + [m.upper() for m in models]

#     for row_idx, (_, window_row) in enumerate(sampled_manifest.iterrows()):
#         cl   = window_row["cell_line"]
#         ch   = window_row["chrom"]
#         s, e = int(window_row["start"]), int(window_row["end"])
#         s_mb = s / 1e6; e_mb = e / 1e6
#         title_base = f"{cl}  {ch}:{s_mb:.2f}-{e_mb:.2f}Mb"

#         print(f"  Window {row_idx+1}/{n_win}: {title_base}")
#         tracks_t, target_np = fetch_window(window_row)
#         target_oe = compute_oe(target_np)

#         # Ground truth
#         axes_raw[row_idx, 0].imshow(target_np, cmap="Reds",
#                                      vmin=0, vmax=vmax_raw, aspect="auto")
#         axes_raw[row_idx, 0].set_title(
#             f"{title_base}\n{col_labels[0]}", fontsize=8)
#         axes_oe[row_idx, 0].imshow(target_oe, cmap="RdBu_r",
#                                     vmin=-vmax_oe, vmax=vmax_oe, aspect="auto")
#         axes_oe[row_idx, 0].set_title(
#             f"{title_base}\n{col_labels[0]}", fontsize=8)

#         for col_idx, mn in enumerate(models, start=1):
#             with torch.no_grad():
#                 pred_t  = models[mn](tracks_t)
#             pred_np = pred_t.squeeze(0).numpy()
#             pred_oe = compute_oe(pred_np)

#             axes_raw[row_idx, col_idx].imshow(pred_np, cmap="Reds",
#                                                vmin=0, vmax=vmax_raw, aspect="auto")
#             axes_raw[row_idx, col_idx].set_title(col_labels[col_idx], fontsize=8)

#             axes_oe[row_idx, col_idx].imshow(pred_oe, cmap="RdBu_r",
#                                               vmin=-vmax_oe, vmax=vmax_oe, aspect="auto")
#             axes_oe[row_idx, col_idx].set_title(col_labels[col_idx], fontsize=8)

#         # Remove axis ticks for cleanliness
#         for ax_row in [axes_raw[row_idx], axes_oe[row_idx]]:
#             for ax in ax_row:
#                 ax.set_xticks([]); ax.set_yticks([])

#     # Shared colorbars
#     for fig, label, fname, cmap, vmin, vmax in [
#         (fig_raw, "log1p(contacts)",     "pred_vs_gt_log1p.png",
#         "Reds",   0,       vmax_raw),
#         (fig_oe,  "O/E deviation",       "pred_vs_gt_oe.png",
#         "RdBu_r", -vmax_oe, vmax_oe),
#     ]:
#         fig.suptitle(
#             f"Ground truth vs. model predictions  |  Test set (chr17)  |  "
#             f"1kb resolution, 100kb windows\n"
#             f"Columns: {', '.join(col_labels)}",
#             fontsize=10, y=0.98)
        
#         # Fix: use constrained_layout or add_axes instead of tight_layout + ax=fig.axes
#         sm = plt.cm.ScalarMappable(
#             cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
#         sm.set_array([])
        
#         # Add colorbar in its own axis on the right
#         cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])  # [left, bottom, width, height]
#         fig.colorbar(sm, cax=cbar_ax, label=label)
        
#         fig.subplots_adjust(right=0.9)  # make room for colorbar
#         out = fig_dir / fname
#         fig.savefig(out, dpi=150, bbox_inches="tight")
#         plt.close(fig)
#         print(f"\nSaved: {out}")

#     # # Shared colorbars
#     # for fig, label, fname, cmap, vmin, vmax in [
#     #     (fig_raw, "log1p(contacts)",     "pred_vs_gt_log1p.png",
#     #      "Reds",   0,       vmax_raw),
#     #     (fig_oe,  "O/E deviation",       "pred_vs_gt_oe.png",
#     #      "RdBu_r", -vmax_oe, vmax_oe),
#     # ]:
#     #     fig.suptitle(
#     #         f"Ground truth vs. model predictions  |  Test set (chr17)  |  "
#     #         f"1kb resolution, 100kb windows\n"
#     #         f"Columns: {', '.join(col_labels)}",
#     #         fontsize=10, y=1.01)
#     #     sm = plt.cm.ScalarMappable(
#     #         cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
#     #     sm.set_array([])
#     #     fig.colorbar(sm, ax=fig.axes, label=label,
#     #                  shrink=0.6, pad=0.02)
#     #     fig.tight_layout()
#     #     out = fig_dir / fname
#     #     fig.savefig(out, dpi=150, bbox_inches="tight")
#     #     plt.close(fig)
#     #     print(f"\nSaved: {out}")

#     # ── Figure 2: per-model difference maps ──────────────────────────────
#     # For each model: show |prediction - ground truth| heatmap per window
#     fig_diff, axes_diff = plt.subplots(
#         n_win, n_mods, figsize=(4 * n_mods, 4 * n_win),
#         squeeze=False)

#     for row_idx, (_, window_row) in enumerate(sampled_manifest.iterrows()):
#         cl   = window_row["cell_line"]
#         ch   = window_row["chrom"]
#         s, e = int(window_row["start"]), int(window_row["end"])
#         tracks_t, target_np = fetch_window(window_row)

#         for col_idx, mn in enumerate(models):
#             with torch.no_grad():
#                 pred_np = models[mn](tracks_t).squeeze(0).numpy()
#             diff = np.abs(pred_np - target_np)
#             axes_diff[row_idx, col_idx].imshow(
#                 diff, cmap="Oranges", vmin=0, vmax=1.5, aspect="auto")
#             title = (f"{cl} {ch}:{s//1000}k\n|{mn.upper()} - GT|"
#                      if row_idx == 0 else f"|{mn.upper()} - GT|")
#             axes_diff[row_idx, col_idx].set_title(title, fontsize=8)
#             axes_diff[row_idx, col_idx].set_xticks([])
#             axes_diff[row_idx, col_idx].set_yticks([])

#     sm_diff = plt.cm.ScalarMappable(
#         cmap="Oranges", norm=plt.Normalize(vmin=0, vmax=1.5))
#     sm_diff.set_array([])
#     fig_diff.colorbar(sm_diff, ax=fig_diff.axes,
#                       label="|prediction - ground truth|  (log1p space)",
#                       shrink=0.6, pad=0.02)
#     fig_diff.suptitle("Absolute prediction error per model  |  Test set (chr17)",
#                        fontsize=10)
#     #fig_diff.tight_layout()
#     fig_diff.subplots_adjust(right=0.9)
#     out_diff = fig_dir / "prediction_error.png"
#     fig_diff.savefig(out_diff, dpi=150, bbox_inches="tight")
#     plt.close(fig_diff)
#     print(f"Saved: {out_diff}")

#     vol_runs.commit()
#     print(f"\nAll figures saved to hic-runs volume at /figures/")
#     print("Download with:")
#     print("  modal volume get hic-runs /figures ./figures_local/")


# @app.local_entrypoint()
# def main(model: str = "all", n_windows: int = 4):
#     """
#     modal run visualize.py                           # all 3 models, 4 windows
#     modal run visualize.py --model unet --n-windows 6
#     """
#     model_names = (["cnn", "unet", "transformer"] if model == "all"
#                    else [model])
#     print(f"Generating figures for: {model_names}  ({n_windows} test windows)")
#     make_figures.remote(model_names, n_windows)
#     print("\nDownload figures:")
#     print("  modal volume get hic-runs /figures ./figures_local/")

"""
visualize.py
------------
Generates prediction vs. ground truth contact map figures for all three
trained models, using windows from the test set (chr17).

Produces two types of figures:
  1. Raw log1p matrices: what the model actually predicts
  2. O/E matrices: observed/expected, dividing out the distance-decay
     trend so only loop/TAD structure remains

The supervisor asked about O/E specifically. We train on raw log1p (not
O/E), so the model must implicitly learn to reproduce the distance-decay
trend in addition to loop structure. O/E prediction would be a cleaner
formulation -- the figures here show both so the difference is visible.

Run:
    modal run visualize.py
    modal run visualize.py --model unet --n-windows 6

Outputs saved to hic-runs volume at /figures/, then downloaded locally:
    modal volume get hic-runs /figures ./figures_local/
"""
from __future__ import annotations
import sys
from pathlib import Path
import modal

# Reuse the same volumes and image as the training pipeline
app = modal.App("hic-visualize")

vol_data  = modal.Volume.from_name("hic-data",  create_if_missing=False)
vol_cache = modal.Volume.from_name("hic-cache", create_if_missing=False)
vol_runs  = modal.Volume.from_name("hic-runs",  create_if_missing=False)

VOLUME_DATA  = Path("/mnt/data")
VOLUME_CACHE = Path("/mnt/cache")
VOLUME_RUNS  = Path("/mnt/runs")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libcurl4-openssl-dev", "build-essential", "zlib1g-dev")
    .pip_install(
        "torch==2.3.1", "numpy", "pandas",
        "pyBigWig", "hic-straw", "pyyaml",
        "matplotlib", "scipy", "requests"
    )
    .add_local_dir(Path(__file__).parent / "src", remote_path="/src")
    .add_local_file(
        Path(__file__).parent / "configs" / "encode_sources.yaml",
        remote_path="/configs/encode_sources.yaml")
)


def _cfg():
    import yaml
    with open("/configs/encode_sources.yaml") as f:
        return yaml.safe_load(f)


def compute_oe(matrix: "np.ndarray") -> "np.ndarray":
    """
    Compute O/E (observed / expected) matrix.

    Expected contact at distance d = mean of all pixels at diagonal
    offset d in the matrix. Dividing by the expected removes the
    distance-decay trend, leaving only loop/TAD structure.

    We work in log1p space, so O/E becomes:
        log1p(obs) - mean_at_distance_d(log1p(obs))
    i.e. a signed deviation from the mean at each distance, rather than
    a ratio. This is sometimes called "balanced" or "distance-corrected"
    representation and is what stratum_adjusted_corr computes over.
    """
    import numpy as np
    n = matrix.shape[0]
    oe = matrix.copy()
    for d in range(n):
        # All pixels at distance d from the diagonal
        if d == 0:
            diag_vals = np.diag(matrix, 0)
        else:
            diag_vals = np.concatenate([np.diag(matrix, d), np.diag(matrix, -d)])
        mean_d = diag_vals.mean()
        if d == 0:
            np.fill_diagonal(oe, matrix.diagonal() - mean_d)
        else:
            for i in range(n - d):
                oe[i, i+d] = matrix[i, i+d] - mean_d
                oe[i+d, i] = matrix[i+d, i] - mean_d
    return oe


@app.function(image=image,
              volumes={VOLUME_DATA: vol_data, VOLUME_CACHE: vol_cache,
                       VOLUME_RUNS: vol_runs},
              timeout=60*60*1, cpu=4, memory=8192)
def make_figures(model_names: list[str], n_windows: int = 4):
    """
    Load checkpoints, run inference on test windows, save figures.
    Figures go to /mnt/runs/figures/ on the hic-runs volume.
    """
    import json
    import numpy as np
    import pandas as pd
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    sys.path.insert(0, "/src")
    from genomic_io import hic_matrix, read_bigwig

    cfg    = _cfg()
    raw    = VOLUME_DATA / "raw"
    fig_dir = VOLUME_RUNS / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load normalizer
    norm_path = VOLUME_RUNS / "normalizer.json"
    if not norm_path.exists():
        raise FileNotFoundError(
            "normalizer.json not found in hic-runs volume. "
            "Run modal_pipeline.py first to train the models.")
    with open(norm_path) as f:
        norm_raw = json.load(f)
    normalizer = {cl: {
        "mean": np.array(v["mean"], dtype=np.float32),
        "std":  np.array(v["std"],  dtype=np.float32),
    } for cl, v in norm_raw.items()}

    # Load test manifest
    manifest_path = VOLUME_DATA / "loop_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    test_manifest = manifest[manifest["split"] == "test"].reset_index(drop=True)
    print(f"Test windows: {len(test_manifest)} total")
    print(test_manifest[["cell_line","chrom","start","end"]].value_counts("cell_line").to_string())

    # Sample windows: try to get n_windows/2 per cell line for balance
    sampled = []
    for cl in test_manifest["cell_line"].unique():
        cl_rows = test_manifest[test_manifest["cell_line"] == cl]
        n = min(n_windows // 2, len(cl_rows))
        sampled.append(cl_rows.sample(n=n, random_state=42))
    sampled_manifest = pd.concat(sampled).reset_index(drop=True)
    print(f"\nSampling {len(sampled_manifest)} windows for visualization")

    def fetch_window(row):
        """Fetch tracks + Hi-C for one window.
        Returns: normalized tracks tensor, O/E target array, raw log1p array.
        O/E is what the model now predicts; raw log1p kept for reference panel."""
        cl = row["cell_line"]
        ch, s, e = row["chrom"], int(row["start"]), int(row["end"])
        n  = int(row["n_bins"])
        bs = int(row["bin_size"])

        # Check cache (new oe_v1 format)
        cache_dir = VOLUME_CACHE / "windows"
        bs = int(row["bin_size"])
        cache_f   = cache_dir / f"{cl}_{ch}_{s}_{e}_oe_v1_{bs}.npz"
        if cache_f.exists():
            data      = np.load(cache_f)
            tracks    = data["tracks"]
            log1p_mat = data["log1p_mat"]
        else:
            ctcf  = read_bigwig(raw / f"{cl}_ctcf.bigWig",    ch, s, e, n)
            h3k   = read_bigwig(raw / f"{cl}_h3k27ac.bigWig", ch, s, e, n)
            dnase = read_bigwig(raw / f"{cl}_dnase.bigWig",   ch, s, e, n)
            tracks = np.stack([ctcf, h3k, dnase], axis=0)
            mat, _ = hic_matrix(cfg["cell_lines"][cl]["hic_url"], ch, s, e, bs)
            log1p_mat = np.log1p(mat).astype(np.float32)

        # Compute O/E (same as training pipeline)
        n_bins = log1p_mat.shape[0]
        oe = log1p_mat.copy()
        for d in range(n_bins):
            if d == 0:
                mean_d = np.diag(log1p_mat).mean()
                np.fill_diagonal(oe, np.diag(log1p_mat) - mean_d)
            else:
                rows = np.arange(n_bins - d); cols = rows + d
                vals = np.concatenate([log1p_mat[rows, cols], log1p_mat[cols, rows]])
                mean_d = vals.mean()
                oe[rows, cols] -= mean_d
                oe[cols, rows] -= mean_d

        norm  = normalizer[cl]
        t_norm = (tracks - norm["mean"][:, None]) / norm["std"][:, None]
        return torch.from_numpy(t_norm).float().unsqueeze(0), oe, log1p_mat

    def load_model(model_name):
        ckpt_path = VOLUME_RUNS / model_name / "best_model.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"No checkpoint found at {ckpt_path}. "
                f"Run modal_pipeline.py --model {model_name} first.")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        n_bins = ckpt["n_bins"]
        name   = ckpt["model_name"]
        if name == "cnn":
            from model_cnn import CNNHiCModel
            model = CNNHiCModel(n_tracks=3, n_bins=n_bins)
        elif name == "unet":
            from model_unet import UNetHiCModel
            model = UNetHiCModel(n_tracks=3, n_bins=n_bins)
        elif name == "transformer":
            from model_transformer import TransformerHiCModel
            model = TransformerHiCModel(n_tracks=3, n_bins=n_bins, dropout=0.0)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        print(f"  Loaded {name} checkpoint from epoch {ckpt['epoch']} "
              f"(test_mse={ckpt.get('test_mse',float('nan')):.4f}  "
              f"test_corr={ckpt.get('test_corr',float('nan')):.4f})")
        return model, ckpt

    # Load all requested models
    models = {}
    ckpts  = {}
    for mn in model_names:
        print(f"\nLoading {mn}...")
        try:
            models[mn], ckpts[mn] = load_model(mn)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")

    if not models:
        raise RuntimeError("No model checkpoints found. Train models first.")

    # ── Figure 1: per-window panels (raw log1p) ───────────────────────────
    # Each row = one genomic window
    # Each column = Ground Truth | CNN pred | UNet pred | Transformer pred
    n_win  = len(sampled_manifest)
    n_mods = len(models)
    n_cols  = 1 + n_mods  # GT + one per model

    vmax_oe  = 1.5   # O/E deviations; loop pixels typically +0.5 to +2.0
    vmax_raw = 3.0   # raw log1p for reference panel

    # Figure 1 (primary): O/E ground truth vs predictions
    # This is what the model actually predicts -- the residual after
    # removing the distance-decay trend. Loop pixels appear as bright spots.
    fig_oe, axes_oe = plt.subplots(
        n_win, n_cols, figsize=(4 * n_cols, 4 * n_win), squeeze=False)

    # Figure 2 (reference): raw log1p for context
    fig_raw, axes_raw = plt.subplots(
        n_win, n_cols, figsize=(4 * n_cols, 4 * n_win), squeeze=False)

    col_labels = ["Ground\nTruth"] + [m.upper() for m in models]

    for row_idx, (_, window_row) in enumerate(sampled_manifest.iterrows()):
        cl   = window_row["cell_line"]
        ch   = window_row["chrom"]
        s, e = int(window_row["start"]), int(window_row["end"])
        s_mb = s / 1e6; e_mb = e / 1e6
        title_base = f"{cl}  {ch}:{s_mb:.2f}-{e_mb:.2f}Mb"

        print(f"  Window {row_idx+1}/{n_win}: {title_base}")
        tracks_t, target_oe, target_raw = fetch_window(window_row)

        # Ground truth columns
        axes_oe[row_idx, 0].imshow(target_oe, cmap="RdBu_r",
                                    vmin=-vmax_oe, vmax=vmax_oe, aspect="auto")
        axes_oe[row_idx, 0].set_title(f"{title_base}\n{col_labels[0]}", fontsize=8)

        axes_raw[row_idx, 0].imshow(target_raw, cmap="Reds",
                                     vmin=0, vmax=vmax_raw, aspect="auto")
        axes_raw[row_idx, 0].set_title(f"{title_base}\n{col_labels[0]}", fontsize=8)

        for col_idx, mn in enumerate(models, start=1):
            with torch.no_grad():
                pred_np = models[mn](tracks_t).squeeze(0).numpy()

            # O/E figure: model predicts O/E directly
            axes_oe[row_idx, col_idx].imshow(pred_np, cmap="RdBu_r",
                                              vmin=-vmax_oe, vmax=vmax_oe, aspect="auto")
            axes_oe[row_idx, col_idx].set_title(col_labels[col_idx], fontsize=8)

            # Raw figure: add back the distance mean to get raw-space prediction
            pred_raw = pred_np.copy()
            n_b = pred_raw.shape[0]
            for d in range(n_b):
                if d == 0:
                    mean_d = np.diag(target_raw).mean()
                    np.fill_diagonal(pred_raw, np.diag(pred_np) + mean_d)
                else:
                    rows = np.arange(n_b - d); cols = rows + d
                    mean_d = np.concatenate([
                        target_raw[rows, cols], target_raw[cols, rows]
                    ]).mean()
                    pred_raw[rows, cols] = pred_np[rows, cols] + mean_d
                    pred_raw[cols, rows] = pred_np[cols, rows] + mean_d
            axes_raw[row_idx, col_idx].imshow(pred_raw, cmap="Reds",
                                               vmin=0, vmax=vmax_raw, aspect="auto")
            axes_raw[row_idx, col_idx].set_title(col_labels[col_idx], fontsize=8)

        for ax_row in [axes_oe[row_idx], axes_raw[row_idx]]:
            for ax in ax_row:
                ax.set_xticks([]); ax.set_yticks([])

    # Shared colorbars
    for fig, label, fname, cmap, vmin, vmax in [
        (fig_raw, "log1p(contacts)",     "pred_vs_gt_log1p.png",
         "Reds",   0,       vmax_raw),
        (fig_oe,  "O/E deviation",       "pred_vs_gt_oe.png",
         "RdBu_r", -vmax_oe, vmax_oe),
    ]:
        fig.suptitle(
            f"Ground truth vs. model predictions  |  Test set (chr17)  |  "
            f"1kb resolution, 100kb windows\n"
            f"Columns: {', '.join(col_labels)}",
            fontsize=10, y=1.01)
        sm = plt.cm.ScalarMappable(
            cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        fig.colorbar(sm, ax=fig.axes, label=label,
                     shrink=0.6, pad=0.02)
        fig.tight_layout()
        out = fig_dir / fname
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nSaved: {out}")

    # ── Figure 2: per-model error maps (O/E space) ───────────────────────
    fig_diff, axes_diff = plt.subplots(
        n_win, n_mods, figsize=(4 * n_mods, 4 * n_win),
        squeeze=False)

    for row_idx, (_, window_row) in enumerate(sampled_manifest.iterrows()):
        cl   = window_row["cell_line"]
        ch   = window_row["chrom"]
        s, e = int(window_row["start"]), int(window_row["end"])
        tracks_t, target_oe, _ = fetch_window(window_row)   # O/E is ground truth

        for col_idx, mn in enumerate(models):
            with torch.no_grad():
                pred_np = models[mn](tracks_t).squeeze(0).numpy()
            # Error in O/E space: signed difference (pred - GT)
            # Positive = model over-predicted contacts; negative = under-predicted
            diff = pred_np - target_oe
            axes_diff[row_idx, col_idx].imshow(
                diff, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
            title = (f"{cl} {ch}:{s//1000}k\n{mn.upper()} - GT (O/E)"
                     if row_idx == 0 else f"{mn.upper()} - GT (O/E)")
            axes_diff[row_idx, col_idx].set_title(title, fontsize=8)
            axes_diff[row_idx, col_idx].set_xticks([])
            axes_diff[row_idx, col_idx].set_yticks([])

    sm_diff = plt.cm.ScalarMappable(
        cmap="RdBu_r", norm=plt.Normalize(vmin=-1.0, vmax=1.0))
    sm_diff.set_array([])
    fig_diff.colorbar(sm_diff, ax=fig_diff.axes,
                      label="prediction error (O/E space)  |  red=over, blue=under",
                      shrink=0.6, pad=0.02)
    fig_diff.suptitle(
        "Signed prediction error per model  |  Test set (chr17)  |  O/E space",
        fontsize=10)
    fig_diff.tight_layout()
    out_diff = fig_dir / "prediction_error_oe.png"
    fig_diff.savefig(out_diff, dpi=150, bbox_inches="tight")
    plt.close(fig_diff)
    print(f"Saved: {out_diff}")

    vol_runs.commit()
    print(f"\nAll figures saved to hic-runs volume at /figures/")
    print("Download with:")
    print("  modal volume get hic-runs /figures ./figures_local/")


@app.local_entrypoint()
def main(model: str = "all", n_windows: int = 4):
    """
    modal run visualize.py                           # all 3 models, 4 windows
    modal run visualize.py --model unet --n-windows 6
    """
    model_names = (["cnn", "unet", "transformer"] if model == "all"
                   else [model])
    print(f"Generating figures for: {model_names}  ({n_windows} test windows)")
    make_figures.remote(model_names, n_windows)
    print("\nDownload figures:")
    print("  modal volume get hic-runs /figures ./figures_local/")