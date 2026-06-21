"""
metrics.py
----------
Loss functions and evaluation metrics for comparing predicted vs ground-truth
Hi-C contact maps (both already in log1p space, as produced by HiCWindowDataset).

Why not just MSE alone:

MSE treats every pixel as equally important, but Hi-C maps are dominated
by genomic distance: pixels near the diagonal (short-range contacts) have
much higher values and variance than pixels far from the diagonal
(long-range contacts), simply because of polymer physics (closer-in-
sequence DNA contacts more often), not because of any interesting
biological signal. A model that just learns "predict high near the
diagonal, low far away" gets a deceptively good MSE without learning
anything about which SPECIFIC TADs or loops exist in a specific window --
which is presumably the actual scientific question.

This is why the field's standard evaluation is the Akita-style approach:
compute metrics AFTER removing or controlling for the distance-dependent
mean trend. Two such metrics are implemented here:

1. distance_stratified_mse: MSE computed separately for each
   diagonal-distance band, then averaged across bands. This stops the
   (easy, distance-trivial) short-range pixels from drowning out errors
   in the (hard, biologically interesting) long-range pixels in the
   aggregate score.

2. stratum_adjusted_correlation: for each diagonal-distance band,
   subtract the mean true value at that distance from both prediction
   and target (a crude per-distance detrending), then compute Pearson
   correlation across all pixels at all distances pooled together. This
   directly measures "does the model capture WHICH pixels are unusually
   high/low for their distance" (i.e. specific loops/TADs), which a raw
   correlation or raw MSE would not isolate, since raw correlation is
   already very high for any model that's roughly right about the strong
   decay trend with distance.

Both metrics are computed per-window and then averaged across the
evaluation set; they assume input matrices are already symmetric
(true for the Dataset's targets, and for these models' outputs).
"""
from __future__ import annotations

import numpy as np
import torch


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Plain pixelwise MSE. Used as the actual training loss (simple,
    well-behaved gradients) -- the stratified metrics below are for
    EVALUATION/REPORTING, not for backprop, since they involve
    per-distance-band statistics that are awkward and unnecessary to
    differentiate through during training."""
    return torch.nn.functional.mse_loss(pred, target)


def _distance_bands(n_bins: int, n_bands: int = 10) -> np.ndarray:
    """
    Assigns every (i, j) pixel to one of n_bands bands based on |i - j|,
    using bands of roughly EQUAL PIXEL COUNT (not equal distance range),
    since the number of pixels at a given distance shrinks linearly as
    distance grows (only n_bins - d pixel-pairs exist at distance d), so
    equal-width distance bands would have very different sample sizes
    and noise levels per band.
    """
    idx = np.arange(n_bins)
    dist = np.abs(idx[:, None] - idx[None, :])  # (n_bins, n_bins)
    flat_dist = dist.flatten()
    # rank-based binning -> equal pixel count per band
    order = np.argsort(flat_dist)
    band_id_flat = np.empty_like(flat_dist)
    band_id_flat[order] = np.floor(np.linspace(0, n_bands - 1e-9, len(flat_dist))).astype(int)
    return band_id_flat.reshape(n_bins, n_bins)


def distance_stratified_mse(pred: np.ndarray, target: np.ndarray, n_bands: int = 10) -> float:
    """pred, target: (n_bins, n_bins) numpy arrays, already log1p space."""
    n_bins = pred.shape[0]
    bands = _distance_bands(n_bins, n_bands)
    band_mses = []
    for b in range(n_bands):
        mask = bands == b
        if mask.sum() == 0:
            continue
        band_mses.append(float(np.mean((pred[mask] - target[mask]) ** 2)))
    return float(np.mean(band_mses)) if band_mses else float("nan")


def stratum_adjusted_correlation(pred: np.ndarray, target: np.ndarray, n_bands: int = 10) -> float:
    """pred, target: (n_bins, n_bins) numpy arrays, already log1p space."""
    n_bins = pred.shape[0]
    bands = _distance_bands(n_bins, n_bands)
    pred_adj = pred.copy()
    target_adj = target.copy()
    for b in range(n_bands):
        mask = bands == b
        if mask.sum() == 0:
            continue
        target_adj[mask] = target[mask] - target[mask].mean()
        pred_adj[mask] = pred[mask] - pred[mask].mean()

    p = pred_adj.flatten()
    t = target_adj.flatten()
    if p.std() < 1e-8 or t.std() < 1e-8:
        return float("nan")
    return float(np.corrcoef(p, t)[0, 1])


def evaluate_batch(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """
    pred, target: (B, n_bins, n_bins) torch tensors.
    Returns per-metric values AVERAGED ACROSS THE BATCH. Call this once per
    batch during eval and average the returned dicts across batches
    (simple mean-of-means is fine here since every window contributes the
    same number of pixels, n_bins^2).
    """
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    B = pred_np.shape[0]

    mse_vals, dist_mse_vals, corr_vals = [], [], []
    for i in range(B):
        mse_vals.append(float(np.mean((pred_np[i] - target_np[i]) ** 2)))
        dist_mse_vals.append(distance_stratified_mse(pred_np[i], target_np[i]))
        corr_vals.append(stratum_adjusted_correlation(pred_np[i], target_np[i]))

    return {
        "mse": float(np.mean(mse_vals)),
        "distance_stratified_mse": float(np.nanmean(dist_mse_vals)),
        "stratum_adjusted_corr": float(np.nanmean(corr_vals)),
    }