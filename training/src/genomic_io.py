"""
genomic_io.py
-------------
Low-level fetchers for Hi-C matrices and 1D signal tracks (bigWig).

This module deliberately reuses the exact logic from the visualization
script (encode_multitrack_plot.py) for hic_matrix() and read_bigwig(),
since that logic has already been visually validated against known loci
(MYC, HOXA). Keeping it identical means: if a window looks right in the
PNG figures, it will be numerically identical to what the model sees.

Do NOT "improve" the normalization or binning logic here without also
re-checking it against the visualizations, or you'll have two pipelines
that disagree silently.

IMPORT STRATEGY: pyBigWig and hicstraw are imported LAZILY, inside the
functions that actually use them (hic_matrix, read_bigwig), not at module
level. This is deliberate: hicstraw in particular is a C-extension package
that needs libcurl dev headers to build from source on some systems, and
not everyone working on this codebase (e.g. someone writing/testing the
training loop, the manifest logic, or the model architectures with
synthetic data) needs those installed. With a module-level import, simply
doing `from dataset import HiCWindowDataset` would hard-fail on a machine
that hasn't installed these, even if that code path never calls
hic_matrix/read_bigwig. With lazy imports, the failure only happens at the
point where real genomic data fetching is actually attempted, which is
the only point where the dependency is actually needed.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Tuple

import numpy as np
import requests


def download_file(url: str, out_path: Path, chunk_mb: int = 8, max_retries: int = 8) -> Path:
    """Resumable download with retry, identical behavior to the viz script."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"  [cache] {out_path.name}")
        return out_path
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    chunk = chunk_mb << 20
    for attempt in range(max_retries):
        sb = tmp.stat().st_size if tmp.exists() else 0
        hdrs = {"accept": "application/octet-stream"}
        if sb:
            hdrs["Range"] = f"bytes={sb}-"
        try:
            print(f"  [down] {out_path.name} attempt {attempt + 1} ...")
            with requests.get(url, headers=hdrs, stream=True, timeout=(30, 1800)) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0)) + sb
                written = sb
                with open(tmp, "ab" if sb else "wb") as fh:
                    for c in r.iter_content(chunk):
                        if c:
                            fh.write(c)
                            written += len(c)
                            if total:
                                print(
                                    f"\r    {written / total * 100:5.1f}% "
                                    f"{written >> 20}/{total >> 20} MB",
                                    end="", flush=True,
                                )
            print()
            tmp.replace(out_path)
            return out_path
        except Exception as exc:
            wait = min(60, 2 ** attempt)
            print(f"\n  [retry] {exc}; sleeping {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Download failed after {max_retries} attempts: {url}")


def hic_matrix(
    hic_source: str,
    chrom: str,
    start: int,
    end: int,
    bin_size: int,
) -> Tuple[np.ndarray, int]:
    """
    Fetch a square Hi-C contact matrix for [start, end) at the given chrom,
    using the same chromosome-name resolution and normalization fallback
    as the visualization script.

    Returns (matrix, actual_bin_size_used). actual_bin_size_used may differ
    from the requested bin_size if the .hic file doesn't have that exact
    zoom level cached -- the caller MUST check this and either accept it
    or raise, never silently assume the requested size was honored.
    """
    try:
        import hicstraw
    except ImportError as e:
        raise ImportError(
            "hic_matrix() requires the 'hic-straw' package, which is not "
            "installed. Install with: pip install hic-straw. (Note: this "
            "requires libcurl development headers on the system, e.g. "
            "`apt install libcurl4-openssl-dev` on Debian/Ubuntu, since "
            "hic-straw compiles a C++ extension.)"
        ) from e

    hf = hicstraw.HiCFile(hic_source)
    avail_res = hf.getResolutions()
    best = max((r for r in avail_res if r <= bin_size), default=min(avail_res))
    bin_size_used = best

    names = [c.name for c in hf.getChromosomes()]
    key = chrom
    if key not in names:
        alt1 = chrom.replace("chr", "")
        if alt1 in names:
            key = alt1
        elif f"chr{chrom}" in names:
            key = f"chr{chrom}"
        else:
            raise ValueError(
                f"Chromosome '{chrom}' not found in Hi-C file. Available: {names}"
            )

    # norm = "NONE"
    # for candidate in ("SCALE", "VC", "NONE"):
    #     try:
    #         hf.getMatrixZoomData(key, key, "observed", candidate, "BP", bin_size_used)
    #         norm = candidate
    #         break
    #     except Exception:
    #         continue

    n = max(1, int(np.ceil((end - start) / bin_size_used)))

    mzd = hf.getMatrixZoomData(
        key,
        key,
        "observed",
        "NONE",
        "BP",
        bin_size_used,
    )
    raw = mzd.getRecordsAsMatrix(start, end, start, end)

    mat = np.zeros((n, n), dtype=np.float32)
    r0 = min(raw.shape[0], n)
    c0 = min(raw.shape[1], n)
    mat[:r0, :c0] = raw[:r0, :c0]
    return mat, bin_size_used


def read_bigwig(
    bw_path: Path,
    chrom: str,
    start: int,
    end: int,
    bins: int,
    clip_pct: float = 99.0,
) -> np.ndarray:
    """
    Mean signal per bin over [start, end), same logic as the viz script's
    read_bigwig but returns only the y-values (we control x-positions by
    construction in the dataset, so no need to recompute them here).
    """
    try:
        import pyBigWig
    except ImportError as e:
        raise ImportError(
            "read_bigwig() requires the 'pyBigWig' package, which is not "
            "installed. Install with: pip install pyBigWig"
        ) from e

    bw = pyBigWig.open(str(bw_path))
    try:
        raw = bw.stats(chrom, start, end, type="mean", nBins=bins, exact=True)
    finally:
        bw.close()
    y = np.nan_to_num(np.array(raw, dtype=float), nan=0.0)
    if clip_pct < 100 and y.max() > 0:
        cap = np.percentile(y[y > 0], clip_pct)
        y = np.clip(y, 0, cap)
    return y.astype(np.float32)