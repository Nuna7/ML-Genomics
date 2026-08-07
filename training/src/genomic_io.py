"""
genomic_io.py
-------------
Low-level fetchers for Hi-C matrices and 1D signal tracks (bigWig).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Tuple

import numpy as np
import requests

_HICFILE_CACHE: dict = {}
_HICFILE_CALL_COUNT: dict = {}
_HICFILE_MAX_CALLS = 50

def _get_hicfile(hic_source: str):
    import hicstraw
    count = _HICFILE_CALL_COUNT.get(hic_source, 0)
    if hic_source not in _HICFILE_CACHE or count >= _HICFILE_MAX_CALLS:
        _HICFILE_CACHE[hic_source] = hicstraw.HiCFile(hic_source)
        _HICFILE_CALL_COUNT[hic_source] = 0
    _HICFILE_CALL_COUNT[hic_source] += 1
    return _HICFILE_CACHE[hic_source]
 
 
def _looks_like_missing_norm(exc: Exception) -> bool:
    """Only these error shapes indicate 'this normalization vector isn't
    computed for this file/resolution' -- genuinely safe to fall back on.
    Anything else (bad_alloc, network errors, etc.) is a real failure that
    falling back to a different normalization will NOT fix."""
    msg = str(exc).lower()
    return (
        "did not contain" in msg
        or "normalization" in msg and "not" in msg
        or "no such" in msg
    )

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
    normalization_preference: tuple = ("SCALE", "VC_SQRT", "NONE"),
) -> Tuple[np.ndarray, int]:
    try:
        import hicstraw
    except ImportError as e:
        raise ImportError(
            "hic_matrix() requires the 'hic-straw' package. "
            "Install with: pip install hic-straw."
        ) from e
 
    hf = _get_hicfile(hic_source)   # <-- reused across calls, not reopened
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
 
    n = max(1, int(np.ceil((end - start) / bin_size_used)))
 
    mzd = None
    norm_used = None
    last_err = None
    for norm in normalization_preference:
        try:
            mzd_try = hf.getMatrixZoomData(key, key, "observed", norm, "BP", bin_size_used)
            raw_try = mzd_try.getRecordsAsMatrix(start, end, start, end)
            if raw_try.size == 0 or not np.isfinite(raw_try).any():
                raise ValueError(f"'{norm}' returned empty/non-finite matrix")
            mzd, raw, norm_used = mzd_try, raw_try, norm
            break
        except Exception as e:
            if not _looks_like_missing_norm(e):
                # Real failure (memory, network, corrupt state) -- don't
                # burn through the remaining normalizations pretending
                # this is a "not available" case. Surface it immediately.
                raise RuntimeError(
                    f"hic_matrix() failed on {chrom}:{start}-{end} while "
                    f"requesting '{norm}' -- this does not look like a "
                    f"missing-normalization error, so not falling back. "
                    f"Original error: {type(e).__name__}: {e}"
                ) from e
            last_err = e
            continue
 
    if mzd is None:
        raise RuntimeError(
            f"No usable normalization found for {hic_source} at {chrom}:{start}-{end} "
            f"(tried {normalization_preference}). Last error: {last_err}"
        )
 
    if norm_used != normalization_preference[0]:
        print(f"  [WARN] {chrom}:{start}-{end}: '{normalization_preference[0]}' "
              f"unavailable, fell back to '{norm_used}'.")
 
    mat = np.zeros((n, n), dtype=np.float32)
    r0 = min(raw.shape[0], n)
    c0 = min(raw.shape[1], n)
    mat[:r0, :c0] = raw[:r0, :c0]
    mat = np.nan_to_num(mat, nan=0.0)
    return mat, bin_size_used

def read_bigwig(
    bw_path: Path,
    chrom: str,
    start: int,
    end: int,
    bins: int,
    clip_pct: float = 99.0,
) -> np.ndarray:
    try:
        import pyBigWig
    except ImportError as e:
        raise ImportError(...) from e

    bw = pyBigWig.open(str(bw_path))
    try:
        # Chromosome name normalization (match hic_matrix logic)
        chrom_key = chrom
        chroms = bw.chroms()
        if chrom_key not in chroms:
            alt = chrom.replace("chr", "")
            if alt in chroms:
                chrom_key = alt
            elif f"chr{alt}" in chroms:
                chrom_key = f"chr{alt}"
            else:
                # Fallback / debug
                print(f"[WARN] Chromosome {chrom} not found in {bw_path.name}. Available: {list(chroms.keys())[:10]}...")
                # Return zeros to avoid crash (or raise)
                return np.zeros(bins, dtype=np.float32)

        # Clamp to bigWig chromosome length
        clen = chroms[chrom_key]
        start = int(start)
        end = int(end)

        if start < 0:
            start = 0
        if start >= clen:
            print(f"[WARN] Interval starts beyond chromosome end: {chrom_key}:{start}-{end} (len={clen})")
            return np.zeros(bins, dtype=np.float32)

        end = min(end, clen)
        if end <= start:
            print(f"[WARN] Invalid interval after clamp: {chrom_key}:{start}-{end} (len={clen})")
            return np.zeros(bins, dtype=np.float32)

        raw = bw.stats(chrom_key, start, end, type="mean", nBins=bins, exact=True)
    finally:
        bw.close()

    y = np.nan_to_num(np.array(raw, dtype=float), nan=0.0)
    if clip_pct < 100 and y.max() > 0:
        cap = np.percentile(y[y > 0], clip_pct)
        y = np.clip(y, 0, cap)
    return y.astype(np.float32)