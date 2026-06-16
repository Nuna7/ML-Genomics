from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import requests
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

try:
    import pyBigWig
except ImportError:
    sys.exit("pip install pyBigWig")
try:
    import hicstraw
except ImportError:
    sys.exit("pip install hic-straw")


# ENCODE sources
ENCODE_BASE = "https://www.encodeproject.org"
HIC_URL     = f"{ENCODE_BASE}/files/ENCFF291JZM/@@download/ENCFF291JZM.hic"
CTCF_URL    = f"{ENCODE_BASE}/files/ENCFF000YMA/@@download/ENCFF000YMA.bigWig"
h3k_url = "https://www.encodeproject.org/files/ENCFF779QTH/@@download/ENCFF779QTH.bigWig"
dnase_url = "https://www.encodeproject.org/files/ENCFF414OGC/@@download/ENCFF414OGC.bigWig"


def download_file(url: str, out_path: Path, chunk_mb: int = 8, max_retries: int = 8) -> Path:
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
            print(f"  [↓] {out_path.name} attempt {attempt+1} …")
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
                                print(f"\r    {written/total*100:5.1f}% {written>>20}/{total>>20} MB", end="", flush=True)
            print()
            tmp.replace(out_path)
            return out_path
        except Exception as exc:
            wait = min(60, 2**attempt)
            print(f"\n  [retry] {exc}; sleeping {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Download failed: {url}")


def hic_matrix(hic_source: str, chrom: str, x_start: int, x_end: int, y_start: int, y_end: int, bin_size: int) -> Tuple[np.ndarray, int]:
    hf = hicstraw.HiCFile(hic_source)
    avail_res = hf.getResolutions()
    best = max((r for r in avail_res if r <= bin_size), default=min(avail_res))
    if best != bin_size:
        print(f"  [hic] using {best} bp")
    bin_size = best

    names = [c.name for c in hf.getChromosomes()]
    print(f"  [hic] Available chroms: {names[:10]}...")
    key = chrom
    if key not in names:
        alt1 = chrom.replace("chr", "")
        if alt1 in names:
            key = alt1
        elif f"chr{chrom}" in names:
            key = f"chr{chrom}"
    print(f"  [hic] Using chromosome key: {key}")

    norm = "NONE"
    for candidate in ("SCALE", "VC", "NONE"):
        try:
            hf.getMatrixZoomData(key, key, "observed", candidate, "BP", bin_size)
            norm = candidate
            break
        except Exception:
            continue

    nx = max(1, int(np.ceil((x_end - x_start) / bin_size)))
    ny = max(1, int(np.ceil((y_end - y_start) / bin_size)))

    mzd = hf.getMatrixZoomData(key, key, "observed", norm, "BP", bin_size)
    raw = mzd.getRecordsAsMatrix(x_start, x_end, y_start, y_end)

    mat = np.zeros((nx, ny), dtype=np.float32)
    r0, c0 = min(raw.shape[0], nx), min(raw.shape[1], ny)
    mat[:r0, :c0] = raw[:r0, :c0]
    return mat, bin_size


def read_bigwig(bw_path: Path, chrom: str, start: int, end: int, bins: int, clip_pct: float = 99.0) -> Tuple[np.ndarray, np.ndarray]:
    bw = pyBigWig.open(str(bw_path))
    try:
        raw = bw.stats(chrom, start, end, type="mean", nBins=bins, exact=True)
    finally:
        bw.close()
    y = np.nan_to_num(np.array(raw, dtype=float), nan=0.0)
    if clip_pct < 100 and y.max() > 0:
        cap = np.percentile(y[y > 0], clip_pct)
        y = np.clip(y, 0, cap)
    half = (end - start) / bins / 2.0
    x = np.linspace(start + half, end - half, bins)
    return x, y


def fetch_seq_ucsc(chrom: str, start: int, end: int) -> str:
    url = f"https://api.genome.ucsc.edu/getData/sequence?genome=hg38;chrom={chrom};start={start};end={end}"
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        return r.json()["dna"].upper()
    except Exception as exc:
        print(f"  [seq] UCSC fetch failed: {exc}")
        return ""


BASE_COLORS = {"A": "#2ca02c", "C": "#1f77b4", "G": "#ff7f0e", "T": "#d62728"}
BASE_ORDER = ["A", "C", "G", "T"]
MAX_ONEHOT = 25_000

def _onehot_image(seq: str) -> np.ndarray:
    L = len(seq)
    lut = {"A": 0, "C": 1, "G": 2, "T": 3}
    display = np.zeros((4, L, 4), dtype=float)
    for row, base in enumerate(BASE_ORDER):
        display[row, :, :3] = matplotlib.colors.to_rgb(BASE_COLORS[base])
    for i, b in enumerate(seq):
        j = lut.get(b)
        if j is not None:
            display[j, i, 3] = 1.0
    return display


def _gc_curve(seq: str, win: int) -> Tuple[np.ndarray, np.ndarray]:
    xs, gc = [], []
    for s in range(0, len(seq), win):
        chunk = seq[s:s+win]
        if chunk:
            xs.append(s + win/2)
            gc.append((chunk.count("G") + chunk.count("C")) / len(chunk))
    return np.array(xs, float), np.array(gc, float)


_HIC_CMAP = LinearSegmentedColormap.from_list("hic", ["#ffffff", "#fff3e8", "#ffaa66", "#cc2200", "#4d0000"])
TRACK_COLORS = {"CTCF": "#2166ac", "H3K27ac": "#d6604d", "DNase": "#4dac26"}


def _clean(ax: plt.Axes, keep_left: bool = True, keep_bottom: bool = True) -> None:
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    if not keep_left:
        ax.spines['left'].set_visible(False)
    if not keep_bottom:
        ax.spines['bottom'].set_visible(False)
    ax.tick_params(labelsize=8)


def _format_mb(x, pos):
    """Force Mb formatting, no 1e8 offsets"""
    return f"{x/1e6:.3f}"

def draw_signal_top(ax: plt.Axes, x: np.ndarray, y: np.ndarray, color: str, label: str, g_start: int, g_end: int) -> None:
    ax.fill_between(x, y, alpha=0.85, color=color, lw=0.5)
    ax.set_xlim(g_start, g_end)
    ymax = y.max() * 1.15 if y.max() > 0 else 1
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
    ax.set_ylabel(label, fontsize=10, rotation=0, ha="right", va="center", labelpad=12, fontweight="bold")
    ax.tick_params(axis="x", labelbottom=False)
    _clean(ax)


def draw_onehot_top(ax: plt.Axes, seq: str, g_start: int, g_end: int) -> None:
    span = g_end - g_start
    if seq and span <= MAX_ONEHOT:
        img = _onehot_image(seq)
        ax.imshow(img, aspect="auto", origin="upper", extent=[g_start, g_end, -0.5, 3.5], interpolation="nearest")
        ax.set_yticks(range(4))
        ax.set_yticklabels(BASE_ORDER, fontsize=7.5)
        ax.set_ylabel("DNA\nbases", fontsize=10, rotation=0, ha="right", va="center", labelpad=12, fontweight="bold")
    else:
        win = max(300, span // 400)
        xs, gc = _gc_curve(seq, win)
        ax.plot(xs + g_start, gc, lw=1.2, color="#7b2d8b")
        ax.axhline(0.5, color="#ccc", lw=0.6, ls="--")
        ax.set_ylim(0, 1)
        ax.set_ylabel("GC %", fontsize=10, rotation=0, ha="right", va="center", labelpad=18, fontweight="bold")
    ax.set_xlim(g_start, g_end)
    ax.tick_params(axis="x", labelbottom=False)
    _clean(ax)

def draw_signal_left(ax: plt.Axes, coord: np.ndarray, signal: np.ndarray,
                     color: str, label: str, g_start: int, g_end: int) -> None:
    coord = np.asarray(coord, dtype=float)
    signal = np.asarray(signal, dtype=float)

    pos = signal[signal > 0]
    xmax = np.percentile(pos, 99) if pos.size else 1.0
    xmax = max(1.0, xmax * 1.05)

    ax.plot(signal, coord, color=color, lw=0.9)
    ax.fill_betweenx(coord, 0, signal, color=color, alpha=0.15, lw=0)

    ax.set_ylim(g_start, g_end)
    ax.set_xlim(xmax, 0)

    ax.xaxis.set_major_locator(ticker.MaxNLocator(2))
    #fmt = ticker.ScalarFormatter(useOffset=False)
    #fmt.set_scientific(False)
    #ax.xaxis.set_major_formatter(fmt)

    ax.set_xlabel("signal", fontsize=8, labelpad=2)
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="x", labeltop=True, labelbottom=False, pad=1)
    ax.tick_params(axis="y", labelleft=False)

    ax.set_title(label, fontsize=10, pad=6, fontweight="bold", color=color)
    _clean(ax, keep_left=False, keep_bottom=True)

def draw_onehot_left(ax: plt.Axes, seq: str, g_start: int, g_end: int) -> None:
    span = g_end - g_start
    if seq and span <= MAX_ONEHOT:
        img = _onehot_image(seq)
        ax.imshow(img.transpose(1, 0, 2), aspect="auto", origin="lower",
                  extent=[-0.5, 3.5, g_start, g_end], interpolation="nearest")
        ax.set_xticks(range(4))
        ax.set_xticklabels(BASE_ORDER, fontsize=7.5)
        ax.set_title("DNA", fontsize=10, loc="center", pad=8, fontweight="bold")
    else:
        win = max(300, span // 400)
        xs, gc = _gc_curve(seq, win)
        ax.fill_betweenx(xs + g_start, 0, gc, alpha=0.75, color="#7b2d8b")
        ax.axvline(0.5, color="#ccc", lw=0.6, ls="--")
        ax.set_xlim(0, 1)
        ax.set_title("GC %", fontsize=10, loc="center", pad=8, fontweight="bold")
        ax.invert_xaxis()
    
    # Also kill offset here for the GC% case
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
    ax.ticklabel_format(style='plain', useOffset=False, axis='x')
    
    ax.set_ylim(g_start, g_end)
    ax.tick_params(axis="x", labelsize=7, pad=2)
    ax.tick_params(axis="y", labelleft=False)
    _clean(ax, keep_left=False, keep_bottom=True)

def plot_region(
    chrom: str,
    x_start: int, x_end: int,
    y_start: int, y_end: int,
    bin_size: int,
    hic_url: str,
    ctcf_bw: Path, h3k27ac_bw: Path, dnase_bw: Path,
    out_path: Path,
    region_name: str = "",
    show_seq: bool = True,
    fig_width: float = 21.0,
    clip_pct: float = 95.0,
) -> None:

    symmetric = (x_start == y_start and x_end == y_end)
    x_span = x_end - x_start
    y_span = y_end - y_start

    sig_x = max(500, (x_span // bin_size) * 6)
    sig_y = max(500, (y_span // bin_size) * 6)

    print("[1/4] Loading signals x-window...")
    xc_x, yc_x = read_bigwig(ctcf_bw, chrom, x_start, x_end, sig_x, clip_pct)
    xh_x, yh_x = read_bigwig(h3k27ac_bw, chrom, x_start, x_end, sig_x, clip_pct)
    xd_x, yd_x = read_bigwig(dnase_bw, chrom, x_start, x_end, sig_x, clip_pct)

    print("[2/4] Loading signals y-window...")
    if symmetric:
        xc_y = yc_y = xc_x; xh_y = yh_y = xh_x; xd_y = yd_y = xd_x
    else:
        xc_y, yc_y = read_bigwig(ctcf_bw, chrom, y_start, y_end, sig_y, clip_pct)
        xh_y, yh_y = read_bigwig(h3k27ac_bw, chrom, y_start, y_end, sig_y, clip_pct)
        xd_y, yd_y = read_bigwig(dnase_bw, chrom, y_start, y_end, sig_y, clip_pct)

    print("[3/4] Hi-C matrix...")
    try:
        mat, actual_bin = hic_matrix(hic_url, chrom, x_start, x_end, y_start, y_end, bin_size)
    except Exception as e:
        print(f"  [hic] WARNING: {e}. Continuing without matrix.")
        mat = np.zeros((10,10))
        actual_bin = bin_size

    seq_x = seq_y = ""
    if show_seq:
        print("[4/4] DNA sequence...")
        seq_x = fetch_seq_ucsc(chrom, x_start, x_end)
        seq_y = seq_x if symmetric else fetch_seq_ucsc(chrom, y_start, y_end)

    # ==================== GEOMETRY ====================
    SIGNAL_W = 0.95
    ONEHOT_W = 1.25
    
    PAD_LEFT = 2      # Increased significantly to prevent overlap
    PAD_INNER = 0.35
    PAD_RIGHT = 0.85

    SIGNAL_H = 0.88
    ONEHOT_H = 1.25
    PAD_TOP = 1.35
    PAD_INNER_V = 0.14
    PAD_BOT = 0.95

    N_SIG = 3
    left_w = PAD_LEFT + ONEHOT_W + N_SIG * (SIGNAL_W + PAD_INNER) + PAD_INNER
    HIC_W = fig_width - left_w - PAD_RIGHT
    HIC_H = HIC_W
    top_h = PAD_TOP + ONEHOT_H + N_SIG * (SIGNAL_H + PAD_INNER_V) + PAD_INNER_V
    fig_height = top_h + HIC_H + PAD_BOT

    fig = plt.figure(figsize=(fig_width, fig_height), facecolor="white", dpi=300)

    def _ax(left_in, bot_in, w_in, h_in):
        W, H = fig_width, fig_height
        return fig.add_axes([left_in/W, bot_in/H, w_in/W, h_in/H])

    hic_left_in = left_w
    hic_bot_in = PAD_BOT

    ax_hic = _ax(hic_left_in, hic_bot_in, HIC_W, HIC_H)

    # Colorbar
    CB_W, CB_H = 0.14, HIC_H * 0.55
    ax_cb = _ax(hic_left_in + HIC_W + 0.25, hic_bot_in + HIC_H*0.22, CB_W, CB_H)

    # Top tracks
    top_tracks_def = [
        ("DNA bases", None, None, None, True),
        ("DNase", xd_x, yd_x, TRACK_COLORS["DNase"], False),
        ("H3K27ac", xh_x, yh_x, TRACK_COLORS["H3K27ac"], False),
        ("CTCF", xc_x, yc_x, TRACK_COLORS["CTCF"], False),
    ]
    cur_bot = hic_bot_in + HIC_H + PAD_INNER_V
    ax_top = {}
    for name, xv, yv, col, is_oh in top_tracks_def:
        h = ONEHOT_H if is_oh else SIGNAL_H
        ax = _ax(hic_left_in, cur_bot, HIC_W, h)
        ax_top[name] = ax
        if is_oh:
            draw_onehot_top(ax, seq_x, x_start, x_end)
        else:
            draw_signal_top(ax, xv, yv, col, name, x_start, x_end)
        cur_bot += h + PAD_INNER_V

   
    # Top x labels
    topmost = ax_top["CTCF"]
    topmost.tick_params(axis="x", labeltop=True, labelbottom=False, length=4)
    topmost.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    topmost.set_xlabel(f"x: {chrom} (GRCh38, Mb)", fontsize=11, labelpad=10)
    topmost.xaxis.set_label_position("top")

    # Disable offset BEFORE setting custom formatter
    topmost.xaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
    topmost.ticklabel_format(style='plain', useOffset=False) # Now safe
    topmost.xaxis.set_major_formatter(ticker.FuncFormatter(_format_mb))

    # Left tracks
    left_tracks_def = [
        ("CTCF", xc_y, yc_y, TRACK_COLORS["CTCF"], False),
        ("H3K27ac", xh_y, yh_y, TRACK_COLORS["H3K27ac"], False),
        ("DNase", xd_y, yd_y, TRACK_COLORS["DNase"], False),
        ("DNA bases", None, None, None, True),
    ]
    cur_left = hic_left_in - PAD_INNER
    ax_left = {}
    for name, xv, yv, col, is_oh in left_tracks_def:
        w = ONEHOT_W if is_oh else SIGNAL_W
        cur_left -= (w + PAD_INNER)
        ax = _ax(cur_left, hic_bot_in, w, HIC_H)
        ax_left[name] = ax
        if is_oh:
            draw_onehot_left(ax, seq_y, y_start, y_end)
        else:
            draw_signal_left(ax, xv, yv, col, name, y_start, y_end)

    # Y labels on the leftmost track (CTCF), with more padding
    leftmost = ax_left["DNA bases"]
    leftmost.yaxis.set_ticks_position('left')
    leftmost.tick_params(axis="y", labelleft=True, labelright=False, length=4, pad=12, labelsize=9)
    leftmost.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v/1e6:.3f}"))
    leftmost.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    leftmost.set_ylabel(f"y: {chrom} (GRCh38, Mb)", fontsize=11, labelpad=40)


    # Hi-C
    log_mat = np.log1p(mat)
    if symmetric and mat.shape[0] > 1:
        mask = np.tril(np.ones_like(log_mat, dtype=bool), k=-1)
        display = np.ma.array(log_mat, mask=mask)
    else:
        display = log_mat

    nonzero = log_mat[log_mat > 0]
    vmax = max(0.6, np.percentile(nonzero, 97) if len(nonzero) > 0 else 1.0)
    
    im = ax_hic.imshow(
        display,
        cmap=_HIC_CMAP,
        interpolation="nearest",
        aspect="equal",
        origin="upper",
        extent=[x_start, x_end, y_end, y_start],    
        vmin=0,
        vmax=vmax,
    )

    ax_hic.set_xlim(x_start, x_end)
    ax_hic.set_ylim(y_end, y_start)

    # Kill offset the right way: set ScalarFormatter first, disable offset, then swap to FuncFormatter
    ax_hic.ticklabel_format(style='plain', useOffset=False)
    ax_hic.xaxis.set_major_formatter(ticker.FuncFormatter(_format_mb))
    ax_hic.yaxis.set_major_formatter(ticker.FuncFormatter(_format_mb))

    ax_hic.set_xlabel(f"x: {chrom} (GRCh38, Mb)", fontsize=11, labelpad=6)
    ax_hic.set_ylabel(f"y: {chrom} (GRCh38, Mb)", fontsize=11, labelpad=6)
    ax_hic.tick_params(labelsize=9, length=4)
    ax_hic.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax_hic.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

    plt.colorbar(im, cax=ax_cb, orientation="vertical", ticks=[0, 0.5, 1.0, 1.5, 2.0, vmax])
    ax_cb.set_ylabel("log(contacts+1)", fontsize=10, labelpad=5)
    
    # Title
    label = region_name or (f"{chrom}:{x_start:,}–{x_end:,}" if symmetric else f"{chrom} x:{x_start:,}–{x_end:,} y:{y_start:,}–{y_end:,}")
    fig.text(0.5, 0.985, f"{label} · K562 (ENCODE, GRCh38) · Hi-C bin={actual_bin:,} bp",
             ha="center", va="top", fontsize=13, fontweight="bold")

    # Legend
    handles = [mpatches.Patch(color=BASE_COLORS[b], label=b) for b in BASE_ORDER]
    fig.legend(handles=handles, ncol=4, fontsize=8.5, title="one-hot DNA", title_fontsize=9,
               loc="lower right", bbox_to_anchor=(0.99, 0.015), frameon=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=350, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✓ Saved {out_path} ({fig_width}x{fig_height:.1f} in)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chrom", default="chr8")
    p.add_argument("--start", type=int, default=None)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--x-start", type=int, default=127600000)
    p.add_argument("--x-end", type=int, default=127800000)
    p.add_argument("--y-start", type=int, default=127800000)
    p.add_argument("--y-end", type=int, default=128000000)
    p.add_argument("--bin-size", type=int, default=5000)
    p.add_argument("--region-name", default="")
    p.add_argument("--workdir", default="encode_work")
    p.add_argument("--out", default="figures/myc_loop_v8.png")
    p.add_argument("--no-sequence", action="store_true")
    p.add_argument("--fig-width", type=float, default=21.0)
    p.add_argument("--clip-pct", type=float, default=95.0)
    args = p.parse_args()

    if args.start is not None and args.end is not None:
        args.x_start = args.y_start = args.start
        args.x_end = args.y_end = args.end

    raw = Path(args.workdir) / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    print("ENCODE multi-track plotter v8 - Fixed y-axis overlap")

    print("\nDownloading / caching data...")
    ctcf_path = download_file(CTCF_URL, raw / "ctcf.bigWig")
    h3k_path = download_file(h3k_url, raw / "h3k27ac.bigWig")
    dnase_path = download_file(dnase_url, raw / "dnase.bigWig")

    plot_region(
        chrom=args.chrom,
        x_start=args.x_start, x_end=args.x_end,
        y_start=args.y_start, y_end=args.y_end,
        bin_size=args.bin_size,
        hic_url=HIC_URL,
        ctcf_bw=ctcf_path,
        h3k27ac_bw=h3k_path,
        dnase_bw=dnase_path,
        out_path=Path(args.out),
        region_name=args.region_name,
        show_seq=not args.no_sequence,
        fig_width=args.fig_width,
        clip_pct=args.clip_pct,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())