"""
modal_pipeline.py
-----------------
Hi-C contact map prediction from 1D epigenomic tracks.
1kb resolution, 100x100 output, K562 + HepG2, GPU training on Modal.

Run:
    modal run modal_pipeline.py                  # all 3 models
    modal run modal_pipeline.py --model cnn      # one model
    modal run modal_pipeline.py --prepare-only   # data prep only

Before re-running after deleting volumes or changing cell lines:
    modal volume rm hic-data /loop_manifest.csv
    modal volume rm hic-runs /normalizer.json
"""
from __future__ import annotations
import sys, math
from pathlib import Path
import modal

app = modal.App("hic-loop-prediction")

vol_data  = modal.Volume.from_name("hic-data",  create_if_missing=True)
vol_cache = modal.Volume.from_name("hic-cache", create_if_missing=True)
vol_runs  = modal.Volume.from_name("hic-runs",  create_if_missing=True)

VOLUME_DATA  = Path("/mnt/data")
VOLUME_CACHE = Path("/mnt/cache")
VOLUME_RUNS  = Path("/mnt/runs")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libcurl4-openssl-dev", "build-essential", "zlib1g-dev")
    .pip_install(
        "torch==2.3.1", "numpy", "pandas",
        "pyBigWig", "hic-straw", "pyyaml", "requests", "pyliftover",
    )
    .add_local_dir(Path(__file__).parent / "src", remote_path="/src")
    .add_local_file(
        Path(__file__).parent / "configs" / "encode_sources.yaml",
        remote_path="/configs/encode_sources.yaml")
    .add_local_file(
        Path(__file__).parent / "assets" / "hg19ToHg38.over.chain.gz",
        remote_path="/assets/hg19ToHg38.over.chain.gz")
)


def _cfg():
    import yaml
    with open("/configs/encode_sources.yaml") as f:
        return yaml.safe_load(f)


# ── Data prep ────────────────────────────────────────────────────────────────
@app.function(image=image,
              volumes={VOLUME_DATA: vol_data, VOLUME_CACHE: vol_cache},
              timeout=60*60*2, cpu=4)
def prepare_data():
    import gzip, random, shutil, csv
    sys.path.insert(0, "/src")
    from genomic_io import download_file
    from genome_constants import chrom_sizes_hg38
    from pyliftover import LiftOver

    cfg = _cfg()
    exp = cfg["experiment"]
    raw = VOLUME_DATA / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    manifest_path = VOLUME_DATA / "loop_manifest.csv"

    if manifest_path.exists():
        import pandas as pd
        existing = set(pd.read_csv(manifest_path)["cell_line"].unique())
        needed   = set(cfg["cell_lines"].keys())
        if existing == needed:
            print(f"[prepare] Manifest exists for {existing}. Skipping.")
            return
        print(f"[prepare] Cell lines changed ({existing} -> {needed}). Rebuilding.")
        manifest_path.unlink()

    # Liftover chain (bundled as local asset, not downloaded at runtime)
    chain_gz   = raw / cfg["liftover"]["chain_filename"]
    chain_path = chain_gz.with_suffix("")
    if not chain_gz.exists():
        shutil.copy("/assets/hg19ToHg38.over.chain.gz", chain_gz)
    if not chain_path.exists():
        with gzip.open(chain_gz, "rb") as fi, open(chain_path, "wb") as fo:
            shutil.copyfileobj(fi, fo)
    lo = LiftOver(str(chain_path))
    print("[prepare] Liftover chain ready.")

    # Download all tracks and loop files
    for cl_name, cl_cfg in cfg["cell_lines"].items():
        print(f"\n[prepare] {cl_name}: downloading tracks and loops...")
        download_file(cl_cfg["ctcf_url"],    raw / f"{cl_name}_ctcf.bigWig")
        download_file(cl_cfg["h3k27ac_url"], raw / f"{cl_name}_h3k27ac.bigWig")
        download_file(cl_cfg["dnase_url"],   raw / f"{cl_name}_dnase.bigWig")
        download_file(cl_cfg["loop_url"],    raw / cl_cfg["loop_filename"])
        download_file(cl_cfg["hic_url"],     raw / f"{cl_name}.hic")
        print(f"[prepare] {cl_name}: done.")

    def parse_hiccups_gz(gz_path):
        """Rao 2014 HiCCUPS format: hg19 tab-separated gzip.
        Columns: chr1 x1 x2 chr2 y1 y2 ... (intra-chromosomal only)."""
        loops = []
        with gzip.open(gz_path, "rt") as fh:
            for line in fh:
                if line.startswith("#") or line.startswith("chr1\t"):
                    continue
                p = line.strip().split("\t")
                if len(p) < 6: continue
                c1, x1, x2 = p[0], int(p[1]), int(p[2])
                c2, y1, y2 = p[3], int(p[4]), int(p[5])
                if c1 != c2: continue
                chrom = c1 if c1.startswith("chr") else f"chr{c1}"
                loops.append({
                    "chrom": chrom,
                    "mid":  (x1 + x2 + y1 + y2) // 4,
                    "span": (y1 + y2) // 2 - (x1 + x2) // 2,
                })
        return loops

    def parse_encode_bedpe_gz(gz_path):
        """ENCODE bedpe.gz format: already hg38, gzipped.
        Columns: chr1 s1 e1 chr2 s2 e2 [name score ...]"""
        loops = []
        with gzip.open(gz_path, "rt") as fh:
            for line in fh:
                if line.startswith("#"): continue
                p = line.strip().split("\t")
                if len(p) < 6: continue
                c1, x1, x2 = p[0], int(p[1]), int(p[2])
                c2, y1, y2 = p[3], int(p[4]), int(p[5])
                if c1 != c2: continue
                chrom = c1 if c1.startswith("chr") else f"chr{c1}"
                loops.append({
                    "chrom": chrom,
                    "mid":  (x1 + x2 + y1 + y2) // 4,
                    "span": (y1 + y2) // 2 - (x1 + x2) // 2,
                })
        return loops

    def liftover(loops):
        result = []
        for l in loops:
            mapped = lo.convert_coordinate(l["chrom"], int(l["mid"]))
            if not mapped: continue
            result.append({**l,
                "chrom": mapped[0][0],
                "mid":   int(round(float(mapped[0][1]))),
            })
        return result

    def place_window(mid, chrom, window_size, jitter, rng, sizes):
        clen = sizes.get(chrom)
        if clen is None: return None
        center = mid + rng.randint(-jitter, jitter)
        start  = center - window_size // 2
        end    = start  + window_size
        if start < 0 or end > clen: return None
        return start, end

    rng         = random.Random(exp["seed"])
    sizes       = chrom_sizes_hg38()
    window_size = exp["window_size"]
    jitter      = exp["jitter"]
    bin_size    = exp["bin_size"]
    n_bins      = window_size // bin_size
    train_chroms = set(exp["train_chroms"])
    test_chroms  = set(exp["test_chroms"])

    rows = []
    for cl_name, cl_cfg in cfg["cell_lines"].items():
        loop_source = cl_cfg["loop_source"]
        loop_file   = raw / cl_cfg["loop_filename"]

        if loop_source == "geo_hiccups_hg19":
            loops_raw  = parse_hiccups_gz(loop_file)
            print(f"\n[prepare] {cl_name}: {len(loops_raw)} loops (hg19) -> liftover...")
            loops_hg38 = liftover(loops_raw)
            print(f"[prepare] {cl_name}: {len(loops_hg38)} loops (hg38)")
        elif loop_source == "encode_bedpe_hg38":
            loops_hg38 = parse_encode_bedpe_gz(loop_file)
            print(f"\n[prepare] {cl_name}: {len(loops_hg38)} loops (hg38, ENCFF050EKS/ENCSR194SRI)")
        else:
            raise ValueError(f"Unknown loop_source '{loop_source}' for {cl_name}")

        for split, chroms, n_target in [
            ("train", train_chroms, exp["train_loops_per_cell_line"]),
            ("test",  test_chroms,  exp["test_loops_per_cell_line"]),
        ]:
            eligible = [l for l in loops_hg38 if l["chrom"] in chroms]
            rng.shuffle(eligible)
            placed = 0
            for loop in eligible:
                if placed >= n_target: break
                result = place_window(
                    loop["mid"], loop["chrom"], window_size, jitter, rng, sizes)
                if result is None: continue
                start, end = result
                rows.append({
                    "split": split, "cell_line": cl_name,
                    "chrom": loop["chrom"], "start": start, "end": end,
                    "bin_size": bin_size, "n_bins": n_bins,
                    "loop_mid": loop["mid"], "loop_span_bp": loop["span"],
                })
                placed += 1
            status = "WARNING: only" if placed < n_target else " "
            print(f"  {status} {placed}/{n_target} windows: {cl_name}/{split}")

    fieldnames = ["split","cell_line","chrom","start","end",
                  "bin_size","n_bins","loop_mid","loop_span_bp"]
    with open(manifest_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)

    vol_data.commit()
    print(f"\n[prepare] Done: {len(rows)} windows -> {manifest_path}")


# ── Train one model ───────────────────────────────────────────────────────────
@app.function(image=image, gpu="A10G",
              volumes={VOLUME_DATA: vol_data, VOLUME_CACHE: vol_cache,
                       VOLUME_RUNS: vol_runs},
              timeout=60*60*6, cpu=4, memory=65536)
def train_model(model_name: str) -> dict:
    import csv, json, time
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset
    sys.path.insert(0, "/src")
    from metrics import mse_loss, huber_loss, evaluate_batch
    from genomic_io import hic_matrix, read_bigwig

    cfg    = _cfg()
    tr_cfg = cfg["training"]
    raw    = VOLUME_DATA / "raw"

    # ── Dataset ──────────────────────────────────────────────────────────────
    class LoopDataset(Dataset):
        def __init__(self, manifest_path, split, normalizer=None,
                     cache_dir=None, clip_pct=99.0):
            import pandas as pd
            full = pd.read_csv(manifest_path)
            self.manifest   = full[full["split"]==split].reset_index(drop=True)
            self.normalizer = normalizer
            self.clip_pct   = clip_pct
            self.cache_root = Path(cache_dir) if cache_dir else None
            if self.cache_root: self.cache_root.mkdir(parents=True, exist_ok=True)

        def __len__(self): return len(self.manifest)

        def _paths(self, cl):
            return {
                "ctcf": str(raw / f"{cl}_ctcf.bigWig"),
                "h3k27ac": str(raw / f"{cl}_h3k27ac.bigWig"),
                "dnase": str(raw / f"{cl}_dnase.bigWig"),
                "hic": str(raw / f"{cl}.hic"),
            }

        def _cache(self, row):
            if self.cache_root is None: return None
            # oe_v1_{bin_size}: encodes both target type and resolution.
            # Prevents 1kb cached files loading for 5kb windows.
            bs = int(row["bin_size"])
            return self.cache_root / (
                f"{row['cell_line']}_{row['chrom']}_"
                f"{row['start']}_{row['end']}_oe_v1_{bs}.npz")

        def get_tracks(self, idx):
            row = self.manifest.iloc[idx]
            p   = self._paths(row["cell_line"])
            n   = int(row["n_bins"])
            ch, s, e = row["chrom"], int(row["start"]), int(row["end"])
            ctcf  = read_bigwig(p["ctcf"],    ch, s, e, n, self.clip_pct)
            h3k   = read_bigwig(p["h3k27ac"], ch, s, e, n, self.clip_pct)
            dnase = read_bigwig(p["dnase"],   ch, s, e, n, self.clip_pct)
            return np.stack([ctcf, h3k, dnase], axis=0)

        def __getitem__(self, idx):
            row  = self.manifest.iloc[idx]
            cl   = row["cell_line"]
            ch, s, e = row["chrom"], int(row["start"]), int(row["end"])
            bs   = int(row["bin_size"])
            cf   = self._cache(row)

            if cf is not None and cf.exists():
                data   = np.load(cf)
                tracks = data["tracks"]
                log1p_mat = data["log1p_mat"]   # cached as raw log1p; O/E computed below
            else:
                tracks = self.get_tracks(idx)
                mat, actual = hic_matrix(self._paths(cl)["hic"], ch, s, e, bs)
                if actual != bs:
                    raise RuntimeError(
                        f"{cl} {ch}:{s}-{e}: requested {bs}bp but got {actual}bp. "
                        f"Change bin_size in encode_sources.yaml and delete the "
                        f"manifest: modal volume rm hic-data /loop_manifest.csv")
                log1p_mat = np.log1p(mat).astype(np.float32)
                if cf is not None:
                    np.savez_compressed(cf,
                        tracks=tracks.astype(np.float32),
                        log1p_mat=log1p_mat)   # store log1p; O/E computed at load time

            # Compute O/E: subtract the mean log1p contact at each genomic distance.
            # This removes the dominant distance-decay trend, leaving only loop/TAD
            # structure as the signal the model must predict.
            # O/E targets are signed (range ~[-1.5, +2.5]) and mean-zero per distance band.
            n = log1p_mat.shape[0]
            target = log1p_mat.copy()
            for d in range(n):
                if d == 0:
                    mean_d = np.diag(log1p_mat).mean()
                    np.fill_diagonal(target, np.diag(log1p_mat) - mean_d)
                else:
                    rows = np.arange(n - d); cols = rows + d
                    vals = np.concatenate([log1p_mat[rows, cols], log1p_mat[cols, rows]])
                    mean_d = vals.mean()
                    target[rows, cols] -= mean_d
                    target[cols, rows] -= mean_d

            if self.normalizer is not None:
                s_n = self.normalizer[cl]
                tracks = (tracks - s_n["mean"][:, None]) / s_n["std"][:, None]

            return {
                "tracks":    torch.from_numpy(tracks).float(),
                "target":    torch.from_numpy(target).float(),
                "cell_line": cl,
            }

    manifest  = VOLUME_DATA / "loop_manifest.csv"
    run_dir   = VOLUME_RUNS  / model_name
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = VOLUME_CACHE / "windows"

    norm_path = VOLUME_RUNS / "normalizer.json"
    config_cls = set(cfg["cell_lines"].keys())
    need_refit = True
    if norm_path.exists():
        with open(norm_path) as f: normalizer = json.load(f)
        if set(normalizer.keys()) == config_cls:
            need_refit = False
            print(f"[train] Loaded normalizer for {config_cls}.")
        else:
            print(f"[train] Normalizer covers {set(normalizer.keys())}, "
                f"need {config_cls}. Refitting.")

    if need_refit:
        import pandas as pd
        # use reset_index so positions == raw_ds positions
        train_df = pd.read_csv(manifest)
        train_df = train_df[train_df["split"] == "train"].reset_index(drop=True)
        raw_ds = LoopDataset(manifest, "train", normalizer=None, cache_dir=cache_dir)

        normalizer = {}
        for cl in train_df["cell_line"].unique():
            cl_pos = train_df[train_df["cell_line"] == cl].index.tolist()
            n_s = min(80, len(cl_pos))
            sample_idx = np.linspace(0, len(cl_pos)-1, n_s).astype(int)
            sums = np.zeros(3); sq_sums = np.zeros(3); count = 0
            for si in sample_idx:
                global_pos = cl_pos[si] # 0.. len(train_df)-1
                t = raw_ds.get_tracks(global_pos)
                sums += t.sum(axis=1)
                sq_sums += (t**2).sum(axis=1)
                count += t.shape[1]
            mean = sums / count
            std = np.sqrt(np.clip(sq_sums/count - mean**2, 1e-8, None))
            normalizer[cl] = {"mean": mean.tolist(), "std": std.tolist()}
            print(f" {cl}: CTCF_mean={mean[0]:.3f} "
                f"H3K27ac_mean={mean[1]:.3f} DNase_mean={mean[2]:.3f}")
        with open(norm_path, "w") as f: json.dump(normalizer, f, indent=2)
        vol_runs.commit()

    norm_np = {cl: {
        "mean": np.array(v["mean"], dtype=np.float32),
        "std":  np.array(v["std"],  dtype=np.float32),
    } for cl, v in normalizer.items()}

    # ── Dataloaders ───────────────────────────────────────────────────────
    train_ds = LoopDataset(manifest, "train", normalizer=norm_np, cache_dir=cache_dir)
    test_ds  = LoopDataset(manifest, "test",  normalizer=norm_np, cache_dir=cache_dir)
    print(f"[train] {len(train_ds)} train, {len(test_ds)} test windows")

    bs = tr_cfg["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False,
                              num_workers=0, pin_memory=True)
    n_bins = int(train_ds.manifest["n_bins"].iloc[0])

    # ── Model ─────────────────────────────────────────────────────────────
    device = "cuda"
    if model_name == "cnn":
        from model_cnn import CNNHiCModel
        model     = CNNHiCModel(n_tracks=3, n_bins=n_bins).to(device)
        wd        = float(tr_cfg["weight_decay"])
        clip_norm = None
    elif model_name == "unet":
        from model_unet import UNetHiCModel
        model     = UNetHiCModel(n_tracks=3, n_bins=n_bins).to(device)
        wd        = float(tr_cfg["weight_decay"])
        clip_norm = None
    elif model_name == "transformer":
        from model_transformer import TransformerHiCModel
        dropout   = float(tr_cfg.get("transformer_dropout", 0.3))
        model     = TransformerHiCModel(n_tracks=3, n_bins=n_bins,
                                        dropout=dropout).to(device)
        wd        = float(tr_cfg.get("transformer_weight_decay", 1e-3))
        clip_norm = float(tr_cfg.get("transformer_grad_clip_norm", 0.5))
        print(f"[train] Transformer: dropout={dropout} wd={wd} clip={clip_norm}")
    else:
        raise ValueError(f"Unknown model: {model_name}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] {model_name}: {n_params:,} params")

    epochs        = tr_cfg["epochs"][model_name]
    patience      = tr_cfg.get("early_stopping_patience", 10)
    warmup        = (tr_cfg.get("transformer_warmup_steps", 0)
                    if model_name == "transformer" else 0)
    total_steps   = len(train_loader) * epochs

    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=float(tr_cfg["lr"]), weight_decay=wd)

    # Warmup + cosine decay schedule (cosine only for Transformer)
    def lr_lambda(step):
        if warmup <= 0:
            return 1.0
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return max(0.05, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if warmup > 0:
        print(f"[train] LR: warmup {warmup} steps, cosine decay to 5% "
              f"over {total_steps-warmup} remaining steps")

    # ── Training loop ─────────────────────────────────────────────────────
    log_path  = run_dir / "training_log.csv"
    ckpt_path = run_dir / "best_model.pt"
    best_test_loss    = float("inf")
    epochs_no_improve = 0
    stopped_at        = None

    fields = ["epoch", "train_huber", "train_mse", "train_corr",
               "test_huber", "test_mse", "test_corr",
               "epoch_seconds", "lr"]

    with open(log_path, "w", newline="") as log_fh:
        writer = csv.DictWriter(log_fh, fieldnames=fields)
        writer.writeheader()

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            # Train
            model.train()
            tr_hub = tr_mse = tr_corr = n_tr = 0; last_lr = None
            for batch in train_loader:
                tracks = batch["tracks"].to(device, non_blocking=True)
                target = batch["target"].to(device, non_blocking=True)
                optimizer.zero_grad()
                pred = model(tracks)
                loss = huber_loss(pred, target, delta=0.5)
                loss.backward()
                if clip_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step(); scheduler.step()
                last_lr  = scheduler.get_last_lr()[0]
                m        = evaluate_batch(pred, target)
                tr_hub  += loss.item()
                tr_mse  += mse_loss(pred, target).item()
                tr_corr += m["stratum_adjusted_corr"]
                n_tr    += 1
            tr_hub /= max(1, n_tr)
            tr_mse /= max(1, n_tr)
            tr_corr/= max(1, n_tr)

            # Eval
            model.eval(); te_hub = te_mse = te_corr = n_te = 0
            with torch.no_grad():
                for batch in test_loader:
                    tracks = batch["tracks"].to(device, non_blocking=True)
                    target = batch["target"].to(device, non_blocking=True)
                    pred   = model(tracks)
                    m      = evaluate_batch(pred, target)
                    te_hub  += huber_loss(pred, target, delta=0.5).item()
                    te_mse  += mse_loss(pred, target).item()
                    te_corr += m["stratum_adjusted_corr"]
                    n_te    += 1
            te_hub /= max(1, n_te)
            te_mse /= max(1, n_te)
            te_corr/= max(1, n_te)

            dt = time.time() - t0
            print(f"  epoch {epoch:3d}/{epochs}  "
                  f"tr_hub={tr_hub:.4f} tr_mse={tr_mse:.4f} corr={tr_corr:.3f}  "
                  f"te_hub={te_hub:.4f} te_mse={te_mse:.4f} corr={te_corr:.3f}  "
                  f"lr={last_lr:.2e}  ({dt:.1f}s)")
            writer.writerow({
                "epoch": epoch,
                "train_huber": tr_hub, "train_mse": tr_mse, "train_corr": tr_corr,
                "test_huber":  te_hub, "test_mse":  te_mse, "test_corr":  te_corr,
                "epoch_seconds": dt, "lr": last_lr,
            })
            log_fh.flush()

            # Checkpoint on best test Huber (not MSE -- see metrics.py)
            if te_hub < best_test_loss:
                best_test_loss    = te_hub
                epochs_no_improve = 0
                torch.save({
                    "model_name":       model_name,
                    "n_bins":           n_bins,
                    "epoch":            epoch,
                    "model_state_dict": model.state_dict(),
                    "train_huber":      tr_hub,
                    "train_mse":        tr_mse,
                    "test_huber":       te_hub,
                    "test_mse":         te_mse,
                    "test_corr":        te_corr,
                }, ckpt_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    stopped_at = epoch
                    print(f"[train] Early stop epoch {epoch} "
                          f"(no improvement for {patience} epochs)")
                    break

    # Per-cell-line breakdown on best checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    per_cl = {cl: {"hub": 0., "corr": 0., "corr_n": 0, "n": 0}
               for cl in cfg["cell_lines"]}
    nan_windows = {cl: [] for cl in cfg["cell_lines"]}  # for locus reporting
    with torch.no_grad():
        for batch in test_loader:
            tracks = batch["tracks"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            pred   = model(tracks)
            for i in range(len(tracks)):
                cl = batch["cell_line"][i]
                if cl not in per_cl: continue
                m = evaluate_batch(pred[i:i+1], target[i:i+1])
                per_cl[cl]["hub"] += huber_loss(
                    pred[i:i+1], target[i:i+1], delta=0.5).item()
                per_cl[cl]["n"]   += 1
                corr_val = m["stratum_adjusted_corr"]
                if np.isnan(corr_val):
                    # Record which window degenerated instead of silently
                    # poisoning the running average. chrom/start/end aren't
                    # in `batch` by default -- see note below to add them.
                    nan_windows[cl].append(i)
                else:
                    per_cl[cl]["corr"]    += corr_val
                    per_cl[cl]["corr_n"]  += 1

    cl_results = {}
    print(f"\n  Per-cell-line (best checkpoint, epoch {ckpt['epoch']}):")
    for cl, v in per_cl.items():
        n = max(1, v["n"])
        corr_n = v["corr_n"]
        mean_corr = v["corr"] / corr_n if corr_n > 0 else float("nan")
        cl_results[cl] = {
            "test_huber": v["hub"] / n,
            "test_corr":  mean_corr,
            "corr_valid_windows": corr_n,
            "n": n,
        }
        n_nan = len(nan_windows[cl])
        flag = f"  *** {n_nan}/{n} windows NaN (degenerate/constant) ***" if n_nan else ""
        print(f"    {cl}: huber={v['hub']/n:.4f}  corr={mean_corr:.4f}  "
              f"(valid {corr_n}/{n}){flag}")
        
    vol_runs.commit(); vol_cache.commit()
    return {
        "model":            model_name,
        "best_test_huber":  best_test_loss,
        "best_epoch":       int(ckpt["epoch"]),
        "best_test_corr":   float(ckpt["test_corr"]),
        "stopped_early_at": stopped_at,
        "per_cell_line":    cl_results,
    }


# ── Local entrypoint ─────────────────────────────────────────────────────────
@app.local_entrypoint()
def main(model: str = "all", prepare_only: bool = False):
    """
    modal run modal_pipeline.py
    modal run modal_pipeline.py --model cnn
    modal run modal_pipeline.py --prepare-only

    If re-running after a previous cell-line config (e.g. K562-only):
      modal volume rm hic-data /loop_manifest.csv
      modal volume rm hic-runs /normalizer.json
    """
    print("=" * 65)
    print("Hi-C loop prediction  (K562 + HepG2, 1kb, 100x100)")
    print("=" * 65)

    print("\n[Step 1] Preparing data...")
    prepare_data.remote()
    print("[Step 1] Done.")
    if prepare_only:
        return

    models = ["cnn", "unet", "transformer"] if model == "all" else [model]
    print(f"\n[Step 2] Training: {models}")
    results = []
    for m in models:
        print(f"\n  -> {m.upper()} on A10G...")
        r = train_model.remote(m); results.append(r)
        s = f" (early@{r['stopped_early_at']})" if r["stopped_early_at"] else ""
        print(f"  <- {m.upper()}{s}: "
              f"best_test_huber={r['best_test_huber']:.4f}  "
              f"corr={r['best_test_corr']:.3f}  epoch={r['best_epoch']}")

    print("\n" + "=" * 65)
    print("SUMMARY (sorted by best test Huber loss)")
    print("=" * 65)
    print(f"{'Model':12s} {'Epoch':>6s} {'TestHuber':>10s} "
          f"{'TestCorr':>9s}  K562_corr  HepG2_corr")
    print("-" * 65)
    for r in sorted(results, key=lambda x: x["best_test_huber"]):
        cl     = r["per_cell_line"]
        k_corr = cl.get("K562",  {}).get("test_corr", float("nan"))
        h_corr = cl.get("HepG2", {}).get("test_corr", float("nan"))
        s      = f"  *stopped@{r['stopped_early_at']}" if r["stopped_early_at"] else ""
        print(f"{r['model']:12s} {r['best_epoch']:>6d} "
              f"{r['best_test_huber']:>10.4f} {r['best_test_corr']:>9.4f}  "
              f"{k_corr:>9.4f}  {h_corr:>9.4f}{s}")

    print()
    print("TestHuber = Huber loss (delta=0.5), not MSE.")
    print("stratum_adjusted_corr = distance-corrected correlation (primary metric).")
    print("K562_corr vs HepG2_corr: if gap > 0.05, model is cell-line-biased.")
    print("Retrieve: modal volume get hic-runs / ./runs_local/")