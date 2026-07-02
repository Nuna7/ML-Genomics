"""
modal_pipeline.py 
--------------------------------
K562 + HepG2, 1kb resolution, 100x100 output matrices.

Loop sources:
  K562:  Rao 2014 HiCCUPS (GSE63525 GEO), hg19 -> lifted to hg38
  HepG2: 3D Genome Browser pre-called loops (GSE105381, already hg38)
         Bundled as assets/HepG2/HepG2_GSE105381.bedpe (2.8MB, 28,950 loops)
         Source: 3dgenome.fsm.northwestern.edu/datasets?id=98
         No runtime loop calling. No 22GB download.

HepG2 Hi-C for training: ENCFF020DPP (22.6GB, ENCODE4, hg38)
  Streamed via hicstraw URL - only 100kb windows fetched per training example.

Run:
  modal run modal_pipeline.py
  modal run modal_pipeline.py --model cnn
  modal run modal_pipeline.py --prepare-only

Before re-running after a K562-only v1 run:
  modal volume rm hic-data /loop_manifest.csv
  modal volume rm hic-runs /normalizer.json
"""
from __future__ import annotations
import sys
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
        "torch==2.3.1",
        "numpy", "pandas", "pyBigWig", "hic-straw",
        "pyyaml", "requests", "pyliftover",
    )
    .add_local_dir(Path(__file__).parent / "src", remote_path="/src")
    .add_local_file(
        Path(__file__).parent / "configs" / "encode_sources.yaml",
        remote_path="/configs/encode_sources.yaml")
    .add_local_file(
        Path(__file__).parent / "assets" / "hg19ToHg38.over.chain.gz",
        remote_path="/assets/hg19ToHg38.over.chain.gz")
    .add_local_file(
        Path(__file__).parent / "assets" / "HepG2" / "HepG2_GSE105381.bedpe",
        remote_path="/assets/HepG2_GSE105381.bedpe")
)


def _load_config():
    import yaml
    with open("/configs/encode_sources.yaml") as f:
        return yaml.safe_load(f)


# ── STEP 1: Data prep ────────────────────────────────────────────────────────
@app.function(image=image,
              volumes={VOLUME_DATA: vol_data, VOLUME_CACHE: vol_cache},
              timeout=60*60*2, cpu=4)
def prepare_data():
    import gzip, random, shutil, csv
    sys.path.insert(0, "/src")
    from genomic_io import download_file
    from genome_constants import chrom_sizes_hg38

    cfg = _load_config()
    exp = cfg["experiment"]
    raw = VOLUME_DATA / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    manifest_path = VOLUME_DATA / "loop_manifest.csv"

    if manifest_path.exists():
        import pandas as pd
        existing_cls = set(pd.read_csv(manifest_path)["cell_line"].unique())
        config_cls = set(cfg["cell_lines"].keys())
        if existing_cls == config_cls:
            print(f"[prepare] Manifest exists with {existing_cls}. Skipping.")
            return
        print(f"[prepare] Cell lines changed ({existing_cls} -> {config_cls}). Rebuilding.")
        manifest_path.unlink()

    # Liftover chain (for K562 hg19 loops)
    chain_gz   = raw / cfg["liftover"]["chain_filename"]
    chain_path = chain_gz.with_suffix("")
    if not chain_gz.exists():
        shutil.copy("/assets/hg19ToHg38.over.chain.gz", chain_gz)
    if not chain_path.exists():
        with gzip.open(chain_gz, "rb") as fi, open(chain_path, "wb") as fo:
            shutil.copyfileobj(fi, fo)
    print("[prepare] Liftover chain ready.")

    from pyliftover import LiftOver
    lo = LiftOver(str(chain_path))

    # Download epigenomic tracks
    for cl_name, cl_cfg in cfg["cell_lines"].items():
        print(f"\n[prepare] {cl_name}: downloading tracks...")
        download_file(cl_cfg["ctcf_url"],    raw / f"{cl_name}_ctcf.bigWig")
        download_file(cl_cfg["h3k27ac_url"], raw / f"{cl_name}_h3k27ac.bigWig")
        download_file(cl_cfg["dnase_url"],   raw / f"{cl_name}_dnase.bigWig")
        # K562 loop file from GEO
        if cl_cfg.get("loop_source") == "geo":
            download_file(cl_cfg["loop_url"], raw / cl_cfg["loop_filename"])
        # HepG2 loop file: copy from bundled asset
        elif cl_cfg.get("loop_source") == "bundled":
            dest = raw / cl_cfg["loop_filename"]
            if not dest.exists():
                shutil.copy(cl_cfg["loop_asset"], dest)
                print(f"  [copy] {cl_cfg['loop_asset']} -> {dest}")
        print(f"[prepare] {cl_name}: done.")

    def parse_hiccups_gz(gz_path):
        """K562: Rao 2014 HiCCUPS format, hg19, gzipped."""
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
                mid = (x1 + x2 + y1 + y2) // 4
                span = (y1 + y2) // 2 - (x1 + x2) // 2
                loops.append({"chrom": chrom, "mid": mid, "span": span})
        return loops

    def parse_bedpe_hg38(bedpe_path):
        """HepG2: 3D Genome Browser pre-called loops, already hg38, no liftover needed."""
        loops = []
        with open(bedpe_path) as fh:
            for line in fh:
                if line.startswith("#"): continue
                p = line.strip().split("\t")
                if len(p) < 6: continue
                c1, x1, x2 = p[0], int(p[1]), int(p[2])
                c2, y1, y2 = p[3], int(p[4]), int(p[5])
                if c1 != c2: continue
                chrom = c1 if c1.startswith("chr") else f"chr{c1}"
                mid = (x1 + x2 + y1 + y2) // 4
                span = (y1 + y2) // 2 - (x1 + x2) // 2
                loops.append({"chrom": chrom, "mid": mid, "span": span})
        return loops

    def liftover_hg19_to_hg38(loops):
        result = []
        for l in loops:
            mapped = lo.convert_coordinate(l["chrom"], int(l["mid"]))
            if not mapped: continue
            new_chrom = mapped[0][0]
            new_mid   = int(round(float(mapped[0][1])))
            result.append({**l, "chrom": new_chrom, "mid": new_mid})
        return result

    def place_window(mid, chrom, window_size, jitter, rng, sizes):
        clen = sizes.get(chrom)
        if clen is None: return None
        center = mid + rng.randint(-jitter, jitter)
        start  = center - window_size // 2
        end    = start  + window_size
        if start < 0 or end > clen: return None
        return start, end

    rng = random.Random(exp["seed"])
    sizes       = chrom_sizes_hg38()
    window_size = exp["window_size"]
    jitter      = exp["jitter"]
    bin_size    = exp["bin_size"]
    n_bins      = window_size // bin_size
    train_chroms = set(exp["train_chroms"])
    test_chroms  = set(exp["test_chroms"])

    rows = []
    for cl_name, cl_cfg in cfg["cell_lines"].items():
        loop_source = cl_cfg.get("loop_source")
        loop_file   = raw / cl_cfg["loop_filename"]

        if loop_source == "geo":
            loops_raw   = parse_hiccups_gz(loop_file)
            print(f"\n[prepare] {cl_name}: {len(loops_raw)} loops (hg19) -> lifting to hg38...")
            loops_hg38  = liftover_hg19_to_hg38(loops_raw)
            print(f"[prepare] {cl_name}: {len(loops_hg38)} loops (hg38)")
        else:  # bundled, already hg38
            loops_hg38  = parse_bedpe_hg38(loop_file)
            print(f"\n[prepare] {cl_name}: {len(loops_hg38)} loops (hg38, pre-called, GSE105381)")

        for split, chroms, n_target in [
            ("train", train_chroms, exp["train_loops_per_cell_line"]),
            ("test",  test_chroms,  exp["test_loops_per_cell_line"]),
        ]:
            eligible = [l for l in loops_hg38 if l["chrom"] in chroms]
            rng.shuffle(eligible)
            placed = 0
            for loop in eligible:
                if placed >= n_target: break
                result = place_window(loop["mid"], loop["chrom"],
                                      window_size, jitter, rng, sizes)
                if result is None: continue
                start, end = result
                rows.append({
                    "split": split, "cell_line": cl_name,
                    "chrom": loop["chrom"], "start": start, "end": end,
                    "bin_size": bin_size, "n_bins": n_bins,
                    "loop_mid": loop["mid"], "loop_span_bp": loop["span"],
                })
                placed += 1
            status = "WARNING: only" if placed < n_target else "  "
            print(f"  {status} {placed}/{n_target} windows: {cl_name}/{split}")

    fieldnames = ["split","cell_line","chrom","start","end",
                  "bin_size","n_bins","loop_mid","loop_span_bp"]
    with open(manifest_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)

    vol_data.commit()
    print(f"\n[prepare] Done: {len(rows)} windows -> {manifest_path}")


# ── STEP 2: Train one model ───────────────────────────────────────────────────
@app.function(image=image, gpu="A10G",
              volumes={VOLUME_DATA: vol_data, VOLUME_CACHE: vol_cache, VOLUME_RUNS: vol_runs},
              timeout=60*60*6, cpu=4, memory=16384)
def train_model(model_name: str) -> dict:
    import csv, json, time
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset
    sys.path.insert(0, "/src")
    from metrics import mse_loss, huber_loss, evaluate_batch
    from genomic_io import hic_matrix, read_bigwig

    cfg    = _load_config()
    tr_cfg = cfg["training"]
    raw    = VOLUME_DATA / "raw"

    class LoopDataset(Dataset):
        def __init__(self, manifest_path, split, normalizer=None,
                     cache_dir=None, clip_pct=99.0):
            import pandas as pd
            full = pd.read_csv(manifest_path)
            self.manifest    = full[full["split"] == split].reset_index(drop=True)
            self.normalizer  = normalizer
            self.clip_pct    = clip_pct
            self.cache_root  = Path(cache_dir) if cache_dir else None
            if self.cache_root:
                self.cache_root.mkdir(parents=True, exist_ok=True)

        def __len__(self): return len(self.manifest)

        def _track_paths(self, cl):
            return {
                "ctcf":    raw / f"{cl}_ctcf.bigWig",
                "h3k27ac": raw / f"{cl}_h3k27ac.bigWig",
                "dnase":   raw / f"{cl}_dnase.bigWig",
                "hic":     cfg["cell_lines"][cl]["hic_url"],
            }

        def _cache_file(self, row):
            if self.cache_root is None: return None
            return self.cache_root / (
                f"{row['cell_line']}_{row['chrom']}"
                f"_{row['start']}_{row['end']}.npz"
            )

        def get_tracks(self, idx):
            row   = self.manifest.iloc[idx]
            paths = self._track_paths(row["cell_line"])
            n     = int(row["n_bins"])
            ch, s, e = row["chrom"], int(row["start"]), int(row["end"])
            ctcf  = read_bigwig(paths["ctcf"],    ch, s, e, n, self.clip_pct)
            h3k   = read_bigwig(paths["h3k27ac"], ch, s, e, n, self.clip_pct)
            dnase = read_bigwig(paths["dnase"],   ch, s, e, n, self.clip_pct)
            return np.stack([ctcf, h3k, dnase], axis=0)

        def __getitem__(self, idx):
            row  = self.manifest.iloc[idx]
            cl   = row["cell_line"]
            ch, s, e = row["chrom"], int(row["start"]), int(row["end"])
            bs   = int(row["bin_size"])
            paths = self._track_paths(cl)
            cf    = self._cache_file(row)

            if cf is not None and cf.exists():
                data   = np.load(cf)
                tracks = data["tracks"]
                target = data["target"]
            else:
                tracks = self.get_tracks(idx)
                mat, actual = hic_matrix(paths["hic"], ch, s, e, bs)
                if actual != bs:
                    raise RuntimeError(
                        f"Requested {bs}bp but got {actual}bp for {cl} {ch}:{s}-{e}.\n"
                        f"Change bin_size to {actual} in encode_sources.yaml and delete "
                        f"the manifest: modal volume rm hic-data /loop_manifest.csv"
                    )
                target = np.log1p(mat).astype(np.float32)
                if cf is not None:
                    np.savez_compressed(cf,
                        tracks=tracks.astype(np.float32), target=target)

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

    # Normalizer: fit on train only, rebuild if cell lines changed
    norm_path  = VOLUME_RUNS / "normalizer.json"
    config_cls = set(cfg["cell_lines"].keys())
    need_refit = True
    if norm_path.exists():
        with open(norm_path) as f: normalizer = json.load(f)
        if set(normalizer.keys()) == config_cls:
            need_refit = False
            print(f"[train] Loaded normalizer for {config_cls}.")
        else:
            print(f"[train] Normalizer mismatch; refitting for {config_cls}.")

    if need_refit:
        import pandas as pd
        raw_ds = LoopDataset(manifest, "train", normalizer=None, cache_dir=cache_dir)
        tm = pd.read_csv(manifest); tm = tm[tm["split"] == "train"]
        normalizer = {}
        for cl in tm["cell_line"].unique():
            idxs    = tm[tm["cell_line"] == cl].index.tolist()
            n_s     = min(80, len(idxs))
            sample  = np.linspace(0, len(idxs)-1, n_s).astype(int)
            sums    = np.zeros(3); sq_sums = np.zeros(3); count = 0
            for i in sample:
                t = raw_ds.get_tracks(idxs[i])
                sums    += t.sum(axis=1)
                sq_sums += (t**2).sum(axis=1)
                count   += t.shape[1]
            mean = sums / count
            std  = np.sqrt(np.clip(sq_sums / count - mean**2, 1e-8, None))
            normalizer[cl] = {"mean": mean.tolist(), "std": std.tolist()}
            print(f"  {cl}: means={[f'{v:.3f}' for v in mean]}")
        with open(norm_path, "w") as f: json.dump(normalizer, f, indent=2)
        vol_runs.commit()

    norm_np = {cl: {
        "mean": np.array(v["mean"], dtype=np.float32),
        "std":  np.array(v["std"],  dtype=np.float32),
    } for cl, v in normalizer.items()}

    train_ds = LoopDataset(manifest, "train", normalizer=norm_np, cache_dir=cache_dir)
    test_ds  = LoopDataset(manifest, "test",  normalizer=norm_np, cache_dir=cache_dir)
    print(f"[train] {len(train_ds)} train, {len(test_ds)} test windows")

    bs = tr_cfg["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False,
                              num_workers=2, pin_memory=True)
    n_bins = int(train_ds.manifest["n_bins"].iloc[0])

    device = "cuda"
    if model_name == "cnn":
        from model_cnn import CNNHiCModel
        model     = CNNHiCModel(n_tracks=3, n_bins=n_bins).to(device)
        wd        = float(tr_cfg["weight_decay"])
        clip_norm = None
        lr = float(tr_cfg.get("lr", 1e-3))
    elif model_name == "unet":
        from model_unet import UNetHiCModel
        model     = UNetHiCModel(n_tracks=3, n_bins=n_bins).to(device)
        wd        = float(tr_cfg["weight_decay"])
        clip_norm = None
        lr = float(tr_cfg.get("lr", 1e-3))
    elif model_name == "transformer":
        from model_transformer import TransformerHiCModel
        dropout   = float(tr_cfg.get("transformer_dropout", 0.3))
        model     = TransformerHiCModel(n_tracks=3, n_bins=n_bins,
                                        dropout=dropout).to(device)
        wd        = float(tr_cfg.get("transformer_weight_decay", 1e-3))
        clip_norm = float(tr_cfg.get("transformer_grad_clip_norm", 0.5))
        print(f"[train] Transformer: dropout={dropout} wd={wd} clip={clip_norm}")
        lr = float(tr_cfg.get("transformer_lr", 1e-4))
    else:
        raise ValueError(f"Unknown model: {model_name}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] {model_name}: {n_params:,} params")

    epochs   = tr_cfg["epochs"][model_name]
    patience = tr_cfg.get("early_stopping_patience", 10)
    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=lr, weight_decay=wd)

    warmup = (tr_cfg.get("transformer_warmup_steps", 0)
              if model_name == "transformer" else 0)
    total_steps = len(train_loader) * epochs
    if warmup > 0:
        print(f"[train] Warmup: {warmup} steps, then cosine decay to 0 "
              f"over remaining {total_steps - warmup} steps "
              f"({len(train_loader)}/epoch, {total_steps} total)")

    import math
    def lr_lambda(step):
        if warmup <= 0:
            return 1.0
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        progress = min(1.0, progress)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return max(0.05, cosine)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    log_path  = run_dir / "training_log.csv"
    ckpt_path = run_dir / "best_model.pt"
    best_test_loss   = float("inf")
    epochs_no_improve = 0
    stopped_at       = None

    fields = ["epoch","train_huber","train_mse","train_corr",
               "test_huber","test_mse","test_corr",
               "epoch_seconds","lr"]
    with open(log_path, "w", newline="") as log_fh:
        writer = csv.DictWriter(log_fh, fieldnames=fields)
        writer.writeheader()

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            # --- train ---
            model.train()
            tr_huber = tr_mse = tr_corr = n_tr = 0; last_lr = None
            for batch in train_loader:
                tracks = batch["tracks"].to(device, non_blocking=True)
                target = batch["target"].to(device, non_blocking=True)
                optimizer.zero_grad()
                pred = model(tracks)
                loss = huber_loss(pred, target, delta=1.0)   # actual backprop objective
                loss.backward()
                if clip_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step(); scheduler.step()
                last_lr   = scheduler.get_last_lr()[0]
                m         = evaluate_batch(pred, target)
                tr_huber += loss.item()
                tr_mse   += mse_loss(pred, target).item()    # reporting-only, not backprop
                tr_corr  += m["stratum_adjusted_corr"]
                n_tr     += 1
            tr_huber /= max(1, n_tr); tr_mse /= max(1, n_tr); tr_corr /= max(1, n_tr)

            # --- eval ---
            model.eval(); te_huber = te_mse = te_corr = n_te = 0
            with torch.no_grad():
                for batch in test_loader:
                    tracks = batch["tracks"].to(device, non_blocking=True)
                    target = batch["target"].to(device, non_blocking=True)
                    pred   = model(tracks)
                    m      = evaluate_batch(pred, target)
                    te_huber += huber_loss(pred, target, delta=1.0).item()
                    te_mse   += mse_loss(pred, target).item()
                    te_corr  += m["stratum_adjusted_corr"]
                    n_te     += 1
            te_huber /= max(1, n_te); te_mse /= max(1, n_te); te_corr /= max(1, n_te)

            dt = time.time() - t0
            print(f"  epoch {epoch:3d}/{epochs}  "
                  f"train_huber={tr_huber:.4f} train_mse={tr_mse:.4f} corr={tr_corr:.3f}  "
                  f"test_huber={te_huber:.4f} test_mse={te_mse:.4f} corr={te_corr:.3f}  "
                  f"lr={last_lr:.2e}  ({dt:.1f}s)")
            writer.writerow({"epoch": epoch,
                             "train_huber": tr_huber, "train_mse": tr_mse, "train_corr": tr_corr,
                             "test_huber":  te_huber,  "test_mse":  te_mse,  "test_corr":  te_corr,
                             "epoch_seconds": dt, "lr": last_lr})
            log_fh.flush()

            if te_huber < best_test_loss:
                best_test_loss    = te_huber
                epochs_no_improve = 0
                torch.save({
                    "model_name":       model_name,
                    "n_bins":           n_bins,
                    "epoch":            epoch,
                    "model_state_dict": model.state_dict(),
                    "train_huber":      tr_huber,
                    "train_mse":        tr_mse,
                    "test_huber":       te_huber,
                    "test_mse":         te_mse,
                    "test_corr":        te_corr,
                }, ckpt_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    stopped_at = epoch
                    print(f"[train] Early stop at epoch {epoch} "
                          f"(no improvement for {patience} epochs)")
                    break

    # Per-cell-line breakdown on best checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    per_cl = {cl: {"loss": 0., "corr": 0., "n": 0}
               for cl in cfg["cell_lines"]}
    with torch.no_grad():
        for batch in test_loader:
            tracks = batch["tracks"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            pred   = model(tracks)
            for i in range(len(tracks)):
                cl = batch["cell_line"][i]
                if cl not in per_cl: continue
                m = evaluate_batch(pred[i:i+1], target[i:i+1])
                per_cl[cl]["loss"] += mse_loss(
                    pred[i:i+1], target[i:i+1]).item()
                per_cl[cl]["corr"] += m["stratum_adjusted_corr"]
                per_cl[cl]["n"]    += 1

    cl_results = {}
    print(f"\n  Per-cell-line (epoch {ckpt['epoch']}):")
    for cl, v in per_cl.items():
        n = max(1, v["n"])
        cl_results[cl] = {
            "test_loss": v["loss"] / n,
            "test_corr": v["corr"] / n,
            "n": v["n"],
        }
        print(f"    {cl}: loss={v['loss']/n:.4f}  corr={v['corr']/n:.4f}  n={v['n']}")

    vol_runs.commit(); vol_cache.commit()
    return {
        "model":            model_name,
        "best_test_loss":   best_test_loss,
        "best_epoch":       int(ckpt["epoch"]),
        "final_test_corr":  float(ckpt["test_corr"]),
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
    """
    print("=" * 65)
    print("Hi-C loop prediction v3  (K562 + HepG2, 1kb, 100x100)")
    print("=" * 65)

    print("\n[Step 1] Preparing data...")
    prepare_data.remote()
    print("[Step 1] Done.")
    if prepare_only: return

    models_to_train = (["cnn", "unet", "transformer"]
                       if model == "all" else [model])
    print(f"\n[Step 2] Training: {models_to_train}")

    results = []
    for m in models_to_train:
        print(f"\n  -> {m.upper()} on A10G GPU...")
        r = train_model.remote(m); results.append(r)
        s = f" (early@{r['stopped_early_at']})" if r["stopped_early_at"] else ""
        print(f"  <- {m.upper()}{s}: "
              f"best_test_loss={r['best_test_loss']:.4f}  "
              f"corr={r['final_test_corr']:.3f}  "
              f"epoch={r['best_epoch']}")

    print("\n" + "=" * 65 + "\nSUMMARY\n" + "=" * 65)
    print(f"{'Model':12s} {'Epoch':>6s} {'TestHuber':>9s} "
          f"{'TestCorr':>9s}  K562_corr  HepG2_corr")
    print("-" * 65)
    for r in sorted(results, key=lambda x: x["best_test_loss"]):
        cl     = r["per_cell_line"]
        k_corr = cl.get("K562",  {}).get("test_corr", float("nan"))
        h_corr = cl.get("HepG2", {}).get("test_corr", float("nan"))
        s      = f"*stopped@{r['stopped_early_at']}" if r["stopped_early_at"] else ""
        print(f"{r['model']:12s} {r['best_epoch']:>6d} "
              f"{r['best_test_loss']:>9.4f} {r['final_test_corr']:>9.4f}  "
              f"{k_corr:>9.4f}  {h_corr:>9.4f}  {s}")

    print("\nNote: TestHuber is Huber loss (delta=1.0), not MSE -- not directly")
    print("comparable to earlier runs' MSE-based loss numbers. See per-run")
    print("training_log.csv for both train/test Huber AND MSE columns.")
    print("\nKey question: if K562_corr >> HepG2_corr, the model learned "
          "cell-line-specific patterns rather than general loop structure.")
    print("Retrieve: modal volume get hic-runs / ./runs_local/")