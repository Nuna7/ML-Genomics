# 

#!/usr/bin/env bash
# run_full_pipeline.sh
# ---------------------
# End-to-end orchestration: download data -> generate manifest -> train
# all three models -> produce comparison report.
#
# This is meant to run on YOUR machine (CPU or single GPU), not in any
# sandboxed environment -- it needs real network access to ENCODE/UCSC and
# meaningful wall-clock time (expect this to take from under an hour to
# several hours depending on your hardware and how many epochs you set
# below, since this script trains three separate models in sequence).
#
# Before running, edit the CONFIG section below.
#
# EPOCH BUDGETS: a real run on this codebase (4 train chromosomes, 1Mb
# windows @ 10kb, M1 CPU) showed CNN and UNet both essentially plateau
# (last-5-epoch val_loss std under 0.01) by epoch 20, while the
# Transformer's val_loss was still visibly noisy (std ~0.029, roughly
# 3-4x the other two) at epoch 20 and its val-correlation curve was still
# climbing, not flat. TRANSFORMER_EPOCHS is set higher than the other two
# for exactly this reason -- see model_transformer.py for why this
# architecture converges slower (near-uniform attention at
# initialization). All three are capped at 40 epochs here as a practical
# ceiling for repeated CPU runs; if the Transformer's curve still hasn't
# flattened by 40 epochs, that's worth reporting as-is rather than
# pushing epochs further before your meeting.
#
# WARMUP / GRAD CLIPPING: only applied to the Transformer run by default,
# since these were added specifically in response to two things observed
# in a real run: (1) the Transformer's near-uniform attention at init
# benefits from not taking large early steps before attention has
# sharpened (--warmup-steps), and (2) a single epoch showed a ~9x val_loss
# spike with a perfectly smooth train_loss that same epoch (most likely
# an unlucky val batch, but --grad-clip-norm is cheap insurance against
# the related failure mode of one extreme training batch producing an
# outsized gradient step). CNN/UNet did not show this instability in the
# real run, so they're left at their original (no warmup, no clipping)
# settings below -- uncomment the extra args on their training calls if
# you want to test these tricks on them too, but there's no evidence yet
# that they need it.

set -euo pipefail

# ============================ CONFIG ============================
WORKDIR="../encode_work"
MANIFEST="manifest.csv"
RUNS_DIR="runs"
REPORT_DIR="comparison_report"

TRAIN_CHROMS="chr1 chr2 chr3 chr8"
VAL_CHROMS="chr10"
TEST_CHROMS="chr17"
WINDOW_SIZE=1000000
STRIDE=1000000
BIN_SIZE=10000

BATCH_SIZE=8
LR=1e-3

# NOTE: deliberately different per model -- see comment above. 40 is the
# practical ceiling agreed for repeated CPU runs.
CNN_EPOCHS=40
UNET_EPOCHS=40
TRANSFORMER_EPOCHS=40

# Transformer-only stabilization settings (see comment above). With 833
# train windows / batch_size=8 -> ~105 steps/epoch -> ~4200 total steps
# at 40 epochs; 200 warmup steps is roughly 5%, a standard starting
# fraction. Adjust if your manifest size differs from this project's
# default chromosome split.
TRANSFORMER_WARMUP_STEPS=200
TRANSFORMER_GRAD_CLIP_NORM=1.0

DEVICE="cpu"   # change to "cuda" if you have a GPU and torch.cuda.is_available() is True
# ==================================================================

ENCODE_BASE="https://www.encodeproject.org"
HIC_URL="${ENCODE_BASE}/files/ENCFF291JZM/@@download/ENCFF291JZM.hic"
CTCF_URL="${ENCODE_BASE}/files/ENCFF000YMA/@@download/ENCFF000YMA.bigWig"
H3K_URL="${ENCODE_BASE}/files/ENCFF779QTH/@@download/ENCFF779QTH.bigWig"
DNASE_URL="${ENCODE_BASE}/files/ENCFF414OGC/@@download/ENCFF414OGC.bigWig"

mkdir -p "${WORKDIR}/raw" "${RUNS_DIR}" "${REPORT_DIR}"

echo "=============================================="
echo "[1/4] Downloading ENCODE tracks (cached after first run)"
echo "=============================================="
python3 - <<PYEOF
import sys
sys.path.insert(0, "src")
from genomic_io import download_file
from pathlib import Path

download_file("${CTCF_URL}", Path("${WORKDIR}/raw/ctcf.bigWig"))
download_file("${H3K_URL}", Path("${WORKDIR}/raw/h3k27ac.bigWig"))
download_file("${DNASE_URL}", Path("${WORKDIR}/raw/dnase.bigWig"))
print("All bigWig tracks downloaded/cached.")
print("NOTE: the .hic file itself is large; this pipeline streams it via")
print("hicstraw directly from the URL rather than downloading it whole, so")
print("there is no separate download step for it here -- the first")
print("training run will be slower as it fetches matrix data over the")
print("network for each window; consider downloading it locally with")
print("download_file() too if you'll run training many times.")
PYEOF

echo ""
echo "=============================================="
echo "[2/4] Generating window manifest (chromosome-level train/val/test split)"
echo "=============================================="
python3 src/make_windows.py \
    --train-chroms ${TRAIN_CHROMS} \
    --val-chroms ${VAL_CHROMS} \
    --test-chroms ${TEST_CHROMS} \
    --window-size ${WINDOW_SIZE} \
    --stride ${STRIDE} \
    --bin-size ${BIN_SIZE} \
    --out "${MANIFEST}"

echo ""
echo "=============================================="
echo "[3/4] Training all three models"
echo "=============================================="

echo ""
echo "--- Training cnn for ${CNN_EPOCHS} epochs ---"
python3 src/train.py \
    --manifest "${MANIFEST}" \
    --model cnn \
    --ctcf-bw "${WORKDIR}/raw/ctcf.bigWig" \
    --h3k27ac-bw "${WORKDIR}/raw/h3k27ac.bigWig" \
    --dnase-bw "${WORKDIR}/raw/dnase.bigWig" \
    --hic-source "${HIC_URL}" \
    --epochs "${CNN_EPOCHS}" \
    --batch-size ${BATCH_SIZE} \
    --lr ${LR} \
    --device "${DEVICE}" \
    --out-dir "${RUNS_DIR}/cnn_run1"
    # Add --grad-clip-norm / --warmup-steps here too if you later want to
    # test whether these tricks help CNN -- no evidence yet that it needs them.

echo ""
echo "--- Training unet for ${UNET_EPOCHS} epochs ---"
python3 src/train.py \
    --manifest "${MANIFEST}" \
    --model unet \
    --ctcf-bw "${WORKDIR}/raw/ctcf.bigWig" \
    --h3k27ac-bw "${WORKDIR}/raw/h3k27ac.bigWig" \
    --dnase-bw "${WORKDIR}/raw/dnase.bigWig" \
    --hic-source "${HIC_URL}" \
    --epochs "${UNET_EPOCHS}" \
    --batch-size ${BATCH_SIZE} \
    --lr ${LR} \
    --device "${DEVICE}" \
    --out-dir "${RUNS_DIR}/unet_run1"

echo ""
echo "--- Training transformer for ${TRANSFORMER_EPOCHS} epochs "
echo "    (warmup-steps=${TRANSFORMER_WARMUP_STEPS}, grad-clip-norm=${TRANSFORMER_GRAD_CLIP_NORM}) ---"
python3 src/train.py \
    --manifest "${MANIFEST}" \
    --model transformer \
    --ctcf-bw "${WORKDIR}/raw/ctcf.bigWig" \
    --h3k27ac-bw "${WORKDIR}/raw/h3k27ac.bigWig" \
    --dnase-bw "${WORKDIR}/raw/dnase.bigWig" \
    --hic-source "${HIC_URL}" \
    --epochs "${TRANSFORMER_EPOCHS}" \
    --batch-size ${BATCH_SIZE} \
    --lr ${LR} \
    --warmup-steps ${TRANSFORMER_WARMUP_STEPS} \
    --grad-clip-norm ${TRANSFORMER_GRAD_CLIP_NORM} \
    --device "${DEVICE}" \
    --out-dir "${RUNS_DIR}/transformer_run1"

echo ""
echo "=============================================="
echo "[4/4] Producing comparison report"
echo "=============================================="
python3 src/compare_models.py \
    --cnn-dir "${RUNS_DIR}/cnn_run1" \
    --unet-dir "${RUNS_DIR}/unet_run1" \
    --transformer-dir "${RUNS_DIR}/transformer_run1" \
    --out-dir "${REPORT_DIR}"

echo ""
echo "=============================================="
echo "DONE. See:"
echo "  ${REPORT_DIR}/comparison_curves.png  -- loss/correlation curves for all 3 models"
echo "  ${REPORT_DIR}/summary_table.md       -- best-epoch metrics table"
echo ""
echo "NEXT STEP: run evaluate.py against held-out TEST chromosomes (chr17 by"
echo "default) using each model's best_model.pt checkpoint -- this script"
echo "intentionally never touches the test split, since val metrics are for"
echo "model selection only and reusing them as your final reported number"
echo "would be a (milder, but real) form of the same leakage problem"
echo "chromosome holdout exists to prevent."
echo "=============================================="