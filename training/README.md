# 1D tracks → 2D Hi-C contact map: architecture comparison

This project trains and compares three model architectures (CNN, UNet,
Transformer) on the task your supervisor specified: predict a 2D Hi-C
contact map from 1D epigenomic tracks (CTCF, H3K27ac, DNase) over the
same genomic window.

## What's actually being compared, and what isn't

This is a controlled comparison of **2D refinement architecture**, not a
comparison of three completely independent pipelines. All three models
share the exact same:
- input tracks and normalization
- 1D-to-2D "outer product with interactions" expansion mechanism
  (`[emb_i, emb_j, emb_i * emb_j, |emb_i - emb_j|]`)
- loss function (MSE in log1p-contact space)
- evaluation metrics
- symmetrization step on the output

They differ only in **how the 1D sequence is encoded** (dilated CNN vs.
dilated CNN vs. Transformer with relative-position attention bias -- note
CNN and UNet use the *same* 1D encoder, see `model_unet.py`'s docstring)
and **how the resulting 2D grid is refined** (plain conv stack vs. UNet
encoder-decoder vs. light conv stack). This is deliberate: if the three
models differed in multiple ways at once, you wouldn't be able to
attribute a performance difference to any one architectural choice.

## Honest scope of this comparison

Before looking at any results, it's worth being upfront about what this
setup can and can't tell you:

- **Can tell you**: which of these three specific architectures fits this
  specific data (K562, 4 train chromosomes, 1Mb windows @ 10kb) better or
  worse, and gives you a real, executed, debugged starting point for
  further work.
- **Cannot tell you**: whether any of these architectures "solves" 1D→3D
  genome folding prediction in general. That would need training on many
  more chromosomes, likely multiple cell lines, and comparison against
  the actual published models in this space (Akita, C.Origami, Orca) --
  this project is sized for a CPU/single-GPU first pass, not a
  publication-grade benchmark.

## Project layout

```
genomics_project/
├── README.md                  <- this file
├── run_full_pipeline.sh       <- orchestration script: runs everything end to end
├── manifest.csv               <- example generated window list (regenerate via make_windows.py)
├── src/
│   ├── genome_constants.py    <- chromosome size constants (no heavy deps)
│   ├── genomic_io.py          <- Hi-C / bigWig fetchers (lazy pyBigWig/hicstraw imports)
│   ├── make_windows.py        <- generates train/val/test window manifest with chromosome holdout
│   ├── dataset.py             <- PyTorch Dataset + train-only normalizer
│   ├── metrics.py             <- distance-stratified MSE, stratum-adjusted correlation
│   ├── model_cnn.py           <- Architecture 1: dilated CNN
│   ├── model_unet.py          <- Architecture 2: dilated CNN encoder + UNet 2D refinement
│   ├── model_transformer.py   <- Architecture 3: Transformer w/ relative position bias
│   ├── train.py                <- trains one model, logs train/val metrics per epoch
│   ├── compare_models.py      <- builds comparison plots/table from training logs
│   └── evaluate.py            <- FINAL test-set evaluation (run once, at the end, per model)
└── runs/                       <- created when you train; one subfolder per model run
```

## Setup

```bash
pip install torch numpy pandas matplotlib requests pyBigWig hic-straw tabulate

# hic-straw compiles a C++ extension and needs libcurl dev headers:
#   Ubuntu/Debian: sudo apt install libcurl4-openssl-dev
#   macOS:         brew install curl  (usually already present)
```

If you only have CPU, that's fine -- the window size (1Mb @ 10kb = 100x100
output) and model sizes (167K-628K params) were chosen specifically to be
CPU-trainable in reasonable time, not requiring a GPU.

## Running everything

```bash
chmod +x run_full_pipeline.sh
./run_full_pipeline.sh
```

This will: download/cache the 3 bigWig tracks, generate the window
manifest with chromosome-level train(chr1,2,3,8)/val(chr10)/test(chr17)
holdout, train all three models in sequence, and produce a comparison
report in `comparison_report/`.

**Edit the CONFIG section at the top of `run_full_pipeline.sh` first** --
in particular note that `TRANSFORMER_EPOCHS` is set much higher than
`CNN_EPOCHS`/`UNET_EPOCHS` by default. This is not an oversight: see
"Why the Transformer gets more epochs" below.

To run pieces individually instead (e.g. while developing), see the
`Usage` docstring at the top of each script in `src/`.

## After training: final test evaluation

`run_full_pipeline.sh` deliberately stops after producing the comparison
report on train/val curves, and does NOT touch the test chromosomes
(chr17 by default). Run `evaluate.py` once per model, after you've
finished looking at val curves and picked checkpoints:

```bash
for MODEL in cnn unet transformer; do
  python3 src/evaluate.py \
    --checkpoint runs/${MODEL}_run1/best_model.pt \
    --normalizer runs/${MODEL}_run1/normalizer.json \
    --manifest manifest.csv \
    --ctcf-bw encode_work/raw/ctcf.bigWig \
    --h3k27ac-bw encode_work/raw/h3k27ac.bigWig \
    --dnase-bw encode_work/raw/dnase.bigWig \
    --hic-source "https://www.encodeproject.org/files/ENCFF291JZM/@@download/ENCFF291JZM.hic" \
    --out runs/${MODEL}_run1/test_results.json
done
```

See `evaluate.py`'s docstring for why this is a separate script rather
than a flag on `train.py` -- it's about preventing a subtle, human-in-
the-loop form of test set leakage, not a code-organization preference.

## Key design decisions, and why

**Chromosome-level train/val/test split, not random window split.**
Hi-C and signal tracks are spatially autocorrelated; a random split lets
the model partially memorize neighboring training windows. `make_windows.py`
has a hard guard that refuses to generate a manifest if any chromosome
appears in more than one split.

**10kb resolution, 1Mb windows (100x100 output matrices).** Smaller than
the 2Mb/2048-bin windows used by Akita, specifically sized to be
comfortably CPU-trainable while still being large enough to contain a
full TAD (typically 100kb-1Mb) within one window.

**Targets are log1p-transformed**, never raw contact counts -- Hi-C
counts are extremely heavy-tailed and training on raw counts means the
loss is dominated by a few diagonal pixels.

**Track normalization statistics are fit ONLY on the train split.**
`TrackNormalizer.fit()` has a runtime guard that raises if you
accidentally pass it anything other than a train-only dataset.

**Evaluation uses distance-stratified metrics, not raw MSE/correlation.**
Hi-C signal is dominated by genomic distance (closer bins always contact
more, regardless of any interesting biology). Raw correlation is
deceptively high for any model that just learns "predict the average
distance-decay curve." `metrics.py`'s `stratum_adjusted_correlation`
removes the distance trend within equal-pixel-count distance bands before
computing correlation, to measure whether the model captures *specific*
loops/TADs rather than just the generic decay-with-distance pattern.

## Why the Transformer gets more epochs

This isn't a hedge -- it's an empirically observed property, found while
building and sanity-checking these models. On a tiny-batch overfitting
test (4 fixed examples, no real data needed):

| Model | Steps to ~halve loss |
|---|---|
| CNN | ~15-20 |
| UNet | ~10 |
| Transformer | ~150-200 |

The cause: at initialization, this Transformer's relative-position bias
starts at exactly zero (by design -- it's a learned parameter, not a
prior), so attention is governed purely by randomly-initialized Q/K
projections, which produces near-uniform attention weights (measured:
max attention weight per row ≈0.016 vs. a uniform baseline of 0.01 for
100 positions). Every output position starts as roughly the *average* of
all input positions, and the network has to learn to sharpen attention
away from this before it can use position-specific information at all.
CNNs and the UNet's CNN-based encoder don't have this problem: spatial
locality is built into the convolution operation from the very first
layer, with no "uniform attention" phase to escape.

This is a real, known tradeoff between architecture families (often
discussed in the literature as part of why Transformers benefit from
longer warmup/more data), not a special property of this codebase --
but it specifically means: **if you train all three models for the same
number of epochs and the Transformer looks worse, check whether its
val_loss curve has actually plateaued before concluding the architecture
itself is worse.** `compare_models.py` prints an explicit warning if
training run lengths differ by more than 2x, for exactly this reason.

## Known limitations / good next steps

- Single cell line (K562). Generalization across cell lines is untested
  and likely to be the binding limitation, not architecture choice.
- 4 training chromosomes is enough to compare architectures meaningfully
  but is thin for claiming strong generalization; chr8 in particular is
  used both for training data here AND was the locus in your original
  visualization work (MYC), so don't read too much into any
  qualitatively-nice-looking prediction near MYC specifically -- it's
  training-distribution-adjacent, not a true held-out example.
- No data augmentation (e.g. reverse-complement of the genomic window) is
  implemented; this is a common and fairly cheap addition for genomics
  sequence models which can help with the modest training set size here.
- The 1D-to-2D outer-product expansion is O(L²) in both compute and
  memory; this is fine for L=100 (this project's window size) but would
  need revisiting (e.g. a more memory-efficient pairwise mechanism) before
  scaling to Akita-scale L=2048 windows.