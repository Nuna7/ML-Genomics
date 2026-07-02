# Experiment: Predicting Hi-C Contact Maps from 1D Epigenomic Tracks

## Overview

The experiment trains three neural network architectures to predict 2D
Hi-C contact matrices from 1D epigenomic signal tracks at 1kb resolution.
Windows are centered in the vicinity of known chromatin loops across two
cell lines (K562 and HepG2), enabling evaluation of cross-cell-line
generalization.

---

## 1. Data

### 1.1 Cell lines

Two cell lines are used:

- **K562**: chronic myelogenous leukemia, the most densely profiled cell
  line in ENCODE. The standard benchmark for computational Hi-C models.
- **HepG2**: hepatocellular carcinoma. Included to test whether models
  learn general epigenome-to-3D-structure relationships or K562-specific
  patterns.

### 1.2 Hi-C data

| Cell line | Accession | Experiment |
|---|---|---|
| K562 | ENCFF291JZM | ENCSR000CXJ | 
| HepG2 | ENCFF020DPP | ENCSR194SRI |

Both files are in `.hic` format and accessed via URL streaming through
`hicstraw`. Only the requested 100kb windows
are fetched at query time, keeping per-window data transfer small.

### 1.3 Epigenomic tracks (1D inputs)

Three tracks per cell line, all in `.bigWig` format, GRCh38:

| Track | K562 accession | HepG2 accession |
|---|---|---|---|
| CTCF ChIP-seq | ENCFF000YMA | ENCFF266BGZ | 
| H3K27ac ChIP-seq | ENCFF779QTH | ENCFF084DIM | 
| DNase-seq | ENCFF414OGC | ENCFF938OBZ | 

These three tracks were selected to capture CTCF-mediated structural
loops (CTCF), active regulatory contacts (H3K27ac), and open chromatin
at anchors (DNase). All are downloaded automatically by `prepare_data()`.

### 1.4 Loop calls

| Cell line | Accession | Source | Format | Coordinate system |
|---|---|---|---|---|
| K562 | GSE63525 | NCBI GEO FTP | `.txt.gz` | hg19 → lifted to hg38 |
| HepG2 | ENCFF050EKS | ENCSR194SRI | `.bedpe.gz` | hg38 (no liftover needed) |

**K562 liftover**: Rao 2014 loop calls are in hg19. The UCSC hg19→hg38
chain file (`hg19ToHg38.over.chain.gz`) is bundled as a local asset
(`assets/hg19ToHg38.over.chain.gz`) rather than downloaded at runtime,
because UCSC's download server intermittently rate-limits or times out
from cloud IP ranges. Obtain it once with:
```bash
curl -L "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz" \
     -o assets/hg19ToHg38.over.chain.gz
```
Place at `assets/hg19ToHg38.over.chain.gz` before running the pipeline.

**HepG2 loops**: `ENCFF050EKS` - Download is automatic inside
`prepare_data()`.

---

## 2. Preprocessing

### 2.1 Window placement

For each cell line, loop midpoints are identified from the loop call
source. A 100kb window is placed around each loop midpoint with
**random jitter of ±20kb** applied to the center before window
placement. The jitter prevents the model from learning a trivial
positional regularity (loop always at the center pixel) instead of
learning to predict structure from epigenomic features.

Windows whose jittered placement would extend past a chromosome boundary
are rejected and skipped entirely — not clipped — to avoid windows with
inconsistent effective size.

### 2.2 Train/test split

The dataset uses chromosome-level holdout. Windows from the same
chromosome never appear in both train and test.

- **Train chromosomes**: chr1, chr2, chr3, chr8
- **Test chromosomes**: chr17

100 loop-windows per cell line are placed on training chromosomes
(200 total), and 25 per cell line on the test chromosome (50 total).
This matches the experimental design: ~100 loops in 2 cell lines for
training, ~50 held-out regions on different chromosomes for prediction.

No validation split is used. With only 50 test windows, a three-way
split would leave too few examples per split to be informative.
Early stopping uses the test split as the monitoring criterion, which
is a mild deviation from strict data hygiene — acceptable given the
exploratory, small-data scope of this phase.

### 2.3 Track preprocessing

For each 1D track and each window, `pyBigWig` computes mean signal per
1kb bin (100 bins per window), using exact (not approximate) bin
statistics. Values are clipped at the 99th percentile of nonzero values
within that window before normalization, to reduce the influence of
extreme outlier bins.

Per-track z-score normalization is applied using statistics computed
**only from training windows, separately for each cell line**. K562 and
HepG2 have substantially different absolute signal levels (e.g. K562
CTCF mean ≈ 5.1 signal units vs. HepG2 ≈ 0.7 in the runs conducted);
a single global normalizer would over- or under-normalize one cell line.
The normalizer is fit once and saved to the `hic-runs` volume, then
reloaded for subsequent model runs.

### 2.4 Hi-C target preprocessing

For each window, the Hi-C contact matrix is fetched at 1kb resolution
via `hicstraw` (using SCALE normalization when available, VC otherwise,
NONE as fallback). Values are **log1p-transformed**: `target = log1p(raw_counts)`.

At 1kb resolution, most off-diagonal pixels have raw counts of 0-3,
and loop pixels typically have 5-20. Log1p compresses this range
(log1p(20) ≈ 3.0).

### 2.5 Final data shape per example

| Tensor | Shape | Dtype | Content |
|---|---|---|---|
| `tracks` (input) | (3, 100) | float32 | [CTCF, H3K27ac, DNase], z-score normalized |
| `target` (label) | (100, 100) | float32 | log1p(Hi-C contacts), symmetric |

---

## 3. Model architectures

All three models take `tracks: (B, 3, 100)` and output `matrix: (B, 100, 100)`.

They share a common structure:
1. **1D encoder**: extract per-bin embeddings from the 3 input tracks
2. **Pairwise expansion**: for every pair of bins (i, j), construct a
   feature vector by concatenating `[emb_i, emb_j, emb_i * emb_j, |emb_i - emb_j|]`
3. **2D refinement**: apply convolutions to the resulting (L×L) feature
   grid to predict the contact matrix
4. **Symmetrization**: enforce `output = 0.5 * (output + output.T)`
   since Hi-C contact matrices are physically symmetric
5. **softplus output activation**: `output = log(1 + exp(pre_activation))`
   ensures non-negative predictions matching log1p targets, with no
   upper saturation

### 3.1 CNN (model_cnn.py)

**1D encoder**: stack of residual dilated 1D convolutions with
exponentially increasing dilation rates (1, 2, 4, 8, ...). Dilation
captures structure at multiple scales within the same window — a 1Mb
TAD spans 1000 bins and would require hundreds of standard convolutional
layers to see; dilated convolutions achieve this with ~6 layers.

**2D refinement**: two 3×3 Conv2d layers with BatchNorm and GELU,
followed by a 1×1 projection to the final 100×100 output.

**Parameter count**: ~167K

### 3.2 UNet (model_unet.py)

**1D encoder**: identical to CNN — same dilated residual blocks. The
difference between CNN and UNet lies entirely in the 2D refinement stage,
making their performance difference interpretable as the value of
multi-scale 2D processing.

**2D refinement**: UNet encoder-decoder with two downsampling/upsampling
steps and skip connections. Deep layers specialize in large-scale
structure (compartments, TADs); skip connections preserve sharp local
detail (loop anchors) that would otherwise be blurred by downsampling.
The UNet structure is motivated by the observation that Hi-C maps contain
structure simultaneously at many scales — single-pixel loop signals, TAD
boundaries spanning tens of pixels, and broad compartment-level background.

**Parameter count**: ~628K

### 3.3 Transformer (model_transformer.py)

**1D encoder**: multi-head self-attention over the 100-bin sequence, with
a **learned relative-position bias** added to attention logits as a
function of |i − j| (bin distance). This is analogous to T5-style
relative position bias and is included because Hi-C contact frequency
depends strongly and smoothly on genomic distance — encoding this
inductive prior directly in the attention mechanism is preferable to
having the model rediscover it from data alone with limited training
examples.

**2D refinement**: light 2-layer conv stack on the expanded pairwise grid.

**Parameter count**: ~234K

**Dropout**: 0.3 (higher than typical Transformer defaults of 0.1)
because the training set is small (~200 windows). Applied inside both
the attention and the feedforward sublayers.

---

## 4. Training

### 4.1 Loss function

**Backpropagation uses Huber loss** (`delta=1.0`), not MSE.

Huber loss is quadratic for errors below `delta` (matching MSE's gradient
behavior for the common case of small prediction errors) and linear for
errors above `delta` (capping the per-pixel gradient contribution at
±delta, regardless of how large the error actually is).

The motivation is due to the result of prior training run
using MSE loss showed the Transformer's test loss reaching 28x its
training loss starting at exactly the epoch where LR warmup ended.
Investigation showed the mechanism: during warmup, large gradient steps
from MSE (where gradient ∝ 2·error, so a prediction error of 7 units
produces a gradient of 14 per pixel) pushed pre-activations to extremes
where, if the output head was sigmoid-bounded, gradients would vanish.
When the output head was removed entirely and Huber was used instead,
the gradient at any single pixel is capped at ±1.0 regardless of error
magnitude, preventing the positive-feedback loop between large errors
and large gradient steps.

- Still Transformer's always failed to generalise to test.

**Evaluation reports both Huber loss and MSE** (as separate columns
`train_huber`/`test_huber` and `train_mse`/`test_mse` in the training
log). MSE is kept as a metric but Huber is used for checkpoint selection
because it is the actual optimized objective.

### 4.2 Output activation

All three models use **softplus**: `output = log(1 + exp(pre_activation))`.

Softplus is non-negative (matching that targets are `log1p(counts) ≥ 0`)
and has no upper saturation. Its gradient is `sigmoid(pre_activation)`,
which is ≈1.0 for large positive pre-activations and only approaches 0
for large negative pre-activations (where the model correctly predicts
near-zero contact, so small gradients are appropriate).

### 4.3 Optimizer and learning rate

**Adam** with:
- Base learning rate: 1e-3
- Weight decay: 1e-5 (CNN, UNet) or 1e-3 (Transformer)
- Batch size: 32

**Learning rate schedule**:
- CNN and UNet: constant 1e-3 throughout
- Transformer: linear warmup over 100 optimizer steps, then cosine
  decay to a floor of 5% of peak LR over the remaining training steps


### 4.4 Early stopping

Training stops if the test Huber loss does not improve for 10 consecutive
epochs. Each model is allowed up to 60 epochs. The checkpoint saved is
the one with the lowest test Huber loss seen during training, not the
last epoch's weights.

---

## 5. Evaluation metrics

### 5.1 Distance-stratified MSE

Hi-C contact frequency is dominated by genomic distance: pixels near the
diagonal (short-range contacts) are systematically higher than distant
pixels, simply due to polymer physics. Plain MSE on such data is dominated
by diagonal pixels and doesn't reflect whether the model captures
biologically interesting loop/TAD structure.

Distance-stratified MSE divides pixels into 10 bands by |i − j| (using
equal-pixel-count bands, not equal-distance-range bands, since the number
of pixel pairs at a given distance decreases linearly with distance).
MSE is computed within each band and averaged across bands, giving equal
weight to short-range and long-range structure in the final score.

### 5.2 Stratum-adjusted correlation

Within each distance band, the band's mean value is subtracted from both
prediction and target before computing Pearson correlation across all
pixels. This removes the global distance-decay trend, so the metric
specifically measures whether the model correctly predicts *which pixels
are elevated above the background for their distance* — i.e. specific
loops and TAD boundaries — rather than just the generic decay curve.

This is the primary evaluation metric, reported as `stratum_adjusted_corr`
in the training log. A model that perfectly predicts the distance-decay
trend but has no knowledge of specific loops will score near 0.

---

## 6. Results (current run)

- See /runs and /comparison_report for details.

---

## 7. Known limitations

- **Small dataset**: 200 training windows total. Enough to compare
  architectures and establish an end-to-end pipeline, not enough to
  claim strong generalization.
- **No validation split**: test set does double duty for early stopping
  and final reporting. Acceptable given the small-data, exploratory
  context.
- **Transformer stability**

---

## 8. Reproducing this experiment

### Dependencies
```bash
pip install modal pyyaml
modal setup
```

### Manual assets (one-time)
```bash
# Liftover chain
curl -L "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz" \
     -o assets/hg19ToHg38.over.chain.gz
```

### Running
```bash
# Verify data prep before committing to GPU time
modal run modal_pipeline.py --prepare-only

# Train all three models
modal run modal_pipeline.py

# Retrieve results
modal volume get hic-runs / ./runs_local/
```

### File structure
```
hic_final/
├── modal_pipeline.py          single entry point
├── configs/
│   └── encode_sources.yaml   all ENCODE accessions and training config
├── assets/
│   └── hg19ToHg38.over.chain.gz   must be placed manually (see above)
└── src/
    ├── genome_constants.py    chromosome sizes (no external deps)
    ├── genomic_io.py          hicstraw + pyBigWig wrappers
    ├── metrics.py             mse_loss, huber_loss, evaluate_batch
    ├── model_cnn.py           dilated CNN architecture
    ├── model_unet.py          UNet architecture
    └── model_transformer.py   Transformer with relative position bias
```