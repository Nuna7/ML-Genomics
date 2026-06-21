# Experiment documentation: 1D epigenomic tracks → 2D Hi-C contact map

This document describes, in detail, what data this experiment uses, how
it's preprocessed, what shape it's in by the time it reaches a model, what
each model architecture does, and how models are evaluated. It does
**not** include results — those are reported separately (training logs,
`comparison_report/`) and interpreted by hand for the specific run you're
presenting.

For setup/run instructions, see `README.md`. This file is about *what*
the experiment is and *why* it's built this way, not how to execute it.

---

## 1. Task definition

**Input**: three 1D genomic signal tracks (CTCF binding, H3K27ac, DNase
accessibility) over a genomic window, at a fixed bin resolution.

**Output**: a 2D Hi-C contact matrix over the same window, at the same
bin resolution — i.e. predict how frequently every pair of genomic bins
in the window contacts every other pair, from epigenomic signal alone.

**Cell line**: K562 (chronic myelogenous leukemia cell line), chosen
because it's one of the most densely ENCODE-profiled cell lines, meaning
all the required track types are available from a single, consistent
source.

---

## 2. Data sources

All data comes from ENCODE (`https://www.encodeproject.org`), fetched at
training time (not bundled with this repo):

| Track | ENCODE accession | File type | What it measures |
|---|---|---|---|
| Hi-C contacts | ENCFF291JZM | `.hic` | 3D genome contact frequency |
| CTCF ChIP-seq | ENCFF000YMA | `.bigWig` | CTCF transcription factor binding (a primary architectural protein for TAD/loop boundaries) |
| H3K27ac ChIP-seq | ENCFF779QTH | `.bigWig` | Active enhancer/promoter histone mark |
| DNase-seq | ENCFF414OGC | `.bigWig` | Chromatin accessibility |

These four tracks were chosen because CTCF and DNase are the most direct
known correlates of 3D genome structure (CTCF in particular is the
primary protein implicated in loop anchor formation via loop extrusion),
and H3K27ac adds a complementary "active regulatory element" signal not
fully captured by the other two. This is the same minimal track set used
in the original visualization work this project builds on.

Reference genome: GRCh38 (hg38), matching ENCODE's coordinate system for
these files.

---

## 3. Preprocessing

### 3.1 Genomic windowing and the train/val/test split

The genome is divided into fixed-size, non-overlapping windows (default:
1Mb each), and **windows are assigned to train/val/test by whole
chromosome**, not individually:

- Train: chr1, chr2, chr3, chr8
- Val: chr10
- Test: chr17

**Why chromosome-level, not random, splitting**: Hi-C contact frequency
and the 1D signal tracks are spatially autocorrelated along the genome —
a window starting at chr8:1,000,000 looks very similar to one starting at
chr8:1,010,000. If windows were assigned to train/val/test at random,
the model could partially memorize neighboring training windows when
evaluated on a "held out" window right next door, inflating apparent
performance without reflecting real generalization. Holding out entire
chromosomes removes this risk: no test window shares a chromosome with
any training window, so a high score there reflects the model
generalizing to genuinely unseen genomic regions, not interpolating
between memorized neighbors.

This is enforced in code (`make_windows.py`), not just by convention: the
manifest generator refuses to write a manifest if any chromosome appears
in more than one split.

### 3.2 Resolution and window size

- Bin size: 10,000 bp (10kb)
- Window size: 1,000,000 bp (1Mb) → 100×100 bins per window

10kb was chosen as a standard resolution in this literature (matching,
e.g., Akita). The 1Mb window (rather than the 2Mb windows used by some
published models) was chosen specifically to keep training feasible on
CPU/single-GPU hardware, while still being large enough to contain a
complete TAD (typically 100kb–1Mb) within a single window.

### 3.3 Track (input) preprocessing

For each 1D track and each window:
1. The mean signal value is computed per bin (100 bins per window),
   using `pyBigWig`'s `stats(..., type="mean", exact=True)`.
2. Values are clipped at a configurable percentile (default: 99th
   percentile of nonzero values within that window) to reduce the
   influence of extreme outlier bins on training.
3. Per-track **z-score normalization** (mean 0, std 1) is applied, using
   normalization statistics computed **only from the training split**.

**Why train-only normalization statistics**: if mean/std were computed
from train+val+test combined, information about the val/test
distribution would leak into the normalization the model is trained
under — a milder but real form of the same leakage problem chromosome
holdout exists to prevent. This is enforced in code: `TrackNormalizer.fit()`
raises an error if called on anything other than a pure-train-split
dataset.

### 3.4 Target (Hi-C) preprocessing

For each window:
1. The raw Hi-C contact matrix is fetched via `hicstraw` at the
   requested bin size (with normalization fallback SCALE → VC → NONE,
   whichever the `.hic` file actually supports at that resolution).
2. Values are **log1p-transformed**: `target = log1p(raw_counts)`.

**Why log1p, not raw counts**: Hi-C contact counts are extremely
heavy-tailed — the diagonal (very-short-range contacts) can be
100–1000x the typical off-diagonal background. Training directly on raw
counts means the loss is dominated by a handful of near-diagonal pixels,
and the model has little incentive to learn the much smaller-magnitude
but biologically interesting structure further from the diagonal (TADs,
loops). log1p compresses this range while preserving order, and is the
standard transform in this literature.

Targets are **not** additionally min-max rescaled per-window — log1p
output is already in a reasonably bounded range for this data, and a
per-window rescale would make windows numerically incomparable to each
other (a window with a different overall contact density would be
rescaled differently, destroying the meaning of "high vs low contact"
across windows).

### 3.5 Final data shape

By the time a single example reaches a model:

| Tensor | Shape | dtype | Notes |
|---|---|---|---|
| `tracks` (input) | `(3, 100)` | float32 | [CTCF, H3K27ac, DNase], z-score normalized |
| `target` (label) | `(100, 100)` | float32 | log1p(Hi-C contacts), symmetric |

Batched: `tracks` is `(B, 3, 100)`, `target` is `(B, 100, 100)`.

### 3.6 Failure handling

If a requested bin size isn't natively available in the `.hic` file at a
given window, the pipeline raises rather than silently substituting a
different resolution — a silent substitution would mean some windows in
a batch have a different effective resolution than others, which would
corrupt both training and evaluation without any visible error. Similarly,
genuinely missing/invalid data for a window raises rather than returning
zeros, since a model trained on silently-zeroed windows would learn
"sometimes Hi-C is just zero," which is true in some genomic contexts
(e.g. assembly gaps) but not a useful signal to learn from a data
artifact.

---

## 4. Model architectures

All three models share an identical **interface**:

```
tracks: (B, 3, n_bins)  -->  model  -->  matrix: (B, n_bins, n_bins)
```

and an identical **1D-to-2D bridge mechanism**: after encoding the 1D
tracks into per-bin embeddings, every pair of bin embeddings `(i, j)` is
expanded into a per-pixel feature vector:

```
pair_features[i, j] = concat(emb_i, emb_j, emb_i * emb_j, |emb_i - emb_j|)
```

This is the standard mechanism for turning a sequence encoder into a
pairwise/matrix predictor (used by Akita and similar models), and is
shared identically across all three architectures specifically so that
any performance difference between them is attributable to differences
in the 1D encoder or 2D refinement stage — not to a confound from also
changing how the 1D→2D bridge works.

All three models also **symmetrize their output** as a final step:
`output = 0.5 * (raw_output + raw_output.T)`. Hi-C contact matrices are
physically symmetric (contact frequency between bin i and bin j equals
that between bin j and bin i), so this is a useful inductive bias rather
than something the model has to learn from data alone.

### 4.1 CNN (`model_cnn.py`)

**1D encoder**: a stack of residual, dilated 1D convolutions. Dilation
(exponentially increasing across layers, e.g. 1, 2, 4, 8...) is used
specifically because Hi-C structure spans scales from ~50kb to ~1Mb
within the same window — a plain CNN with small fixed receptive fields
would need an impractically deep stack to "see" a 1Mb-scale TAD;
exponentially increasing dilation achieves a wide effective receptive
field with a shallow network.

**2D refinement**: a small stack of plain 2D convolutions on the
expanded pairwise grid, refining each pixel using local context from
nearby pixels.

This is the architecture family used by Akita, and serves as the
most-established baseline of the three.

### 4.2 UNet (`model_unet.py`)

**1D encoder**: identical structure to the CNN model's encoder (same
dilated residual block design), so any difference between CNN and UNet
results is attributable to the 2D refinement stage specifically, not to
also having a different 1D encoder.

**2D refinement**: a UNet-style encoder-decoder with two downsampling /
upsampling steps and skip connections, operating on the expanded
pairwise grid.

**Why UNet here**: Hi-C maps contain structure at very different
physical scales simultaneously — sharp point-to-point loops (a handful
of pixels), TAD boundaries (tens of pixels), and broad compartment-level
background (the whole map). A UNet's downsampling path lets deep layers
specialize in large-scale structure while skip connections preserve
sharp local detail that would otherwise be blurred out — the same
motivation that makes UNet effective for multi-scale dense prediction in
other domains (its origin in biomedical image segmentation reflects the
same idea: fine-grained cell boundaries vs. coarse tissue structure).

### 4.3 Transformer (`model_transformer.py`)

**1D encoder**: a small Transformer encoder (multi-head self-attention +
feedforward layers) over the sequence of 100 bins, with one addition:
a **learned relative-position attention bias**, added to attention
logits as a function of `|i - j|` (bin distance), similar in spirit to
T5-style relative position bias.

**Why relative-position bias specifically**: Hi-C contact frequency
depends strongly and smoothly on genomic distance (closer bins almost
always contact more, with rapid decay) — distance is itself a primary
signal in this task, not just an addressing mechanism. A vanilla
Transformer's absolute positional embeddings don't directly encode
"these two positions are 200kb apart"; the relative-position bias gives
the model this information directly rather than requiring it to be
rediscovered from data alone with limited training examples.

**2D refinement**: a lighter conv stack than the UNet's, applied to the
expanded pairwise grid.

**Why test a Transformer at all**: convolutions build long-range context
indirectly, by stacking layers until a receptive field large enough to
span the whole window is reached. Self-attention lets every bin attend
directly to every other bin in a single layer — a more direct mechanism
for the long-range dependencies that loops represent (two bins far apart
in linear sequence that are functionally coupled in 3D space). This
architecture tests whether that direct mechanism is more effective than
the CNN/UNet's indirect, stacked-receptive-field approach.

**A documented, verified property of this specific implementation**: at
initialization, the relative-position bias starts at exactly zero (it's
a learned parameter), so early-training attention is governed purely by
randomly-initialized query/key projections — which produces *near-uniform*
attention (every output position starts close to the average of all
input positions, rather than attending to specific positions). This was
confirmed empirically: max attention weight per row at initialization
was ≈0.016, barely above the uniform baseline of 0.01 for 100 positions.
Practically, this means the Transformer needs noticeably more
optimization steps than the CNN/UNet to reach a comparable point in
training, since it first has to learn to sharpen attention away from
this near-uniform starting point. Two optional training tricks
(`--warmup-steps`, `--grad-clip-norm` in `train.py`) were added
specifically to address this — see `train.py`'s docstring for detail on
why these two were chosen.

### 4.4 Parameter counts

Approximate (depends slightly on configured window/hidden-dim):

| Model | Parameters |
|---|---|
| CNN | ~167K |
| UNet | ~628K |
| Transformer | ~234K |

All three are well within "no large model" scope — small enough to train
on CPU in reasonable time, and chosen specifically not to be a genomics
foundation model.

---

## 5. Training setup

- **Loss**: pixelwise MSE between predicted and target matrices, both in
  log1p space. This is the optimized quantity (backprop target); it is
  *not* the only thing reported, see Evaluation below for why.
- **Optimizer**: Adam, with optional weight decay.
- **Optional stabilization** (off by default, recommended for the
  Transformer): gradient norm clipping, and linear learning-rate warmup
  over a configurable number of *optimizer steps* (not epochs — see
  `train.py` for why this distinction matters at this dataset size).
- **Per-epoch evaluation**: every epoch, the model is evaluated on the
  validation split (held-out chromosome chr10). The checkpoint with the
  lowest validation loss is saved.
- **Test split is untouched during training/model-selection.** chr17 is
  reserved for a single, final, separate evaluation pass
  (`evaluate.py`), run once per model after development is finished —
  not used to make any architecture or hyperparameter decisions. This is
  deliberate: even looking at test-set numbers repeatedly while still
  iterating is a real (if subtle) form of data leakage through the human
  in the loop, not just through code.

---

## 6. Evaluation methodology

Three metrics are reported per window and averaged across the evaluation
set:

### 6.1 Plain MSE
Pixelwise mean squared error in log1p space. Simple and standard, but
has a known weakness for this task (see below).

### 6.2 Distance-stratified MSE
Hi-C signal is dominated by genomic distance: pixels near the diagonal
(short-range) have systematically higher values and variance than
distant pixels, simply due to polymer physics, not biology. A model that
only learns "predict high near the diagonal, low far away" can get a
deceptively good plain MSE without learning anything about which
*specific* TADs or loops are present in a *specific* window.

To address this, pixels are grouped into distance bands by `|i - j|`,
using **equal-pixel-count bands** (not equal-distance-range bands, since
the number of pixel pairs at a given distance shrinks linearly as
distance grows). MSE is computed separately within each band, then
averaged across bands — this prevents the easy, distance-trivial
short-range pixels from drowning out errors in the harder, more
biologically interesting long-range pixels.

### 6.3 Stratum-adjusted correlation
Within each distance band, the band's mean value is subtracted from both
prediction and target (a crude per-distance detrending), and Pearson
correlation is then computed across all pixels at all distances pooled
together. This directly measures whether the model captures *which*
specific pixels are unusually high or low *for their distance* — i.e.
specific loop/TAD structure — rather than just the generic "closer =
higher contact" trend that any reasonable model picks up almost
trivially. Raw (non-adjusted) correlation is deceptively high for nearly
any model on this task, since it's dominated by the same distance trend;
the adjustment is what makes this metric actually discriminating between
architectures.

### 6.4 Reporting: best-epoch vs. last-K-epoch average

`compare_models.py` reports both:
- **`best_val_loss`**: the single best epoch's validation loss. Useful,
  but sensitive to noise — on a modestly-sized validation set, a single
  epoch can land on an easier batch by chance and look better than the
  model "really" performs.
- **`lastK_val_loss_mean` / `lastK_val_loss_std`**: average and standard
  deviation of validation loss over the final K epochs (default 5). The
  mean is a more robust estimate of where a model has actually settled;
  the **std is a direct, automatic check of whether training has
  actually plateaued** — a high std relative to other models in the same
  comparison is a concrete, non-subjective signal that a model's numbers
  are not yet a fair basis for comparison, and it needs more training
  before its result is trustworthy.

This distinction matters in practice for this exact task: architectures
with weaker built-in locality priors (the Transformer in particular) can
still be visibly unstable in validation loss well after a CNN or UNet
has settled, at the same epoch count. Comparing only best-single-epoch
numbers across models with different convergence speeds risks
mistaking "hasn't finished training yet" for "is a worse architecture."

---

## 7. Known scope and limitations of this experiment

This is explicitly a **first-pass, CPU/single-laptop-scale experiment**,
not a publication-grade benchmark. Specifically:

- **Single cell line (K562).** Cross-cell-line generalization is
  untested. This is likely the single biggest limitation on how far
  these results generalize — more so than which architecture is used.
- **Four training chromosomes.** Enough to produce real, meaningfully
  different training curves across three architectures and to compare
  them against each other, but thin for claiming strong generalization
  to the rest of the genome. The published models in this space (Akita,
  C.Origami, Orca) train on substantially more data, typically across
  multiple cell lines.
- **chr8 is both a training chromosome here and the locus of the
  original visualization work (MYC).** Any qualitatively nice-looking
  prediction near MYC specifically is training-distribution-adjacent,
  not a genuinely held-out example — chr17 (the test chromosome) is the
  only fair held-out evaluation.
- **No data augmentation** (e.g. reverse-complementing the genomic
  window) is implemented. This is a common, relatively cheap addition
  for genomics sequence models that could help given the modest training
  set size here, and is a reasonable next step before scaling up data.
- **The pairwise expansion mechanism is O(L²)** in compute and memory.
  Fine at L=100 (this project's window size), but would need revisiting
  before scaling toward larger windows (e.g. Akita-scale L=2048).

These are intentional, documented scope choices for an initial
CPU-feasible comparison — not oversights — and are the natural next
steps once this phase moves to GPU and a larger dataset.