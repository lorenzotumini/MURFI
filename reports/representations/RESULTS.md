# Laptop representation pilot — 2026-11-05

The CPU pipeline is implemented and has been run on the three archived
digit-0 checkpoint trios, seeds 0, 1, and 2. No training, GPU, fault injection,
or dataset download was needed. ResNet extraction is prepared for the desktop;
it has not been run on actual ResNet checkpoints here.

## What was compared

Each trio contains the original model, the incorrect-label suppression model
named `unlearned` in the archives, and a retained-only retrained model. We
extract the 64-dimensional hidden ReLU activation before the classifier.

The analysis uses shared images and keeps three disjoint groups:

- Up to 60 original training images per digit, for training-record diagnostics.
- The archived discovery split for held-out comparisons: 180 images per seed,
  including 18, 17, and 18 digit-0 images respectively.
- Retained images from the selection split for fitting the visualization's
  orthogonal alignment and common PCA axes. These do not enter metric estimates.

The archived test split was not used. Discovery and selection data were already
used in previous experiments, making this an exploratory analysis.

## Observations

Debiased linear CKA on held-out digit-0 examples:

| Seed | n | Original / unlearned | Original / retrained | Unlearned / retrained |
|---|---:|---:|---:|---:|
| 0 | 18 | 0.9512 | 0.9708 | 0.9223 |
| 1 | 17 | 0.9302 | 0.9793 | 0.9014 |
| 2 | 18 | 0.9508 | 0.9510 | 0.9579 |

Shared 10-neighbor fractions for held-out digit-0 anchors, with all 180
held-out images as the candidate pool:

| Seed | Original / unlearned | Original / retrained | Unlearned / retrained |
|---|---:|---:|---:|
| 0 | 0.9167 | 0.9278 | 0.8778 |
| 1 | 0.8941 | 0.8588 | 0.8059 |
| 2 | 0.8944 | 0.8889 | 0.8778 |

In seed 0, original digit-0 recall is 100%, while unlearned and retrained recall
are both 0%. All three have 96.91% retained accuracy on these held-out images.
Despite equal forget recall, unlearned and retrained predictions disagree on
72.22% of the forgotten examples: matching one behavioral metric does not imply
matching the output distribution, either.

All three pairs show similar hidden geometry. In particular, the retrained
model is also close to the original. Therefore, high original/unlearned
similarity alone cannot identify unsuccessful unlearning. CKA and neighbor
overlap also do not always rank the pairs identically. The results establish a
working diagnostic pipeline, not a universal signature of residual knowledge.

![Shared PCA view for seed 0](digit0-seed0/pca.png)

The axes are common across the three panels. Forgotten digit-0 examples are
marked with crosses. Calibration PCA explains about 49.6% of calibration
variance in two dimensions for seed 0, so substantial geometry remains unseen.
Calibration rank is below 64 for these ReLU features; rotations outside the
calibration span are underdetermined. See each `plot_diagnostics.json`.

## Evidence and reproduction

The additional [class-level views and heatmap reading guide](relations-seed0/REPORT.md)
show full-dimensional cosine relationships without alignment or PCA. Results
are also available for [seed 1](relations-seed1/REPORT.md) and
[seed 2](relations-seed2/REPORT.md).

For seed 0, mean similarity between **distinct zero images** is 0.9592 in the
original, 0.9539 in the suppressed model, and 0.9431 after retraining. Mean
zero/one similarity changes from 0.5995 to 0.7413 and 0.6728 respectively. Thus
the zero representations remain coherent while their angular relationships
with other classes change. These are descriptive values on the same small
held-out set; they do not establish memorization or a causal mechanism.

- [Seed 0 report](digit0-seed0/REPORT.md), [metrics](digit0-seed0/metrics.json).
- [Seed 1 report](digit0-seed1/REPORT.md), [metrics](digit0-seed1/metrics.json).
- [Seed 2 report](digit0-seed2/REPORT.md), [metrics](digit0-seed2/metrics.json).
- [Compact cross-seed values](summary.json).
- [Setup, exact commands, metrics, and desktop handoff](../../docs/representation-analysis.md).

Every per-seed folder includes CSV tables, four PNG figures, and provenance
with source checkpoint/reference hashes. Portable feature bundles are in the
ignored local `runs/representations/digit0-seed{seed}/features.npz` paths and can
be regenerated from the archived weights. Git transfers the reports and code
only after the new files are added through the usual workflow.

Validation: ten new representation tests pass, including orthogonal invariance,
shuffled correspondence, high-dimensional finite-sample bias, degenerate inputs,
calibration separation, saved digits extraction, and end-to-end figure export.
Seven existing CPU bit/fault tests also pass. Synthetic original/unlearned
features related only by rotation give CKA and neighbor alignment of 1.
Figures were inspected visually. PyTorch/CUDA tests were not run on this laptop.

## Limits and next experiment

There are only 17–18 held-out forget examples per seed. Scores are descriptive,
without confidence intervals or an equivalence threshold. Original and retrained
models share an initialization seed, and the existing seeds change data splits.
These three configurations cannot estimate pure retrain/retrain variability.

On the desktop, extract the ResNet trio using the same analysis defaults, inspect
all four stages, and add independently retrained models on an identical fixed
retained dataset. Confirm that the original/retrained difference is detectable
before interpreting original/unlearned similarity. Repeat with a validated
published unlearning method before extending conclusions beyond suppression.
