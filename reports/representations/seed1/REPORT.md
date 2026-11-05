# Representation comparison

Source: **archived_digits**; seed: **1**; forget class: **0**. incorrect-label suppression baseline.

- Exploratory: archived splits/checkpoints were used in earlier experiments.
- No independent retrain replicate in this fixture; no erasure claim.
- Selection split retained examples calibrate PCA; excluded from metrics.

## Behavior on held-out inputs

| Model | Group | n | Accuracy | NLL |
|---|---|---:|---:|---:|
| original | forget | 17 | 100.00% | 0.0117 |
| original | retain | 163 | 97.55% | 0.0795 |
| unlearned | forget | 17 | 0.00% | 8.3113 |
| unlearned | retain | 163 | 96.93% | 0.1124 |
| retrained | forget | 17 | 0.00% | 12.1185 |
| retrained | retain | 163 | 98.16% | 0.0829 |

## Representation metrics

These metrics use unreduced activations before any fitted alignment. Higher values mean
more similar geometry or neighborhoods, not better unlearning. CKA is centered within
each group; neighbor candidates include all classes in the partition.

| Layer | Group | Pair | n | CKA | Debiased CKA | Shuffled CKA | Shared 10-NN | Prediction disagreement |
|---|---|---|---:|---:|---:|---:|---:|---:|
| hidden | forget | original--unlearned | 17 | 0.9489 | 0.9302 | 0.2682 | 0.8941 | 100.00% |
| hidden | retain | original--unlearned | 163 | 0.9795 | 0.9789 | 0.0406 | 0.9012 | 2.45% |
| hidden | forget | original--retrained | 17 | 0.9825 | 0.9793 | 0.2612 | 0.8588 | 100.00% |
| hidden | retain | original--retrained | 163 | 0.9922 | 0.9920 | 0.0349 | 0.9025 | 1.84% |
| hidden | forget | unlearned--retrained | 17 | 0.9240 | 0.9014 | 0.2643 | 0.8059 | 82.35% |
| hidden | retain | unlearned--retrained | 163 | 0.9707 | 0.9698 | 0.0409 | 0.8693 | 3.07% |

## Class relationships

Cosine and PCA views use **hidden**. Each heatmap cell averages similarities between
images of its row and column classes. Diagonal cells compare distinct images of the same
class, excluding self-matches. Cosine measures direction, ignoring magnitude. ReLU features
are nonnegative, so raw cosines cannot be negative. Class means can hide variation.

![Class-average cosine](similarity_by_class.png)

## Changes and sensitivity to centering

Columns show the model subtraction in their titles. Red means increased cosine similarity;
blue means decreased similarity. Raw and centered rows share one signed color scale.
Centering subtracts each model mean on retained calibration examples only. The mean is not
fitted on these held-out images. No rotation or projection is used for these heatmaps.

![Raw and centered cosine changes](similarity_change.png)

| Coordinates | Model difference | Mean change among retained classes | Mean absolute change | Positive / valid pairs |
|---|---|---:|---:|---:|
| raw | unlearned minus original | 0.0851 | 0.0851 | 36 / 36 |
| raw | retrained minus original | -0.0074 | 0.0096 | 8 / 36 |
| raw | retrained minus unlearned | -0.0925 | 0.0925 | 0 / 36 |
| centered | unlearned minus original | 0.0025 | 0.0228 | 19 / 36 |
| centered | retrained minus original | -0.0001 | 0.0170 | 16 / 36 |
| centered | retrained minus unlearned | -0.0026 | 0.0330 | 17 / 36 |

Signed means may cancel; centering also constrains aggregate geometry. Disappearance
of a broad signed increase does not prove that changes were entirely a common offset.

| Model | Mean Euclidean distance between retained classes | Mean retained activation norm |
|---|---:|---:|
| original | 8.8006 | 11.1752 |
| unlearned | 7.7837 | 11.5137 |
| retrained | 9.0719 | 11.3834 |

Euclidean distances weight image pairs equally and are sensitive to scale. Pair counts
are not independent statistical samples; these comparisons are descriptive.

## PCA overview

![Shared aligned PCA](pca.png)

Two axes explain **48.6%** of calibration variance. Rotations/reflections and PCA
are fitted on retained calibration examples only. A similar 2D position is not full-space equivalence.

- Orthogonal maps and common PCA fitted only on retained calibration rows.
- No labels are used to fit PCA or rotations; labels color the display.
- A 2D projection can hide changes; numerical metrics use full feature spaces.
- Calibration is rank-deficient: rotations outside its span are underdetermined.

## Limits and numerical evidence

- CKA uses separately centered groups; kNN uses all classes in each partition.
- Permutation means diagnose finite-sample bias; they are not p-values.
- No confidence interval or equivalence claim from a single model trio.
- Debiased CKA uses unbiased HSIC; its ratio is not an unbiased estimator.
- Cosine ties use input row order; inspect neighbor_quality for degeneracy.

A retrained model may retain a coherent forget-class representation through shared features.
High similarity alone cannot demonstrate residual training-data influence. Compare against
independent retraining on fixed retained data before claiming equivalence or removal.

All results and provenance are in [metrics.json](metrics.json). Detailed tables:
[comparisons](comparisons.csv), [behavior](behavior.csv), [class relations](relations.csv).
[Plot diagnostics](plot_diagnostics.json) record calibration rank and alignment residuals.
