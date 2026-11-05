# Representation comparison

Source: **archived_digits**; seed: **2**; forget class: **0**. incorrect-label suppression baseline.

- Exploratory: archived splits/checkpoints were used in earlier experiments.
- No independent retrain replicate in this fixture; no erasure claim.
- Selection split retained examples calibrate PCA; excluded from metrics.

## Behavior on held-out inputs

| Model | Group | n | Accuracy | NLL |
|---|---|---:|---:|---:|
| original | forget | 18 | 94.44% | 0.1811 |
| original | retain | 162 | 96.91% | 0.1310 |
| unlearned | forget | 18 | 0.00% | 9.0048 |
| unlearned | retain | 162 | 95.06% | 0.1566 |
| retrained | forget | 18 | 0.00% | 10.0238 |
| retrained | retain | 162 | 96.91% | 0.1243 |

## Representation metrics

These metrics use unreduced activations before any fitted alignment. Higher values mean
more similar geometry or neighborhoods, not better unlearning. CKA is centered within
each group; neighbor candidates include all classes in the partition.

| Layer | Group | Pair | n | CKA | Debiased CKA | Shuffled CKA | Shared 10-NN | Prediction disagreement |
|---|---|---|---:|---:|---:|---:|---:|---:|
| hidden | forget | original--unlearned | 18 | 0.9669 | 0.9508 | 0.2222 | 0.8944 | 94.44% |
| hidden | retain | original--unlearned | 162 | 0.9879 | 0.9875 | 0.0399 | 0.9191 | 1.85% |
| hidden | forget | original--retrained | 18 | 0.9679 | 0.9510 | 0.2865 | 0.8889 | 94.44% |
| hidden | retain | original--retrained | 162 | 0.9960 | 0.9958 | 0.0467 | 0.9265 | 0.00% |
| hidden | forget | unlearned--retrained | 18 | 0.9675 | 0.9579 | 0.3059 | 0.8778 | 61.11% |
| hidden | retain | unlearned--retrained | 162 | 0.9866 | 0.9861 | 0.0330 | 0.9012 | 1.85% |

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
| raw | unlearned minus original | 0.0729 | 0.0729 | 36 / 36 |
| raw | retrained minus original | 0.0125 | 0.0125 | 33 / 36 |
| raw | retrained minus unlearned | -0.0604 | 0.0604 | 0 / 36 |
| centered | unlearned minus original | 0.0004 | 0.0254 | 20 / 36 |
| centered | retrained minus original | -0.0008 | 0.0114 | 15 / 36 |
| centered | retrained minus unlearned | -0.0012 | 0.0272 | 14 / 36 |

Signed means may cancel; centering also constrains aggregate geometry. Disappearance
of a broad signed increase does not prove that changes were entirely a common offset.

| Model | Mean Euclidean distance between retained classes | Mean retained activation norm |
|---|---:|---:|
| original | 7.9684 | 10.0449 |
| unlearned | 7.5334 | 10.8019 |
| retrained | 7.8621 | 10.0964 |

Euclidean distances weight image pairs equally and are sensitive to scale. Pair counts
are not independent statistical samples; these comparisons are descriptive.

## PCA overview

![Shared aligned PCA](pca.png)

Two axes explain **47.6%** of calibration variance. Rotations/reflections and PCA
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
