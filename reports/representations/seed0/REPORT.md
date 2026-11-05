# Representation comparison

Source: **archived_digits**; seed: **0**; forget class: **0**.

Method: incorrect-label suppression baseline.

[Metric definitions and limitations](../../../docs/representation-analysis.md).

- Exploratory: archived splits/checkpoints were used in earlier experiments.
- No independent retrain replicate in this fixture; no erasure claim.
- Selection split retained examples calibrate PCA; excluded from metrics.

## Behavior on held-out inputs

| Model | Group | n | Accuracy | NLL |
|---|---|---:|---:|---:|
| original | forget | 18 | 100.00% | 0.0002 |
| original | retain | 162 | 96.91% | 0.0892 |
| unlearned | forget | 18 | 0.00% | 5.5871 |
| unlearned | retain | 162 | 96.91% | 0.0915 |
| retrained | forget | 18 | 0.00% | 13.2720 |
| retrained | retain | 162 | 96.91% | 0.0899 |

## Representation metrics

Full-space comparisons, before visualization alignment. Higher similarity
does not establish better unlearning. Neighbor candidates include all classes.

| Layer | Group | Pair | n | CKA | Debiased CKA | Shuffled CKA | Shared 10-NN | Prediction disagreement |
|---|---|---|---:|---:|---:|---:|---:|---:|
| hidden | forget | original--unlearned | 18 | 0.9660 | 0.9512 | 0.2903 | 0.9167 | 100.00% |
| hidden | retain | original--unlearned | 162 | 0.9884 | 0.9881 | 0.0363 | 0.9086 | 1.23% |
| hidden | forget | original--retrained | 18 | 0.9790 | 0.9708 | 0.2634 | 0.9278 | 100.00% |
| hidden | retain | original--retrained | 162 | 0.9926 | 0.9924 | 0.0336 | 0.9235 | 0.62% |
| hidden | forget | unlearned--retrained | 18 | 0.9430 | 0.9223 | 0.2785 | 0.8778 | 72.22% |
| hidden | retain | unlearned--retrained | 162 | 0.9784 | 0.9778 | 0.0331 | 0.8864 | 1.23% |

## Class relationships

Cosine and PCA views use **hidden**. Diagonal cells compare
distinct images of the same class; self-matches are excluded.

![Class-average cosine](similarity_by_class.png)

## Changes and sensitivity to centering

Column titles specify the subtraction order. Red indicates increased cosine;
blue indicates decreased cosine. All panels share one signed scale.
Centered coordinates subtract the retained-calibration mean for each model.

![Raw and centered cosine changes](similarity_change.png)

| Coordinates | Model difference | Mean change among retained classes | Mean absolute change | Positive / valid pairs |
|---|---|---:|---:|---:|
| raw | unlearned minus original | 0.0456 | 0.0456 | 36 / 36 |
| raw | retrained minus original | -0.0111 | 0.0153 | 5 / 36 |
| raw | retrained minus unlearned | -0.0567 | 0.0578 | 2 / 36 |
| centered | unlearned minus original | 0.0025 | 0.0237 | 18 / 36 |
| centered | retrained minus original | -0.0004 | 0.0185 | 16 / 36 |
| centered | retrained minus unlearned | -0.0029 | 0.0339 | 14 / 36 |

| Model | Mean Euclidean distance between retained classes | Mean retained activation norm |
|---|---:|---:|
| original | 9.3036 | 11.7961 |
| unlearned | 8.3878 | 11.5058 |
| retrained | 8.6237 | 10.7492 |

## PCA overview

![Shared aligned PCA](pca.png)

Two axes explain **49.6%** of calibration variance.
Alignment and PCA use retained calibration examples only. A similar 2D
position does not establish full-space equivalence.

Calibration is rank-deficient: rotations outside its span are underdetermined.

## Numerical evidence

[metrics.json](metrics.json) contains all measurements, quality checks and provenance.
Detailed tables: [comparisons](comparisons.csv), [behavior](behavior.csv),
[class relations](relations.csv). [Plot diagnostics](plot_diagnostics.json)
record calibration rank and alignment residuals.
