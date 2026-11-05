# Representation comparison

Source: **archived_digits**; seed: **2**; forget class: **0**.

Method: incorrect-label suppression baseline.

[Metric definitions and limitations](../../../docs/representation-analysis.md).

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

Full-space comparisons, before visualization alignment. Higher similarity
does not establish better unlearning. Neighbor candidates include all classes.

| Layer | Group | Pair | n | CKA | Debiased CKA | Shuffled CKA | Shared 10-NN | Prediction disagreement |
|---|---|---|---:|---:|---:|---:|---:|---:|
| hidden | forget | original--unlearned | 18 | 0.9669 | 0.9508 | 0.2222 | 0.8944 | 94.44% |
| hidden | retain | original--unlearned | 162 | 0.9879 | 0.9875 | 0.0399 | 0.9191 | 1.85% |
| hidden | forget | original--retrained | 18 | 0.9679 | 0.9510 | 0.2865 | 0.8889 | 94.44% |
| hidden | retain | original--retrained | 162 | 0.9960 | 0.9958 | 0.0467 | 0.9265 | 0.00% |
| hidden | forget | unlearned--retrained | 18 | 0.9675 | 0.9579 | 0.3059 | 0.8778 | 61.11% |
| hidden | retain | unlearned--retrained | 162 | 0.9866 | 0.9861 | 0.0330 | 0.9012 | 1.85% |

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
| raw | unlearned minus original | 0.0729 | 0.0729 | 36 / 36 |
| raw | retrained minus original | 0.0125 | 0.0125 | 33 / 36 |
| raw | retrained minus unlearned | -0.0604 | 0.0604 | 0 / 36 |
| centered | unlearned minus original | 0.0004 | 0.0254 | 20 / 36 |
| centered | retrained minus original | -0.0008 | 0.0114 | 15 / 36 |
| centered | retrained minus unlearned | -0.0012 | 0.0272 | 14 / 36 |

| Model | Mean Euclidean distance between retained classes | Mean retained activation norm |
|---|---:|---:|
| original | 7.9684 | 10.0449 |
| unlearned | 7.5334 | 10.8019 |
| retrained | 7.8621 | 10.0964 |

## PCA overview

![Shared aligned PCA](pca.png)

Two axes explain **47.6%** of calibration variance.
Alignment and PCA use retained calibration examples only. A similar 2D
position does not establish full-space equivalence.

Calibration is rank-deficient: rotations outside its span are underdetermined.

## Numerical evidence

[metrics.json](metrics.json) contains all measurements, quality checks and provenance.
Detailed tables: [comparisons](comparisons.csv), [behavior](behavior.csv),
[class relations](relations.csv). [Plot diagnostics](plot_diagnostics.json)
record calibration rank and alignment residuals.
