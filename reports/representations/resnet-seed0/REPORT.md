# Representation comparison

Source: **repository_resnet18_cifar10**; seed: **0**; forget class: **0**. incorrect-label suppression baseline.

- Existing search/selection splits have been used previously; this is exploratory.
- Extra retrained checkpoints must be independently trained on the same retained data.

## Behavior on held-out inputs

| Model | Group | n | Accuracy | NLL |
|---|---|---:|---:|---:|
| original | forget | 100 | 97.00% | 0.1391 |
| original | retain | 900 | 94.78% | 0.2035 |
| unlearned | forget | 100 | 0.00% | 5.4523 |
| unlearned | retain | 900 | 95.44% | 0.2073 |
| retrained | forget | 100 | 0.00% | 9.1822 |
| retrained | retain | 900 | 94.33% | 0.2025 |

## Representation metrics

These metrics use unreduced activations before any fitted alignment. Higher values mean
more similar geometry or neighborhoods, not better unlearning. CKA is centered within
each group; neighbor candidates include all classes in the partition.

| Layer | Group | Pair | n | CKA | Debiased CKA | Shuffled CKA | Shared 10-NN | Prediction disagreement |
|---|---|---|---:|---:|---:|---:|---:|---:|
| layer1 | forget | original--unlearned | 100 | 0.9996 | 0.9996 | 0.0594 | 0.9640 | 97.00% |
| layer1 | retain | original--unlearned | 900 | 0.9995 | 0.9995 | 0.0078 | 0.9693 | 1.78% |
| layer1 | forget | original--retrained | 100 | 0.9204 | 0.9165 | 0.0581 | 0.5810 | 98.00% |
| layer1 | retain | original--retrained | 900 | 0.9213 | 0.9207 | 0.0079 | 0.5584 | 5.56% |
| layer1 | forget | unlearned--retrained | 100 | 0.9195 | 0.9155 | 0.0662 | 0.5820 | 60.00% |
| layer1 | retain | unlearned--retrained | 900 | 0.9230 | 0.9224 | 0.0068 | 0.5621 | 5.33% |
| layer2 | forget | original--unlearned | 100 | 0.9993 | 0.9992 | 0.1086 | 0.9630 | 97.00% |
| layer2 | retain | original--unlearned | 900 | 0.9994 | 0.9994 | 0.0132 | 0.9574 | 1.78% |
| layer2 | forget | original--retrained | 100 | 0.8808 | 0.8742 | 0.0834 | 0.6000 | 98.00% |
| layer2 | retain | original--retrained | 900 | 0.9353 | 0.9345 | 0.0137 | 0.6212 | 5.56% |
| layer2 | forget | unlearned--retrained | 100 | 0.8822 | 0.8754 | 0.0816 | 0.6100 | 60.00% |
| layer2 | retain | unlearned--retrained | 900 | 0.9351 | 0.9343 | 0.0133 | 0.6171 | 5.33% |
| layer3 | forget | original--unlearned | 100 | 0.9920 | 0.9908 | 0.1446 | 0.8810 | 97.00% |
| layer3 | retain | original--unlearned | 900 | 0.9965 | 0.9964 | 0.0262 | 0.9168 | 1.78% |
| layer3 | forget | original--retrained | 100 | 0.8156 | 0.7974 | 0.1355 | 0.4140 | 98.00% |
| layer3 | retain | original--retrained | 900 | 0.8741 | 0.8713 | 0.0252 | 0.4812 | 5.56% |
| layer3 | forget | unlearned--retrained | 100 | 0.8188 | 0.8000 | 0.1126 | 0.3990 | 60.00% |
| layer3 | retain | unlearned--retrained | 900 | 0.8760 | 0.8732 | 0.0259 | 0.4750 | 5.33% |
| layer4 | forget | original--unlearned | 100 | 0.6036 | 0.5396 | 0.0619 | 0.4000 | 97.00% |
| layer4 | retain | original--unlearned | 900 | 0.9716 | 0.9716 | 0.0094 | 0.6950 | 1.78% |
| layer4 | forget | original--retrained | 100 | 0.3867 | 0.3572 | 0.0559 | 0.1220 | 98.00% |
| layer4 | retain | original--retrained | 900 | 0.9386 | 0.9382 | 0.0090 | 0.2784 | 5.56% |
| layer4 | forget | unlearned--retrained | 100 | 0.2465 | 0.2169 | 0.0575 | 0.1150 | 60.00% |
| layer4 | retain | unlearned--retrained | 900 | 0.9153 | 0.9149 | 0.0083 | 0.2717 | 5.33% |

![Metrics across layers](representation_metrics.png)

## Class relationships

Cosine and PCA views use **layer4**. Each heatmap cell averages similarities between
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
| raw | unlearned minus original | 0.0080 | 0.0170 | 25 / 36 |
| raw | retrained minus original | 0.0089 | 0.0148 | 29 / 36 |
| raw | retrained minus unlearned | 0.0009 | 0.0189 | 21 / 36 |
| centered | unlearned minus original | 0.0004 | 0.0218 | 18 / 36 |
| centered | retrained minus original | -0.0002 | 0.0134 | 18 / 36 |
| centered | retrained minus unlearned | -0.0006 | 0.0214 | 16 / 36 |

Signed means may cancel; centering also constrains aggregate geometry. Disappearance
of a broad signed increase does not prove that changes were entirely a common offset.

| Model | Mean Euclidean distance between retained classes | Mean retained activation norm |
|---|---:|---:|
| original | 7.1192 | 5.9513 |
| unlearned | 7.2829 | 6.0621 |
| retrained | 7.0351 | 5.9218 |

Euclidean distances weight image pairs equally and are sensitive to scale. Pair counts
are not independent statistical samples; these comparisons are descriptive.

## PCA overview

![Shared aligned PCA](pca.png)

Two axes explain **30.7%** of calibration variance. Rotations/reflections and PCA
are fitted on retained calibration examples only. A similar 2D position is not full-space equivalence.

- Orthogonal maps and common PCA fitted only on retained calibration rows.
- No labels are used to fit PCA or rotations; labels color the display.
- A 2D projection can hide changes; numerical metrics use full feature spaces.

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
