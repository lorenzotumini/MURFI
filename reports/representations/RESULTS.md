# Representation analysis: digits and CIFAR-10

The experiments compare clean original, incorrect-label-suppressed, and
retained-only retrained models on matched inputs. Digits results cover forgotten
class 0 and seeds 0–2; CIFAR-10 covers airplane and seed 0. No fault injection or
additional training is part of this analysis.

The [method guide](../../docs/representation-analysis.md) defines the metrics,
calibration procedure, and reproduction commands. Per-run reports contain the
full tables and figures; this page discusses comparisons across runs.

## Digits: similar geometry despite different behavior

The 64-dimensional hidden ReLU activations are evaluated on the archived
discovery split: 180 held-out images per seed, including 18, 17, and 18 digit-0
images. Retained selection examples calibrate alignment/PCA and are excluded
from metric estimates. Training-record diagnostics use up to 60 examples per
digit. The test split is not used; prior use of discovery/selection data makes
the analysis exploratory.

Held-out digit-0 representation comparisons:

| Seed | Pair | Debiased CKA | Shared 10-neighbor fraction |
|---|---|---:|---:|
| 0 | Original / unlearned | 0.9512 | 0.9167 |
| 0 | Original / retrained | 0.9708 | 0.9278 |
| 0 | Unlearned / retrained | 0.9223 | 0.8778 |
| 1 | Original / unlearned | 0.9302 | 0.8941 |
| 1 | Original / retrained | 0.9793 | 0.8588 |
| 1 | Unlearned / retrained | 0.9014 | 0.8059 |
| 2 | Original / unlearned | 0.9508 | 0.8944 |
| 2 | Original / retrained | 0.9510 | 0.8889 |
| 2 | Unlearned / retrained | 0.9579 | 0.8778 |

All three pairs have similar hidden geometry, including original/retrained.
High original/unlearned similarity therefore cannot by itself identify failed
unlearning. CKA and neighbor overlap also do not always rank pairs identically.

In seed 0, original digit-0 recall is 100%, while unlearned and retrained recall
are both 0%. All three have 96.91% retained accuracy. Nevertheless, the latter
two disagree on 72.22% of forgotten-example predictions. Matching recall does
not imply matching output distributions.

Digit-0 representations remain coherent while their relations to other classes
change: in seed 0, mean cosine between distinct zero images is 0.9592, 0.9539,
and 0.9431 for original, suppressed, and retrained models. Mean zero/one cosine
is 0.5995, 0.7413, and 0.6728. These are descriptive values, not evidence of
memorization or a causal mechanism.

### Sensitivity to the feature-space origin

Suppression changes retained-class pair cosines as follows:

| Seed | Raw mean change | Centered mean change | Centered mean absolute change |
|---|---:|---:|---:|
| 0 | +0.0456 | +0.0025 | 0.0237 |
| 1 | +0.0851 | +0.0025 | 0.0228 |
| 2 | +0.0729 | +0.0004 | 0.0254 |

Centering on retained calibration means makes the broadly positive change mixed,
but nonzero differences remain. Signed means can cancel, and centering
constrains aggregate geometry; a common offset is not a complete explanation
established by this comparison. Mean Euclidean separation between different
retained classes decreases about 5–12% after suppression.

The seed-0 PCA view explains about 49.6% of calibration variance in two axes.
Calibration rank is below the 64-dimensional feature space, leaving rotations
outside its span underdetermined. The per-run diagnostics record these limits.

## CIFAR-10: differences increase in later layers

The existing seed-0 ResNet-18 checkpoints are evaluated on 100 airplane and
900 retained images from the search split. A separate 900 retained selection
examples fit alignment/PCA. These splits have been used previously, so this is
also exploratory.

For held-out airplane examples, original/suppressed debiased CKA falls from
0.9996 at layer1 to 0.5396 at layer4. Final-layer comparisons are:

| Pair | Airplane debiased CKA | Airplane shared 10-neighbor fraction | Retained debiased CKA |
|---|---:|---:|---:|
| Original / unlearned | 0.5396 | 0.400 | 0.9716 |
| Original / retrained | 0.3572 | 0.122 | 0.9382 |
| Unlearned / retrained | 0.2169 | 0.115 | 0.9149 |

In the shared PCA view, airplane examples form compact clouds in the original
and suppressed models but spread more broadly in the retrained model. Retained
class clouds look broadly similar. This contrast is more pronounced than in
digits and suggests a hypothesis for further study. The two axes explain only
30.7% of calibration variance; the picture does not establish full-space
separation or erasure.

Airplane accuracy is 97% in the original and 0% in both other models, which
still disagree on 60% of airplane predictions. Retained accuracies are 94.78%,
95.44%, and 94.33%. These behavioral and geometric differences cannot yet be
attributed uniquely to suppression rather than retraining variability.

## Limits and next experiment

The digits analysis has only 17–18 held-out forgotten examples per seed. The
ResNet analysis has one checkpoint trio. Neither includes independent retrains
on an identical retained dataset, confidence intervals, or an equivalence
threshold. Original and retrained models share initialization seeds; changing
seeds in the current training program also changes data splits.

Add fixed-split independent retraining controls and repeat the ResNet analysis
across configurations. Establish whether original/retrained differences are
detectable before interpreting original/unlearned similarity. A published
unlearning method is needed before extending conclusions beyond suppression.

## Evidence

- Digits: [seed 0](seed0/REPORT.md), [seed 1](seed1/REPORT.md),
  [seed 2](seed2/REPORT.md), and [cross-seed summary](summary.json).
- CIFAR-10: [ResNet seed 0](resnet-seed0/REPORT.md).

Each report links its complete numerical measurements, CSV tables, plot
diagnostics, and checkpoint/reference provenance. The digits feature bundles
can be regenerated from the archived model fixtures. ResNet extraction requires
the original checkpoint files, which are not distributed in this repository.
