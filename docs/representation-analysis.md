# Representation analysis

Compare original, unlearned, and retained-only retrained models on identical
inputs. Extraction reads clean checkpoints without training, changing weights,
or running fault search. The unlearned models use the project's incorrect-label
suppression baseline.

The workflow has two steps: export activations and class probabilities to a
`features.npz` bundle, then analyze that bundle on a CPU. The analysis step needs
neither the original weights nor GPU libraries.

[Experiment results](../reports/representations/RESULTS.md) contain the findings;
this guide defines the procedure and how to interpret its measurements.

## Setup

Run commands from the repository root with the CPU dependencies installed:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
```

ResNet extraction additionally requires a matching PyTorch/torchvision pair;
see the [GPU setup guide](targeted-flips-gpu.md). It supports either
`--device cuda:0` or `--device cpu`. Existing feature bundles need only the CPU
dependencies for analysis and plotting.

## Extract digits representations

The archived NPZ fixtures contain clean original, unlearned, and retrained
models. The dataset is packaged with scikit-learn, so extraction requires no
training or download:

```bash
python representation_analysis.py digits --seed 0 --forget-class 0 \
  --output runs/representations/digits-seed0/features.npz
python representation_analysis.py analyze \
  --features runs/representations/digits-seed0/features.npz \
  --output-dir runs/representations/digits-seed0/analysis
```

Seeds 0, 1, and 2 and forgotten digits 0, 1, and 2 have archived fixtures in
`reports/random-flips/`. Split indices come from
`reports/targeted-flips/cpu/digits-main.json.gz`.

The exporter samples at most 60 training examples per class and uses the archived
discovery split for held-out evaluation, about 18 examples per class. All
retained selection examples form a separate alignment-calibration set.
`--per-class` caps training and held-out counts separately; `--probe-seed`
controls sampling. `--evaluation test` explicitly opens the archived test split.
The default analysis does not use it. Previously used splits make this analysis
exploratory even when the feature extraction itself is new.

## Extract ResNet representations

Provide an existing training run containing `original.pt`, `unlearned.pt`,
`retrained.pt`, `split.json`, and `train-config.json`:

```bash
python representation_resnet.py --root runs/seed0-class0 --data data \
  --device cuda:0 --output runs/representations/resnet-seed0/features.npz
python representation_analysis.py analyze \
  --features runs/representations/resnet-seed0/features.npz \
  --output-dir runs/representations/resnet-seed0/analysis
```

The exporter evaluates models sequentially with the original CIFAR normalization
and no augmentation. It averages spatial dimensions after `layer1` through
`layer4`; the final 512-dimensional vector is the pre-classifier representation.
Checkpoints load with `weights_only=True`. Architecture, seed, forgotten class,
and split consistency are checked, and checkpoint hashes are recorded.

Defaults are up to 100 examples per class from training, 100 per class from
search for held-out evaluation, and 100 per retained class from selection for
calibration. `--per-class` and `--calibration-per-class` control those caps.
Calibration examples do not enter metric estimates. Use a smaller `--batch-size`
if inference exceeds available memory.

CIFAR-10 must already exist under `--data` unless `--download` is supplied.
`--evaluation test` uses the official test split and should follow frozen
analysis choices. Prior use of that test set in this project means it is not a
new untouched dataset.

External checkpoints require a validated adapter: matching shapes alone do not
establish preprocessing, original-model provenance, or identical forget/retain
data. This exporter accepts the project's checkpoint format.

## Analyze and inspect the output

Choose fresh bundle paths and output directories; existing artifacts are never
overwritten. `features.npz` includes ordered sample IDs, labels, partitions,
pooled activations, probabilities, and provenance. It is the complete input to
`analyze`. For the default ResNet sample count, activations occupy about 32 MiB
before compression across the three models.

Exact neighbor search and Gram matrices have quadratic time/memory cost in probe
count. Increasing probes to the entire dataset is not a streaming operation.

| Artifact | Contents |
|---|---|
| `REPORT.md` | Run-specific tables, figures, and calibration diagnostics |
| `metrics.json` | All comparisons, class relations, counts, and provenance |
| `comparisons.csv` | Every model pair, layer, partition, and class/group |
| `behavior.csv` | Accuracy and negative log-likelihood |
| `relations.csv` | Raw/centered class-average cosines and valid pair counts |
| `similarity_by_class.png` | Held-out class relationships, excluding self-pairs |
| `similarity_change.png` | Raw/centered model differences on a shared scale |
| `pca.png` | Retained-calibration alignment and common PCA, when defined |
| `representation_metrics.png` | CKA/neighbor curves for multi-layer bundles |
| `plot_diagnostics.json` | Calibration rank, alignment residuals, and PCA variance |

Use `analyze --layer layer2` to select another layer for cosine/PCA views.
Numerical CKA and neighbor comparisons always cover all exported layers.
`--neighbors` defaults to 5, 10, and 25; `--null-repeats` defaults to five
shuffled-row CKA comparisons. `analyze --seed` controls those permutations.

## Metrics and figure interpretation

**Behavior.** Accuracy and negative log-likelihood use the model's class
probabilities. Pairwise prediction disagreement and Jensen–Shannon divergence
(in nats) supplement class recall: equal recall need not mean equal predictions.

**CKA.** Linear centered kernel alignment compares matched examples using their
full representations, separately centered within each evaluated group. Debiased
CKA uses unbiased HSIC terms in a normalized ratio; the ratio itself is not
unbiased and may be negative. Collapsed or insufficient groups yield `null`.
Shuffled-row means diagnose finite-sample similarity and are not p-values.

**Shared neighbors.** Cosine nearest neighbors exclude self-matches and zero
vectors. Forget/retain groups choose anchors; candidates include all classes in
the same train or held-out partition. Insufficient candidate pools yield
undefined scores. Valid counts and tie fractions are recorded; ties use sample
order. These scores describe neighborhood overlap, not prediction agreement.

**Class cosines.** Each heatmap cell averages held-out image-pair cosines within
one model. Diagonal cells compare distinct images of the same class. Cosine
ignores magnitude; raw ReLU features are nonnegative and use a shared 0-to-1
scale. High cosine is neither prediction confidence nor proof of memorization,
and class averages can hide individual variation.

**Raw and centered changes.** Column titles specify subtraction order; red
means increased cosine and blue means decreased cosine. The centered row
subtracts each model's mean over retained calibration examples. No rotation or
projection enters these heatmaps. All difference panels share one signed scale
within a run; the scale may differ across runs. Centered panels are omitted when
calibration examples are absent.

Signed changes can cancel. Centering also constrains aggregate geometry, so a
smaller signed increase does not prove that the change was only an additive
offset. Mean absolute changes, Euclidean distances, and activation norms provide
complementary checks. Cosine-change summaries weight class pairs equally;
Euclidean summaries weight image pairs equally. Pair counts are not independent
sample sizes.

**PCA.** Orthogonal alignment and common PCA axes are fitted only on retained
calibration examples. All model panels share axes and limits. Rank-deficient
calibration leaves rotations outside its span underdetermined; reported variance
is calibration variance. A similar 2D position does not establish full-space
equivalence. UMAP and linear classifier probes are not implemented.

## Retraining variability and limits

Repeated `--retrained-repeat path/to/retrained.pt` arguments add independent
retrained controls. Each extra checkpoint needs a matching adjacent `split.json`.
Its initialization seed may differ, but the retained dataset must stay fixed.
The analysis automatically includes all model pairs, including retrain/retrain.

The training program changes the dataset split with `--seed`, so ordinary
cross-seed runs do not isolate retraining variability. Shared initialization
between original and retrained models can also contribute to similar geometry.
The archived representation experiments have no independent fixed-split
retraining controls or equivalence threshold.

High original/unlearned similarity alone does not establish failed unlearning.
Check whether original/retrained differences are detectable, compare
unlearned/retrained against retrain/retrain, and retain behavioral measurements.
A retained-only model can represent a deleted class through shared features.
Neither CKA nor a 2D picture proves erasure, privacy, or residual training-data
influence.

## Software checks

```bash
python -m unittest discover -s tests -p test_representations.py -v
python representation_analysis.py synthetic \
  --output runs/representations/rotation-control/features.npz
python representation_analysis.py analyze \
  --features runs/representations/rotation-control/features.npz \
  --output-dir runs/representations/rotation-control/analysis
```

This synthetic control makes original/unlearned representations differ only by
an orthogonal rotation; CKA and shared neighbors should be one. The synthetic
retrained role is unrelated noise. These are software checks, not unlearning
evidence.

For a ResNet extraction check without checkpoints or dataset downloads:

```bash
python representation_resnet.py --smoke --device cpu \
  --output runs/representations/resnet-smoke/features.npz
```

This runs actual ResNet-18 models with random weights and synthetic images. Its
bundle is labeled synthetic and can be passed to the same analysis command.
