# Representation comparison: laptop now, desktop later

This experiment compares original, unlearned, and retained-only retrained
models on identical inputs. It is independent of fault injection: extraction
does not train models, mutate weights, or run the fault search.

There are two stages:

1. Export pooled activations, class probabilities, sample identities, and
   provenance to a portable `features.npz` file.
2. Analyze that file on a CPU. GPU libraries and original weights are not
   required for this stage.

## Laptop setup and completed pilot

Use Python 3.12 or newer with compatible NumPy/scikit-learn wheels. From the
repository root in PowerShell, using an existing Python installation:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-cpu.txt
```

If the environment was created with `uv` without pip, use:

```powershell
uv pip install --python .venv/Scripts/python.exe -r requirements-cpu.txt
```

The current laptop already has `.venv` with the CPU dependencies installed.
On Linux use `.venv/bin/python` in place of `.venv/Scripts/python.exe`.
Virtual environments are machine-specific; recreate them on the desktop.

The saved digits models are small NumPy checkpoints already in
`reports/random-flips/digit{class}-seed{seed}.npz`. They are the clean models,
before faults. The script loads the packaged scikit-learn digits dataset, so
neither a dataset download nor model training is needed.

```powershell
.venv/Scripts/python.exe representation_analysis.py digits --seed 0 --forget-class 0 --output runs/representations/my-digit0-seed0/features.npz
.venv/Scripts/python.exe representation_analysis.py analyze --features runs/representations/my-digit0-seed0/features.npz --output-dir runs/representations/my-digit0-seed0/analysis
```

Change `--seed` to 1 or 2, and optionally `--forget-class` to 1 or 2. Always
choose fresh output paths; existing artifacts are not overwritten. Completed
laptop runs and observations are described in
[the pilot report](../reports/representations/RESULTS.md).

Defaults: at most 60 training examples per class, the archived discovery split
for held-out evaluation (about 18 per class), and all retained selection
examples for alignment calibration. `--per-class` caps training and held-out
counts separately. `--evaluation test` explicitly opens the archived test split;
the default pilot does not use it. These splits have been used in prior work,
so even a new analysis remains exploratory.

## Desktop: export existing ResNet checkpoints

Install the CPU analysis requirements plus the matching CUDA-enabled
PyTorch/torchvision pair described in [the GPU guide](targeted-flips-gpu.md).
Use the repository's existing training output, containing:

```text
runs/seed0-class0/
  original.pt
  unlearned.pt
  retrained.pt
  split.json
  train-config.json
```

Then run (the same commands work with your desktop Python executable):

```powershell
python representation_resnet.py --root runs/seed0-class0 --data data --device cuda:0 --output runs/representations/resnet-seed0/features.npz
python representation_analysis.py analyze --features runs/representations/resnet-seed0/features.npz --output-dir runs/representations/resnet-seed0/analysis
```

The exporter runs each model sequentially, in evaluation/inference mode. It
uses the original CIFAR normalization, no augmentation, and averages spatial
dimensions after `layer1` through `layer4`. The final stage's pooled 512-vector
is the pre-classifier representation. Checkpoint shapes, seed, forget class,
and split consistency are checked. Checkpoints load with `weights_only=True`.
SHA-256 hashes and dependency versions are recorded in the bundle.

Default probes: up to 100 examples per class from the training split, up to
100 per class from the search split for held-out evaluation, and up to 100
per retained class from selection for calibration. No calibration example
is included in metric computation. The shared candidate pool for neighbors
contains all classes within the relevant train/held-out partition.

Use `--batch-size 32` for lower GPU memory use. Data must exist locally unless
`--download` is explicitly supplied. Use `--evaluation test` only after freezing
analysis choices; it uses the official CIFAR-10 test split. Existing test results
in this repository mean it is not a new untouched dataset.

For a quick desktop installation check, without weights or dataset downloads:

```powershell
python representation_resnet.py --smoke --device cpu --output runs/representations/resnet-smoke/features.npz
python representation_analysis.py analyze --features runs/representations/resnet-smoke/features.npz --output-dir runs/representations/resnet-smoke/analysis
```

This uses randomly initialized **actual ResNet-18 models** and synthetic
images. Its artifacts are clearly labeled synthetic and are not MU evidence.
The laptop validation did not exercise PyTorch, CUDA, or real ResNet extraction;
these checks remain for the desktop.

External SalUn or other checkpoints require a separately validated adapter:
architecture compatibility alone does not establish preprocessing, original
checkpoint provenance, or identical forget/retain data. This exporter accepts
the repository's checkpoint format and describes its unlearned model as the
existing incorrect-label suppression baseline.

## Move between machines

Transfer the changed source files, requirements file, tests, and this guide
through your usual repository workflow. `runs/`, `data/`, and `.venv/` are ignored by Git.

Copy just `features.npz` from the desktop to run CPU analysis on the laptop.
For the default ResNet sample count, activations occupy about 32 MiB before
compression for three models. The bundle includes all sample IDs and metadata,
so its corresponding JSON/report folder is not needed as an input.

The analysis uses exact neighbor search and Gram matrices, with quadratic
time/memory in probe count; start with the default counts. Increasing probes
to the entire dataset is not a streaming operation.

## One report per run

`representation_analysis.py analyze` computes everything and creates one
`REPORT.md`. There is no separate relations command or report. The archived
pilot reports are [seed 0](../reports/representations/seed0/REPORT.md),
[seed 1](../reports/representations/seed1/REPORT.md), and
[seed 2](../reports/representations/seed2/REPORT.md).

| Artifact | Contents |
|---|---|
| `REPORT.md` | Behavior, representation metrics, figures, centering checks, and interpretation |
| `metrics.json` | All comparisons, class relations, Euclidean checks, counts, and provenance |
| `comparisons.csv` | Every model pair, layer, partition, and class/group |
| `behavior.csv` | Accuracy and negative log-likelihood |
| `relations.csv` | Raw and centered class-average cosines, with valid pair counts |
| `similarity_by_class.png` | Final-layer class relationships, excluding self-pairs |
| `similarity_change.png` | Model differences, raw and centered, on one shared color scale |
| `pca.png` | Retained-calibration alignment and common PCA; illustrative only |
| `representation_metrics.png` | Combined CKA/neighbor curves, only when multiple layers exist |
| `plot_diagnostics.json` | Calibration rank, held-out alignment residual, PCA variance |

Single-layer digits runs present CKA and neighbors in a table. Redundant
individual-image heatmaps, forget profiles, and separate neighbor/CKA figures
are no longer generated. The detailed numerical metrics are retained.
Use `analyze --layer layer2` to choose another exported layer for cosine/PCA
views; numerical CKA and neighbor comparisons always cover all layers.

### How to read the figures

Each class-similarity cell compares images of its row and column classes
within one model. It averages all held-out image pairs; the diagonal compares
**different** images of the same class. Cosine is the dot product divided by
vector lengths, so it ignores magnitude. Raw ReLU features are nonnegative
and use a shared 0-to-1 scale. High cosine is not prediction confidence or
proof of memorization. Class averaging can hide individual-example variation.

The difference figure's column titles give the subtraction order. Red means
increased cosine; blue means decreased. The top row uses raw features, while
the centered row subtracts each model mean over retained calibration examples.
No rotation or dimensionality reduction enters either calculation. All panels
in this figure share one signed scale. Across different runs, that scale may
change. Centered plots are omitted if the bundle has no calibration examples.

The accompanying table includes both mean signed and mean absolute changes
among retained classes. Signed changes can cancel; centering constrains
aggregate geometry, so disappearance of a broad increase is not proof of a
pure additive offset. Euclidean distance and activation-norm summaries provide
an additional scale-sensitive check. Distances weight image pairs equally;
class-cosine change summaries weight class pairs equally. Pair counts are not
independent sample sizes.

Linear CKA centers each evaluated group. Debiased CKA uses unbiased HSIC in a
normalized ratio, but that ratio is not unbiased and can be negative. Collapsed
or insufficient samples yield `null`. Shuffled-row CKA means are descriptive
finite-sample references, not p-values; `--null-repeats` controls their count.

Shared-neighbor scores use cosine with self excluded and k=5,10,25. Forget
and retain groups select anchors; candidate neighbors include all classes in
the partition. Zero vectors are excluded, insufficient pools yield undefined
scores, and counts/tie fractions are recorded. Ties use sample order.

PCA and orthogonal alignment use retained calibration examples only. All model
panels share axes and axis limits. Rank-deficient calibration leaves rotations
outside its span underdetermined. The reported variance is calibration variance.
A similar 2D position does not establish full-space equivalence. UMAP and linear
classifier probes are not part of this implementation.

## Retraining variability and interpretation

### Independent retraining controls

For a retraining variability control, the exporter accepts repeated
`--retrained-repeat path/to/retrained.pt` arguments. Each extra checkpoint must
have a matching adjacent `split.json`. Its checkpoint seed may differ; the
dataset must remain fixed. The analysis includes all pairwise comparisons,
including retrain/retrain, automatically. It does not declare equivalence or
construct a confidence band from a single replicate.

The existing training program changes the dataset split when `--seed` changes.
Those runs are useful separate pilot configurations but are **not** pure
retraining variability controls. Producing independent retrains on fixed splits
requires a training change or separately prepared checkpoints; this pipeline
does not silently train them. Shared initializations between original and
retrained models are another reason their geometry can be close.

High similarity to the original is not evidence of failed unlearning on its
own. Check whether the original/retrained pair is distinguishable, compare
unlearned/retrained against retrain/retrain, and retain behavioral evaluations.
A retained-only model can still represent a deleted class through shared features.
Neither CKA nor a 2D picture proves erasure or privacy. The present baseline
is suppression, and this pilot has no independent retraining controls.

## Verification

```powershell
python -m unittest discover -s tests -p test_representations.py -v
python representation_analysis.py synthetic --output runs/representations/rotation-control/features.npz
python representation_analysis.py analyze --features runs/representations/rotation-control/features.npz --output-dir runs/representations/rotation-control/analysis
```

The synthetic control makes original and unlearned representations differ only
by an orthogonal rotation; both CKA variants and neighbor overlap should be 1.
The synthetic retrained role is unrelated noise. Tests additionally cover
row permutation, small-sample/high-dimension bias, zero/collapsed vectors,
disjoint splits, nonfinite data rejection, retained-only alignment fitting,
archived digits activation extraction, and artifact overwrite protection.
