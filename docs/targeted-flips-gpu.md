# Targeted flips: GPU / CIFAR-10

`targeted_flips_gpu.py` searches for FP32 weight-bit flips in a CIFAR-adapted
ResNet-18. A GPU is recommended for the full experiment; CPU execution also
supports smoke checks and slower runs. Run all commands from the repository root.

[Experiment chooser](../README.md) · [Recorded results](../reports/targeted-flips/gpu/RESULTS.md)

The default greedy search is a starting configuration. The archived strongest
seed-0 result used a different beam-search configuration; its exact settings
are recorded in the result report and frozen protocol.

## Hardware and scope

The experiment runs on a single CUDA-capable GPU. The default batch size is
128; reduce it if GPU memory is insufficient. CPU execution is supported,
but full training will be much slower.

These are **software-injected faults**, implemented as XORs in the FP32 tensor
storage. They are persistent during inference. This does not establish that a
physical mechanism can induce the same changes on a GPU. It does not
model stuck-at MAC faults, memory-address restrictions, or transient timing.

## Linux setup

Use a recent supported Python, for example Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install a matching CUDA-enabled `torch` / `torchvision` pair using the command
from the [official PyTorch installation selector](https://pytorch.org/get-started/locally/)
for Linux, pip, and a CUDA build compatible with your GPU and NVIDIA driver.

```bash
python -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0)); print(torch.ones(1, device="cuda"))'
python -m pip install -r requirements-cpu.txt
python -m unittest discover -s tests -v
python targeted_flips_gpu.py --stage smoke --device cpu --root runs/smoke
```

## First run

```bash
python targeted_flips_gpu.py --stage train --device cuda:0 --root runs/seed0-class0
python targeted_flips_gpu.py --stage search --device cuda:0 --root runs/seed0-class0
```

The first command downloads CIFAR-10 and trains:

1. **Original:** all ten classes, 100 epochs by default.
2. **Unlearned:** initialized from the original checkpoint, then fine-tuned on
   retained examples and randomly incorrect labels for the forgotten class.
   This is a simple RL-style baseline, **not a reproduction of SalUn**.
3. **Retrained:** trained from scratch without examples of the forgotten class,
   using the same architecture, ten outputs, and initialization seed.

The default forgotten class is 0, airplane. Change it with `--forget-class`.
For independent experiments on multiple GPUs, use a separate run root and
`--device` value for each process.

If interrupted after original training and unlearning have both completed, use
`--stage retrain` with the same run root and training settings. This preserves
those two checkpoints, restarts retraining from its initialization (not its
interrupted epoch), and then runs the search. Any partial retraining checkpoint
is archived. Once all training finishes, `--stage search` reuses all three
checkpoints without further training.

Training stops before the search if the original validation forget or retain
accuracy is below 70%, or if no unlearned checkpoint achieves forget accuracy
at most 10% while retaining accuracy within five percentage points of the
original. These are baseline gates, not guaranteed outcomes. Failed gates mean
the training/unlearning baseline needs work; they do not answer the fault
question. Adjust unlearning parameters using validation data only and a fresh
run directory. Inspect `*-training.json` for the trajectory.

The search stage uses gradients to rank bit candidates, evaluates their actual
effects, and accepts one bit change at a time. It never uses an optimizer to
update weights. It searches both the unlearned and retrained models with the
same maximum budget (32 bits). Their selected counts may differ. It also
evaluates 20 independent random-fault trials on the unlearned model, using the
selected targeted bit count. Increase this to 100 for a larger random-control sample:

```bash
python targeted_flips_gpu.py --stage search --root runs/seed0-class0 --random-trials 100
```

Run this instead of the previous search command: the script refuses to
overwrite an existing search. Every accepted bit is logged, and an audit
verifies that no unlogged parameter or BatchNorm-buffer changes occurred.

The first search allows all parameter matrices, including the output head.
Repeat with the output head excluded to check for trivial classifier repair:

```bash
python targeted_flips_gpu.py --stage search --root runs/seed0-class0 --exclude-head
```

For exploratory searches on existing checkpoints, give each configuration a
unique `--search-name` and use `--validation-only` to keep the official test
split closed. For example:

```bash
python targeted_flips_gpu.py --stage search --device cuda:0 --root runs/seed0-class0 \
  --search-name search-beam64 --validation-only --algorithm beam \
  --budget 64 --beam-width 3 --branch-factor 3 \
  --candidate-weights 1024 --candidate-tests 64
```

The custom beam search keeps several distinct sets of flips rather than one
greedy path. Each extension must improve discovery loss and pass the same
retained-accuracy and false-positive constraints against the untouched model.
Beam paths rank by validation forgotten-class accuracy, then discovery loss;
all accepted children are considered for final checkpoint selection. This is
a heuristic, not an exhaustive search. Increasing the budget does not force
additional flips when every proposed extension fails.

`--forget-weight 4` weights forgotten-example cross-entropy four times as
strongly as retained-example cross-entropy, without relaxing the validation
constraints. `--exclude-head` remains available for both algorithms. Histories
are saved after each depth; diagnostics distinguish reaching the budget from
running out of acceptable extensions. Validation-only results use
`evaluation_split: selection` and contain no test metrics. Comparing multiple
configurations on this selection split is exploratory tuning; confirmation
requires a separately fixed evaluation and additional seeds/classes.

For shorter, bounded searches, add `--cache-eval --ranking bit-change
--screen-per-class 32 --validation-candidates 4 --max-candidate-evals 512
--max-full-validations 48 --max-search-seconds 180`. The candidate ranking
uses the largest beneficial gradient-times-change over valid storage flips,
rather than weight magnitude as a proxy. Evaluation images are cached on the
selected device (about 60 MiB for the 5,000 selection images).

The screening subset comes from discovery examples outside the gradient
probe. A custom utility score balances relative discovery-loss reduction
against screening retain-accuracy loss and false-positive growth. Screening
allows an extra `--screen-slack` (default two percentage points) to account
for sampling noise, but every accepted flip must still pass the original
constraints on the full selection split. Screening and shortlisting can miss
good candidates; this is an exploratory efficiency tradeoff, not an exact
replacement for exhaustive validation or a reproduction of TFL.

Limits apply separately to each model's targeted search. Candidate limits
count measured flip objectives; full-validation limits count candidate
evaluations on the selection split. Diagnostics record both, screening
evaluations, elapsed time, and `search_incomplete` when a cap is reached.
The time limit is checked between operations; initialization, final audits,
checkpoint saving, and random controls add overhead. Compare models under
matching maximum budgets and report the work each actually used.

To evaluate a validation-selected configuration without rerunning its search:

```bash
python targeted_flips_gpu.py --stage evaluate --root runs/seed0-class0 \
  --search-name search-beam64 --seed 0 --forget-class 0 --device cuda:0 --cache-eval
```

This loads the saved selected states, verifies their bit traces against the
base checkpoints, and evaluates the official test split. Before evaluation it
records checkpoint, configuration, split, and evaluator hashes in
`frozen-test-manifest.json`. It saves `frozen-test-results.json` separately from
validation results and refuses a second evaluation in the same directory.
Eligibility and random-trial counts come from the saved search configuration.
For seed replication, freeze all settings before viewing these results, retain
failed training/unlearning gates, and select each seed's flips on validation
data only. Reusing a previously examined test set is frozen confirmation,
not a new untouched test set.

## Pretrained checkpoints

Matched original/unlearned checkpoints could avoid the expensive training
stage and let us inspect larger models. They **do not remove the memory cost
of inference, gradients, and candidate evaluation**. A large model that only
just fits for inference may still be unsuitable for this search.

This version does not import arbitrary third-party checkpoints. `--stage search`
reuses checkpoints produced by this script, with its architecture and saved
dataset split. Before adding a published pair we must verify:

- Same architecture, output labels, numerical format, and preprocessing.
- The precise forgotten examples/class and the unlearning procedure.
- Which examples were used for training, unlearning, and evaluation.
- A compatible retrained control, or the ability to train that control.
- Whether checkpoint access and forgotten labeled examples match the assumed
  researcher/attacker access.

The [official SalUn repository](https://github.com/OPTML-Group/Unlearn-Saliency)
provides a useful implementation to consider for the next baseline. Availability
of a complete, compatible downloadable checkpoint bundle has not been verified.
A matching filename alone does not establish checkpoint compatibility. See
the [checkpoint candidates](checkpoint-candidates.md) for potential future baselines.

## Data separation and fault model

For each CIFAR-10 class, the 5,000 official training examples are split into
4,000 training, 500 search, and 500 selection examples. The official test split
is loaded only after targeted model selection is complete.

The search uses a fixed balanced probe of up to 32 search images per class;
selection uses the separate selection split. Original and unlearned model
selection also use this validation split. Fault selection has access to labeled
forgotten examples from the search split; **this is a white-box, supervised
robustness experiment**, not a data-free or black-box attack. It does not use
the original model's weights to choose faults.

The fault space is all 32 bits of FP32 matrix weights, excluding BatchNorm
parameters and biases. NaN/Inf values and values exceeding 16 times that
tensor's initial maximum absolute weight are excluded. Each coordinate-bit
pair may be changed only once. Random faults use the same validity rules and
matrix eligibility, but uniform scalar/bit selection; targeted and random
faults need not have the same layer distribution. The random baseline is
conditional on these rules, not a measured physical SEU distribution.

Selection requires retained accuracy to stay within two percentage points of
the starting checkpoint and the false-positive rate for the forgotten label
to rise by at most one percentage point. Among accepted search steps, the
checkpoint with highest validation forgotten-class accuracy is selected;
smaller budgets win ties. A zero-bit result means no improving checkpoint
was selected. Search failure is limited to this heuristic and budget.

## Reading the result

Results are saved to `runs/.../search/results.json`, with a separate directory
for the head-exclusion ablation. Report:

- Forgotten-class recall/accuracy, before and after faults.
- Macro accuracy over the retained classes and each class's accuracy.
- Forgotten-label precision, false-positive rate, and confusion matrix.
- Number of bits, affected layers, and the numerical changes.
- Targeted recovery versus random trials and recovery of the retrained control.

A proposed first screening criterion, fixed before inspecting test outcomes:
forgotten-class accuracy rises by at least **20 percentage points**, retained
accuracy falls by at most **2 points**, and forgotten-label false-positive
rate rises by at most **1 point**. Apply these thresholds on the held-out test
data too; validation success alone is insufficient.

A stronger indication of residual-knowledge recovery is a larger accuracy
gain in the unlearned model than in the retrained control (a proposed margin
is 10 percentage points), plus a result above the random-trial distribution.
These numerical thresholds are research choices, not established standards.
With only 20 random trials, avoid a strong tail-probability claim.

Repeat at least three seeds and three forgotten classes before drawing a
general conclusion. Report confidence intervals, failed baseline gates, and
failed searches, rather than only successful examples. Check that the
retrained control has adequate retained-class accuracy. Its original
forgotten-class accuracy need not be exactly zero because features transfer
between classes. Apply the same search to it: optimization with labeled
forgotten examples can introduce information as well as reveal it.

Successful recovery here would justify testing a published unlearning method,
stronger search/negative controls, and then hardware-constrained faults.
It would not establish physical exploitability, reversal of privacy guarantees,
or generalization to LLM safety behavior. A negative result would rule out
only this model, unlearning baseline, search procedure, and budget.

## Relevant primary sources

- [An Illusion of Unlearning? (AISTATS 2026)](https://proceedings.mlr.press/v300/gao26e.html):
  motivates examining residual internal representations and classifier effects.
- [SalUn official code](https://github.com/OPTML-Group/Unlearn-Saliency):
  a next baseline to reproduce; this baseline is independently implemented.
- [PrisonBreak](https://arxiv.org/abs/2412.07192): demonstrates targeted bit
  selection and a separate hardware realization for a different research target.
- [GPUHammer](https://gpuhammer.com/): reports experiments on an RTX A6000
  with GDDR6; physical feasibility remains specific to the tested hardware.

## Reproducibility and limitations

Run configuration, split indices, software versions, checkpoint metrics, and
fault traces are saved. Training and fault search use FP32, with TF32 disabled.
The script requests deterministic algorithms where possible, but does not
guarantee identical results across GPU architectures. Record the NVIDIA driver
version (`nvidia-smi`) and record the installed package versions:

```bash
mkdir -p runs
python -m pip freeze > runs/environment-lock.txt
nvidia-smi > runs/nvidia-smi.txt
```

CPU synthetic smoke testing is distinct from the full CIFAR-10/GPU experiment.
Do not interpret smoke outputs as experimental evidence.

