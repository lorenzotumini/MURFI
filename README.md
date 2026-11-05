# Machine Unlearning Reversal via Fault Injection

Can bit flips in a model's stored weights restore recognition of a forgotten
class while preserving recognition of the others? MURFI explores two questions:
**targeted flips**, chosen to recover that behavior, and **random flips**, sampled
without looking for a favorable outcome.

These experiments use software XORs on FP32 parameters and a simple relabeling
baseline for unlearning. They do not demonstrate physical GPU faults or
certified removal of knowledge.

## Targeted flips

Choose an experiment according to the hardware available:

| Option | Model and data | Entry point | Guide | Results |
|---|---|---|---|---|
| CPU | Small NumPy network, packaged handwritten digits | `targeted_flips_cpu.py` | [CPU guide](docs/targeted-flips-cpu.md) | [Digits results](reports/targeted-flips/cpu/RESULTS.md) |
| GPU recommended | PyTorch ResNet-18, CIFAR-10 | `targeted_flips_gpu.py` | [GPU guide](docs/targeted-flips-gpu.md) | [CIFAR-10 results](reports/targeted-flips/gpu/RESULTS.md) |

Both search for useful flips, evaluate a retrained control, and include random
controls matched to the targeted fault rules. They use different models and
datasets; the hardware options are different experiments, not interchangeable
backends for the same experiment.

### Start with the CPU experiment

No GPU, PyTorch or dataset download is needed. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
python targeted_flips_cpu.py --seeds 0 --classes 0 --output runs/targeted-flips-cpu.json
```

This trains the three models, searches for flips and saves metrics and exact
traces. The terminal prints a short result. Use a fresh output path for each
run. Add `--exclude-head` to restrict flips to the hidden-layer weights; see the
[CPU guide](docs/targeted-flips-cpu.md) for the full multi-seed campaign.

### Run the GPU experiment

Follow the [GPU setup guide](docs/targeted-flips-gpu.md) to install a matching
PyTorch/torchvision pair. Then:

```bash
python targeted_flips_gpu.py --stage train --device cuda:0 --root runs/seed0-class0
python targeted_flips_gpu.py --stage search --device cuda:0 --root runs/seed0-class0
```

The guide covers synthetic smoke checks, head exclusion, exploratory search,
and frozen evaluation. The default search is a starting configuration; the
[recorded GPU result](reports/targeted-flips/gpu/RESULTS.md) identifies the exact
configuration used for that result.

## Random flips

`random_flips.py` asks how often uniformly sampled storage faults produce
selective recovery. It uses the CPU digits fixtures, includes biases, and keeps
large or nonfinite corruptions in the results. This differs from the filtered
random controls inside the targeted experiments.

With the CPU dependencies installed:

```bash
python random_flips.py --output-dir runs/random-flips
```

The default command runs the full campaign and refuses to overwrite an existing
directory. See the [random-flip guide](docs/random-flips.md) for the protocol and
[results](reports/random-flips/RESULTS.md) for the archived measurements.

## Further reading

- **[Findings and limitations](FINDINGS.md)** — Start here for the main
  conclusions from targeted and random flips, and the limits of their
  interpretation.
- **[Representation results](reports/representations/RESULTS.md)** — Comparisons
  of original, suppressed, and retrained models on digits and CIFAR-10, with
  links to detailed tables and plots.
- **[Representation analysis guide](docs/representation-analysis.md)** — How
  to reproduce those comparisons and interpret CKA, nearest-neighbor overlap,
  class similarities, and PCA.
- **[Candidate external checkpoints](docs/checkpoint-candidates.md)** —
  Possibilities for future experiments and the validation needed before use.

Experiment reports in `reports/` include supporting evidence. Downloaded
datasets belong in `data/`; new run outputs belong in `runs/`. Both directories
are ignored by Git.

## Running the tests

To run the full test suite, install both the CPU dependencies and the
PyTorch/torchvision pair described in the GPU guide, then run:

```bash
python -m unittest discover -s tests -v
```

The tests cover FP32 bit mutations, search constraints, evaluation of frozen
checkpoints without rerunning search, and representation metrics and artifacts.
