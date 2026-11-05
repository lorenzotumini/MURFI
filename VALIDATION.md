# Validation record

## Desktop GPU update, 2026-11-05

- Existing `.venv` validated on Linux with Python 3.12.14, PyTorch
  2.15.0+cu132, torchvision 0.29.2+cu132 and both RTX 3070 Ti / RTX 5060 Ti.
- CPU and both CUDA synthetic smoke runs completed successfully.
- CIFAR-10 training, unlearning, retraining and targeted/random fault searches
  completed for seed 0. Frozen-test results are in the
  [ResNet report](reports/resnet/RESULTS.md); seed-1/2 replication is pending.
- All 14 tests passed after adding beam search, exact bit-change ranking,
  cached evaluation, capped screening and frozen checkpoint evaluation.
- Frozen-evaluation checks verify selected-state and validation-file hashes
  remain unchanged, no search is rerun, and a second test evaluation is refused.
- Third-party checkpoint import, published-MU reproduction, LLM experiments
  and physical fault injection remain unperformed.

## Historical laptop handoff, 2026-11-04

The limitations below describe the laptop handoff only; the desktop update
above supersedes its unperformed CUDA/CIFAR items.

Date: 2026-11-04.

- Python syntax compilation passed.
- All three standard-library binary representation tests passed.
- The synthetic CPU smoke run completed original training, relabeling,
  retraining, targeted searches, random trials, tensor-mutation audits, and
  JSON serialization. A targeted candidate was accepted during the retrained
  search, exercising the actual mutation/audit path.
- A CIFAR-adapted ResNet-18 CPU forward pass produced finite 10-class outputs.
- All 32 bit positions were toggled in actual FP32 tensor storage and checked
  against their expected integer XOR; a second toggle restored the original
  storage word exactly.

Local validation used an isolated Python 3.12 environment with CPU-only
PyTorch 2.15.0 and torchvision 0.29.2. These are a record of the local test
environment, not a requirement to install those exact versions on the desktop.

**Not yet performed:** CIFAR-10 download/training, a complete scientific run,
CUDA execution, Linux execution, third-party checkpoint import, or physical
fault injection. There is no measured class-recovery result yet. The temporary
laptop test environment and package cache were removed at handoff.
