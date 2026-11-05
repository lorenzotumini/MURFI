# Targeted flips: CPU / digits

`targeted_flips_cpu.py` trains a small FP32
64-input / 64-hidden-unit ReLU / 10-output network on scikit-learn's packaged
8x8 handwritten digits. It needs NumPy, scikit-learn, and their normal
dependency threadpoolctl. It needs neither PyTorch nor dataset downloads.
Run all commands from the repository root.

[Experiment chooser](../README.md) · [Recorded results](../reports/targeted-flips/cpu/RESULTS.md)

Install the CPU dependencies, then run the two targeted conditions:

```bash
python -m pip install -r requirements-cpu.txt
python targeted_flips_cpu.py --seeds 0 1 2 --classes 0 1 2 --random-trials 100 --output runs/digits-main.json
python targeted_flips_cpu.py --seeds 0 1 2 --classes 0 1 2 --random-trials 100 --exclude-head --output runs/digits-no-head.json
```

The script checks its analytic gradients against finite differences before
running and audits the exact binary changes for every targeted and random
model. It writes split indices, configuration, dependency versions, unlearning
validation histories, accepted fault traces, selected bit counts, and held-out
test metrics. Use a new output path for each run; it refuses to overwrite.

The data are stratified into approximately 60% training, 10% fault discovery,
10% model/fault selection, and 20% testing. Final testing has about 36 examples
per digit, so accuracy estimates are noisy. Model selection and fault
selection never use test inputs or labels. These are sample splits, not
writer-disjoint splits, and repeated seeds share underlying dataset examples.

The original and retrained models use 100 Adam epochs with learning rate
0.003. The retrained model has never seen training images of the forgotten
digit but retains all ten outputs. Unlearning starts from the original and
uses random incorrect labels for the forgotten digit alongside correct labels
for the other digits. It evaluates 30-epoch trajectories at learning rates
0.0001, 0.0003, 0.001, and 0.003. Selection requires forgotten accuracy at most
10% and retained macro accuracy within five percentage points of the original.
This is a simple RL-style toy baseline, not a reproduction of SalUn.

The search uses labeled discovery images, gradients, and actual candidate
evaluation. It changes only FP32 matrix bits, without optimizer steps. All bit
positions are eligible, subject to finite weights and the same 16-times-layer-
maximum bound used by the GPU targeted experiment. It tests a shortlist of candidates
and accepts up to 32 flips. It selects the validation checkpoint with highest
forgotten accuracy subject to a two-point retain-accuracy loss limit and a
one-point false-positive-rate increase limit. The head-exclusion condition
allows changes only in the input-to-hidden matrix. Biases are excluded in
both conditions. Random trials use the selected unlearned model's bit count
and the same bit validity rules. Retrained controls receive the same search
budget ceiling and select their own bit counts.

Search access to labeled forgotten examples means recovery can reflect
learning induced by supervised search. The retrained control and head
ablation help investigate this confound, but do not eliminate every
alternative explanation. Software recovery here does not establish physical
fault feasibility or generalize to modern unlearning algorithms and LLMs.
