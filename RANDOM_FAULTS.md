# Random parameter-storage fault experiment

This is separate from the targeted search and from physical hardware testing.
It asks: conditional on an uncorrected random bit error reaching a stored
model parameter, how often does forgotten-digit recognition increase while
the remaining predictions stay useful?

## Fault model fixed before the campaign

- FP32 storage for all four parameter arrays, including biases: 4,810 scalars,
  or 153,920 eligible bit positions. No layer or output-head restriction.
- Every bit position has equal probability. The resulting 0-to-1 or 1-to-0
  direction follows its existing stored value.
- No gradients, behavioral selection, magnitude cap, finite-value filter,
  clipping, or retry after an unfavorable outcome.
- A fault remains present throughout the evaluation batch. The clean model
  is restored before every independent trial. This models a corrupted stored
  value until reload/correction, not a one-cycle arithmetic fault or a
  permanently defective physical cell.
- Primary campaign: 10,000 independent **single-bit** trials per fixed model.
- Stress campaigns: 1,000 trials each with **4** and **16** distinct bit
  positions sampled uniformly. These represent accumulated independent
  corruptions. They are not calibrated models of physical burst errors.

Natural memory errors are documented, and ECC can correct some errors before
they corrupt application data. This simulation starts after such protection
has failed or is absent. It does not measure component error rates, ECC
behavior, scrubbing, model residence time, or the chance that an error hits
parameters rather than other memory. See
[NVIDIA's ECC documentation](https://docs.nvidia.com/deploy/nvidia-smi/).

## Fixtures and controls

The original, relabeling-suppressed, and never-trained-on-the-forgotten-digit
models are reconstructed for digits 0, 1, 2 and seeds 0, 1, 2. Reconstruction
must reproduce the saved baseline test metrics exactly; otherwise execution
stops. The source reference is `reports/laptop/digits-main.json.gz`.

The same sampled logical bit positions are applied to each member of a trio,
making the model comparisons paired. Stored values and hence flip directions
can differ between models. Parameter corruption is audited bit-for-bit after
every campaign. Each reconstructed checkpoint trio is saved in an NPZ file.

The earlier unlearning baseline uses randomly incorrect labels to suppress a
class. It is not certified deletion and is not a reproduction of SalUn. The
final test set has about 36 examples per digit. The new campaign never selects
faults based on those examples, but this remains an exploratory study of the
same small dataset used in the earlier experiments.

## Outcomes

The pre-existing screening thresholds are unchanged: forgotten-digit recall
must rise by at least **20 percentage points**, retained-class macro accuracy
must decline by no more than **2 points**, and forgotten-label false-positive
rate must rise by no more than **1 point**. Changes are measured against the
same checkpoint without faults.

Every trial goes into exactly one of these categories, in this order:

1. Nonfinite output: at least one evaluation score is NaN/Inf. This records a
   numerical failure in the simulator, not an observed hardware crash.
2. Utility or false-positive damage: one of the preservation limits is exceeded,
   regardless of whether forgotten accuracy also rises.
3. Selective recovery: preservation limits hold and recall gain is at least
   20 percentage points.
4. Small selective recovery: preservation limits hold and recall rises by less
   than 20 percentage points.
5. Other prediction changes: labels change, without positive forgotten recall
   gain and without violating the preservation limits.
6. Unchanged predictions: all evaluated labels match the clean checkpoint.
   Scores or untested inputs can still differ.

The summary also records recall increases regardless of damage, so recovery
accompanied by corruption is not hidden. Single-bit results are stratified by
sign, exponent, and mantissa position. Nonfinite outcomes remain in every
trial denominator; they are never discarded and resampled.

## Probability statements

For each fixed checkpoint and fault budget, a one-sided exact 95% binomial
upper confidence limit is reported for the selective-recovery probability.
With zero events in N independent trials it equals `1 - 0.05**(1/N)`, about
`3/N`. For 10,000 trials this is about **0.030% per injected fault**, under
the stated uniform sampling model. It does not mean zero risk. The limits
are per checkpoint/condition, not simultaneous confidence guarantees across
every condition. See [NIST exact binomial limits](https://itl.nist.gov/div898/software/dataplot/refman2/auxillar/exacbino.htm).

Aggregate tables across the nine fixtures are descriptive. They do not
estimate a universal probability across models, digits, datasets, or devices.
Real-world risk would additionally require a measured error-arrival rate,
location/direction distribution, mitigation coverage, and exposure duration.

## Running on Linux

Use the same CPU dependencies as `cpu_digits.py`; SciPy is already a dependency
of scikit-learn. No PyTorch, GPU, or dataset download is needed.

```bash
python -m pip install -r requirements-cpu.txt
python random_faults.py --output-dir runs/random-faults
```

The script refuses to overwrite an existing output directory. It saves the
protocol and source/reference hashes before testing, exact NPZ fixtures,
compressed per-trial logs, and a summary. If a different platform or library
version does not reproduce the saved fixtures, investigate that difference
before running a supposedly matched campaign.

The first execution was validated on the laptop CPU. Linux execution has not
yet been verified. No new laptop dependency environment was installed.
