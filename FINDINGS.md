# Unlearning and weight faults: findings and limitations

This summary covers the digits classifier and ResNet-18 on CIFAR-10. 
No LLMs or physical GPU faults were tested.

## Main findings

1. **Targeted bit flips can restore recognition of a suppressed class in these
   experimental classifiers.** Even with the output head excluded, recovery
   remains high on digits and reaches 34.2% test recall on CIFAR-10 seed 0.
   However, independent seeds 1 and 2 fail the frozen criteria: recovery is
   not reliable across seeds.
2. **Random faults produced no substantial selective recovery in the completed
   campaigns.** Most single-bit faults left predictions unchanged; other
   prediction changes and damage became more common with multiple faults.
3. **Low accuracy on the forgotten class does not establish that knowledge was
   erased.** The procedure explicitly encourages incorrect answers and should
   be described as a simple suppression baseline using relabeling.
4. **Practical relevance depends on the attacker's access.** A particularly
   relevant scenario is modifying a victim's model without ordinary write
   permission. No advantage over alternatives available with the same
   privileges has been demonstrated.

## 1. What was tested

The CPU experiments used scikit-learn's 8×8 handwritten digits and an FP32
network with 64 inputs, 64 hidden ReLU units, and 10 outputs: **4,810 parameters**.
The targets were digits 0, 1, and 2, each with three seeds: nine configurations.

Each configuration included:

- an **original model**, trained on all classes;
- a **suppressed model**, initialized from the original and updated using
  randomly chosen, always-incorrect labels for the forgotten digit while
  retaining correct labels for the other digits;
- a **control retrained from scratch**, without training examples of the
  forgotten digit, using the same architecture and all ten outputs.

The loss is cross-entropy against the assigned labels. Selection of the
suppressed checkpoint required forgotten-class accuracy at most 10% and
retained-class accuracy within five percentage points of the original.
This is neither a reproduction of SalUn nor a certified removal method.

The split was approximately 60% training, 10% fault discovery, 10% checkpoint
selection, and 20% final testing. The test set contains about **36 images per
digit**. Seeds share the dataset; they are not independent datasets.

"Forgotten-class accuracy" means **recall**: the fraction of images of that
digit assigned the correct label. Retained accuracy is the mean accuracy over
the nine retained classes. The false-positive rate measures the fraction of
images from other classes incorrectly assigned the forgotten label.

## 2. Targeted flips: substantial recovery, with interpretation still open

Search uses gradients and labeled examples of the forgotten class to choose
useful bit flips. It does not consult the original weights to undo unlearning
updates exactly. It seeks **behavioral recovery**, with a maximum budget of
32 flips in the digits experiments.

| Mean across nine configurations | All weight matrices eligible | Output head excluded |
|---|---:|---:|
| Forgotten-class recall before → after | 0.9% → **94.1%** | 0.9% → **89.1%** |
| Retained-class accuracy before → after | 96.7% → 96.2% | 96.7% → 95.4% |
| Retrained-control recall after its own search | 32.3% | **0.3%** |
| Selected flips | 4–8 | 14–32 |
| Configurations meeting all test constraints | 7/9 | 6/9 |

The selective-recovery criterion was fixed at a recall gain of at least
**20 percentage points**, retained accuracy loss at most two points, and
false-positive growth at most one point. Aggregate means do not imply that
every individual experiment satisfied these constraints.

The control underwent the same search procedure, with the same maximum budget
and independent selection of the flip count. **Search on the control also sees
labeled examples of the digit absent from its training data.** Its outcome
therefore measures how much recognition the search itself can introduce.

The gap with the head excluded is consistent with useful information persisting
in the suppressed model. It is not a causal demonstration: the control's
forgotten-class output never received positive training examples. Freezing that
output can make recovery inherently harder. Search is also a form of supervised
optimization, even without conventional training updates. Sign and exponent
bits are eligible; a few bit flips need not imply small numerical changes.

Details: [CPU targeted-flip results](reports/targeted-flips/cpu/RESULTS.md) and
[CPU protocol](docs/targeted-flips-cpu.md).

### CIFAR-10 / ResNet-18 and frozen replications

The model uses the same simple relabeling baseline, not a reproduced published
unlearning method. Original and retrained models use 100 epochs; suppression
uses 15. The target is airplane, with nine retained classes.

| Seed-0 evaluation on the official test split | Airplane recall | Retained accuracy | Airplane false-positive rate |
|---|---:|---:|---:|
| Original | 95.40% | 94.28% | 0.57% |
| Suppressed, before faults | 0.00% | 94.57% | 0.00% |
| Suppressed, five flips with head excluded | **34.20%** | **93.23%** | **0.86%** |
| Retrained control, selected checkpoint | 0.00% | 94.53% | 0.00% |

This corresponds to **342 correct airplane predictions out of 1,000**, with
retained accuracy loss of 1.33 points: all predefined criteria pass on test.
None of the 20 random trials at five flips recovers airplane recognition.
All five selected flips toggle exponent bit 26, multiplying the affected
weight by 256: the change is sparse but numerically large.

The initial greedy search recovered only 0.4% on test. Subsequent exploratory
configurations were compared on validation data, reaching 35.4% recall with
the head excluded. The report includes worse alternatives and incomplete
campaigns. The limit of 96 full validations stopped the best search at five
flips, below its 128-flip budget.

Selected checkpoints and configuration were frozen and verified before final
test evaluation, without rerunning search. However, the same test set had
already been viewed in the initial experiment: this is **frozen confirmation
on a previously examined test set**, not a new untouched dataset. Search uses
labeled airplane examples; it does not provide causal proof of recovery solely
from residual knowledge.

Seeds 1 and 2 used a fixed replication protocol. **Seed 1 did not recover
airplane recognition:** search explored six depths and exhausted its 1,280
candidate evaluations without increasing validation recall, selecting zero
flips. Test recall was 0.0%, with retained accuracy of 94.89%. The retrained
control also remained at 0.0%. This seed's 20 random trials use zero flips and
are not informative controls for nonzero faults.

Seed 2 selected six flips and reached **6.5% test recall**, with retained
accuracy of 92.16%: a loss of **2.52 points**, above the two-point limit. It
therefore fails both the recovery and utility criteria. These six flips also
multiply weights by 256 through exponent bit 26. The retrained control remains
at 0.0%; none of the 20 random trials at six flips recovers airplane recognition.

**One success across three seeds, but zero successes in the two independent
replications.** All models passed the training/unlearning gates, without
protocol changes or exclusion of negative seeds. Searches on suppressed models
were resource-limited: full validations for seed 0, candidate evaluations for
seed 1, and time for seed 2. This limits the generality of the positive result;
it does not demonstrate irreversible forgetting in the negative seeds.
Validation utility constraints did not fully generalize to test data in seed 2.

Details and archived JSON: [CIFAR-10 results](reports/targeted-flips/gpu/RESULTS.md).

## 3. Random flips: no substantial recovery observed

The unfiltered campaign includes **all 153,920 FP32 parameter bits**, including
biases. It does not filter large changes, NaN/Inf values, or unfavorable outcomes.
Every trial starts from the clean checkpoint and retains the corruption for
the full test batch. The same logical bit positions are applied to all three
models in each configuration.

Aggregate results across the nine suppressed models:

| Flips per trial | Trials | Unchanged predictions | Utility or false-positive damage | Nonfinite output | Substantial selective recoveries |
|---|---:|---:|---:|---:|---:|
| 1 | 90,000 | 88.588% | 1.547% | 0.072% | **0** |
| 4 | 9,000 | 61.822% | 5.656% | 0.156% | **0** |
| 16 | 9,000 | 16.222% | 21.900% | 1.078% | **0** |

These columns do not cover every outcome: other prediction changes and small
recall gains with preserved utility also occurred. Among single-bit trials,
89 faults increased forgotten-class recall. Of these, 85 violated at least one
utility or false-positive constraint; the remaining four corrected **just one
additional image**, about 2.8 percentage points.

Including original and retrained controls, the campaign contains **324,000
model evaluations**, representing 108,000 fault trials paired across the three
model types. None showed substantial selective recovery under the criterion.
Nonfinite outcomes remain in the denominators rather than being discarded.

For each checkpoint, zero events in 10,000 single-bit trials gives an exact
one-sided 95% binomial upper bound of approximately **0.030% per injected
fault**, under the adopted sampling model. This is a per-checkpoint/condition
bound, not a simultaneous guarantee across all configurations.

This probability is conditional on an uncorrected error reaching the parameters.
**It is not a probability per inference, GPU-hour, or device lifetime.** Physical
error rates, distributions of vulnerable cells, ECC, scrubbing, and residence
times were not measured. Trials with four or sixteen flips are stress tests of
accumulated corruption, not calibrated models of natural events or physical
bursts.

The initial random comparison in the targeted-flip study used different filters
and budgets. The two campaigns must not be pooled as though they used the same
fault model.

Details: [random-flip results](reports/random-flips/RESULTS.md) and
[random-flip protocol](docs/random-flips.md). Archived evidence includes NPZ
checkpoints, trial logs, and verification records: 663 independently
reconstructed faulted models reproduced their recorded metrics.

## 4. Interpreting "ignorance" metrics

A model unfamiliar with a class need not guess uniformly. Ten percent is the
expected accuracy of a predictor choosing uniformly among ten labels, not an
automatic property of unlearning. This model uses argmax and must always choose
a digit; it has no "unknown" output.

For the control trained without the forgotten class, predicting that label is
always an error during training. Its output can therefore be disfavored and
recall can approach zero even without prior positive examples. In the suppressed
model, intentionally incorrect labels directly discourage predicting the class.

An illustrative check on one original model also produced high softmax scores
for unrelated inputs: a completely white image was labeled digit 2 with a 99.8%
score; a hand-drawn A was labeled digit 4 with a 97.5% score. Among 1,000 uniform
noise images, 57.5% had a maximum score above 90%. These scores do not reliably
indicate membership in the known classes. These are illustrative examples, not
a systematic out-of-distribution evaluation.
[Recorded values](reports/targeted-flips/cpu/ood-check.json).

Further experiments should compare retrained controls, prediction distributions,
and internal representations in addition to class accuracy. Zero accuracy,
uncertainty, and removal of the influence of training data are different goals.

## 5. Hardware relevance and alternative techniques

The literature documents targeted hardware attacks on models, but **these
experiments used only software XORs on parameters**. They did not induce
physical faults, establish the vulnerability of a particular GPU, or model
stuck-at faults in multiply-accumulate units.

[PrisonBreak](https://arxiv.org/html/2412.07192v3) demonstrates targeted GPU faults
against LLM alignment under specific co-location and hardware assumptions.
It does not demonstrate this project's unlearning-reversal objective. Being
able to select a model bit does not establish that it can be physically flipped
in the required direction.

Relevant access scenarios include:

- **Unrestricted weight access:** compare with restoring an available original
  model, fine-tuning, and modifying the classifier head. Fewer changed bits do
  not establish lower total cost.
- **Inference API access only:** hardware faults are not directly available;
  attacks on inputs accepted by the service are the relevant comparison.
- **Unprivileged code on shared hardware, without write access to the victim's
  model:** a fault could provide an otherwise unavailable capability, if the
  hardware and configuration permit it.
- **Physical access to a protected device:** a separate scenario with its own
  fault-injection costs and constraints.

Exploits that permit arbitrary writes are also alternatives to consider, not
only attacks on model weights. A practical advantage over alternatives with
the same access **remains to be demonstrated**.

## 6. Priorities for further experiments

1. Verify provenance, preprocessing, and compatibility of the potential
   CIFAR-10 original / SalUn / retrained trio identified in the
   [checkpoint candidates](docs/checkpoint-candidates.md). Those weights have
   not yet been downloaded or validated for these experiments.
2. Repeat with a published method, an explicit definition of unlearning, and
   controls that distinguish residual recovery, supervised relearning, and
   classifier-head repair.
3. Keep the two questions separate: targeted-flip effectiveness and the
   frequency of selective outcomes under a random-fault model fixed before
   the experiment.
4. Introduce verifiable hardware constraints on locations, directions, and
   collateral corruption. Evaluate parameter faults and computation faults
   separately.
5. Compare techniques available with the same privileges and data, measuring
   success, utility, search cost, reliability, and persistence.

**The supported conclusion is that a few search-selected bit flips can reverse
class suppression at least partially in these models, while uniformly random
faults produced no substantial recovery in the completed campaigns. Effective
knowledge removal, generality of the phenomenon, and a practical advantage
for hardware attacks remain open questions.**
