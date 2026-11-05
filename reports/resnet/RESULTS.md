# CIFAR-10 / ResNet-18: targeted bit-flip recovery

Recorded: **2026-11-05, 12:14 Europe/Rome**. This is a snapshot, not a final
multi-seed result. Seed 0 has completed frozen test evaluation; seeds 1 and 2
are training under a recorded protocol. Earlier digits findings are in the
[laptop report](../laptop/CPU_RESULTS.md).

## Main findings

- **Five targeted FP32 storage-bit flips restored airplane recognition from
  0.0% to 34.2% on the official test split**, with the classifier head excluded.
- Retained-class accuracy fell from **94.57% to 93.23%** (1.33 percentage
  points). The airplane false-positive rate rose from **0.00% to 0.86%**.
  All three predefined screening criteria passed on test data.
- The equally configured retrained control selected zero flips and stayed at
  **0.0% airplane accuracy**. All **20 matched-count random-fault trials** also
  had 0.0% airplane accuracy. These are bounded searches, not impossibility
  results or estimates of a natural hardware fault rate.
- The five edits are sparse but numerically large: **all toggle exponent bit
  26, multiplying the affected FP32 weights by 256**. They affect layer3/layer4
  convolutional or downsample matrices, not the output head.
- Larger search budgets alone did not ensure better outcomes. Broader
  candidates, beam paths, ranking and screening changed the selected paths;
  some configurations regressed. The best search hit its full-validation
  cap, not its 128-bit ceiling.

## Setup and outcome definitions

CIFAR-adapted ResNet-18: 3×3 stride-1 first convolution, no initial max-pool,
10 outputs, FP32. The forgotten class is 0 (airplane). Per class, the original
training split supplies 4,000 training, 500 discovery and 500 selection images.
The official test contains 1,000 airplanes and 9,000 retained-class images.

Original and retain-only retrained models use 100 SGD epochs; unlearning uses
15 epochs with random incorrect labels on airplanes and correct labels on
retained examples. Checkpoints are selected on validation only. **This is our
simple relabeling baseline, not a reproduced state-of-the-art unlearning
method or certified removal procedure.**

Forgotten accuracy means airplane recall. Retained accuracy is the macro mean
over the nine other classes. Airplane false-positive rate is the fraction of
non-airplane inputs labeled airplane. Screening thresholds, fixed before the
first CIFAR run, are a recall gain of at least 20 percentage points, retained
accuracy loss at most two points, and false-positive growth at most one point.

## Frozen seed-0 test results

The selected checkpoints and configuration were hashed before this evaluation.
The evaluator loaded and audited those exact states; it did not rerun search,
choose a different flip count, or tune to these test outcomes.

| State | Selected flips | Airplane accuracy | Retained accuracy | Airplane false positives |
|---|---:|---:|---:|---:|
| Original | 0 | 95.40% | 94.28% | 0.57% |
| Unlearned, before faults | 0 | 0.00% | 94.57% | 0.00% |
| Unlearned, frozen head-excluded selection | **5** | **34.20%** | **93.23%** | **0.86%** |
| Retrained without airplanes, before/search-selected | 0 | 0.00% | 94.53% | 0.00% |
| Random faults on unlearned model, 20 trials | 5 per trial | 0.00% in every trial | Per-trial metrics archived | 0.00% in every trial |

The targeted checkpoint recognizes **342/1,000 airplanes** and incorrectly
labels **77/9,000 retained images** as airplane. Its validation airplane recall
was 35.4%; the frozen test value is 34.2%.

The official test set had already been examined for the initial baseline
experiment. Consequently this is **frozen confirmation on a previously viewed
test set**, not evaluation on a newly untouched dataset. Seed 0 also supplied
the exploratory configuration choice. No cross-seed reliability claim follows
from this one completed seed.

## Exploratory configuration history

All percentages below are **validation airplane accuracy**, not test accuracy.
These configurations differ in multiple settings and sometimes GPU; this is
an exploratory ledger, not a controlled attribution of each component's gain.

| Configuration | Best selected flips | Airplane accuracy | Status |
|---|---:|---:|---|
| Initial greedy, 8 candidate tests | 3 | 0.4% | Complete; initial test also 0.4% |
| Wider greedy, 64 candidate tests | 3 | 5.0% | Complete |
| Beam, 64-bit ceiling | 6 | 16.4% | Unlearned search finished; campaign stopped during controls |
| Beam, airplane loss weight ×4 | 3 | 5.0% | Unlearned search finished; campaign stopped during controls |
| That campaign's head-excluded variant | — | — | Never started |
| Cached/screened beam, magnitude ranking | 4 | 4.6% | Complete; retrained control hit validation cap |
| Cached/screened beam, bit-change ranking | 5 | 29.4% | Complete |
| Wider bit-change beam, head allowed | 3 | 6.8% | Complete; retrained control hit validation cap |
| Wider bit-change beam, head excluded | **5** | **35.4%** | Complete; unlearned search hit validation cap |

The final configuration uses beam width 4, branch factor 3, 2,048 shortlisted
weights, 64 measured candidate flips per expansion, a 320-image gradient probe
and a separate 320-image discovery screening subset. At most eight screened
candidates per expansion receive full validation. Preservation limits remain
unchanged. Each model receives maxima of 128 flips, 1,280 candidate objectives,
96 full validations and 180 seconds; actual work can differ.

The selected unlearned search used 1,088 candidate objectives, 1,083 screens
and 96 full validations in 138.8 seconds, reaching depth five. Its retrained
control used 832 candidate objectives and 94 full validations in 107.5 seconds,
reaching depth four before finding no acceptable extensions. Matching caps do
not imply equal search difficulty or exhaustive coverage.

Caching evaluation tensors and screening a small discovery subset made the
bounded comparisons practical: the smaller two-configuration campaign took
about two minutes overall; the wider campaign took about five. This is not an
isolated caching speedup measurement, since search settings also changed.

## What these results do and do not establish

The evidence supports **behavioral recovery through a few targeted software
weight faults in this relabeling baseline**, including when the output head
cannot change. It is consistent with useful residual information persisting
after suppression, but does not prove that causal interpretation.

The search sees labeled forgotten-class images and can introduce behavior
through supervised optimization. The retrained control is important, but its
class output never received positive airplane training examples; excluding
the head can therefore create asymmetric recovery difficulty. The random
control is conditional on finite, layer-magnitude-bounded FP32 matrix faults,
not the unfiltered digits random-fault campaign. Do not pool their trials.

Our greedy/beam implementation combines standard heuristic ideas; it is not a
faithful implementation of TFL, SOLEFLIP or a published unlearning method, and
we make no algorithmic novelty or state-of-the-art comparison claim. High
retained utility in an MU benchmark is a different quantity from our post-fault
forgotten-class recall. No LLM unlearning or physical GPU fault injection was
tested. All changes were software XORs at ideal arbitrary weight addresses.

## Frozen replication and outstanding work

Seeds **1 and 2** are in progress with unchanged training/search settings and
no outcome-based tuning. They select flips using validation only and then run
one frozen test evaluation. The protocol retains failed training/unlearning
gates rather than dropping or replacing seeds. Searches hitting caps are
reported as incomplete. Each seed also receives its own retrained control and
20 random trials. Seeds change initialization, split and RNG, but share the
same dataset and official test images; these are not independent datasets.

Pending: seed-1/2 outcomes and across-seed variation; other forgotten classes;
stronger published MU baselines; comparisons with ordinary supervised repair;
and physically constrained fault models. No result from a pending seed is
included in the headline finding.

## Saved evidence

- [Frozen test metrics, confusion matrices and exact traces](seed0-frozen-test.json).
- [Selected validation results](seed0-selection-results.json).
- [Test manifest and checkpoint/evaluator hashes](seed0-frozen-test-manifest.json).
- [Exploratory search ledger, including stopped and unstarted runs](search-ledger.json).
- [Frozen replication protocol](frozen-protocol.json) and [file checksum](frozen-protocol.json.sha256).
- [Replication status snapshot](confirmation-status-snapshot.json), current at recording time.

Live replication outputs are in `runs/frozen-confirmation-20261105-121007/`;
seed-0 checkpoints remain in `runs/resnet18-seed0-class0-20261105-021912/`.
These local run directories are git-ignored. The compact JSON evidence above
is copied into `reports/resnet/` for durable review; checkpoint weights remain
local. Update this snapshot after the pending seeds complete.
