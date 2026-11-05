# Targeted flips: GPU / CIFAR-10 results

Experiment date: **2026-11-05**. All three seeds completed frozen
test evaluation under the recorded protocol. Only the exploratory seed 0
passes the predefined screen; neither independent replication passes. Earlier digits findings are in the
[CPU targeted-flip report](../cpu/RESULTS.md).

## Main findings

- **Substantial selective recovery did not replicate under the frozen search
  budgets.** Seed 0 passed, seed 1 selected zero flips and had 0.0% test recall,
  and seed 2 reached 6.5% recall with six flips but lost 2.52 points of retained
  accuracy. This is one pass out of three runs, **zero passes out of the two
  independent replications**. All unlearned searches reached a resource cap;
  the negative outcomes do not prove irreversible forgetting.
- **In seed 0, five targeted FP32 storage-bit flips restored airplane recognition from
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
the exploratory configuration choice. Neither independent replication meets the recovery/preservation criteria;
cross-seed reliability is not established.

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

## Completed frozen replication

| Seed | Role | Selected flips | Test airplane accuracy | Retained accuracy | Retained loss (points) | Airplane false positives | Test screen |
|---|---|---:|---:|---:|---:|---:|---|
| 0 | Exploratory configuration seed; frozen confirmation | 5 | 34.20% | 93.23% | 1.33 | 0.86% | Pass |
| 1 | Independent frozen replication | 0 | 0.00% | 94.89% | 0.00 | 0.00% | Fail: no recovery |
| 2 | Independent frozen replication | 6 | 6.50% | 92.16% | 2.52 | 0.10% | Fail: insufficient recovery and excess utility loss |

All three completed the training/unlearning gates; no seed was discarded or
retuned. Retrained controls selected zero flips and had 0.0% test airplane
recall in every seed. Their retained test accuracies were 94.53%, 94.77% and
94.29%, respectively.

Seed 1 explored six accepted depths without increasing validation airplane
recall, so validation selection retained the unchanged zero-bit checkpoint.
Seed 2 selected six flips at its time limit. It reached 5.4% validation airplane
recall and 93.49% retained accuracy, meeting the validation preservation limits,
but its test retained loss exceeded two points. Thus preservation on selection
data did not guarantee preservation on test. Its six edits also toggle
exponent bit 26 and multiply weights by 256, all in layer4 matrices.

| Seed | Unlearned search: candidate objectives / full validations | Stop | Retrained search: candidate objectives / full validations | Stop |
|---|---|---|---|---|
| 0 | 1,088 / 96 | Full-validation cap | 832 / 94 | No acceptable extension |
| 1 | 1,280 / 63 | Candidate-evaluation cap | 1,216 / 96 | Full-validation cap |
| 2 | 1,156 / 92 | Time cap | 1,216 / 96 | Full-validation cap |

All unlearned searches are resource-capped and marked incomplete. Matching
maximum budgets did not yield identical work; GPU assignments differ and a
wall-time cap also applies. These are negative results for this heuristic and
budget, not exhaustive statements about remaining recoverable information.

Each seed has 20 random trials with its selected targeted count. No trial
recovered airplane recognition. Seed 0's trials have five flips and seed 2's
have six. Seed 1's trials have **zero flips**, reproduce the baseline, and are
**not informative nonzero-fault controls**. Do not treat all 60 trials as
nonzero random-fault experiments or combine them with the unfiltered digits
campaign.

The descriptive mean test airplane recall is 13.57% across the three runs;
the median is 6.5%. This includes the exploratory configuration seed and is
not an unbiased estimate of generalization. The independent replications are
0.0% and 6.5%, with neither passing the predefined screen. The headline must
therefore remain **a seed-dependent vulnerability, not reliable multi-seed
unlearning reversal**.

All selections used validation only, followed by one frozen test evaluation.
The same dataset and official test images are shared; seeds vary initialization,
split and RNG rather than supplying independent datasets. No outcomes caused
hyperparameter changes or replacement seeds. The earlier test exposure and
supervised search limitations described above still apply.

Outstanding: other forgotten classes; stronger published MU baselines;
comparisons with ordinary supervised repair; and physically constrained fault
models. Further experiments should be recorded as a new exploratory protocol,
not a continuation that changes this frozen replication's outcome.

## Saved evidence

- [Frozen test metrics, confusion matrices and exact traces](seed0-frozen-test.json).
- [Selected validation results](seed0-selection-results.json).
- [Test manifest and checkpoint/evaluator hashes](seed0-frozen-test-manifest.json).
- [Exploratory search ledger, including stopped and unstarted runs](search-ledger.json).
- [Frozen replication protocol](frozen-protocol.json) and [file checksum](frozen-protocol.json.sha256).
- [Historical seed-0 status snapshot](confirmation-status-snapshot.json).
- [Seed-1 frozen test](seed1-frozen-test.json), [validation selection](seed1-selection-results.json),
  [test manifest](seed1-frozen-test-manifest.json), and [search trajectory](seed1-unlearned-search.json).
- [Seed-2 frozen test](seed2-frozen-test.json), [validation selection](seed2-selection-results.json),
  [test manifest](seed2-frozen-test-manifest.json), and [search trajectory](seed2-unlearned-search.json).
- [Final three-seed replication summary](replication-summary.json).

The archived evidence includes metrics, configurations, fault traces, and
checkpoint hashes. The checkpoint weights themselves are not distributed in
this repository.
