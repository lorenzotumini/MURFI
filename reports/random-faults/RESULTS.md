# Unfiltered random-fault campaign: exploratory CPU results

Run on 2026-11-04. No GPU, new dependency installation, or physical fault injection was used.

The campaign reconstructed all nine original/unlearned/retrained model trios from the earlier digits study and verified their baseline metrics. It used uniformly selected FP32 parameter-storage bits, including biases, with no magnitude or finite-value filtering. Each trial starts from a clean checkpoint.

The fixed screening criterion is at least +20 percentage points of forgotten-digit recall, at most -2 points of retained macro accuracy, and at most +1 point of forgotten-label false-positive rate.

## Suppressed-model results

| Flips per trial | Trials across nine checkpoints | Unchanged predictions | Other prediction changes | Small recall gain with preserved utility | Substantial selective recovery | Utility/FPR damage | Nonfinite output |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 90,000 | 79,729 (88.588%) | 8,810 (9.789%) | 4 (0.004%) | 0 (0.000%) | 1,392 (1.547%) | 65 (0.072%) |
| 4 | 9,000 | 5,564 (61.822%) | 2,909 (32.322%) | 4 (0.044%) | 0 (0.000%) | 509 (5.656%) | 14 (0.156%) |
| 16 | 9,000 | 1,460 (16.222%) | 5,465 (60.722%) | 7 (0.078%) | 0 (0.000%) | 1,971 (21.900%) | 97 (1.078%) |

For single-bit faults, 89 trials increased forgotten-digit recall at all: 85 exceeded at least one utility/FPR limit, and four remained within the limits. All four useful increases corrected just one additional test image (about 2.8 percentage points), below the predefined 20-point recovery threshold. This is much smaller than the earlier targeted recovery and is not strong evidence of recovered class knowledge.

The 4- and 16-flip conditions are accumulated-error stress tests. Their trial counts cannot be translated into probabilities of natural multi-bit events or physical bursts.

## Original and retrained controls

The same logical bit positions were applied to each member of a model trio. The table below aggregates all three model types. These totals are descriptive across nine fixed fixtures, not a universal model/device probability.

| Model | Flips | Trials | Substantial selective recovery | Utility/FPR damage | Nonfinite output |
|---|---:|---:|---:|---:|---:|
| original | 1 | 90,000 | 0 | 1,427 | 62 |
| original | 4 | 9,000 | 0 | 511 | 17 |
| original | 16 | 9,000 | 0 | 1,974 | 105 |
| unlearned | 1 | 90,000 | 0 | 1,392 | 65 |
| unlearned | 4 | 9,000 | 0 | 509 | 14 |
| unlearned | 16 | 9,000 | 0 | 1,971 | 97 |
| retrained | 1 | 90,000 | 0 | 1,421 | 38 |
| retrained | 4 | 9,000 | 0 | 506 | 12 |
| retrained | 16 | 9,000 | 0 | 1,984 | 77 |

## What zero observed events permits us to say

Every fixed checkpoint had zero substantial selective-recovery events in 10,000 single-bit trials. Its individual one-sided exact 95% upper bound is therefore approximately **0.030% per injected single-bit error** under the uniform storage-bit model (about 3 in 10,000). These are per-checkpoint limits, not a simultaneous statement across all checkpoints. The corresponding bound for each 1,000-trial stress condition is about 0.299%.

The probability is conditional on an uncorrected error reaching the modeled parameter storage. It is not a probability per inference, per GPU-hour, per radiation event, or per machine lifetime. Actual error-arrival rates, affected components, nonuniform bit susceptibility, ECC, detection, repair, and model reloads were not modeled.

## Interpretation and limits

The results support the expectation that substantial selective recovery is rare in these toy fixtures under uniform, unfiltered random storage faults. Most single-bit errors left the evaluated labels unchanged; increasing the number of faults produced much more damage. Zero observed recovery events does not prove impossibility.

The models remain small classifiers trained with a simple random-incorrect-label suppression procedure. They do not establish behavior for stronger unlearning algorithms. Only the stored FP32 parameters were faulted; activation, cache, register, arithmetic, and stuck-at faults remain separate experiments. The evaluation has about 36 images per digit and cannot establish recovery on the population of handwritten digits.

## Verification and artifacts

- All 108,000 logged fault trials and 324,000 model evaluations were checked against summary counters.
- 663 independently reconstructed faulted models reproduced their logged metrics, including numerical-failure and small-gain cases.
- Four unit tests passed, covering full parameter/bias bit mapping, retention of nonfinite faults, outcome classification, and the zero-event confidence calculation.
- Bitwise audits confirmed that each campaign restored the exact clean checkpoint.
- `protocol.json` fixes thresholds, sampling, and source/reference hashes; `summary.json` contains detailed counts, field breakdowns, and examples.
- `trials.jsonl.gz` contains all raw trials. The nine NPZ files preserve the exact original/unlearned/retrained checkpoints.
- `verification.json` records validation counts.

Total campaign execution time, including fixture reconstruction: 126.5 seconds. See the repository-root `RANDOM_FAULTS.md` for the full method and Linux commands.
