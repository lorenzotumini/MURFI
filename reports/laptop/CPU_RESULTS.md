# Exploratory laptop experiment: results

Run on 2026-11-04 using existing laptop NumPy/scikit-learn. No PyTorch installation or GPU was needed.

These are real held-out toy-dataset results, not synthetic smoke outputs. They are exploratory evidence for continuing the study, not confirmation of physical fault injection or robust unlearning reversal.

Both conditions used digits 0, 1, and 2; seeds 0, 1, and 2; a 32-bit maximum search budget; and 100 random trials per run. The earlier digit-0-only exploratory file is retained separately. No hyperparameter changes were made between the two reported conditions.

## Aggregate results

| Condition | Forget accuracy before | Forget accuracy after | Retain accuracy before | Retain accuracy after | Retrained-control forget accuracy after | Selected flips | Test screening criterion passed |
|---|---:|---:|---:|---:|---:|---:|---:|
| All weight matrices | 0.9% | 94.1% | 96.7% | 96.2% | 32.3% | 4–8 | 7/9 |
| Output head excluded | 0.9% | 89.1% | 96.7% | 95.4% | 0.3% | 14–32 | 6/9 |

The screening criterion is forgotten accuracy improving by at least 20 percentage points, retained macro accuracy declining by at most 2 points, and forgotten-label false-positive rate rising by at most 1 point. It was inherited from the CIFAR pilot protocol. Validation constraints did not always generalize to test data; the failed cases above are included.

## What this suggests

- Targeted software bit changes can recover forgotten-digit behavior in this small relabeling baseline.
- Allowing output-head changes also raises recognition in some never-trained-on-that-digit controls. The all-matrices result alone is therefore difficult to attribute exclusively to residual knowledge.
- Excluding the output head preserves a substantial recovery gap: mean forgotten accuracy is 89.1% in the unlearned models versus 0.3% in the retrained controls. This makes the head-exclusion condition worth carrying forward.
- None of the 1,800 sampled random-fault models increased forgotten accuracy over its own baseline. This is conditional on the implemented idealized fault distribution and is not an estimated physical fault rate.

## Limits and next step

This tiny dataset has only about 36 final test images per digit. The nine runs share a dataset and are not nine independent datasets. The unlearning procedure is intentionally simple; the search sees forgotten labeled examples. Candidate selection has arbitrary software access to FP32 bits and no physical address constraints. Retrained and unlearned models may have different search difficulty, even with the same search procedure. These results do not establish a causal proof that stored knowledge was recovered.

Next, reproduce a published CIFAR-10 unlearning checkpoint trio under its documented preprocessing, retain the head-exclusion ablation, and repeat with stronger unlearning methods and controls. A checkpoint candidate audit accompanies these results.

## Individual held-out results

| Condition | Digit | Seed | Flips | Forget before → after | Retain change (points) | FPR change (points) | Retrained-control forget after | Test criterion |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| All matrices | 0 | 0 | 6 | 0.0% → 97.2% | +0.31 | +0.31 | 16.7% | Pass |
| All matrices | 0 | 1 | 5 | 0.0% → 94.4% | +0.00 | +0.00 | 52.8% | Pass |
| All matrices | 0 | 2 | 8 | 0.0% → 100.0% | -0.93 | +1.23 | 91.7% | Fail |
| All matrices | 1 | 0 | 5 | 0.0% → 97.2% | -1.90 | +2.16 | 36.1% | Fail |
| All matrices | 1 | 1 | 6 | 2.8% → 94.4% | +0.01 | +0.00 | 5.6% | Pass |
| All matrices | 1 | 2 | 4 | 5.6% → 86.1% | -0.93 | +0.62 | 25.0% | Pass |
| All matrices | 2 | 0 | 6 | 0.0% → 97.1% | -0.61 | +0.62 | 17.1% | Pass |
| All matrices | 2 | 1 | 5 | 0.0% → 100.0% | -0.61 | +0.92 | 22.9% | Pass |
| All matrices | 2 | 2 | 4 | 0.0% → 80.0% | +0.00 | +0.00 | 22.9% | Pass |
| No head | 0 | 0 | 15 | 0.0% → 86.1% | -2.16 | +0.00 | 0.0% | Fail |
| No head | 0 | 1 | 27 | 0.0% → 97.2% | -0.93 | +0.00 | 0.0% | Pass |
| No head | 0 | 2 | 32 | 0.0% → 94.4% | +0.92 | +0.00 | 2.8% | Pass |
| No head | 1 | 0 | 20 | 0.0% → 86.1% | -1.26 | +0.31 | 0.0% | Pass |
| No head | 1 | 1 | 14 | 2.8% → 83.3% | -1.52 | +0.31 | 0.0% | Pass |
| No head | 1 | 2 | 14 | 5.6% → 83.3% | -0.29 | +0.31 | 0.0% | Pass |
| No head | 2 | 0 | 24 | 0.0% → 91.4% | -0.35 | +0.00 | 0.0% | Pass |
| No head | 2 | 1 | 22 | 0.0% → 97.1% | -3.71 | +0.00 | 0.0% | Fail |
| No head | 2 | 2 | 19 | 0.0% → 82.9% | -2.16 | +0.00 | 0.0% | Fail |

All configurations, split indices, per-run confusion matrices, validation trajectories, and bit traces are in `digits-main.json.gz` and `digits-no-head.json.gz` (gzip-compressed JSON). Source: `cpu_digits.py`; execution details: `LAPTOP_EXPERIMENT.md`.
