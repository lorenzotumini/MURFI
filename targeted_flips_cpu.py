"""Targeted bit-flip search on a small digits classifier, using only a CPU.

Train original, unlearned and retrained models, then search for FP32 weight
flips that recover a forgotten class while preserving the other classes.
Matched random controls use the same eligibility rules as targeted search.

Requires NumPy and scikit-learn; no PyTorch or dataset download is needed.
See docs/targeted-flips-cpu.md. The separate random_flips.py campaign samples
unfiltered storage faults, including biases and nonfinite values.
"""

import os

# Set these before importing NumPy so a small matrix operation does not start
# a large BLAS thread pool. The CLI also limits already-loaded thread pools.
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits


# Model and evaluation
# --------------------
# The model is deliberately small: 64 image pixels, 64 ReLU hidden units and
# ten digit outputs. Parameters stay in FP32 throughout training and search.
# Their dictionary order also defines the layout used by random controls;
# changing it would change which parameter a sampled storage position reaches.


def initialize_model(seed):
    """Create the two weight matrices and their biases in a fixed order."""
    rng = np.random.default_rng(seed)
    return {
        'w1': (rng.normal(size=(64, 64)) * np.sqrt(2 / 64)).astype('float32'),
        'b1': np.zeros(64, 'float32'),
        'w2': (rng.normal(size=(64, 10)) * np.sqrt(2 / 64)).astype('float32'),
        'b2': np.zeros(10, 'float32'),
    }


def copy_model(model):
    """Copy every parameter array so later bit flips cannot alter the source."""
    return {name: parameter.copy() for name, parameter in model.items()}


def forward(model, x):
    """Return hidden activations and softmax probabilities for a batch."""
    hidden = np.maximum(x @ model['w1'] + model['b1'], 0)
    logits = hidden @ model['w2'] + model['b2']
    logits -= logits.max(axis=1, keepdims=True)
    exponentials = np.exp(logits)
    return hidden, exponentials / exponentials.sum(axis=1, keepdims=True)


def loss_and_gradients(model, x, y, sample_weight=None):
    """Compute cross-entropy and analytic gradients for all four parameters."""
    hidden, probabilities = forward(model, x)
    weights = (
        np.full(len(y), 1 / len(y), dtype='float32')
        if sample_weight is None
        else sample_weight
    )
    loss = float(
        -(np.log(np.maximum(probabilities[np.arange(len(y)), y], 1e-30)) * weights).sum()
    )

    # Backpropagate through softmax/cross-entropy and the hidden ReLU. During
    # training, weights average over the batch. During search, they give the
    # forgotten and retained groups equal total weight despite unequal sizes.
    logit_gradient = probabilities.copy()
    logit_gradient[np.arange(len(y)), y] -= 1
    logit_gradient *= weights[:, None]
    hidden_gradient = (logit_gradient @ model['w2'].T) * (hidden > 0)

    return loss, {
        'w1': x.T @ hidden_gradient,
        'b1': hidden_gradient.sum(0),
        'w2': hidden.T @ logit_gradient,
        'b2': logit_gradient.sum(0),
    }


def evaluate_model(model, x, y, forget_class):
    """Measure forgotten recall, retained macro accuracy and false positives."""
    _, probabilities = forward(model, x)
    if not np.isfinite(probabilities).all():
        return {'valid': False}

    predictions = probabilities.argmax(1)
    confusion = np.bincount(10 * y + predictions, minlength=100).reshape(10, 10)
    per_class = confusion.diagonal() / np.maximum(confusion.sum(1), 1)
    retained_classes = [digit for digit in range(10) if digit != forget_class]
    true_positives = int(confusion[forget_class, forget_class])
    false_positives = int(confusion[:, forget_class].sum() - true_positives)

    return {
        'valid': True,
        'forget': float(per_class[forget_class]),
        'retain': float(per_class[retained_classes].mean()),
        'fpr': false_positives / max(int(confusion[retained_classes].sum()), 1),
        'precision': true_positives / max(true_positives + false_positives, 1),
        'confusion_matrix': confusion.tolist(),
    }


# Training and the unlearning baseline
# -----------------------------------
# All three models have the same ten outputs. The original sees every class;
# the retrained control never sees training examples of the forgotten class.
# Unlearning starts from the original and assigns freshly drawn, always-wrong
# labels to forgotten examples while keeping retained labels correct.
#
# This relabeling procedure suppresses recognition; it does not certify that
# information was erased. Checkpoint selection uses only the selection split.
# The optimizer's arithmetic and random-number order are kept explicit because
# the archived experiments depend on reproducing these trained parameters.


def train_model(
    model, x, y, selection, forget_class, epochs, lr, seed,
    unlearn=False, original=None,
):
    """Train in place with Adam and return the best eligible checkpoint."""
    rng = np.random.default_rng(seed)
    first_moment = {name: np.zeros_like(value) for name, value in model.items()}
    second_moment = copy_model(first_moment)
    best_model, best_metrics, best_score, history = None, None, -np.inf, []
    update_count = 0

    for epoch in range(epochs):
        shuffled_ids = rng.permutation(len(y))
        for start in range(0, len(y), 64):
            batch_ids = shuffled_ids[start:start + 64]
            targets = y[batch_ids].copy()
            if unlearn:
                forgotten = targets == forget_class
                targets[forgotten] = (
                    targets[forgotten]
                    + rng.integers(1, 10, size=forgotten.sum())
                ) % 10

            _, gradients = loss_and_gradients(model, x[batch_ids], targets)
            update_count += 1
            for name in model:
                first_moment[name] *= 0.9
                first_moment[name] += 0.1 * gradients[name]
                second_moment[name] *= 0.999
                second_moment[name] += 0.001 * gradients[name] ** 2
                model[name] -= (
                    lr * (first_moment[name] / (1 - 0.9 ** update_count))
                    / (np.sqrt(second_moment[name] / (1 - 0.999 ** update_count)) + 1e-8)
                )

        selection_metrics = evaluate_model(model, *selection, forget_class)
        history.append(selection_metrics)
        if unlearn:
            viable = (
                selection_metrics['forget'] <= 0.10
                and selection_metrics['retain'] >= original['retain'] - 0.05
            )
            score = selection_metrics['retain'] if viable else -np.inf
        else:
            score = (
                selection_metrics['retain']
                if not np.any(y == forget_class)
                else (selection_metrics['retain'] * 9 + selection_metrics['forget']) / 10
            )

        if score > best_score:
            best_model = copy_model(model)
            best_metrics = selection_metrics
            best_score = score

    return best_model, best_metrics, history


# Exact storage mutations
# -----------------------
# A fault XORs one bit of the existing FP32 word; it is not a floating-point
# update. Toggling the same bit again restores the original storage exactly.
# The audit reconstructs all logged edits and compares integer views, so it
# also detects changes such as +0.0 versus -0.0 that float equality would miss.


def toggle_bit(model, name, index, bit):
    model[name].view(np.uint32).flat[index] ^= np.uint32(1 << bit)


def audit_bit_changes(before, after, trace):
    """Require every parameter bit to match the recorded mutation trace."""
    expected = copy_model(before)
    for flip in trace:
        toggle_bit(expected, flip['parameter'], flip['index'], flip['bit'])

    assert all(
        np.array_equal(expected[name].view('uint32'), after[name].view('uint32'))
        for name in before
    )
    assert all(parameter.dtype == np.float32 for parameter in after.values())


# Targeted search
# ---------------
# Discovery examples provide gradients and measured candidate losses. A
# separate selection split decides whether a flip preserves retained accuracy
# and false-positive rate. Neither stage sees test data.
#
# Search can edit weight matrices, but never biases. With the head excluded,
# only the input-to-hidden matrix is eligible. Sign and exponent bits remain
# eligible, so a small number of bit edits can produce large numerical changes.
# Finite-value and magnitude filters define this experiment's fault model;
# they do not describe a physical distribution of memory errors.


def search_targeted_flips(
    base, discovery, selection, forget_class, budget, exclude_head=False,
):
    """Greedily accept useful flips and return the best selection checkpoint."""
    model, best_model, trace, best_trace = copy_model(base), copy_model(base), [], []
    baseline_metrics = best_metrics = evaluate_model(base, *selection, forget_class)
    x, y = discovery
    forgotten = y == forget_class
    sample_weights = np.where(
        forgotten,
        1 / max(forgotten.sum(), 1),
        1 / max((~forgotten).sum(), 1),
    ).astype('float32')
    parameter_names = ['w1'] if exclude_head else ['w1', 'w2']
    limits = {
        name: max(float(abs(base[name]).max()) * 16, 1e-8)
        for name in parameter_names
    }
    used_bits, history = set(), []

    for step in range(budget):
        current_loss, gradients = loss_and_gradients(model, x, y, sample_weights)

        # First shortlist scalar weights by gradient magnitude times weight
        # magnitude. Then rank their valid bit flips by the estimated loss
        # change. A negative estimate means a promising descent direction;
        # we still measure its actual effect before accepting it.
        ranked_weights = []
        for name in parameter_names:
            scores = abs(gradients[name].ravel()) * np.maximum(
                abs(model[name].ravel()), 1e-8,
            )
            indices = np.argsort(scores)[-64:]
            ranked_weights.extend(
                (float(scores[index]), name, int(index)) for index in indices
            )
        ranked_weights.sort(reverse=True)

        candidates = []
        for _, name, index in ranked_weights[:64]:
            old_value = float(model[name].flat[index])
            for bit in range(32):
                if (name, index, bit) in used_bits:
                    continue
                toggle_bit(model, name, index, bit)
                new_value = float(model[name].flat[index])
                toggle_bit(model, name, index, bit)
                if np.isfinite(new_value) and abs(new_value) <= limits[name]:
                    estimate = float(gradients[name].flat[index]) * (new_value - old_value)
                    if estimate < 0:
                        candidates.append((estimate, name, index, bit))
        candidates.sort()

        # Every trial mutation is undone before another candidate is measured.
        # Gradients only rank flips: search never takes an optimizer step.
        measured = []
        for _, name, index, bit in candidates[:16]:
            toggle_bit(model, name, index, bit)
            loss, _ = loss_and_gradients(model, x, y, sample_weights)
            toggle_bit(model, name, index, bit)
            if np.isfinite(loss) and loss < current_loss:
                measured.append((loss, name, index, bit))
        measured.sort()

        accepted = False
        for loss, name, index, bit in measured:
            old_value = float(model[name].flat[index])
            toggle_bit(model, name, index, bit)
            selection_metrics = evaluate_model(model, *selection, forget_class)

            # Compare with the untouched checkpoint, not the previous step:
            # otherwise many individually small losses could accumulate.
            # Strict improvement when saving the best checkpoint also means
            # that a tie in forgotten recall keeps the earlier, smaller trace.
            if (
                selection_metrics['valid']
                and selection_metrics['retain'] >= baseline_metrics['retain'] - 0.02
                and selection_metrics['fpr'] <= baseline_metrics['fpr'] + 0.01
            ):
                trace.append({
                    'parameter': name,
                    'index': index,
                    'bit': bit,
                    'before': old_value,
                    'after': float(model[name].flat[index]),
                })
                used_bits.add((name, index, bit))
                history.append({
                    'bits': len(trace),
                    'selection': selection_metrics,
                    'discovery_loss': loss,
                })
                if selection_metrics['forget'] > best_metrics['forget']:
                    best_model = copy_model(model)
                    best_trace = list(trace)
                    best_metrics = selection_metrics
                accepted = True
                break
            toggle_bit(model, name, index, bit)

        if not accepted:
            break

    audit_bit_changes(base, model, trace)
    audit_bit_changes(base, best_model, best_trace)
    return best_model, best_trace, history


def sample_random_control(base, count, seed, exclude_head=False):
    """Sample filtered random flips matched to the targeted experiment."""
    # This control shares targeted search's matrix eligibility, finite-value
    # filter and magnitude cap. It is intentionally different from the
    # unfiltered random_flips.py campaign. Its bit count comes from the selected
    # targeted model; each trial starts from a fresh copy of the base model.
    model, trace, used_bits = copy_model(base), [], set()
    parameter_names = ['w1'] if exclude_head else ['w1', 'w2']
    sizes = [model[name].size for name in parameter_names]
    rng = np.random.default_rng(seed)
    limits = {
        name: max(float(abs(base[name]).max()) * 16, 1e-8)
        for name in parameter_names
    }

    while len(trace) < count:
        index = int(rng.integers(sum(sizes)))
        for name, size in zip(parameter_names, sizes):
            if index < size:
                break
            index -= size
        bit = int(rng.integers(32))
        if (name, index, bit) in used_bits:
            continue

        old_value = float(model[name].flat[index])
        toggle_bit(model, name, index, bit)
        new_value = float(model[name].flat[index])
        if np.isfinite(new_value) and abs(new_value) <= limits[name]:
            used_bits.add((name, index, bit))
            trace.append({
                'parameter': name,
                'index': index,
                'bit': bit,
                'before': old_value,
                'after': new_value,
            })
        else:
            toggle_bit(model, name, index, bit)

    audit_bit_changes(base, model, trace)
    return model, trace


# Experiment orchestration
# ------------------------
# Split the dataset before fitting anything: approximately 60% training,
# 10% discovery, 10% selection and 20% test. The original and retrained models
# start from the same initialization seed, but see different training examples.
# Four unlearning learning rates are compared using selection metrics alone.
# Each eligible model then gets its own targeted search under the same ceiling.


def run_experiment(seed, forget_class, budget, random_trials, exclude_head):
    started = time.monotonic()
    x, y = load_digits(return_X_y=True)
    x = (x / 16).astype('float32')
    indices = np.arange(len(y))
    rest, test = train_test_split(
        indices, test_size=0.20, stratify=y, random_state=seed,
    )
    training, validation = train_test_split(
        rest, test_size=0.25, stratify=y[rest], random_state=seed + 100,
    )
    discovery, selection = train_test_split(
        validation, test_size=0.5, stratify=y[validation], random_state=seed + 200,
    )
    splits = [set(split.tolist()) for split in [training, discovery, selection, test]]
    assert sum(map(len, splits)) == len(set.union(*splits)) == len(y)

    selection_data = x[selection], y[selection]
    original, original_metrics, _ = train_model(
        initialize_model(seed), x[training], y[training], selection_data,
        forget_class, 100, 0.003, seed,
    )
    retained = training[y[training] != forget_class]
    retrained, _, _ = train_model(
        initialize_model(seed), x[retained], y[retained], selection_data,
        forget_class, 100, 0.003, seed,
    )

    candidates, unlearning_grid = [], []
    for lr in [0.0001, 0.0003, 0.001, 0.003]:
        unlearned, selection_metrics, history = train_model(
            copy_model(original), x[training], y[training], selection_data,
            forget_class, 30, lr, seed + 400, True, original_metrics,
        )
        unlearning_grid.append({
            'lr': lr,
            'history': history,
            'selected': selection_metrics,
        })
        if unlearned is not None:
            candidates.append((selection_metrics['retain'], lr, unlearned, selection_metrics))

    result = {
        'seed': seed,
        'forget_class': forget_class,
        'unlearning_grid': unlearning_grid,
        'splits': {
            name: split.tolist()
            for name, split in [
                ('training', training),
                ('discovery', discovery),
                ('selection', selection),
                ('test', test),
            ]
        },
    }
    if min(original_metrics['forget'], original_metrics['retain']) < 0.70 or not candidates:
        result['status'] = 'baseline_gate_failed'
        result['original_selection'] = original_metrics
        return result

    _, lr, unlearned, _ = max(candidates, key=lambda candidate: candidate[0])
    result.update(status='completed', unlearn_lr=lr, targeted={}, baseline={}, random=[])

    # Test data are never passed to training, search or checkpoint selection.
    # Recorded baselines are descriptive; search uses its own selection metrics.
    for name, model in [
        ('original', original), ('unlearned', unlearned), ('retrained', retrained),
    ]:
        result['baseline'][name] = evaluate_model(model, x[test], y[test], forget_class)

    for name, model in [('unlearned', unlearned), ('retrained', retrained)]:
        changed, trace, history = search_targeted_flips(
            model, (x[discovery], y[discovery]), selection_data,
            forget_class, budget, exclude_head,
        )
        result['targeted'][name] = {
            'bits': len(trace),
            'trace': trace,
            'history': history,
            'test': evaluate_model(changed, x[test], y[test], forget_class),
        }

    count = result['targeted']['unlearned']['bits']
    for trial in range(random_trials):
        changed, trace = sample_random_control(
            unlearned, count, seed * 10000 + trial + 5000, exclude_head,
        )
        result['random'].append({
            'trial': trial,
            'bits': count,
            'trace': trace,
            'test': evaluate_model(changed, x[test], y[test], forget_class),
        })

    result['elapsed_seconds'] = time.monotonic() - started
    print(
        f'seed={seed} class={forget_class} {result["elapsed_seconds"]:.1f}s: '
        f'forget {result["baseline"]["unlearned"]["forget"]:.3f} -> '
        f'{result["targeted"]["unlearned"]["test"]["forget"]:.3f}, bits={count}',
        flush=True,
    )
    return result


def check_gradients():
    """Check selected analytic derivatives against central finite differences."""
    model = initialize_model(2)
    rng = np.random.default_rng(10)
    x, y = rng.random((20, 64), dtype=np.float32), np.arange(20) % 10
    _, gradients = loss_and_gradients(model, x, y)
    for name, index in [('w1', (4, 3)), ('w2', (5, 2)), ('b1', (7,)), ('b2', (2,))]:
        old_value = model[name][index].copy()
        model[name][index] = old_value + 0.001
        high_loss, _ = loss_and_gradients(model, x, y)
        model[name][index] = old_value - 0.001
        low_loss, _ = loss_and_gradients(model, x, y)
        model[name][index] = old_value
        assert abs((high_loss - low_loss) / 0.002 - float(gradients[name][index])) < 0.002


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--seeds', type=int, nargs='+', default=[0, 1, 2],
        help='Initialization and data-split seeds (default: 0 1 2)',
    )
    parser.add_argument(
        '--classes', type=int, nargs='+', default=[0],
        help='Digits to forget, evaluated separately (default: 0)',
    )
    parser.add_argument(
        '--budget', type=int, default=32,
        help='Maximum targeted flips per model (default: 32)',
    )
    parser.add_argument(
        '--random-trials', type=int, default=30,
        help='Matched random-control trials per configuration (default: 30)',
    )
    parser.add_argument(
        '--exclude-head', action='store_true',
        help='Allow flips only in the input-to-hidden weight matrix',
    )
    parser.add_argument(
        '--output', type=Path, default=Path('runs/targeted-flips-cpu.json'),
        help='New JSON output path (default: runs/targeted-flips-cpu.json)',
    )
    args = parser.parse_args()

    if args.output.exists():
        parser.error('Output exists; choose a new output path.')
    if (
        args.budget < 1
        or args.random_trials < 1
        or any(digit not in range(10) for digit in args.classes)
    ):
        parser.error('Positive budget/trials and digit classes 0-9 required.')
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with threadpool_limits(limits=1):
        check_gradients()
        result = {
            'config': vars(args) | {'output': str(args.output)},
            'environment': {
                'numpy': np.__version__,
                'sklearn': sklearn.__version__,
                'python': platform.python_version(),
            },
            'gradient_check': 'passed',
            'runs': [],
        }
        for forget_class in args.classes:
            for seed in args.seeds:
                result['runs'].append(
                    run_experiment(
                        seed, forget_class, args.budget,
                        args.random_trials, args.exclude_head,
                    )
                )
                # Save after every completed configuration so earlier results
                # survive an interruption. A new invocation requires a new path.
                args.output.write_text(
                    json.dumps(result, indent=2, allow_nan=False), encoding='utf-8',
                )


if __name__ == '__main__':
    main()
