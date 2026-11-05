"""CPU-only exploratory class-unlearning experiment; no PyTorch or downloads.

Requires NumPy and scikit-learn. This is a toy study, not the CIFAR-10 pilot.
"""
import os
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


def initial(seed):
    r = np.random.default_rng(seed)
    return {'w1': (r.normal(size=(64, 64)) * np.sqrt(2/64)).astype('float32'),
            'b1': np.zeros(64, 'float32'),
            'w2': (r.normal(size=(64, 10)) * np.sqrt(2/64)).astype('float32'),
            'b2': np.zeros(10, 'float32')}


def clone(m):
    return {k: v.copy() for k, v in m.items()}


def forward(m, x):
    h = np.maximum(x @ m['w1'] + m['b1'], 0)
    z = h @ m['w2'] + m['b2']
    z -= z.max(axis=1, keepdims=True)
    ex = np.exp(z)
    return h, ex / ex.sum(axis=1, keepdims=True)


def loss_grad(m, x, y, sample_weight=None):
    h, prob = forward(m, x)
    sw = np.full(len(y), 1/len(y), dtype='float32') if sample_weight is None else sample_weight
    loss = float(-(np.log(np.maximum(prob[np.arange(len(y)), y], 1e-30)) * sw).sum())
    dz = prob.copy()
    dz[np.arange(len(y)), y] -= 1
    dz *= sw[:, None]
    dh = (dz @ m['w2'].T) * (h > 0)
    return loss, {'w1': x.T @ dh, 'b1': dh.sum(0), 'w2': h.T @ dz, 'b2': dz.sum(0)}


def metrics(m, x, y, c):
    _, prob = forward(m, x)
    if not np.isfinite(prob).all():
        return {'valid': False}
    pred = prob.argmax(1)
    cm = np.bincount(10*y + pred, minlength=100).reshape(10, 10)
    per = cm.diagonal()/np.maximum(cm.sum(1), 1)
    other = [j for j in range(10) if j != c]
    tp = int(cm[c, c]); fp = int(cm[:, c].sum() - tp)
    return {'valid': True, 'forget': float(per[c]), 'retain': float(per[other].mean()),
            'fpr': fp/max(int(cm[other].sum()), 1), 'precision': tp/max(tp+fp, 1),
            'confusion_matrix': cm.tolist()}


def train(m, x, y, selection, c, epochs, lr, seed, unlearn=False, original=None):
    rng = np.random.default_rng(seed)
    mom = {k: np.zeros_like(v) for k, v in m.items()}
    var = clone(mom)
    best, best_met, best_score, history = None, None, -np.inf, []
    t = 0
    for epoch in range(epochs):
        ids = rng.permutation(len(y))
        for start in range(0, len(y), 64):
            ix = ids[start:start+64]
            targets = y[ix].copy()
            if unlearn:
                forget = targets == c
                targets[forget] = (targets[forget] + rng.integers(1, 10, size=forget.sum())) % 10
            _, grad = loss_grad(m, x[ix], targets)
            t += 1
            for k in m:
                mom[k] *= 0.9
                mom[k] += 0.1 * grad[k]
                var[k] *= 0.999
                var[k] += 0.001 * grad[k]**2
                m[k] -= lr * (mom[k]/(1-0.9**t)) / (np.sqrt(var[k]/(1-0.999**t))+1e-8)
        met = metrics(m, *selection, c)
        history.append(met)
        if unlearn:
            viable = met['forget'] <= 0.10 and met['retain'] >= original['retain']-0.05
            score = met['retain'] if viable else -np.inf
        else:
            score = met['retain'] if not np.any(y == c) else (met['retain']*9+met['forget'])/10
        if score > best_score:
            best, best_met, best_score = clone(m), met, score
    return best, best_met, history


def toggle(m, name, index, bit):
    m[name].view(np.uint32).flat[index] ^= np.uint32(1 << bit)


def audit(before, after, trace):
    expected = clone(before)
    for t in trace:
        toggle(expected, t['parameter'], t['index'], t['bit'])
    assert all(np.array_equal(expected[k].view('uint32'), after[k].view('uint32')) for k in before)
    assert all(v.dtype == np.float32 for v in after.values())


def search(base, discovery, selection, c, budget, exclude_head=False):
    m, best, trace, best_trace = clone(base), clone(base), [], []
    base_met = best_met = metrics(base, *selection, c)
    x, y = discovery
    mask = y == c
    sw = np.where(mask, 1/max(mask.sum(), 1), 1/max((~mask).sum(), 1)).astype('float32')
    keys = ['w1'] if exclude_head else ['w1', 'w2']
    limits = {k: max(float(abs(base[k]).max())*16, 1e-8) for k in keys}
    used, history = set(), []
    for step in range(budget):
        current, grad = loss_grad(m, x, y, sw)
        weights = []
        for k in keys:
            score = abs(grad[k].ravel()) * np.maximum(abs(m[k].ravel()), 1e-8)
            ix = np.argsort(score)[-64:]
            weights.extend((float(score[i]), k, int(i)) for i in ix)
        weights.sort(reverse=True)
        candidates = []
        for _, k, i in weights[:64]:
            old = float(m[k].flat[i])
            for bit in range(32):
                if (k, i, bit) in used:
                    continue
                toggle(m, k, i, bit)
                new = float(m[k].flat[i])
                toggle(m, k, i, bit)
                if np.isfinite(new) and abs(new) <= limits[k]:
                    estimate = float(grad[k].flat[i]) * (new-old)
                    if estimate < 0:
                        candidates.append((estimate, k, i, bit))
        candidates.sort()
        measured = []
        for _, k, i, bit in candidates[:16]:
            toggle(m, k, i, bit)
            value, _ = loss_grad(m, x, y, sw)
            toggle(m, k, i, bit)
            if np.isfinite(value) and value < current:
                measured.append((value, k, i, bit))
        measured.sort()
        accepted = False
        for value, k, i, bit in measured:
            old = float(m[k].flat[i])
            toggle(m, k, i, bit)
            met = metrics(m, *selection, c)
            if met['valid'] and met['retain'] >= base_met['retain']-0.02 and met['fpr'] <= base_met['fpr']+0.01:
                trace.append({'parameter': k, 'index': i, 'bit': bit,
                              'before': old, 'after': float(m[k].flat[i])})
                used.add((k, i, bit))
                history.append({'bits': len(trace), 'selection': met, 'discovery_loss': value})
                if met['forget'] > best_met['forget']:
                    best, best_trace, best_met = clone(m), list(trace), met
                accepted = True
                break
            toggle(m, k, i, bit)
        if not accepted:
            break
    audit(base, m, trace)
    audit(base, best, best_trace)
    return best, best_trace, history


def random_fault(base, count, seed, exclude_head=False):
    m, trace, used = clone(base), [], set()
    keys = ['w1'] if exclude_head else ['w1', 'w2']
    sizes = [m[k].size for k in keys]
    rng = np.random.default_rng(seed)
    limits = {k: max(float(abs(base[k]).max())*16, 1e-8) for k in keys}
    while len(trace) < count:
        index = int(rng.integers(sum(sizes)))
        for k, size in zip(keys, sizes):
            if index < size:
                break
            index -= size
        bit = int(rng.integers(32))
        if (k, index, bit) in used:
            continue
        old = float(m[k].flat[index])
        toggle(m, k, index, bit)
        new = float(m[k].flat[index])
        if np.isfinite(new) and abs(new) <= limits[k]:
            used.add((k, index, bit))
            trace.append({'parameter': k, 'index': index, 'bit': bit, 'before': old, 'after': new})
        else:
            toggle(m, k, index, bit)
    audit(base, m, trace)
    return m, trace


def run(seed, c, budget, random_trials, exclude_head):
    started = time.monotonic()
    x, y = load_digits(return_X_y=True)
    x = (x/16).astype('float32')
    ids = np.arange(len(y))
    rest, test = train_test_split(ids, test_size=0.20, stratify=y, random_state=seed)
    training, validation = train_test_split(rest, test_size=0.25, stratify=y[rest], random_state=seed+100)
    discovery, selection = train_test_split(validation, test_size=0.5, stratify=y[validation], random_state=seed+200)
    splits = [set(s.tolist()) for s in [training, discovery, selection, test]]
    assert sum(map(len, splits)) == len(set.union(*splits)) == len(y)
    sel = x[selection], y[selection]
    original, original_met, _ = train(initial(seed), x[training], y[training], sel, c, 100, 0.003, seed)
    keep = training[y[training] != c]
    retrained, retrained_met, _ = train(initial(seed), x[keep], y[keep], sel, c, 100, 0.003, seed)
    candidates, grid = [], []
    for lr in [0.0001, 0.0003, 0.001, 0.003]:
        unlearned, met, history = train(clone(original), x[training], y[training], sel, c,
                                        30, lr, seed+400, True, original_met)
        grid.append({'lr': lr, 'history': history, 'selected': met})
        if unlearned is not None:
            candidates.append((met['retain'], lr, unlearned, met))
    result = {'seed': seed, 'forget_class': c, 'unlearning_grid': grid,
              'splits': {k: v.tolist() for k, v in [('training', training), ('discovery', discovery), ('selection', selection), ('test', test)]}}
    if min(original_met['forget'], original_met['retain']) < 0.70 or not candidates:
        result['status'] = 'baseline_gate_failed'
        result['original_selection'] = original_met
        return result
    _, lr, unlearned, unlearned_met = max(candidates, key=lambda t: t[0])
    result.update(status='completed', unlearn_lr=lr, targeted={}, baseline={}, random=[])
    # Test data are not passed to training, search, or checkpoint selection.
    for name, m in [('original', original), ('unlearned', unlearned), ('retrained', retrained)]:
        result['baseline'][name] = metrics(m, x[test], y[test], c)
    for name, m in [('unlearned', unlearned), ('retrained', retrained)]:
        changed, trace, history = search(m, (x[discovery], y[discovery]), sel, c, budget, exclude_head)
        result['targeted'][name] = {'bits': len(trace), 'trace': trace, 'history': history,
                                    'test': metrics(changed, x[test], y[test], c)}
    count = result['targeted']['unlearned']['bits']
    for i in range(random_trials):
        changed, trace = random_fault(unlearned, count, seed*10000+i+5000, exclude_head)
        result['random'].append({'trial': i, 'bits': count, 'trace': trace,
                                  'test': metrics(changed, x[test], y[test], c)})
    result['elapsed_seconds'] = time.monotonic()-started
    print(f'seed={seed} class={c} {result["elapsed_seconds"]:.1f}s: '
          f'forget {result["baseline"]["unlearned"]["forget"]:.3f} -> '
          f'{result["targeted"]["unlearned"]["test"]["forget"]:.3f}, bits={count}', flush=True)
    return result


def gradient_check():
    m = initial(2)
    rng = np.random.default_rng(10)
    x, y = rng.random((20,64), dtype=np.float32), np.arange(20)%10
    _, grad = loss_grad(m, x, y)
    for k, index in [('w1', (4, 3)), ('w2', (5, 2)), ('b1', (7,)), ('b2', (2,))]:
        old = m[k][index].copy()
        m[k][index] = old+0.001
        hi, _ = loss_grad(m, x, y)
        m[k][index] = old-0.001
        lo, _ = loss_grad(m, x, y)
        m[k][index] = old
        assert abs((hi-lo)/0.002-float(grad[k][index])) < 0.002


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds', type=int, nargs='+', default=[0,1,2])
    p.add_argument('--classes', type=int, nargs='+', default=[0])
    p.add_argument('--budget', type=int, default=32)
    p.add_argument('--random-trials', type=int, default=30)
    p.add_argument('--exclude-head', action='store_true')
    p.add_argument('--output', type=Path, default=Path('runs/cpu-digits.json'))
    a = p.parse_args()
    if a.output.exists():
        p.error('Output exists; choose a new output path.')
    if a.budget < 1 or a.random_trials < 1 or any(c not in range(10) for c in a.classes):
        p.error('Positive budget/trials and digit classes 0-9 required.')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        gradient_check()
        result = {'config': vars(a) | {'output': str(a.output)},
                  'environment': {'numpy': np.__version__, 'sklearn': sklearn.__version__, 'python': platform.python_version()},
                  'gradient_check': 'passed', 'runs': []}
        for c in a.classes:
            for seed in a.seeds:
                result['runs'].append(run(seed, c, a.budget, a.random_trials, a.exclude_head))
                a.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
