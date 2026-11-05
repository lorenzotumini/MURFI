"""Uniform, unfiltered FP32 storage fault campaigns on the CPU digits fixtures.

See docs/random-flips.md. Fault probabilities here are conditional on an injected
error reaching parameter storage, not rates of naturally occurring GPU errors.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import time
import numpy as np
import scipy
from scipy.stats import beta
from threadpoolctl import threadpool_limits
import targeted_flips_cpu as digits


MODELS = ('original', 'unlearned', 'retrained')
CATEGORIES = ('unchanged_predictions', 'other_prediction_changes', 'small_selective_recovery',
              'selective_recovery', 'utility_or_false_positive_damage', 'nonfinite_output')
THRESHOLDS = {'forget_gain': 0.20, 'retain_loss': 0.02, 'fpr_rise': 0.01}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def upper95(k, n):
    if k == 0:
        return -math.expm1(math.log(0.05)/n)
    if k == n:
        return 1.0
    return float(beta.ppf(0.95, k+1, n-k))


def layout(model):
    keys = list(model)
    ends = np.cumsum([model[k].size*32 for k in keys])
    assert all(v.dtype == np.float32 and v.flags.c_contiguous for v in model.values())
    return keys, ends


def decode(bit_id, keys, ends):
    which = int(np.searchsorted(ends, bit_id, side='right'))
    offset = int(bit_id - (ends[which-1] if which else 0))
    return keys[which], offset//32, offset % 32


def evaluate(m, x, y, c):
    _, prob = digits.forward(m, x)
    if not np.isfinite(prob).all():
        return {'valid': False}, None
    pred = prob.argmax(1)
    cm = np.bincount(10*y+pred, minlength=100).reshape(10, 10)
    per = cm.diagonal()/np.maximum(cm.sum(1), 1)
    other = [j for j in range(10) if j != c]
    tp = int(cm[c, c]); fp = int(cm[:, c].sum()-tp)
    return {'valid': True, 'forget': float(per[c]), 'retain': float(per[other].mean()),
            'fpr': fp/max(int(cm[other].sum()), 1), 'precision': tp/max(tp+fp, 1)}, pred


def classify(met, baseline, pred, base_pred):
    if not met['valid']:
        return 'nonfinite_output'
    usable = (met['retain'] >= baseline['retain']-THRESHOLDS['retain_loss']-1e-12
              and met['fpr'] <= baseline['fpr']+THRESHOLDS['fpr_rise']+1e-12)
    gain = met['forget']-baseline['forget']
    if not usable:
        return 'utility_or_false_positive_damage'
    if gain >= THRESHOLDS['forget_gain']-1e-12:
        return 'selective_recovery'
    if gain > 1e-12:
        return 'small_selective_recovery'
    if np.array_equal(pred, base_pred):
        return 'unchanged_predictions'
    return 'other_prediction_changes'


def fixtures(record):
    """Reconstruct the saved trained fixtures and check their test predictions."""
    x, y = digits.load_digits(return_X_y=True)
    x = (x/16).astype('float32')
    c, seed = record['forget_class'], record['seed']
    ids = {k: np.array(v) for k, v in record['splits'].items()}
    training, sel, test = ids['training'], ids['selection'], ids['test']
    selection = (x[sel], y[sel])
    original, original_met, _ = digits.train_model(digits.initialize_model(seed), x[training], y[training],
                                           selection, c, 100, 0.003, seed)
    keep = training[y[training] != c]
    retrained, _, _ = digits.train_model(digits.initialize_model(seed), x[keep], y[keep],
                                 selection, c, 100, 0.003, seed)
    unlearned, _, _ = digits.train_model(digits.copy_model(original), x[training], y[training],
                                 selection, c, 30, record['unlearn_lr'], seed+400,
                                 True, original_met)
    models = {'original': original, 'unlearned': unlearned, 'retrained': retrained}
    for name, m in models.items():
        if m is None or digits.evaluate_model(m, x[test], y[test], c) != record['baseline'][name]:
            raise RuntimeError(f'{name} fixture does not reproduce saved baseline. '
                               'Check source and dependency versions; do not silently mix fixtures.')
    return models, x[test], y[test]


def new_stats():
    return {'outcomes': dict.fromkeys(CATEGORIES, 0), 'any_forget_gain': 0,
            'large_forget_gain_regardless_of_damage': 0,
            'finite_valid_trials': 0, 'forget_sum': 0.0, 'retain_sum': 0.0,
            'max_forget': 0.0, 'single_bit_fields': {}, 'examples': {}}


def observe(stats, met, baseline, category, bit_ids, field):
    stats['outcomes'][category] += 1
    if field is not None:
        fields = stats['single_bit_fields'].setdefault(field, dict.fromkeys(CATEGORIES, 0))
        fields[category] += 1
    if met['valid']:
        stats['finite_valid_trials'] += 1
        stats['forget_sum'] += met['forget']
        stats['retain_sum'] += met['retain']
        stats['max_forget'] = max(stats['max_forget'], met['forget'])
        stats['any_forget_gain'] += int(met['forget'] > baseline['forget']+1e-12)
        stats['large_forget_gain_regardless_of_damage'] += int(
            met['forget']-baseline['forget'] >= THRESHOLDS['forget_gain']-1e-12)
    if category != 'unchanged_predictions':
        examples = stats['examples'].setdefault(category, [])
        if len(examples) < 3:
            examples.append({'bits': [int(b) for b in bit_ids], 'metrics': met})


def campaign(models, x, y, c, seed, budget, n, log, keys, ends):
    baseline = {name: evaluate(m, x, y, c) for name, m in models.items()}
    stats = {name: new_stats() for name in MODELS}
    # Independent random trials, with the same bit locations paired across the three models.
    rng = np.random.default_rng(9012345 + c*65537 + seed*257 + budget*100003)
    before = {name: digits.copy_model(m) for name, m in models.items()}
    for trial in range(n):
        bits = (rng.integers(0, ends[-1], size=1) if budget == 1 else
                rng.choice(int(ends[-1]), size=budget, replace=False))
        decoded = [decode(int(b), keys, ends) for b in bits]
        field = None
        if budget == 1:
            bit = decoded[0][2]
            field = 'sign' if bit == 31 else 'exponent' if bit >= 23 else 'mantissa'
        row = {'digit': c, 'seed': seed, 'budget': budget, 'trial': trial,
               'bits': bits.tolist(), 'models': {}}
        for name, m in models.items():
            try:
                for k, i, bit in decoded:
                    digits.toggle_bit(m, k, i, bit)
                met, pred = evaluate(m, x, y, c)
            finally:
                for k, i, bit in decoded:
                    digits.toggle_bit(m, k, i, bit)
            category = classify(met, baseline[name][0], pred, baseline[name][1])
            observe(stats[name], met, baseline[name][0], category, bits, field)
            row['models'][name] = met | {'outcome': category}
        log.write(json.dumps(row, separators=(',', ':'), allow_nan=False)+'\n')
    for name, m in models.items():
        digits.audit_bit_changes(before[name], m, [])
        s = stats[name]
        assert sum(s['outcomes'].values()) == n
        s['selective_recovery_rate'] = s['outcomes']['selective_recovery']/n
        s['selective_recovery_upper95'] = upper95(s['outcomes']['selective_recovery'], n)
        valid = s['finite_valid_trials']
        s['mean_forget_if_finite'] = s.pop('forget_sum')/valid if valid else None
        s['mean_retain_if_finite'] = s.pop('retain_sum')/valid if valid else None
    return {'digit': c, 'seed': seed, 'budget': budget, 'trials_per_model': n,
            'baseline': {name: pair[0] for name, pair in baseline.items()}, 'models': stats}


def self_check():
    m = digits.initialize_model(19)
    keys, ends = layout(m)
    assert ends[-1] == 4810*32
    for b in [0, 31, 32, 4096*32, int(ends[-1])-1]:
        k, i, bit = decode(b, keys, ends)
        before = digits.copy_model(m)
        digits.toggle_bit(m, k, i, bit)
        digits.audit_bit_changes(before, m, [{'parameter': k, 'index': i, 'bit': bit}])
        digits.toggle_bit(m, k, i, bit)
        digits.audit_bit_changes(before, m, [])
    base = {'valid': True, 'forget': 0.0, 'retain': 0.95, 'fpr': 0.0}
    pred = np.array([0,1])
    assert classify(base, base, pred, pred) == 'unchanged_predictions'
    assert classify(base | {'forget': 0.2}, base, pred, pred) == 'selective_recovery'
    assert classify(base | {'forget': 0.2, 'retain': 0.90}, base, pred, pred) == 'utility_or_false_positive_damage'
    assert classify({'valid': False}, base, None, pred) == 'nonfinite_output'
    assert abs(upper95(0,10000)-0.00029952835977661195) < 1e-12


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, default=Path('reports/targeted-flips/cpu/digits-main.json.gz'))
    p.add_argument('--output-dir', type=Path, default=Path('runs/random-flips'))
    p.add_argument('--seeds', type=int, nargs='+', default=[0,1,2])
    p.add_argument('--classes', type=int, nargs='+', default=[0,1,2])
    p.add_argument('--single-trials', type=int, default=10000)
    p.add_argument('--stress-trials', type=int, default=1000)
    a = p.parse_args()
    if min(a.single_trials,a.stress_trials) < 1:
        p.error('Trial counts must be positive.')
    if a.output_dir.exists():
        p.error('Output directory already exists; choose a fresh directory.')
    reference = json.loads(gzip.decompress(a.reference.read_bytes()))
    records = [r for r in reference['runs'] if r['seed'] in a.seeds and r['forget_class'] in a.classes]
    if len(records) != len(set(a.seeds))*len(set(a.classes)):
        p.error('Requested class/seed fixtures missing from reference.')
    self_check()
    a.output_dir.mkdir(parents=True)
    # "pilot_sha256" identifies the shared CPU model/training implementation.
    # Keep this protocol key consistent with the archived campaign records.
    protocol = {'config': {k: str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
                'thresholds': THRESHOLDS, 'reference_sha256': sha(a.reference),
                'pilot_sha256': sha(digits.__file__), 'campaign_sha256': sha(__file__),
                'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                                'scipy': scipy.__version__, 'sklearn': digits.sklearn.__version__},
                'fault_model': 'uniform FP32 parameter-storage bit flips; biases included; no filtering; '
                               'persistent throughout test batch; independent reset between trials; '
                               'multi-flip budgets are stress tests, not physical error distributions',
                'confidence': 'per fixed model and budget, one-sided exact 95% binomial upper limit; '
                              'conditional on this sampling model, not on hours of hardware use'}
    (a.output_dir/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    result = {'protocol': protocol, 'campaigns': []}
    started = time.monotonic()
    with threadpool_limits(limits=1), np.errstate(over='ignore',invalid='ignore',under='ignore',divide='ignore'):
        with gzip.open(a.output_dir/'trials.jsonl.gz','wt',encoding='utf-8') as log:
            for r in records:
                c, seed = r['forget_class'], r['seed']
                models, x, y = fixtures(r)
                keys, ends = layout(models['unlearned'])
                np.savez_compressed(a.output_dir/f'digit{c}-seed{seed}.npz',
                                    **{name+'_'+k: v for name,m in models.items() for k,v in m.items()})
                for budget, n in [(1,a.single_trials),(4,a.stress_trials),(16,a.stress_trials)]:
                    st = time.monotonic()
                    report = campaign(models,x,y,c,seed,budget,n,log,keys,ends)
                    report['elapsed_seconds'] = time.monotonic()-st
                    result['campaigns'].append(report)
                    (a.output_dir/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
                    counts={name:report['models'][name]['outcomes']['selective_recovery'] for name in MODELS}
                    print(f'digit={c} seed={seed} flips={budget} trials/model={n} '
                          f'selective={counts} {report["elapsed_seconds"]:.1f}s',flush=True)
    result['elapsed_seconds']=time.monotonic()-started
    (a.output_dir/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(f'Finished in {result["elapsed_seconds"]:.1f}s. Saved {a.output_dir}',flush=True)


if __name__ == '__main__':
    main()
