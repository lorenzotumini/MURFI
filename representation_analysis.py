"""Portable representation diagnostics. See docs/representation-analysis.md."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import csv
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import platform
import re
import sys

import numpy as np


BASE_MODELS = ['original', 'unlearned', 'retrained']
PARTITIONS = ('train', 'heldout', 'calibration')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def feature_key(model, layer):
    return f'features__{model}__{layer}'


def validate_bundle(meta, arrays):
    if meta.get('schema_version') != 1:
        raise ValueError('Unsupported feature bundle schema.')
    models, layers = meta['models'], meta['layers']
    if not set(BASE_MODELS).issubset(models) or not layers:
        raise ValueError('Need original, unlearned, retrained and at least one layer.')
    for names in (models, layers):
        if len(names) != len(set(names)) or any(not re.fullmatch(r'[a-zA-Z0-9_]+', n) for n in names):
            raise ValueError('Model/layer names must be unique alphanumeric identifiers.')
    ids, labels, partitions = (arrays[n] for n in ('sample_ids', 'labels', 'partitions'))
    n = len(ids)
    if n < 4 or ids.ndim != 1 or len(set(ids.tolist())) != n:
        raise ValueError('Need at least four unique, ordered sample IDs.')
    if ids.dtype.kind not in 'US' or labels.dtype.kind not in 'iu':
        raise ValueError('Sample IDs must be strings and labels integer class indices.')
    if labels.shape != (n,) or partitions.shape != (n,) or not set(partitions).issubset(PARTITIONS):
        raise ValueError('Labels/partitions must match the sample axis and known partitions.')
    if np.any(labels < 0) or not 0 <= meta['forget_class'] < len(meta['class_names']):
        raise ValueError('Invalid class index.')
    if np.any(labels >= len(meta['class_names'])):
        raise ValueError('Label outside class_names.')
    if np.any((partitions == 'calibration') & (labels == meta['forget_class'])):
        raise ValueError('Alignment calibration must contain retained examples only.')
    for model in models:
        for layer in layers:
            a = arrays[feature_key(model, layer)]
            if a.ndim != 2 or a.shape[0] != n or a.shape[1] < 1 or not np.isfinite(a).all():
                raise ValueError(f'Invalid/nonfinite activations: {model}/{layer}')
        p = arrays[f'probabilities__{model}']
        if (p.shape != (n, len(meta['class_names'])) or not np.isfinite(p).all()
                or np.any(p < 0) or not np.allclose(p.sum(1), 1, atol=1e-5)):
            raise ValueError(f'Invalid class probabilities: {model}')


def save_bundle(path, meta, arrays):
    validate_bundle(meta, arrays)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental replacement of a frozen extraction.
    with path.open('xb') as stream:
        np.savez_compressed(stream, metadata=np.array(json.dumps(meta, allow_nan=False)), **arrays)


def load_bundle(path):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data['metadata']))
        arrays = {k: data[k] for k in data.files if k != 'metadata'}
    validate_bundle(meta, arrays)
    return meta, arrays


def centered_gram(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(0, keepdims=True)
    return x @ x.T


def cka(x, y, debiased=False):
    """Linear CKA; debiased uses unbiased HSIC (the ratio is not unbiased).

    Returns None for insufficient samples or collapsed geometry. Debiased CKA
    may be negative; it is deliberately not clipped or called a distance.
    """
    if len(x) != len(y):
        raise ValueError('CKA rows must correspond.')
    n = len(x)
    if n < (4 if debiased else 2):
        return None
    k, l = centered_gram(x), centered_gram(y)
    if debiased:
        np.fill_diagonal(k, 0)
        np.fill_diagonal(l, 0)
        def hsic(a, b):
            return (np.sum(a*b) + a.sum()*b.sum()/((n-1)*(n-2))
                    - 2*np.dot(a.sum(1), b.sum(1))/(n-2)) / (n*(n-3))
    else:
        def hsic(a, b):
            return np.sum(a*b)
    kk, ll = hsic(k, k), hsic(l, l)
    if kk <= 0 or ll <= 0:
        return None
    return float(hsic(k, l) / (np.sqrt(kk)*np.sqrt(ll)))


def cosine_neighbors(x, k, block_size=256):
    """Exact cosine neighbors; self and zero-vector candidates are excluded.

    Ties are resolved by sample row order. Invalid anchor rows contain -1.
    Blocked computation bounds the temporary similarity matrix size.
    """
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1)
    valid = norms > 0
    result = np.full((len(x), k), -1, dtype=np.int64)
    if valid.sum() <= k:
        return result, {'zero_vectors': int((~valid).sum()), 'boundary_tie_fraction': None}
    z = x / np.where(valid, norms, 1)[:, None]
    ties = []
    for start in range(0, len(x), block_size):
        end = min(start+block_size, len(x))
        sim = z[start:end] @ z.T
        sim[:, ~valid] = -np.inf
        sim[np.arange(end-start), np.arange(start, end)] = -np.inf
        order = np.argsort(-sim, axis=1, kind='stable')
        for j, i in enumerate(range(start, end)):
            if valid[i]:
                result[i] = order[j, :k]
                if k+1 < valid.sum():
                    ties.append(bool(np.isclose(sim[j, order[j, k-1]], sim[j, order[j, k]],
                                                atol=1e-12, rtol=0)))
    return result, {'zero_vectors': int((~valid).sum()),
                    'boundary_tie_fraction': float(np.mean(ties)) if ties else None}


def neighbor_overlap(a, b):
    scores = np.full(len(a), np.nan)
    valid = (a >= 0).all(1) & (b >= 0).all(1)
    for i in np.flatnonzero(valid):
        scores[i] = len(set(a[i]) & set(b[i])) / a.shape[1]
    return scores


def align_to_reference(features, calibration, reference='original'):
    """Center on retained calibration means, then fit orthogonal maps there."""
    target = np.asarray(features[reference], dtype=np.float64)
    if calibration.sum() < 3 or len({x.shape[1] for x in features.values()}) != 1:
        raise ValueError('PCA overlay needs >=3 calibration rows and equal feature dimensions.')
    target = target - target[calibration].mean(0)
    aligned, diagnostics = {}, {}
    for name, x in features.items():
        z = np.asarray(x, dtype=np.float64) - x[calibration].mean(0)
        if name == reference:
            q = np.eye(z.shape[1])
        else:
            u, _, vt = np.linalg.svd(z[calibration].T @ target[calibration], full_matrices=False)
            q = u @ vt
        aligned[name] = z @ q
        diagnostics[name] = {'calibration_rank': int(np.linalg.matrix_rank(z[calibration])),
                             'feature_dimension': z.shape[1]}
    return aligned, diagnostics


def balanced_ids(candidates, labels, count, rng):
    selected = []
    for c in np.unique(labels):
        pool = np.array([i for i in candidates if labels[i] == c], dtype=int)
        if not len(pool):
            raise ValueError(f'No candidate examples for class {c}.')
        selected.extend(rng.permutation(pool)[:min(count, len(pool))].tolist())
    return np.array(selected, dtype=int)


def extract_digits(args):
    from sklearn.datasets import load_digits
    import sklearn
    reference = json.loads(gzip.decompress(args.reference.read_bytes()))
    records = [r for r in reference['runs'] if r['seed'] == args.seed and r['forget_class'] == args.forget_class]
    if len(records) != 1:
        raise ValueError('Requested digits seed/class absent from archived reference.')
    record = records[0]
    fixture = args.fixtures / f'digit{args.forget_class}-seed{args.seed}.npz'
    x, labels = load_digits(return_X_y=True)
    x = (x / 16).astype(np.float32)
    split = record['splits']
    flat = sum(split.values(), [])
    if len(flat) != len(set(flat)) or set(flat) != set(range(len(x))):
        raise ValueError('Reference splits must be disjoint and cover the digits dataset.')
    rng = np.random.default_rng(args.probe_seed)
    train = balanced_ids(split['training'], labels, args.per_class, rng)
    heldout = balanced_ids(split[args.evaluation], labels, args.per_class, rng)
    cal = np.array([i for i in split['selection'] if labels[i] != args.forget_class])
    ids = np.concatenate([train, heldout, cal])
    arrays = {'sample_ids': np.array([f'digits:{i}' for i in ids]), 'labels': labels[ids],
              'partitions': np.repeat(['train', 'heldout', 'calibration'], [len(train), len(heldout), len(cal)])}
    with np.load(fixture, allow_pickle=False) as data:
        for name in BASE_MODELS:
            h = np.maximum(x[ids] @ data[name+'_w1'] + data[name+'_b1'], 0)
            logits = h @ data[name+'_w2'] + data[name+'_b2']
            p = np.exp(logits - logits.max(1, keepdims=True))
            arrays[feature_key(name, 'hidden')] = h
            arrays[f'probabilities__{name}'] = p / p.sum(1, keepdims=True)
    meta = {'schema_version': 1, 'models': BASE_MODELS, 'layers': ['hidden'],
            'class_names': [str(i) for i in range(10)], 'forget_class': args.forget_class,
            'seed': args.seed, 'probe_seed': args.probe_seed, 'source': 'archived_digits',
            'evaluation_split': args.evaluation, 'method': 'incorrect-label suppression baseline',
            'notes': ['Exploratory: archived splits/checkpoints were used in earlier experiments.',
                      'No independent retrain replicate in this fixture; no erasure claim.',
                      'Selection split retained examples calibrate PCA; excluded from metrics.'],
            'fixture_sha256': sha256(fixture), 'reference_sha256': sha256(args.reference),
            'dataset_sha256': hashlib.sha256(x.tobytes()+labels.tobytes()).hexdigest(),
            'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'sklearn': sklearn.__version__}}
    save_bundle(args.output, meta, arrays)
    return args.output


def extract_synthetic(args):
    rng = np.random.default_rng(args.seed)
    n, d = 240, 16
    labels = np.tile(np.arange(4), n//4)
    parts = np.array(['train']*80 + ['heldout']*80 + ['calibration']*80)
    keep = ~((parts == 'calibration') & (labels == 0))
    x = (rng.normal(size=(n, d)) + rng.normal(size=(4, d))[labels])[keep]
    labels, parts = labels[keep], parts[keep]
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    arrays = {'sample_ids': np.array([f'synthetic:{i}' for i in np.flatnonzero(keep)]),
              'labels': labels, 'partitions': parts}
    for name, h in zip(BASE_MODELS, [x, x @ q, rng.normal(size=x.shape)]):
        arrays[feature_key(name, 'hidden')] = h
        arrays[f'probabilities__{name}'] = np.full((len(x), 4), 0.25)
    meta = {'schema_version': 1, 'models': BASE_MODELS, 'layers': ['hidden'],
            'class_names': ['0', '1', '2', '3'], 'forget_class': 0, 'seed': args.seed,
            'source': 'synthetic', 'method': 'metric control, not trained models',
            'notes': ['Original/unlearned differ by rotation only; retrained is unrelated noise.',
                      'These synthetic roles test software and say nothing about unlearning.']}
    save_bundle(args.output, meta, arrays)
    return args.output


def class_cosines(features, labels, classes):
    """Average example-pair cosines, not cosine between class centroids.

    Same-class blocks exclude self-pairs and count each unordered pair once.
    Zero vectors have undefined cosine and are excluded with counts recorded.
    """
    x = np.asarray(features, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1)
    valid = norms > 0
    z = x / np.where(valid, norms, 1)[:, None]
    cosine = np.clip(z @ z.T, -1, 1)
    means = np.full((len(classes), len(classes)), np.nan)
    counts = np.zeros_like(means, dtype=int)
    for i, first in enumerate(classes):
        rows = np.flatnonzero((labels == first) & valid)
        for j in range(i, len(classes)):
            cols = np.flatnonzero((labels == classes[j]) & valid)
            block = cosine[np.ix_(rows, cols)]
            values = block[np.triu_indices(len(rows), k=1)] if i == j else block.ravel()
            if len(values):
                means[i, j] = means[j, i] = float(values.mean())
            counts[i, j] = counts[j, i] = len(values)
    return means, counts, int((~valid).sum())



def compute_relations(meta, arrays, layer=None):
    """Full-space class relationships, including retained-calibration centering."""
    layer = layer or meta['layers'][-1]
    if layer not in meta['layers']:
        raise ValueError(f'Unknown layer: {layer}')
    held = arrays['partitions'] == 'heldout'
    cal = arrays['partitions'] == 'calibration'
    if not held.any():
        raise ValueError('At least one held-out example is required.')
    yy = arrays['labels'][held]
    classes = np.unique(yy)
    blocks, summaries, euclidean = [], [], []
    modes = ['raw', 'centered'] if cal.any() else ['raw']
    means = {}
    for mode in modes:
        for model in meta['models']:
            h = arrays[feature_key(model, layer)].astype(float)
            if mode == 'centered':
                h = h - h[cal].mean(0)
            matrix, counts, zeros = class_cosines(h[held], yy, classes)
            means[mode, model] = matrix
            for i, first in enumerate(classes):
                for j, second in enumerate(classes):
                    value = matrix[i, j]
                    blocks.append({'mode': mode, 'model': model, 'class_a': int(first),
                                   'class_b': int(second), 'mean_cosine': float(value) if np.isfinite(value) else None,
                                   'distinct_pairs': int(counts[i, j]), 'zero_vectors': zeros})
        keep = classes != meta['forget_class']
        pair_mask = np.triu(keep[:, None] & keep[None, :], 1)
        for first, second in itertools.combinations(meta['models'], 2):
            values = (means[mode, second]-means[mode, first])[pair_mask]
            values = values[np.isfinite(values)]
            summaries.append({'mode': mode, 'pair': f'{second} minus {first}',
                              'retain_cross_mean_change': float(values.mean()) if len(values) else None,
                              'retain_cross_mean_absolute_change': float(np.abs(values).mean()) if len(values) else None,
                              'positive_class_pairs': int((values > 0).sum()), 'valid_class_pairs': len(values)})
    retain = yy != meta['forget_class']
    for model in meta['models']:
        h = arrays[feature_key(model, layer)][held][retain].astype(float)
        y = yy[retain]
        pairs = np.triu(y[:, None] != y[None, :], 1)
        sq = np.sum(h*h, axis=1)
        dist = np.sqrt(np.maximum(sq[:, None]+sq[None, :]-2*h@h.T, 0))
        euclidean.append({'model': model,
                          'mean_cross_retain_distance': float(dist[pairs].mean()) if pairs.any() else None,
                          'mean_activation_norm': float(np.linalg.norm(h, axis=1).mean()) if len(h) else None})
    return {'layer': layer, 'classes': classes.tolist(), 'modes': modes, 'blocks': blocks,
            'changes': summaries, 'euclidean': euclidean,
            'heldout_counts': {str(c): int((yy == c).sum()) for c in classes},
            'notes': ['Class cosines use all held-out examples; self-pairs and zero vectors are excluded.',
                      'Centered cosines subtract each model retained-calibration mean; no alignment is fitted.',
                      'Means can cancel; centered geometry constrains aggregates. Centering is not proof of a pure offset.',
                      'Euclidean distances are scale-sensitive and pair-weighted; cosine summaries weight class pairs equally.',
                      'Pair counts are not independent sample sizes. No confidence or causal claim.']}


def write_report(result, diagnostics, output):
    """One report containing behavior, metrics, geometry, and interpretation."""
    meta, rel = result['metadata'], result['relations']
    k = 10 if 10 in result['ks'] else result['ks'][0]
    def fmt(x):
        return 'undefined' if x is None else f'{x:.4f}'
    lines = ['# Representation comparison', '',
             f"Source: **{meta['source']}**; seed: **{meta.get('seed', 'unspecified')}**; "
             f"forget class: **{meta['forget_class']}**. {meta.get('method', '')}.", '',
             *[f'- {note}' for note in meta.get('notes', [])], '',
             '## Behavior on held-out inputs', '',
             '| Model | Group | n | Accuracy | NLL |', '|---|---|---:|---:|---:|']
    for r in result['behavior']:
        if r['partition'] == 'heldout' and r['group'] in ('forget', 'retain'):
            lines.append(f"| {r['model']} | {r['group']} | {r['n']} | {r['accuracy']:.2%} | {fmt(r['nll'])} |")
    lines += ['', '## Representation metrics', '',
              'These metrics use unreduced activations before any fitted alignment. Higher values mean',
              'more similar geometry or neighborhoods, not better unlearning. CKA is centered within',
              'each group; neighbor candidates include all classes in the partition.', '',
              f'| Layer | Group | Pair | n | CKA | Debiased CKA | Shuffled CKA | Shared {k}-NN | Prediction disagreement |',
              '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in result['comparisons']:
        if r['partition'] == 'heldout' and r['group'] in ('forget', 'retain'):
            lines.append(f"| {r['layer']} | {r['group']} | {r['pair']} | {r['n']} | {fmt(r['cka'])} | "
                         f"{fmt(r['cka_debiased'])} | {fmt(r['cka_permuted_mean'])} | {fmt(r[f'knn_{k}'])} | "
                         f"{r['prediction_disagreement']:.2%} |")
    if (output/'representation_metrics.png').exists():
        lines += ['', '![Metrics across layers](representation_metrics.png)']
    lines += ['', '## Class relationships', '',
              f"Cosine and PCA views use **{rel['layer']}**. Each heatmap cell averages similarities between",
              'images of its row and column classes. Diagonal cells compare distinct images of the same',
              'class, excluding self-matches. Cosine measures direction, ignoring magnitude. ReLU features',
              'are nonnegative, so raw cosines cannot be negative. Class means can hide variation.', '',
              '![Class-average cosine](similarity_by_class.png)', '',
              '## Changes and sensitivity to centering', '',
              'Columns show the model subtraction in their titles. Red means increased cosine similarity;',
              'blue means decreased similarity. Raw and centered rows share one signed color scale.',
              'Centering subtracts each model mean on retained calibration examples only. The mean is not',
              'fitted on these held-out images. No rotation or projection is used for these heatmaps.', '',
              '![Raw and centered cosine changes](similarity_change.png)', '',
              '| Coordinates | Model difference | Mean change among retained classes | Mean absolute change | Positive / valid pairs |',
              '|---|---|---:|---:|---:|']
    for r in rel['changes']:
        lines.append(f"| {r['mode']} | {r['pair']} | {fmt(r['retain_cross_mean_change'])} | "
                     f"{fmt(r['retain_cross_mean_absolute_change'])} | {r['positive_class_pairs']} / {r['valid_class_pairs']} |")
    lines += ['', 'Signed means may cancel; centering also constrains aggregate geometry. Disappearance',
              'of a broad signed increase does not prove that changes were entirely a common offset.', '',
              '| Model | Mean Euclidean distance between retained classes | Mean retained activation norm |',
              '|---|---:|---:|']
    for r in rel['euclidean']:
        lines.append(f"| {r['model']} | {fmt(r['mean_cross_retain_distance'])} | {fmt(r['mean_activation_norm'])} |")
    lines += ['', 'Euclidean distances weight image pairs equally and are sensitive to scale. Pair counts',
              'are not independent statistical samples; these comparisons are descriptive.', '', '## PCA overview', '']
    if (output/'pca.png').exists():
        variance = sum(diagnostics['pca_calibration_explained_variance_ratio'])
        lines += ['![Shared aligned PCA](pca.png)', '',
                  f'Two axes explain **{variance:.1%}** of calibration variance. Rotations/reflections and PCA',
                  'are fitted on retained calibration examples only. A similar 2D position is not full-space equivalence.']
    else:
        lines += [f"PCA skipped: {diagnostics.get('pca_skipped', 'unavailable')}."]
    lines += ['', *[f'- {n}' for n in diagnostics['notes']], '', '## Limits and numerical evidence', '',
              *[f'- {n}' for n in result['notes']], '',
              'A retrained model may retain a coherent forget-class representation through shared features.',
              'High similarity alone cannot demonstrate residual training-data influence. Compare against',
              'independent retraining on fixed retained data before claiming equivalence or removal.', '',
              'All results and provenance are in [metrics.json](metrics.json). Detailed tables:',
              '[comparisons](comparisons.csv), [behavior](behavior.csv), [class relations](relations.csv).',
              '[Plot diagnostics](plot_diagnostics.json) record calibration rank and alignment residuals.', '']
    (output/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def analyze(bundle, output, ks=(5, 10, 25), null_repeats=5, seed=0, layer=None):
    from representation_plots import make_plots
    meta, a = load_bundle(bundle)
    output = Path(output)
    if output.exists():
        raise FileExistsError('Choose a fresh analysis output directory.')
    if not ks or min(ks) < 1 or null_repeats < 1:
        raise ValueError('Neighbor counts and permutation repeats must be positive.')
    labels, parts = a['labels'], a['partitions']
    forget = labels == meta['forget_class']
    pairs = list(itertools.combinations(meta['models'], 2))
    rng = np.random.default_rng(seed)
    rows, behavior, quality = [], [], []
    for partition in ('train', 'heldout'):
        mask = parts == partition
        if not mask.any():
            continue
        yy, ff = labels[mask], forget[mask]
        groups = {'all': np.ones(len(yy), bool), 'forget': ff, 'retain': ~ff}
        groups.update({f'class_{c}': yy == c for c in np.unique(yy)})
        for name in meta['models']:
            prob = a[f'probabilities__{name}'][mask]
            pred = prob.argmax(1)
            for group, g in groups.items():
                if not g.any():
                    continue
                behavior.append({'partition': partition, 'group': group, 'model': name, 'n': int(g.sum()),
                                 'accuracy': float(np.mean(pred[g] == yy[g])),
                                 'nll': float(-np.log(np.maximum(prob[g, yy[g]], 1e-30)).mean())})
        for metric_layer in meta['layers']:
            features = {name: a[feature_key(name, metric_layer)][mask] for name in meta['models']}
            neighbors = {}
            for name, x in features.items():
                for k in ks:
                    nn, diag = cosine_neighbors(x, k)
                    neighbors[name, k] = nn
                    quality.append({'partition': partition, 'layer': metric_layer, 'model': name, 'k': k, **diag})
            for first, second in pairs:
                overlap = {k: neighbor_overlap(neighbors[first, k], neighbors[second, k]) for k in ks}
                p, q = a[f'probabilities__{first}'][mask], a[f'probabilities__{second}'][mask]
                mid = (p + q) / 2
                js = 0.5 * np.sum(p*np.log(np.maximum(p, 1e-30)/np.maximum(mid, 1e-30))
                                 + q*np.log(np.maximum(q, 1e-30)/np.maximum(mid, 1e-30)), axis=1)
                for group, g in groups.items():
                    if not g.any():
                        continue
                    x, y = features[first][g], features[second][g]
                    null = [cka(x, y[rng.permutation(len(y))]) for _ in range(null_repeats)]
                    null = [v for v in null if v is not None]
                    row = {'partition': partition, 'group': group, 'layer': metric_layer,
                           'pair': f'{first}--{second}', 'n': len(x),
                           'cka': cka(x, y), 'cka_debiased': cka(x, y, debiased=True),
                           'cka_permuted_mean': float(np.mean(null)) if null else None,
                           'prediction_disagreement': float(np.mean(p[g].argmax(1) != q[g].argmax(1))),
                           'prediction_js_nats': float(js[g].mean())}
                    for k in ks:
                        values = overlap[k][g]
                        valid = np.isfinite(values)
                        row[f'knn_{k}'] = float(values[valid].mean()) if valid.any() else None
                        row[f'knn_{k}_valid_anchors'] = int(valid.sum())
                    rows.append(row)
    relations = compute_relations(meta, a, layer)
    output.mkdir(parents=True)
    result = {'metadata': meta, 'bundle_sha256': sha256(bundle), 'ks': list(ks),
              'analysis_seed': seed, 'null_repeats': null_repeats,
              'analysis_sha256': sha256(__file__), 'plots_sha256': sha256(Path(__file__).with_name('representation_plots.py')),
              'environment': {'python': platform.python_version(), 'numpy': np.__version__},
              'notes': ['CKA uses separately centered groups; kNN uses all classes in each partition.',
                        'Permutation means diagnose finite-sample bias; they are not p-values.',
                        'No confidence interval or equivalence claim from a single model trio.',
                        'Debiased CKA uses unbiased HSIC; its ratio is not an unbiased estimator.',
                        'Cosine ties use input row order; inspect neighbor_quality for degeneracy.'],
              'comparisons': rows, 'behavior': behavior, 'neighbor_quality': quality, 'relations': relations}
    write_json(output/'metrics.json', result)
    for filename, records in [('comparisons.csv', rows), ('behavior.csv', behavior), ('relations.csv', relations['blocks'])]:
        if records:
            with (output/filename).open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
    plot_notes = make_plots(meta, a, rows, output, ks, relations)
    write_json(output/'plot_diagnostics.json', plot_notes)
    write_report(result, plot_notes, output)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    d = sub.add_parser('digits', help='Extract archived CPU checkpoints; no training/downloads.')
    d.add_argument('--seed', type=int, default=0)
    d.add_argument('--forget-class', type=int, default=0)
    d.add_argument('--probe-seed', type=int, default=2026)
    d.add_argument('--per-class', type=int, default=60)
    d.add_argument('--evaluation', choices=['discovery', 'test'], default='discovery')
    d.add_argument('--reference', type=Path, default=Path('reports/targeted-flips/cpu/digits-main.json.gz'))
    d.add_argument('--fixtures', type=Path, default=Path('reports/random-flips'))
    d.add_argument('--output', type=Path, required=True)
    s = sub.add_parser('synthetic', help='Rotation/noise control; not an unlearning experiment.')
    s.add_argument('--seed', type=int, default=0)
    s.add_argument('--output', type=Path, required=True)
    a = sub.add_parser('analyze', help='CPU analysis of an existing feature bundle.')
    a.add_argument('--features', type=Path, required=True)
    a.add_argument('--output-dir', type=Path, required=True)
    a.add_argument('--neighbors', type=int, nargs='+', default=[5, 10, 25])
    a.add_argument('--null-repeats', type=int, default=5)
    a.add_argument('--seed', type=int, default=0)
    a.add_argument('--layer', help='Layer for cosine/PCA views; default: final exported layer.')
    args = p.parse_args()
    try:
        if args.command == 'digits':
            if args.per_class < 1:
                p.error('--per-class must be positive.')
            result = extract_digits(args)
        elif args.command == 'synthetic':
            result = extract_synthetic(args)
        else:
            analyze(args.features, args.output_dir, tuple(dict.fromkeys(args.neighbors)), args.null_repeats, args.seed, args.layer)
            result = args.output_dir/'REPORT.md'
    except (ValueError, FileExistsError, FileNotFoundError) as e:
        p.error(str(e))
    print(f'Saved {result}')


if __name__ == '__main__':
    main()
