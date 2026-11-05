"""Extract matched representations and run CPU analysis on saved feature bundles.

The `digits` command reads archived NumPy models; `synthetic` creates a metric
control; `analyze` produces numerical tables and figures from either exporter.
Metric implementations live in representation_metrics.py and report rendering
in representation_report.py. See docs/representation-analysis.md for usage.
"""

import os

# Set before NumPy loads BLAS. Probe matrices are small enough that large
# thread pools can cost more than they save, and stable execution aids replay.
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

import numpy as np

from representation_metrics import (
    cka,
    compute_relations,
    cosine_neighbors,
    feature_key,
    neighbor_overlap,
)
from representation_report import write_report


BASE_MODELS = ['original', 'unlearned', 'retrained']
PARTITIONS = ('train', 'heldout', 'calibration')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')


# Feature bundles
# ---------------
# Every model uses the same row order, sample identities and partitions. This
# contract is what makes row-wise CKA and shared-neighbor comparisons meaningful.
# Calibration rows are retained-only and separate from train/held-out metrics;
# their purpose is to fit display coordinates, not to improve measured scores.
# Reject malformed or nonfinite inputs rather than silently dropping examples.


def validate_bundle(metadata, arrays):
    """Check the shared sample axis, partitions and every model's feature arrays."""
    if metadata.get('schema_version') != 1:
        raise ValueError('Unsupported feature bundle schema.')
    models, layers = metadata['models'], metadata['layers']
    if not set(BASE_MODELS).issubset(models) or not layers:
        raise ValueError('Need original, unlearned, retrained and at least one layer.')
    for names in (models, layers):
        if len(names) != len(set(names)) or any(
            not re.fullmatch(r'[a-zA-Z0-9_]+', name) for name in names
        ):
            raise ValueError('Model/layer names must be unique alphanumeric identifiers.')
    sample_ids, labels, partitions = (
        arrays[field] for field in ('sample_ids', 'labels', 'partitions')
    )
    sample_count = len(sample_ids)
    if (
        sample_count < 4
        or sample_ids.ndim != 1
        or len(set(sample_ids.tolist())) != sample_count
    ):
        raise ValueError('Need at least four unique, ordered sample IDs.')
    if sample_ids.dtype.kind not in 'US' or labels.dtype.kind not in 'iu':
        raise ValueError('Sample IDs must be strings and labels integer class indices.')
    if (
        labels.shape != (sample_count,)
        or partitions.shape != (sample_count,)
        or not set(partitions).issubset(PARTITIONS)
    ):
        raise ValueError('Labels/partitions must match the sample axis and known partitions.')
    if np.any(labels < 0) or not 0 <= metadata['forget_class'] < len(metadata['class_names']):
        raise ValueError('Invalid class index.')
    if np.any(labels >= len(metadata['class_names'])):
        raise ValueError('Label outside class_names.')
    if np.any((partitions == 'calibration') & (labels == metadata['forget_class'])):
        raise ValueError('Alignment calibration must contain retained examples only.')
    for model in models:
        for layer in layers:
            features = arrays[feature_key(model, layer)]
            if (
                features.ndim != 2
                or features.shape[0] != sample_count
                or features.shape[1] < 1
                or not np.isfinite(features).all()
            ):
                raise ValueError(f'Invalid/nonfinite activations: {model}/{layer}')
        probabilities = arrays[f'probabilities__{model}']
        if (
            probabilities.shape != (sample_count, len(metadata['class_names']))
            or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0)
            or not np.allclose(probabilities.sum(1), 1, atol=1e-5)
        ):
            raise ValueError(f'Invalid class probabilities: {model}')


def save_bundle(path, metadata, arrays):
    validate_bundle(metadata, arrays)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental replacement of a frozen extraction.
    with path.open('xb') as stream:
        np.savez_compressed(
            stream, metadata=np.array(json.dumps(metadata, allow_nan=False)), **arrays
        )


def load_bundle(path):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data['metadata']))
        arrays = {k: data[k] for k in data.files if k != 'metadata'}
    validate_bundle(metadata, arrays)
    return metadata, arrays


# Extraction from archived digits models
# --------------------------------------
# Sampling and inference happen once per bundle. The saved identities allow
# later analyses to use exactly the same examples without retraining models.
# Keep RNG calls and class order explicit: changing either changes the probes.


def balanced_ids(candidates, labels, count, rng):
    """Sample at most count examples from each class, preserving class order."""
    selected = []
    for class_id in np.unique(labels):
        pool = np.array([i for i in candidates if labels[i] == class_id], dtype=int)
        if not len(pool):
            raise ValueError(f'No candidate examples for class {class_id}.')
        selected.extend(rng.permutation(pool)[: min(count, len(pool))].tolist())
    return np.array(selected, dtype=int)


def extract_digits(args):
    """Export clean, archived model activations without running fault search."""
    from sklearn.datasets import load_digits
    import sklearn

    reference = json.loads(gzip.decompress(args.reference.read_bytes()))
    records = [
        r
        for r in reference['runs']
        if r['seed'] == args.seed and r['forget_class'] == args.forget_class
    ]
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
    calibration_ids = np.array(
        [i for i in split['selection'] if labels[i] != args.forget_class]
    )
    ids = np.concatenate([train, heldout, calibration_ids])
    arrays = {
        'sample_ids': np.array([f'digits:{i}' for i in ids]),
        'labels': labels[ids],
        'partitions': np.repeat(
            ['train', 'heldout', 'calibration'],
            [len(train), len(heldout), len(calibration_ids)],
        ),
    }
    with np.load(fixture, allow_pickle=False) as data:
        for name in BASE_MODELS:
            hidden = np.maximum(x[ids] @ data[name + '_w1'] + data[name + '_b1'], 0)
            logits = hidden @ data[name + '_w2'] + data[name + '_b2']
            probabilities = np.exp(logits - logits.max(1, keepdims=True))
            arrays[feature_key(name, 'hidden')] = hidden
            arrays[f'probabilities__{name}'] = probabilities / probabilities.sum(
                1, keepdims=True
            )
    metadata = {
        'schema_version': 1,
        'models': BASE_MODELS,
        'layers': ['hidden'],
        'class_names': [str(i) for i in range(10)],
        'forget_class': args.forget_class,
        'seed': args.seed,
        'probe_seed': args.probe_seed,
        'source': 'archived_digits',
        'evaluation_split': args.evaluation,
        'method': 'incorrect-label suppression baseline',
        'notes': [
            'Exploratory: archived splits/checkpoints were used in earlier experiments.',
            'No independent retrain replicate in this fixture; no erasure claim.',
            'Selection split retained examples calibrate PCA; excluded from metrics.',
        ],
        'fixture_sha256': sha256(fixture),
        'reference_sha256': sha256(args.reference),
        'dataset_sha256': hashlib.sha256(x.tobytes() + labels.tobytes()).hexdigest(),
        'environment': {
            'python': platform.python_version(),
            'numpy': np.__version__,
            'sklearn': sklearn.__version__,
        },
    }
    save_bundle(args.output, metadata, arrays)
    return args.output


def extract_synthetic(args):
    """Build a rotation-invariant positive control and an unrelated noise model.

    The model-role names match real bundles so the entire analysis path is
    exercised. Their synthetic probabilities are uniform and are not evidence
    about unlearning. Calibration excludes class zero, just as in real runs.
    """
    rng = np.random.default_rng(args.seed)
    n, d = 240, 16
    labels = np.tile(np.arange(4), n // 4)
    partitions = np.array(['train'] * 80 + ['heldout'] * 80 + ['calibration'] * 80)
    keep = ~((partitions == 'calibration') & (labels == 0))
    x = (rng.normal(size=(n, d)) + rng.normal(size=(4, d))[labels])[keep]
    labels, partitions = labels[keep], partitions[keep]
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    arrays = {
        'sample_ids': np.array([f'synthetic:{i}' for i in np.flatnonzero(keep)]),
        'labels': labels,
        'partitions': partitions,
    }
    for name, h in zip(BASE_MODELS, [x, x @ q, rng.normal(size=x.shape)]):
        arrays[feature_key(name, 'hidden')] = h
        arrays[f'probabilities__{name}'] = np.full((len(x), 4), 0.25)
    metadata = {
        'schema_version': 1,
        'models': BASE_MODELS,
        'layers': ['hidden'],
        'class_names': ['0', '1', '2', '3'],
        'forget_class': 0,
        'seed': args.seed,
        'source': 'synthetic',
        'method': 'metric control, not trained models',
        'notes': [
            'Original/unlearned differ by rotation only; retrained is unrelated noise.',
            'These synthetic roles test software and say nothing about unlearning.',
        ],
    }
    save_bundle(args.output, metadata, arrays)
    return args.output


# Analysis and output
# -------------------
# Full-space measurements precede plotting. Train and held-out partitions are
# evaluated independently; calibration rows never enter these estimates.
# Groups restrict the anchors for neighbors, while the candidate pool remains
# the complete partition. CKA instead centers each selected group separately.


def analyze(bundle, output, ks=(5, 10, 25), null_repeats=5, seed=0, layer=None):
    """Measure each model pair and layer, then write one report and its evidence."""
    from representation_plots import make_plots

    metadata, arrays = load_bundle(bundle)
    output = Path(output)
    if output.exists():
        raise FileExistsError('Choose a fresh analysis output directory.')
    if not ks or min(ks) < 1 or null_repeats < 1:
        raise ValueError('Neighbor counts and permutation repeats must be positive.')
    labels, partitions = arrays['labels'], arrays['partitions']
    forget = labels == metadata['forget_class']
    pairs = list(itertools.combinations(metadata['models'], 2))
    # Permutation references diagnose finite-sample similarity, not significance.
    # Loop order determines RNG consumption and is part of reproducibility.
    rng = np.random.default_rng(seed)
    rows, behavior, quality = [], [], []
    for partition in ('train', 'heldout'):
        mask = partitions == partition
        if not mask.any():
            continue
        partition_labels, forgotten = labels[mask], forget[mask]
        groups = {
            'all': np.ones(len(partition_labels), bool),
            'forget': forgotten,
            'retain': ~forgotten,
        }
        groups.update(
            {f'class_{c}': partition_labels == c for c in np.unique(partition_labels)}
        )
        for name in metadata['models']:
            prob = arrays[f'probabilities__{name}'][mask]
            pred = prob.argmax(1)
            for group, group_mask in groups.items():
                if not group_mask.any():
                    continue
                behavior.append(
                    {
                        'partition': partition,
                        'group': group,
                        'model': name,
                        'n': int(group_mask.sum()),
                        'accuracy': float(
                            np.mean(pred[group_mask] == partition_labels[group_mask])
                        ),
                        'nll': float(
                            -np.log(
                                np.maximum(
                                    prob[group_mask, partition_labels[group_mask]], 1e-30
                                )
                            ).mean()
                        ),
                    }
                )
        for metric_layer in metadata['layers']:
            features = {
                name: arrays[feature_key(name, metric_layer)][mask]
                for name in metadata['models']
            }
            neighbors = {}
            for name, x in features.items():
                for k in ks:
                    neighbor_indices, diagnostics = cosine_neighbors(x, k)
                    neighbors[name, k] = neighbor_indices
                    quality.append(
                        {
                            'partition': partition,
                            'layer': metric_layer,
                            'model': name,
                            'k': k,
                            **diagnostics,
                        }
                    )
            for first, second in pairs:
                overlap = {
                    k: neighbor_overlap(neighbors[first, k], neighbors[second, k]) for k in ks
                }
                first_probabilities, second_probabilities = (
                    arrays[f'probabilities__{first}'][mask],
                    arrays[f'probabilities__{second}'][mask],
                )
                midpoint = (first_probabilities + second_probabilities) / 2
                js_divergence = 0.5 * np.sum(
                    first_probabilities
                    * np.log(
                        np.maximum(first_probabilities, 1e-30) / np.maximum(midpoint, 1e-30)
                    )
                    + second_probabilities
                    * np.log(
                        np.maximum(second_probabilities, 1e-30) / np.maximum(midpoint, 1e-30)
                    ),
                    axis=1,
                )
                for group, group_mask in groups.items():
                    if not group_mask.any():
                        continue
                    x, y = features[first][group_mask], features[second][group_mask]
                    permuted_scores = [
                        cka(x, y[rng.permutation(len(y))]) for _ in range(null_repeats)
                    ]
                    permuted_scores = [v for v in permuted_scores if v is not None]
                    row = {
                        'partition': partition,
                        'group': group,
                        'layer': metric_layer,
                        'pair': f'{first}--{second}',
                        'n': len(x),
                        'cka': cka(x, y),
                        'cka_debiased': cka(x, y, debiased=True),
                        'cka_permuted_mean': float(np.mean(permuted_scores))
                        if permuted_scores
                        else None,
                        'prediction_disagreement': float(
                            np.mean(
                                first_probabilities[group_mask].argmax(1)
                                != second_probabilities[group_mask].argmax(1)
                            )
                        ),
                        'prediction_js_nats': float(js_divergence[group_mask].mean()),
                    }
                    for k in ks:
                        values = overlap[k][group_mask]
                        valid = np.isfinite(values)
                        row[f'knn_{k}'] = float(values[valid].mean()) if valid.any() else None
                        row[f'knn_{k}_valid_anchors'] = int(valid.sum())
                    rows.append(row)
    # Finish numerical work before reserving an output directory. Existing
    # artifacts are never reused, so a report cannot silently mix two analyses.
    # Hash every implementation module that contributes to the saved result.
    relations = compute_relations(metadata, arrays, layer)
    output.mkdir(parents=True)
    result = {
        'metadata': metadata,
        'bundle_sha256': sha256(bundle),
        'ks': list(ks),
        'analysis_seed': seed,
        'null_repeats': null_repeats,
        'analysis_sha256': sha256(__file__),
        'plots_sha256': sha256(Path(__file__).with_name('representation_plots.py')),
        'metrics_sha256': sha256(Path(__file__).with_name('representation_metrics.py')),
        'report_sha256': sha256(Path(__file__).with_name('representation_report.py')),
        'environment': {'python': platform.python_version(), 'numpy': np.__version__},
        'notes': [
            'CKA uses separately centered groups; kNN uses all classes in each partition.',
            'Permutation means diagnose finite-sample bias; they are not p-values.',
            'No confidence interval or equivalence claim from a single model trio.',
            'Debiased CKA uses unbiased HSIC; its ratio is not an unbiased estimator.',
            'Cosine ties use input row order; inspect neighbor_quality for degeneracy.',
        ],
        'comparisons': rows,
        'behavior': behavior,
        'neighbor_quality': quality,
        'relations': relations,
    }
    write_json(output / 'metrics.json', result)
    for filename, records in [
        ('comparisons.csv', rows),
        ('behavior.csv', behavior),
        ('relations.csv', relations['blocks']),
    ]:
        if records:
            with (output / filename).open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
    plot_notes = make_plots(metadata, arrays, rows, output, ks, relations)
    write_json(output / 'plot_diagnostics.json', plot_notes)
    write_report(result, plot_notes, output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    digits_parser = subparsers.add_parser(
        'digits', help='Extract archived CPU checkpoints; no training/downloads.'
    )
    digits_parser.add_argument('--seed', type=int, default=0)
    digits_parser.add_argument('--forget-class', type=int, default=0)
    digits_parser.add_argument('--probe-seed', type=int, default=2026)
    digits_parser.add_argument('--per-class', type=int, default=60)
    digits_parser.add_argument(
        '--evaluation', choices=['discovery', 'test'], default='discovery'
    )
    digits_parser.add_argument(
        '--reference',
        type=Path,
        default=Path('reports/targeted-flips/cpu/digits-main.json.gz'),
    )
    digits_parser.add_argument('--fixtures', type=Path, default=Path('reports/random-flips'))
    digits_parser.add_argument('--output', type=Path, required=True)
    synthetic_parser = subparsers.add_parser(
        'synthetic', help='Rotation/noise control; not an unlearning experiment.'
    )
    synthetic_parser.add_argument('--seed', type=int, default=0)
    synthetic_parser.add_argument('--output', type=Path, required=True)
    analysis_parser = subparsers.add_parser(
        'analyze', help='CPU analysis of an existing feature bundle.'
    )
    analysis_parser.add_argument('--features', type=Path, required=True)
    analysis_parser.add_argument('--output-dir', type=Path, required=True)
    analysis_parser.add_argument('--neighbors', type=int, nargs='+', default=[5, 10, 25])
    analysis_parser.add_argument('--null-repeats', type=int, default=5)
    analysis_parser.add_argument('--seed', type=int, default=0)
    analysis_parser.add_argument(
        '--layer', help='Layer for cosine/PCA views; default: final exported layer.'
    )
    args = parser.parse_args()
    try:
        if args.command == 'digits':
            if args.per_class < 1:
                parser.error('--per-class must be positive.')
            result = extract_digits(args)
        elif args.command == 'synthetic':
            result = extract_synthetic(args)
        else:
            analyze(
                args.features,
                args.output_dir,
                tuple(dict.fromkeys(args.neighbors)),
                args.null_repeats,
                args.seed,
                args.layer,
            )
            result = args.output_dir / 'REPORT.md'
    except (ValueError, FileExistsError, FileNotFoundError) as e:
        parser.error(str(e))
    print(f'Saved {result}')


if __name__ == '__main__':
    main()
