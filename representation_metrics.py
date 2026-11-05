"""Representation metrics evaluated on matched examples in full feature space.

CKA and neighbors compare the original coordinates. Class-relation centering
uses a separate retained calibration split; alignment is only for PCA views.
See docs/representation-analysis.md for definitions and interpretation.
"""

import itertools

import numpy as np


def feature_key(model, layer):
    return f'features__{model}__{layer}'


# Centered kernel alignment
# -------------------------
# Rows are matched examples; columns may differ across models. Centering each
# group removes translation before its linear Gram matrix is compared. Ordinary
# CKA can look large with few examples and many dimensions. The debiased option
# corrects HSIC terms, but their normalized ratio is not itself unbiased.
# Undefined cases remain explicit rather than becoming a misleading zero.


def centered_gram(x):
    """Compute the linear Gram matrix after centering this group's features."""
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
        # Unbiased HSIC excludes diagonal self-products and corrects row sums.
        # Its n-3 denominator explains the minimum of four matched examples.
        np.fill_diagonal(k, 0)
        np.fill_diagonal(l, 0)

        def hsic(a, b):
            return (
                np.sum(a * b)
                + a.sum() * b.sum() / ((n - 1) * (n - 2))
                - 2 * np.dot(a.sum(1), b.sum(1)) / (n - 2)
            ) / (n * (n - 3))
    else:

        def hsic(a, b):
            return np.sum(a * b)

    kk, ll = hsic(k, k), hsic(l, l)
    if kk <= 0 or ll <= 0:
        return None
    return float(hsic(k, l) / (np.sqrt(kk) * np.sqrt(ll)))


# Shared neighbors
# ----------------
# Normalize each nonzero vector and rank all other valid examples in the same
# partition. Zero vectors have no direction and cannot be anchors or candidates.
# Returning -1 for an undefined row lets overlap propagate NaN instead of
# reporting a perfect match between two equally empty neighbor lists.


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
        end = min(start + block_size, len(x))
        sim = z[start:end] @ z.T
        sim[:, ~valid] = -np.inf
        sim[np.arange(end - start), np.arange(start, end)] = -np.inf
        order = np.argsort(-sim, axis=1, kind='stable')
        for j, i in enumerate(range(start, end)):
            if valid[i]:
                result[i] = order[j, :k]
                if k + 1 < valid.sum():
                    ties.append(
                        bool(
                            np.isclose(
                                sim[j, order[j, k - 1]],
                                sim[j, order[j, k]],
                                atol=1e-12,
                                rtol=0,
                            )
                        )
                    )
    return result, {
        'zero_vectors': int((~valid).sum()),
        'boundary_tie_fraction': float(np.mean(ties)) if ties else None,
    }


def neighbor_overlap(a, b):
    """Return the shared fraction of two k-neighbor sets for each valid anchor."""
    scores = np.full(len(a), np.nan)
    valid = (a >= 0).all(1) & (b >= 0).all(1)
    for i in np.flatnonzero(valid):
        scores[i] = len(set(a[i]) & set(b[i])) / a.shape[1]
    return scores


# Visualization alignment
# -----------------------
# Fit translations and orthogonal maps on retained calibration rows only.
# The same fitted map is applied to every row, including held-out examples.
# Refitting on those examples would make the display appear artificially close.
# These aligned coordinates never feed back into CKA or neighbor measurements.


def align_to_reference(features, calibration, reference='original'):
    """Center on retained calibration means, then fit orthogonal maps there."""
    target = np.asarray(features[reference], dtype=np.float64)
    if calibration.sum() < 3 or len({x.shape[1] for x in features.values()}) != 1:
        raise ValueError(
            'PCA overlay needs >=3 calibration rows and equal feature dimensions.'
        )
    target = target - target[calibration].mean(0)
    aligned, diagnostics = {}, {}
    for name, x in features.items():
        centered = np.asarray(x, dtype=np.float64) - x[calibration].mean(0)
        if name == reference:
            rotation = np.eye(centered.shape[1])
        else:
            u, _, vt = np.linalg.svd(
                centered[calibration].T @ target[calibration], full_matrices=False
            )
            rotation = u @ vt
        aligned[name] = centered @ rotation
        diagnostics[name] = {
            'calibration_rank': int(np.linalg.matrix_rank(centered[calibration])),
            'feature_dimension': centered.shape[1],
        }
    return aligned, diagnostics


# Class relationships
# -------------------
# Average example-pair similarities within each model. Class-centroid cosines
# would be a different statistic and would hide within-class spread. A diagonal
# cell excludes self-pairs; a cross-class cell uses the entire rectangular block.
# Counts describe contributing pairs, not independent statistical observations.


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


def compute_relations(metadata, arrays, layer=None):
    """Full-space class relationships, including retained-calibration centering."""
    layer = layer or metadata['layers'][-1]
    if layer not in metadata['layers']:
        raise ValueError(f'Unknown layer: {layer}')
    heldout = arrays['partitions'] == 'heldout'
    calibration = arrays['partitions'] == 'calibration'
    if not heldout.any():
        raise ValueError('At least one held-out example is required.')
    heldout_labels = arrays['labels'][heldout]
    classes = np.unique(heldout_labels)
    blocks, summaries, euclidean = [], [], []
    modes = ['raw', 'centered'] if calibration.any() else ['raw']
    means = {}
    for mode in modes:
        for model in metadata['models']:
            features = arrays[feature_key(model, layer)].astype(float)
            if mode == 'centered':
                features = features - features[calibration].mean(0)
            matrix, counts, zeros = class_cosines(features[heldout], heldout_labels, classes)
            means[mode, model] = matrix
            for i, first in enumerate(classes):
                for j, second in enumerate(classes):
                    value = matrix[i, j]
                    blocks.append(
                        {
                            'mode': mode,
                            'model': model,
                            'class_a': int(first),
                            'class_b': int(second),
                            'mean_cosine': float(value) if np.isfinite(value) else None,
                            'distinct_pairs': int(counts[i, j]),
                            'zero_vectors': zeros,
                        }
                    )
        retained_classes = classes != metadata['forget_class']
        pair_mask = np.triu(retained_classes[:, None] & retained_classes[None, :], 1)
        for first, second in itertools.combinations(metadata['models'], 2):
            values = (means[mode, second] - means[mode, first])[pair_mask]
            values = values[np.isfinite(values)]
            summaries.append(
                {
                    'mode': mode,
                    'pair': f'{second} minus {first}',
                    'retain_cross_mean_change': float(values.mean()) if len(values) else None,
                    'retain_cross_mean_absolute_change': float(np.abs(values).mean())
                    if len(values)
                    else None,
                    'positive_class_pairs': int((values > 0).sum()),
                    'valid_class_pairs': len(values),
                }
            )
    # Cosine changes weight class pairs equally. This separate Euclidean check
    # weights image pairs equally and retains activation magnitude information.
    # Use raw activations here. Pair distances are translation-invariant, while
    # activation norms retain dependence on the original feature-space origin.
    retain = heldout_labels != metadata['forget_class']
    for model in metadata['models']:
        features = arrays[feature_key(model, layer)][heldout][retain].astype(float)
        y = heldout_labels[retain]
        pairs = np.triu(y[:, None] != y[None, :], 1)
        squared_norms = np.sum(features * features, axis=1)
        distances = np.sqrt(
            np.maximum(
                squared_norms[:, None] + squared_norms[None, :] - 2 * features @ features.T, 0
            )
        )
        euclidean.append(
            {
                'model': model,
                'mean_cross_retain_distance': float(distances[pairs].mean())
                if pairs.any()
                else None,
                'mean_activation_norm': float(np.linalg.norm(features, axis=1).mean())
                if len(features)
                else None,
            }
        )
    return {
        'layer': layer,
        'classes': classes.tolist(),
        'modes': modes,
        'blocks': blocks,
        'changes': summaries,
        'euclidean': euclidean,
        'heldout_counts': {str(c): int((heldout_labels == c).sum()) for c in classes},
        'notes': [
            'Class cosines use all held-out examples; self-pairs and zero vectors are excluded.',
            'Centered cosines subtract each model retained-calibration mean; no alignment is fitted.',
            'Means can cancel; centered geometry constrains aggregates. Centering is not proof of a pure offset.',
            'Euclidean distances are scale-sensitive and pair-weighted; cosine summaries weight class pairs equally.',
            'Pair counts are not independent sample sizes. No confidence or causal claim.',
        ],
    }
