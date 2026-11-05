"""Class summaries of unreduced cosine relationships; no fitted alignment/PCA."""
import argparse
import csv
import itertools
from pathlib import Path

import numpy as np
from representation_analysis import feature_key, load_bundle, sha256, write_json


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


def render(bundle, output, layer=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    meta, arrays = load_bundle(bundle)
    output = Path(output)
    if output.exists():
        raise FileExistsError('Choose a fresh relations output directory.')
    layer = layer or meta['layers'][-1]
    if layer not in meta['layers']:
        raise ValueError(f'Unknown layer: {layer}')
    heldout = arrays['partitions'] == 'heldout'
    labels = arrays['labels'][heldout]
    if not len(labels):
        raise ValueError('Held-out examples are required.')
    classes = np.unique(labels)
    class_names = [meta['class_names'][int(c)] for c in classes]
    models = meta['models']
    matrices, counts, zeros = {}, {}, {}
    records = []
    for model in models:
        matrices[model], counts[model], zeros[model] = class_cosines(
            arrays[feature_key(model, layer)][heldout], labels, classes)
        for i, first in enumerate(classes):
            for j, second in enumerate(classes):
                value = matrices[model][i, j]
                records.append({'model': model, 'class_a': int(first), 'class_b': int(second),
                                'mean_cosine': float(value) if np.isfinite(value) else None,
                                'distinct_pairs': int(counts[model][i, j])})
    output.mkdir(parents=True)
    result = {'source': meta['source'], 'method': meta.get('method'), 'layer': layer,
              'forget_class': meta['forget_class'], 'models': models,
              'class_names': meta['class_names'], 'bundle_sha256': sha256(bundle),
              'script_sha256': sha256(__file__), 'zero_vectors': zeros,
              'heldout_counts': {str(c): int(np.sum(labels == c)) for c in classes},
              'notes': ['All held-out examples, original uncentered feature vectors; no alignment or projection.',
                        'Same-class means exclude self-pairs; class means can hide example-level variation.',
                        'Zero-vector pairs are excluded. Pair counts are not independent sample sizes.',
                        'Subtracted block means are descriptive and do not establish statistical significance.'],
              'blocks': records}
    write_json(output/'relations.json', result)
    with (output/'relations.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 130, 'savefig.dpi': 160})

    def heatmap(ax, values, title, low, high, cmap):
        im = ax.imshow(np.ma.masked_invalid(values), vmin=low, vmax=high, cmap=cmap)
        ax.set_xticks(range(len(classes)), class_names, rotation=60, ha='right')
        ax.set_yticks(range(len(classes)), class_names)
        ax.set(title=title, xlabel='Digit / class of second image', ylabel='Digit / class of first image')
        for i in range(len(classes)):
            for j in range(len(classes)):
                val = values[i, j]
                pos = (val-low)/(high-low) if np.isfinite(val) else .5
                color = 'white' if pos < .12 or pos > .88 else 'black'
                ax.text(j, i, f'{val:.2f}' if np.isfinite(val) else 'NA', ha='center', va='center',
                        color=color, fontsize=7)
        return im

    source_title = 'Synthetic control' if meta['source'] == 'synthetic' else 'Exploratory held-out representations'
    nonnegative = all(np.all(arrays[feature_key(m, layer)][heldout] >= 0) for m in models)
    low = 0 if nonnegative else -1
    fig, axes = plt.subplots(1, len(models), figsize=(5*len(models)+.6, 5.3), squeeze=False, layout='constrained')
    for ax, model in zip(axes[0], models):
        im = heatmap(ax, matrices[model], model, low, 1, 'YlOrRd' if nonnegative else 'coolwarm')
    fig.colorbar(im, ax=list(axes[0]), shrink=.7, label='Mean cosine of distinct example pairs')
    fig.suptitle(f'{source_title}: {layer}\nClass-level view of the original feature space; diagonal excludes self-pairs')
    fig.savefig(output/'similarity_by_class.png', bbox_inches='tight', pad_inches=.15)
    plt.close(fig)

    pairs = list(itertools.combinations(models, 2))
    differences = [matrices[b]-matrices[a] for a, b in pairs]
    finite = np.concatenate([d[np.isfinite(d)] for d in differences])
    limit = max(float(np.max(np.abs(finite))) if len(finite) else 0, .01)
    fig, axes = plt.subplots(1, len(pairs), figsize=(5*len(pairs)+.6, 5.3), squeeze=False, layout='constrained')
    for ax, (first, second), delta in zip(axes[0], pairs, differences):
        im = heatmap(ax, delta, f'{second} minus {first}', -limit, limit, 'coolwarm')
    fig.colorbar(im, ax=list(axes[0]), shrink=.7, label='Change in mean cosine (signed)')
    fig.suptitle(f'{source_title}: changes in {layer}\nRed = more similar within the second model; blue = less similar; white = little change')
    fig.savefig(output/'similarity_change.png', bbox_inches='tight', pad_inches=.15)
    plt.close(fig)

    forget_index = np.flatnonzero(classes == meta['forget_class'])
    if len(forget_index):
        idx = int(forget_index[0])
        fig, ax = plt.subplots(figsize=(10, 4.8), layout='constrained')
        for model in models:
            ax.plot(range(len(classes)), matrices[model][idx], marker='o', label=model)
        ax.set_xticks(range(len(classes)), class_names, rotation=35, ha='right')
        ax.set(xlabel='Class of comparison image', ylabel='Mean cosine similarity', ylim=(low-.02, 1.02),
               title=f'{source_title}: {class_names[idx]} compared with each class\nFull feature space; distinct pairs only; class labels summarize inputs')
        ax.grid(alpha=.2)
        ax.legend()
        fig.savefig(output/'forget_profile.png')
        plt.close(fig)

    report = ['# Another view of the representation', '',
              f"Source: {meta['source']}; layer: {layer}; forgotten class: {meta['forget_class']}.", '',
              'These views use the full activations with no fitted alignment or dimensionality reduction.', '',
              '## Reading the original similarity.png', '',
              'Each panel is one model. Rows and columns identify the same sequence of input images,',
              'ordered by true class, with up to 12 images per class. A cell compares those two images',
              'within that model, not one neuron or one pair of models. The scale is shared across panels.', '',
              'Cosine similarity is the dot product divided by the product of vector lengths.',
              '+1 means the same direction, 0 means perpendicular, and -1 means opposite directions.',
              'The self-comparison diagonal is 1 for nonzero features and provides no forgetting evidence.',
              'Raw ReLU features are nonnegative, so their cosine cannot be negative: mostly red panels',
              'are expected. The old scale included -1 for generality. Cosine ignores vector magnitude.', '',
              'A dark same-class square means examples of that class have similar activation directions.',
              'An off-diagonal square compares two classes. Compare the same cell across panels to see',
              'whether that relationship changes. The matrix is symmetric; mirrored cells repeat information.', '',
              '## Class summaries', '', '![Class summaries](similarity_by_class.png)', '',
              'Each cell now averages all held-out image pairs from its row and column classes.',
              'The diagonal compares different examples of the same class, excluding trivial self-matches.',
              'The raw ReLU view uses a shared 0-to-1 scale. Class averages can hide variation.', '',
              '## Changes between models', '', '![Differences](similarity_change.png)', '',
              'Each title specifies second model minus first model. Red means that the example pairs',
              'became more similar; blue means less similar. This color scale shows a change, not raw cosine.',
              'The scale is shared among panels within this figure, but may differ across run folders.', '',
              '## Forgotten-class relationships', '', '![Forgotten-class profile](forget_profile.png)', '',
              'This traces the forgotten-class row of the class summary. It can reveal whether the class',
              'stays coherent while its relationships to other classes change. These are geometry summaries,',
              'not class probabilities, evidence of memorization, or confidence intervals.', '']
    (output/'REPORT.md').write_text('\n'.join(report), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--layer')
    args = parser.parse_args()
    try:
        render(args.features, args.output_dir, args.layer)
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(f'Saved {args.output_dir / "REPORT.md"}')


if __name__ == '__main__':
    main()
