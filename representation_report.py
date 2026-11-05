"""Render per-run representation measurements without repeating the method guide."""

import os
from pathlib import Path


def format_metric(value):
    """Keep undefined measurements visible instead of presenting them as zero."""
    return 'undefined' if value is None else f'{value:.4f}'


def method_reference(output):
    """Link in-repository reports to the guide without embedding machine paths."""
    repository = Path(__file__).resolve().parent
    try:
        output.resolve().relative_to(repository)
    except ValueError:
        return 'Metric definitions and limitations: `docs/representation-analysis.md`.'
    relative = os.path.relpath(repository / 'docs/representation-analysis.md', output)
    return f'[Metric definitions and limitations]({Path(relative).as_posix()}).'


def write_report(result, diagnostics, output):
    """Write measured tables and figure captions for one extracted model trio.

    The method guide owns definitions and general interpretation rules. This
    report keeps the information needed to read this particular run: sample
    counts, measured values, extraction notes and calibration limitations.
    It is derived from metrics.json and never changes the numerical evidence.
    """
    output = Path(output)
    metadata = result['metadata']
    relations = result['relations']
    neighbors = 10 if 10 in result['ks'] else result['ks'][0]
    lines = [
        '# Representation comparison',
        '',
        f'Source: **{metadata["source"]}**; seed: **{metadata.get("seed", "unspecified")}**; '
        f'forget class: **{metadata["forget_class"]}**.',
        '',
        f'Method: {metadata.get("method", "unspecified")}.',
        '',
        method_reference(output),
        '',
        *[f'- {note}' for note in metadata.get('notes', [])],
        '',
        '## Behavior on held-out inputs',
        '',
        '| Model | Group | n | Accuracy | NLL |',
        '|---|---|---:|---:|---:|',
    ]
    for row in result['behavior']:
        if row['partition'] == 'heldout' and row['group'] in ('forget', 'retain'):
            lines.append(
                f'| {row["model"]} | {row["group"]} | {row["n"]} | '
                f'{row["accuracy"]:.2%} | {format_metric(row["nll"])} |'
            )

    lines += [
        '',
        '## Representation metrics',
        '',
        'Full-space comparisons, before visualization alignment. Higher similarity',
        'does not establish better unlearning. Neighbor candidates include all classes.',
        '',
        f'| Layer | Group | Pair | n | CKA | Debiased CKA | Shuffled CKA | Shared {neighbors}-NN | Prediction disagreement |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|',
    ]
    for row in result['comparisons']:
        if row['partition'] == 'heldout' and row['group'] in ('forget', 'retain'):
            lines.append(
                f'| {row["layer"]} | {row["group"]} | {row["pair"]} | {row["n"]} | '
                f'{format_metric(row["cka"])} | {format_metric(row["cka_debiased"])} | '
                f'{format_metric(row["cka_permuted_mean"])} | '
                f'{format_metric(row[f"knn_{neighbors}"])} | {row["prediction_disagreement"]:.2%} |'
            )
    if (output / 'representation_metrics.png').exists():
        lines += ['', '![Metrics across layers](representation_metrics.png)']

    lines += [
        '',
        '## Class relationships',
        '',
        f'Cosine and PCA views use **{relations["layer"]}**. Diagonal cells compare',
        'distinct images of the same class; self-matches are excluded.',
        '',
        '![Class-average cosine](similarity_by_class.png)',
        '',
        '## Changes and sensitivity to centering',
        '',
        'Column titles specify the subtraction order. Red indicates increased cosine;',
        'blue indicates decreased cosine. All panels share one signed scale.',
        'Centered coordinates subtract the retained-calibration mean for each model.',
        '',
        '![Raw and centered cosine changes](similarity_change.png)',
        '',
        '| Coordinates | Model difference | Mean change among retained classes | Mean absolute change | Positive / valid pairs |',
        '|---|---|---:|---:|---:|',
    ]
    for row in relations['changes']:
        lines.append(
            f'| {row["mode"]} | {row["pair"]} | '
            f'{format_metric(row["retain_cross_mean_change"])} | '
            f'{format_metric(row["retain_cross_mean_absolute_change"])} | '
            f'{row["positive_class_pairs"]} / {row["valid_class_pairs"]} |'
        )
    lines += [
        '',
        '| Model | Mean Euclidean distance between retained classes | Mean retained activation norm |',
        '|---|---:|---:|',
    ]
    for row in relations['euclidean']:
        lines.append(
            f'| {row["model"]} | {format_metric(row["mean_cross_retain_distance"])} | '
            f'{format_metric(row["mean_activation_norm"])} |'
        )

    lines += ['', '## PCA overview', '']
    if (output / 'pca.png').exists():
        variance = sum(diagnostics['pca_calibration_explained_variance_ratio'])
        lines += [
            '![Shared aligned PCA](pca.png)',
            '',
            f'Two axes explain **{variance:.1%}** of calibration variance.',
            'Alignment and PCA use retained calibration examples only. A similar 2D',
            'position does not establish full-space equivalence.',
        ]
        # Rank deficiency is a property of this run, rather than a universal
        # warning. Other general visualization caveats live in the method guide.
        if any(
            fit['calibration_rank'] < fit['feature_dimension']
            for fit in diagnostics.get('alignment', {}).values()
        ):
            lines += [
                '',
                'Calibration is rank-deficient: rotations outside its span are underdetermined.',
            ]
    else:
        lines += [f'PCA skipped: {diagnostics.get("pca_skipped", "unavailable")}.']

    lines += [
        '',
        '## Numerical evidence',
        '',
        '[metrics.json](metrics.json) contains all measurements, quality checks and provenance.',
        'Detailed tables: [comparisons](comparisons.csv), [behavior](behavior.csv),',
        '[class relations](relations.csv). [Plot diagnostics](plot_diagnostics.json)',
        'record calibration rank and alignment residuals.',
        '',
    ]
    (output / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
