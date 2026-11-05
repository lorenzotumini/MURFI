"""Static figures for representation_analysis; inference-free and CPU-only."""
import numpy as np


def make_plots(meta, arrays, rows, output, ks):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from representation_analysis import feature_key, align_to_reference

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.dpi': 130, 'savefig.dpi': 160})
    models, layers = meta['models'], meta['layers']
    pairs = list(dict.fromkeys(r['pair'] for r in rows))
    heldout = arrays['partitions'] == 'heldout'
    cal = arrays['partitions'] == 'calibration'
    retained = arrays['labels'] != meta['forget_class']
    selected_rows = [r for r in rows if r['partition'] == 'heldout']
    title = 'SYNTHETIC CONTROL' if meta['source'] == 'synthetic' else 'Exploratory representation comparison'

    def series(pair, group, key):
        values = []
        for layer in layers:
            found = [r for r in selected_rows if r['pair'] == pair and r['group'] == group and r['layer'] == layer]
            val = found[0].get(key) if found else None
            values.append(np.nan if val is None else val)
        return values

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), squeeze=False, layout='constrained')
    for col, group in enumerate(['forget', 'retain']):
        for row, key in enumerate(['cka', 'cka_debiased']):
            ax = axes[row, col]
            for pair in pairs:
                line, = ax.plot(layers, series(pair, group, key), marker='o', label=pair.replace('--', ' / '))
                if key == 'cka':
                    ax.plot(layers, series(pair, group, 'cka_permuted_mean'), ':', marker='x', color=line.get_color(), alpha=.6)
            ax.set_title(f'{group.capitalize()} examples: {"linear CKA" if row == 0 else "debiased CKA"}')
            ax.set_ylabel('Similarity')
            ax.grid(alpha=.2)
            ax.tick_params(axis='x', rotation=15)
            # Comparable panels; do not magnify small differences by autoscaling.
            ax.set_ylim(-.05 if row == 0 else -1.05, 1.05)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(title + '\nHeld-out inputs; dotted lines = shuffled-row CKA reference')
    fig.savefig(output/'layerwise.png')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    k = 10 if 10 in ks else ks[0]
    for ax, group in zip(axes, ['forget', 'retain']):
        for pair in pairs:
            ax.plot(layers, series(pair, group, f'knn_{k}'), marker='o', label=pair.replace('--', ' / '))
        n = int(heldout.sum())
        if n > k:
            ax.axhline(k/(n-1), linestyle=':', color='grey', label='Uniform-neighbor chance reference')
        ax.set(title=f'{group.capitalize()} anchors', ylabel=f'Shared {k}-NN fraction', ylim=(-.03, 1.03))
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Local neighborhoods in the original feature spaces\nCandidate pool: all held-out classes; final projection is not used')
    fig.savefig(output/'neighbors.png')
    plt.close(fig)

    last = layers[-1]
    labels = arrays['labels']
    display = np.concatenate([np.flatnonzero(heldout & (labels == c))[:12] for c in np.unique(labels)])
    if not len(display):
        raise ValueError('Plotting requires held-out examples.')
    fig, axes = plt.subplots(1, len(models), figsize=(4*len(models), 4), squeeze=False, layout='constrained')
    boundaries = np.flatnonzero(np.diff(labels[display])) + .5
    for ax, name in zip(axes[0], models):
        h = arrays[feature_key(name, last)][display].astype(float)
        norms = np.linalg.norm(h, axis=1, keepdims=True)
        z = h / np.maximum(norms, 1e-30)
        im = ax.imshow(z @ z.T, vmin=-1, vmax=1, cmap='coolwarm', interpolation='nearest')
        for boundary in boundaries:
            ax.axhline(boundary, color='black', linewidth=.3)
            ax.axvline(boundary, color='black', linewidth=.3)
        ax.set(title=name, xlabel='Same examples, ordered by class', ylabel='Example')
    fig.colorbar(im, ax=list(axes[0]), shrink=.65, label='Cosine similarity')
    fig.suptitle(f'{title}\n{last}: identical example order in every matrix')
    fig.savefig(output/'similarity.png')
    plt.close(fig)

    diagnostics = {'layer': last, 'reference': 'original', 'calibration_examples': int(cal.sum()),
                   'notes': ['Orthogonal maps and common PCA fitted only on retained calibration rows.',
                             'No labels are used to fit PCA or rotations; labels color the display.',
                             'A 2D projection can hide changes; numerical metrics use full feature spaces.']}
    try:
        features = {name: arrays[feature_key(name, last)] for name in models}
        aligned, fit = align_to_reference(features, cal)
        diagnostics['alignment'] = fit
        if any(v['calibration_rank'] < v['feature_dimension'] for v in fit.values()):
            diagnostics['notes'].append('Calibration is rank-deficient: rotations outside its span are underdetermined.')
        reference = aligned['original']
        for name in models:
            mask = heldout & retained
            denom = np.linalg.norm(reference[mask])
            fit[name]['heldout_retain_relative_residual'] = (
                float(np.linalg.norm(aligned[name][mask]-reference[mask])/denom) if denom else None)
        stack = np.concatenate([h[cal] for h in aligned.values()])
        mean = stack.mean(0)
        _, singular, vt = np.linalg.svd(stack-mean, full_matrices=False)
        if len(singular) < 2 or np.sum(singular**2) == 0:
            raise ValueError('Insufficient nonzero variance for 2D PCA.')
        explained = singular[:2]**2 / np.sum(singular**2)
        diagnostics['pca_calibration_explained_variance_ratio'] = explained.tolist()
        fig, axes = plt.subplots(1, len(models), figsize=(4*len(models), 4.6),
                                 squeeze=False, sharex=True, sharey=True, layout='constrained')
        for ax, name in zip(axes[0], models):
            xy = (aligned[name][heldout]-mean) @ vt[:2].T
            yy = labels[heldout]
            for c in np.unique(yy):
                is_forget = c == meta['forget_class']
                ax.scatter(*xy[yy == c].T, s=35 if is_forget else 12,
                           marker='X' if is_forget else 'o', alpha=.85 if is_forget else .5,
                           color=plt.get_cmap('tab10')(int(c) % 10),
                           label=meta['class_names'][int(c)] + (' (forget)' if is_forget else ''))
            ax.set(title=name, xlabel=f'PC1 ({explained[0]:.1%} calibration variance)',
                   ylabel=f'PC2 ({explained[1]:.1%})')
        axes[0, -1].legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.02, 1))
        fig.suptitle(f'{title}\n{last}: one common PCA after retained-only alignment; held-out inputs')
        fig.savefig(output/'pca.png')
        plt.close(fig)
    except ValueError as exc:
        diagnostics['pca_skipped'] = str(exc)
    return diagnostics
