"""The unified report's figures; all statistical calculations live in analysis."""
import itertools
import numpy as np


def make_plots(meta, arrays, rows, output, ks, relations):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from representation_analysis import feature_key, align_to_reference

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.dpi': 130, 'savefig.dpi': 160})
    models, layers = meta['models'], meta['layers']
    pairs = list(itertools.combinations(models, 2))
    heldout = arrays['partitions'] == 'heldout'
    cal = arrays['partitions'] == 'calibration'
    labels = arrays['labels']
    retained = labels != meta['forget_class']
    last = relations['layer']
    title = 'SYNTHETIC CONTROL' if meta['source'] == 'synthetic' else 'Exploratory representation comparison'
    k = 10 if 10 in ks else ks[0]

    # Single-layer runs use the report table instead of redundant single-point plots.
    if len(layers) > 1:
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout='constrained')
        for col, group in enumerate(['forget', 'retain']):
            for row, key in enumerate(['cka_debiased', f'knn_{k}']):
                ax = axes[row, col]
                for first, second in pairs:
                    pair = f'{first}--{second}'
                    values = []
                    for layer in layers:
                        matches = [r for r in rows if r['partition'] == 'heldout' and r['group'] == group
                                   and r['pair'] == pair and r['layer'] == layer]
                        value = matches[0].get(key) if matches else None
                        values.append(np.nan if value is None else value)
                    ax.plot(layers, values, marker='o', label=f'{first} / {second}')
                ax.set(title=f'{group.capitalize()}: {"debiased CKA" if row == 0 else f"shared {k}-NN"}',
                       ylim=(-1.05 if row == 0 else -.05, 1.05))
                ax.grid(alpha=.2)
                ax.tick_params(axis='x', rotation=15)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(title + '\nFull-space representation metrics across layers')
        fig.savefig(output/'representation_metrics.png', bbox_inches='tight')
        plt.close(fig)

    classes = relations['classes']
    names = [meta['class_names'][c] for c in classes]
    positions = {c: i for i, c in enumerate(classes)}
    matrices = {(mode, model): np.full((len(classes), len(classes)), np.nan)
                for mode in relations['modes'] for model in models}
    for r in relations['blocks']:
        value = r['mean_cosine']
        matrices[r['mode'], r['model']][positions[r['class_a']], positions[r['class_b']]] = (
            np.nan if value is None else value)

    def heatmap(ax, values, caption, low, high, cmap):
        im = ax.imshow(np.ma.masked_invalid(values), vmin=low, vmax=high, cmap=cmap)
        ax.set_xticks(range(len(classes)), names, rotation=60, ha='right')
        ax.set_yticks(range(len(classes)), names)
        ax.set(title=caption, xlabel='Class of second image', ylabel='Class of first image')
        if len(classes) <= 12:
            for i in range(len(classes)):
                for j in range(len(classes)):
                    v = values[i, j]
                    pos = (v-low)/(high-low) if np.isfinite(v) else .5
                    ax.text(j, i, f'{v:.2f}' if np.isfinite(v) else 'NA', ha='center', va='center',
                            color='white' if pos < .12 or pos > .88 else 'black', fontsize=7)
        return im

    nonnegative = all(np.all(arrays[feature_key(m, last)][heldout] >= 0) for m in models)
    fig, axes = plt.subplots(1, len(models), figsize=(5*len(models)+.6, 5.3), squeeze=False, layout='constrained')
    for ax, model in zip(axes[0], models):
        im = heatmap(ax, matrices['raw', model], model, 0 if nonnegative else -1, 1,
                     'YlOrRd' if nonnegative else 'coolwarm')
    fig.colorbar(im, ax=list(axes[0]), shrink=.7, label='Mean cosine of distinct image pairs')
    fig.suptitle(f'{title}: {last}\nRaw class relationships; self-comparisons excluded')
    fig.savefig(output/'similarity_by_class.png', bbox_inches='tight', pad_inches=.15)
    plt.close(fig)

    modes = relations['modes']
    differences = {(mode, first, second): matrices[mode, second]-matrices[mode, first]
                   for mode in modes for first, second in pairs}
    finite = np.concatenate([d[np.isfinite(d)] for d in differences.values()])
    limit = max(float(np.max(np.abs(finite))) if len(finite) else 0, .01)
    fig, axes = plt.subplots(len(modes), len(pairs), figsize=(5*len(pairs)+.6, 5.2*len(modes)),
                             squeeze=False, layout='constrained')
    for row, mode in enumerate(modes):
        for col, (first, second) in enumerate(pairs):
            im = heatmap(axes[row, col], differences[mode, first, second],
                         f'{mode.capitalize()}: {second} minus {first}', -limit, limit, 'coolwarm')
    fig.colorbar(im, ax=list(axes.ravel()), shrink=.7, label='Change in mean cosine (shared scale)')
    fig.suptitle(f'{title}: {last}\nRed = increased similarity; blue = decreased. Centered rows subtract retained calibration means.')
    fig.savefig(output/'similarity_change.png', bbox_inches='tight', pad_inches=.15)
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
