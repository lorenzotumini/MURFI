"""Export CIFAR-10 ResNet-18 activations on the desktop; analyze on any CPU.

Only the repository's unmodified checkpoint schema/preprocessing is accepted.
No training or fault search is performed. See docs/representation-analysis.md.
"""
import argparse
import json
import os
from pathlib import Path
import platform

import numpy as np
from representation_analysis import BASE_MODELS, balanced_ids, feature_key, save_bundle, sha256

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


LAYERS = ['layer1', 'layer2', 'layer3', 'layer4']
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def new_model():
    import torch.nn as nn
    from torchvision.models import resnet18
    model = resnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model.float()


def collect(model, loader, device):
    import torch
    features = {name: [] for name in LAYERS}
    handles, probabilities = [], []
    def hook(name):
        def capture(module, inputs, value):
            features[name].append(value.detach().mean(dim=(2, 3)).cpu().numpy())
        return capture
    for name in LAYERS:
        handles.append(getattr(model, name).register_forward_hook(hook(name)))
    try:
        model.to(device).eval()
        with torch.inference_mode():
            for x, _ in loader:
                logits = model(x.to(device))
                probabilities.append(logits.softmax(1).cpu().numpy())
    finally:
        for handle in handles:
            handle.remove()
        model.cpu()
    return {k: np.concatenate(v) for k, v in features.items()}, np.concatenate(probabilities)


def validate_splits(split, total=50000):
    if set(split) != {'train', 'search', 'select'}:
        raise ValueError('Expected repository train/search/select split.json.')
    ids = sum(split.values(), [])
    if len(ids) != len(set(ids)) or set(ids) != set(range(total)):
        raise ValueError('Training/search/selection splits must partition CIFAR-10 training data.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, help='Existing run with original.pt, unlearned.pt, retrained.pt, split.json.')
    p.add_argument('--data', type=Path, default=Path('data'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--per-class', type=int, default=100)
    p.add_argument('--calibration-per-class', type=int, default=100)
    p.add_argument('--probe-seed', type=int, default=2026)
    p.add_argument('--evaluation', choices=['search', 'test'], default='search')
    p.add_argument('--download', action='store_true', help='Explicitly allow downloading CIFAR-10.')
    p.add_argument('--retrained-repeat', type=Path, action='append', default=[],
                   help='Optional independent retained-only checkpoint; adjacent split.json must match.')
    p.add_argument('--smoke', action='store_true', help='Random ResNets/images to check extraction, not scientific results.')
    a = p.parse_args()
    if a.output.exists():
        p.error('Choose a fresh feature output path.')
    if min(a.per_class, a.calibration_per_class, a.batch_size) < 1 or a.workers < 0:
        p.error('Counts must be positive and workers nonnegative.')
    if not a.smoke and a.root is None:
        p.error('--root is required unless --smoke is used.')
    if a.smoke and a.retrained_repeat:
        p.error('--retrained-repeat cannot be combined with --smoke.')
    try:
        import torch
        import torchvision
        from torch.utils.data import DataLoader, TensorDataset, Subset, ConcatDataset
        from torchvision import datasets, transforms
    except ImportError as e:
        p.error(f'Desktop extraction requires a matching torch/torchvision installation: {e}')
    if a.device.startswith('cuda') and not torch.cuda.is_available():
        p.error('CUDA is unavailable; use --device cpu for the random-model smoke check.')
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    rng = np.random.default_rng(a.probe_seed)
    models = list(BASE_MODELS)
    paths, source_files = {}, {}
    if a.smoke:
        labels = np.tile(np.arange(10), 6)
        parts = np.array(['train']*20 + ['heldout']*20 + ['calibration']*20)
        keep = ~((parts == 'calibration') & (labels == 0))
        xx = torch.randn(60, 3, 32, 32)[keep]
        labels, parts = labels[keep], parts[keep]
        dataset = TensorDataset(xx, torch.tensor(labels))
        ids = np.array([f'synthetic:{i}' for i in np.flatnonzero(keep)])
        seed, forget_class = 0, 0
    else:
        split_path = a.root/'split.json'
        split = json.loads(split_path.read_text(encoding='utf-8'))
        validate_splits(split)
        config_path = a.root/'train-config.json'
        config = json.loads(config_path.read_text(encoding='utf-8'))
        seed, forget_class = config['seed'], config['forget_class']
        source_files = {'split_sha256': sha256(split_path), 'train_config_sha256': sha256(config_path)}
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
        train_ds = datasets.CIFAR10(a.data, train=True, download=a.download, transform=transform)
        yy = np.asarray(train_ds.targets)
        train_ids = balanced_ids(split['train'], yy, a.per_class, rng)
        cal_ids = balanced_ids(split['select'], yy, a.calibration_per_class, rng)
        cal_ids = cal_ids[yy[cal_ids] != forget_class]
        if a.evaluation == 'test':
            eval_ds = datasets.CIFAR10(a.data, train=False, download=a.download, transform=transform)
            eval_ids = balanced_ids(range(len(eval_ds)), np.asarray(eval_ds.targets), a.per_class, rng)
            prefix = 'cifar10:test:'
        else:
            eval_ds = train_ds
            eval_ids = balanced_ids(split['search'], yy, a.per_class, rng)
            prefix = 'cifar10:train:'
        dataset = ConcatDataset([Subset(train_ds, train_ids.tolist()), Subset(eval_ds, eval_ids.tolist()),
                                 Subset(train_ds, cal_ids.tolist())])
        labels = np.concatenate([yy[train_ids], np.asarray(eval_ds.targets)[eval_ids], yy[cal_ids]])
        ids = np.array([f'cifar10:train:{i}' for i in train_ids] + [f'{prefix}{i}' for i in eval_ids]
                       + [f'cifar10:train:{i}' for i in cal_ids])
        parts = np.repeat(['train', 'heldout', 'calibration'], [len(train_ids), len(eval_ids), len(cal_ids)])
        paths = {name: a.root/f'{name}.pt' for name in models}
        for i, path in enumerate(a.retrained_repeat, 1):
            repeat_split = json.loads((path.parent/'split.json').read_text(encoding='utf-8'))
            if repeat_split != split:
                p.error('Retrain replicates must use the identical data split; ordinary different-seed runs change it.')
            name = f'retrained_repeat{i}'
            models.append(name)
            paths[name] = path
    loader = DataLoader(dataset, batch_size=a.batch_size, shuffle=False, num_workers=a.workers)
    arrays = {'sample_ids': ids, 'labels': labels, 'partitions': parts}
    checkpoints = {}
    for i, name in enumerate(models):
        torch.manual_seed(i)
        model = new_model()
        if not a.smoke:
            ck = torch.load(paths[name], map_location='cpu', weights_only=True)
            if ck['forget_class'] != forget_class or (name in BASE_MODELS and ck['seed'] != seed):
                p.error(f'Checkpoint seed/forget class mismatch: {name}')
            model.load_state_dict(ck['state'], strict=True)
            checkpoints[name] = {'filename': paths[name].name, 'sha256': sha256(paths[name]), 'seed': ck['seed']}
        print(f'Extracting {name}: {len(dataset)} examples on {a.device}', flush=True)
        features, probabilities = collect(model, loader, a.device)
        for layer, values in features.items():
            arrays[feature_key(name, layer)] = values
        arrays[f'probabilities__{name}'] = probabilities
        del model
    meta = {'schema_version': 1, 'models': models, 'layers': LAYERS, 'class_names': CLASS_NAMES,
            'forget_class': forget_class, 'seed': seed, 'probe_seed': a.probe_seed,
            'source': 'synthetic' if a.smoke else 'repository_resnet18_cifar10',
            'method': 'random model smoke control' if a.smoke else 'incorrect-label suppression baseline',
            'evaluation_split': a.evaluation, 'checkpoints': checkpoints, **source_files,
            'preprocessing': {'mean': MEAN, 'std': STD, 'augmentation': False},
            'pooling': 'spatial mean after each residual stage; layer4 is the pre-head vector',
            'notes': ['Synthetic controls are not MU evidence.' if a.smoke else
                      'Existing search/selection splits have been used previously; this is exploratory.',
                      'Extra retrained checkpoints must be independently trained on the same retained data.'],
            'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                            'torch': torch.__version__, 'torchvision': torchvision.__version__, 'device': a.device}}
    save_bundle(a.output, meta, arrays)
    print(f'Saved {a.output}. Analyze this bundle on either machine.')


if __name__ == '__main__':
    main()
