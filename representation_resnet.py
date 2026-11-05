"""Export matched CIFAR-10 / ResNet-18 activations for representation analysis.

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

# Configure deterministic CUDA matrix operations before torch is imported.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


LAYERS = ['layer1', 'layer2', 'layer3', 'layer4']
CLASS_NAMES = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck',
]
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


# Checkpoint architecture and extraction
# --------------------------------------
# Recreate the CIFAR adaptation exactly: a small stride-one first convolution,
# no initial max-pool, and ten outputs. Loading with strict=True below prevents
# accidentally interpreting another architecture as this experiment's model.


def create_resnet18():
    """Construct the FP32 architecture used by the repository's checkpoints."""
    import torch.nn as nn
    from torchvision.models import resnet18

    model = resnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model.float()


def collect_activations(model, loader, device):
    """Collect spatially pooled residual-stage activations in loader order.

    Evaluation/inference mode keeps BatchNorm buffers and weights unchanged.
    Hooks average spatial dimensions after each residual stage; layer4 then
    matches the pre-classifier vector. Hooks are removed even on failure so
    later calls cannot append duplicate features. Only one model uses the
    device at a time, and exported arrays return to CPU memory immediately.
    """
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
            for images, _ in loader:
                logits = model(images.to(device))
                probabilities.append(logits.softmax(1).cpu().numpy())
    finally:
        for handle in handles:
            handle.remove()
        model.cpu()
    return {k: np.concatenate(v) for k, v in features.items()}, np.concatenate(probabilities)


def validate_splits(split, total=50000):
    """Require a complete, disjoint partition of the original training dataset."""
    if set(split) != {'train', 'search', 'select'}:
        raise ValueError('Expected repository train/search/select split.json.')
    ids = sum(split.values(), [])
    if len(ids) != len(set(ids)) or set(ids) != set(range(total)):
        raise ValueError(
            'Training/search/selection splits must partition CIFAR-10 training data.'
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--root',
        type=Path,
        help='Existing run with original.pt, unlearned.pt, retrained.pt, split.json.',
    )
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--per-class', type=int, default=100)
    parser.add_argument('--calibration-per-class', type=int, default=100)
    parser.add_argument('--probe-seed', type=int, default=2026)
    parser.add_argument('--evaluation', choices=['search', 'test'], default='search')
    parser.add_argument(
        '--download', action='store_true', help='Explicitly allow downloading CIFAR-10.'
    )
    parser.add_argument(
        '--retrained-repeat',
        type=Path,
        action='append',
        default=[],
        help='Optional independent retained-only checkpoint; adjacent split.json must match.',
    )
    parser.add_argument(
        '--smoke',
        action='store_true',
        help='Random ResNets/images to check extraction, not scientific results.',
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a fresh feature output path.')
    if (
        min(args.per_class, args.calibration_per_class, args.batch_size) < 1
        or args.workers < 0
    ):
        parser.error('Counts must be positive and workers nonnegative.')
    if not args.smoke and args.root is None:
        parser.error('--root is required unless --smoke is used.')
    if args.smoke and args.retrained_repeat:
        parser.error('--retrained-repeat cannot be combined with --smoke.')
    try:
        import torch
        import torchvision
        from torch.utils.data import DataLoader, TensorDataset, Subset, ConcatDataset
        from torchvision import datasets, transforms
    except ImportError as e:
        parser.error(
            f'ResNet extraction requires a matching torch/torchvision installation: {e}'
        )
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; use --device cpu for CPU extraction.')
    # Keep model and probe randomness separate. Extra retrained models must
    # share the saved data split even when their initialization seeds differ.
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    rng = np.random.default_rng(args.probe_seed)
    models = list(BASE_MODELS)
    paths, source_files = {}, {}
    if args.smoke:
        labels = np.tile(np.arange(10), 6)
        partitions = np.array(['train'] * 20 + ['heldout'] * 20 + ['calibration'] * 20)
        keep = ~((partitions == 'calibration') & (labels == 0))
        images = torch.randn(60, 3, 32, 32)[keep]
        labels, partitions = labels[keep], partitions[keep]
        dataset = TensorDataset(images, torch.tensor(labels))
        ids = np.array([f'synthetic:{i}' for i in np.flatnonzero(keep)])
        seed, forget_class = 0, 0
    else:
        split_path = args.root / 'split.json'
        split = json.loads(split_path.read_text(encoding='utf-8'))
        validate_splits(split)
        config_path = args.root / 'train-config.json'
        config = json.loads(config_path.read_text(encoding='utf-8'))
        seed, forget_class = config['seed'], config['forget_class']
        source_files = {
            'split_sha256': sha256(split_path),
            'train_config_sha256': sha256(config_path),
        }
        # Inference uses saved training normalization with no augmentation.
        # Prefix IDs by dataset split so train and test index zero remain
        # distinct samples. Calibration never includes the forgotten class.
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
        )
        train_dataset = datasets.CIFAR10(
            args.data, train=True, download=args.download, transform=transform
        )
        train_labels = np.asarray(train_dataset.targets)
        train_ids = balanced_ids(split['train'], train_labels, args.per_class, rng)
        calibration_ids = balanced_ids(
            split['select'], train_labels, args.calibration_per_class, rng
        )
        calibration_ids = calibration_ids[train_labels[calibration_ids] != forget_class]
        if args.evaluation == 'test':
            evaluation_dataset = datasets.CIFAR10(
                args.data, train=False, download=args.download, transform=transform
            )
            evaluation_ids = balanced_ids(
                range(len(evaluation_dataset)),
                np.asarray(evaluation_dataset.targets),
                args.per_class,
                rng,
            )
            prefix = 'cifar10:test:'
        else:
            evaluation_dataset = train_dataset
            evaluation_ids = balanced_ids(split['search'], train_labels, args.per_class, rng)
            prefix = 'cifar10:train:'
        dataset = ConcatDataset(
            [
                Subset(train_dataset, train_ids.tolist()),
                Subset(evaluation_dataset, evaluation_ids.tolist()),
                Subset(train_dataset, calibration_ids.tolist()),
            ]
        )
        labels = np.concatenate(
            [
                train_labels[train_ids],
                np.asarray(evaluation_dataset.targets)[evaluation_ids],
                train_labels[calibration_ids],
            ]
        )
        ids = np.array(
            [f'cifar10:train:{i}' for i in train_ids]
            + [f'{prefix}{i}' for i in evaluation_ids]
            + [f'cifar10:train:{i}' for i in calibration_ids]
        )
        partitions = np.repeat(
            ['train', 'heldout', 'calibration'],
            [len(train_ids), len(evaluation_ids), len(calibration_ids)],
        )
        paths = {name: args.root / f'{name}.pt' for name in models}
        for i, path in enumerate(args.retrained_repeat, 1):
            repeat_split = json.loads((path.parent / 'split.json').read_text(encoding='utf-8'))
            if repeat_split != split:
                parser.error(
                    'Retrain replicates must use the identical data split; ordinary different-seed runs change it.'
                )
            name = f'retrained_repeat{i}'
            models.append(name)
            paths[name] = path
    # A single, nonshuffled loader fixes the sample axis across every model.
    # Additional checkpoints are evaluated sequentially rather than batching
    # models together, keeping device memory bounded by one model's inference.
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )
    arrays = {'sample_ids': ids, 'labels': labels, 'partitions': partitions}
    checkpoints = {}
    for i, name in enumerate(models):
        torch.manual_seed(i)
        model = create_resnet18()
        if not args.smoke:
            checkpoint = torch.load(paths[name], map_location='cpu', weights_only=True)
            if checkpoint['forget_class'] != forget_class or (
                name in BASE_MODELS and checkpoint['seed'] != seed
            ):
                parser.error(f'Checkpoint seed/forget class mismatch: {name}')
            model.load_state_dict(checkpoint['state'], strict=True)
            checkpoints[name] = {
                'filename': paths[name].name,
                'sha256': sha256(paths[name]),
                'seed': checkpoint['seed'],
            }
        print(f'Extracting {name}: {len(dataset)} examples on {args.device}', flush=True)
        features, probabilities = collect_activations(model, loader, args.device)
        for layer, values in features.items():
            arrays[feature_key(name, layer)] = values
        arrays[f'probabilities__{name}'] = probabilities
        del model
    metadata = {
        'schema_version': 1,
        'models': models,
        'layers': LAYERS,
        'class_names': CLASS_NAMES,
        'forget_class': forget_class,
        'seed': seed,
        'probe_seed': args.probe_seed,
        'source': 'synthetic' if args.smoke else 'repository_resnet18_cifar10',
        'method': 'random model smoke control'
        if args.smoke
        else 'incorrect-label suppression baseline',
        'evaluation_split': args.evaluation,
        'checkpoints': checkpoints,
        **source_files,
        'preprocessing': {'mean': MEAN, 'std': STD, 'augmentation': False},
        'pooling': 'spatial mean after each residual stage; layer4 is the pre-head vector',
        'notes': [
            'Synthetic controls are not MU evidence.'
            if args.smoke
            else 'Existing search/selection splits have been used previously; this is exploratory.',
            'Extra retrained checkpoints must be independently trained on the same retained data.',
        ],
        'environment': {
            'python': platform.python_version(),
            'numpy': np.__version__,
            'torch': torch.__version__,
            'torchvision': torchvision.__version__,
            'device': args.device,
        },
    }
    save_bundle(args.output, metadata, arrays)
    print(
        f'Saved {args.output}. Run representation_analysis.py analyze to compute its metrics.'
    )


if __name__ == '__main__':
    main()
