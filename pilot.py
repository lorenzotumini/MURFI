"""Benign CIFAR-10 class-unlearning robustness pilot. See README.md.

Only modifies owned model tensors in software; no physical fault injection.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import struct
import time


def flipped_float(value, bit):
    word = struct.unpack('<I', struct.pack('<f', value))[0]
    return struct.unpack('<f', struct.pack('<I', word ^ (1 << bit)))[0]


def word(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')


def select_beam(nodes, width):
    """Keep distinct bit sets, preferring recall then discovery loss."""
    ordered = sorted(nodes, key=lambda n: (-n['validation']['forget_accuracy'],
                                           n['objective'],
                                           -n['validation']['retain_accuracy']))
    selected, seen = [], set()
    for node in ordered:
        bits = frozenset((t['parameter'], t['index'], t['bit']) for t in node['trace'])
        if bits not in seen:
            selected.append(node)
            seen.add(bits)
        if len(selected) == width:
            break
    return selected


def bit_change_scores(weights, gradients, limit):
    """Largest beneficial first-order change from a valid FP32 storage flip."""
    import torch
    w, g = weights.detach().flatten(), gradients.detach().flatten()
    scores = torch.zeros_like(w)
    words = w.contiguous().view(torch.int32)
    for bit in range(32):
        mask = (1 << bit) if bit < 31 else -(1 << 31)
        changed = words.bitwise_xor(mask).view(torch.float32)
        benefit = -g * (changed - w)
        valid = torch.isfinite(changed) & (changed.abs() <= limit)
        scores = torch.maximum(scores, torch.where(valid, benefit, 0.0))
    return scores


def cached_batches(dl, device):
    return [(x.to(device), y.to(device)) for x, y in dl]


def main():
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, Dataset, Subset
    import torchvision
    from torchvision import datasets, transforms

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['train', 'retrain', 'search', 'evaluate', 'all', 'smoke'], default='all',
                   help='retrain reuses completed original/unlearned checkpoints, restarts retraining, then searches')
    p.add_argument('--root', type=Path, default=Path('runs/seed0-class0'))
    p.add_argument('--data', type=Path, default=Path('data'))
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--forget-class', type=int, choices=range(10), default=0)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--unlearn-epochs', type=int, default=15)
    p.add_argument('--unlearn-lr', type=float, default=0.001)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--budget', type=int, default=32)
    p.add_argument('--random-trials', type=int, default=20)
    p.add_argument('--candidate-weights', type=int, default=128)
    p.add_argument('--candidate-tests', type=int, default=8)
    p.add_argument('--probe-per-class', type=int, default=32)
    p.add_argument('--exclude-head', action='store_true')
    p.add_argument('--retain-drop', type=float, default=0.02)
    p.add_argument('--fpr-rise', type=float, default=0.01)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--algorithm', choices=['greedy', 'beam'], default='greedy')
    p.add_argument('--beam-width', type=int, default=3)
    p.add_argument('--branch-factor', type=int, default=3)
    p.add_argument('--forget-weight', type=float, default=1.0)
    p.add_argument('--validation-only', action='store_true',
                   help='Evaluate on selection data; do not open the official test split')
    p.add_argument('--search-name', help='Unique output subdirectory within the run root')
    p.add_argument('--cache-eval', action='store_true')
    p.add_argument('--ranking', choices=['magnitude', 'bit-change'], default='magnitude')
    p.add_argument('--screen-per-class', type=int, default=0,
                   help='Use additional discovery examples for cheap utility screening')
    p.add_argument('--screen-slack', type=float, default=0.02)
    p.add_argument('--validation-candidates', type=int, default=0,
                   help='Maximum full validations per expansion; zero means unlimited')
    p.add_argument('--max-candidate-evals', type=int, default=0)
    p.add_argument('--max-full-validations', type=int, default=0)
    p.add_argument('--max-search-seconds', type=float, default=0)
    a = p.parse_args()
    frozen_config = None
    if a.stage == 'evaluate':
        if not a.search_name or Path(a.search_name).name != a.search_name or a.search_name in ['.', '..']:
            p.error('--stage evaluate requires the existing --search-name directory.')
        frozen_config = json.loads((a.root / a.search_name / 'config.json').read_text())
        if frozen_config['seed'] != a.seed or frozen_config['forget_class'] != a.forget_class:
            p.error('Frozen search seed/class differs from the requested evaluation.')
        a.exclude_head = frozen_config['exclude_head']
        a.random_trials = frozen_config['random_trials']
        a.validation_only = False
    synthetic = a.stage == 'smoke' or (frozen_config is not None and frozen_config['stage'] == 'smoke')
    if min(a.epochs, a.unlearn_epochs, a.batch_size, a.candidate_weights,
           a.candidate_tests, a.probe_per_class, a.random_trials, a.budget,
           a.beam_width, a.branch_factor) < 1:
        p.error('Epoch counts, sizes, trials and budget must be positive.')
    if not math.isfinite(a.forget_weight) or a.forget_weight <= 0:
        p.error('--forget-weight must be finite and positive.')
    if min(a.screen_per_class, a.validation_candidates, a.max_candidate_evals,
           a.max_full_validations, a.max_search_seconds, a.screen_slack) < 0:
        p.error('Screen sizes, evaluation limits, time limits and slack cannot be negative.')
    if not all(math.isfinite(x) for x in [a.max_search_seconds, a.screen_slack]):
        p.error('Time limit and screen slack must be finite.')
    if a.search_name is not None and (Path(a.search_name).name != a.search_name
                                      or a.search_name in ['', '.', '..']):
        p.error('--search-name must be a single directory name.')
    if a.stage == 'smoke':
        a.epochs = a.unlearn_epochs = 1
        a.budget = 2
        a.random_trials = 2
        a.candidate_weights = 8
        a.candidate_tests = 2
    a.root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(a.threads)
    device = torch.device(a.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    # Request repeatable algorithms where available; exact cross-GPU equality is not promised.
    torch.use_deterministic_algorithms(True, warn_only=True)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(a.seed)

    def model_new():
        if synthetic:
            return nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10)).to(device)
        m = torchvision.models.resnet18(weights=None, num_classes=10)
        m.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        return m.to(device).float()

    def loader(ds, shuffle=False):
        return DataLoader(ds, batch_size=a.batch_size, shuffle=shuffle,
                          num_workers=a.workers, pin_memory=device.type == 'cuda')

    class Relabeled(Dataset):
        def __init__(self, ds):
            self.ds = ds

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, i):
            x, y = self.ds[i]
            # Random incorrect labels, drawn afresh; a simple RL-style baseline.
            if y == a.forget_class:
                y = (y + int(torch.randint(1, 10, ()).item())) % 10
            return x, y

    if synthetic:
        class Toy(Dataset):
            def __init__(self, n, seed):
                g = torch.Generator().manual_seed(seed)
                self.x = torch.randn(n, 3, 32, 32, generator=g)
                self.targets = [i % 10 for i in range(n)]

            def __len__(self):
                return len(self.targets)

            def __getitem__(self, i):
                return self.x[i], self.targets[i]
        train_ds, eval_ds, test_ds = Toy(100, 4), Toy(100, 4), Toy(40, 8)
        train_ids, search_ids, select_ids = list(range(60)), list(range(60, 80)), list(range(80, 100))
    else:
        norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        aug = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                  transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm])
        plain = transforms.Compose([transforms.ToTensor(), norm])
        train_ds = datasets.CIFAR10(a.data, train=True, download=True, transform=aug)
        eval_ds = datasets.CIFAR10(a.data, train=True, download=False, transform=plain)
        # Do not load the official test split until selection has finished.
        rng = random.Random(a.seed)
        train_ids, search_ids, select_ids = [], [], []
        for c in range(10):
            ids = [i for i, y in enumerate(eval_ds.targets) if y == c]
            rng.shuffle(ids)
            train_ids.extend(ids[:4000])
            search_ids.extend(ids[4000:4500])
            select_ids.extend(ids[4500:])
    assert not (set(train_ids) & set(search_ids) or set(train_ids) & set(select_ids)
                or set(search_ids) & set(select_ids))
    retain_ids = [i for i in train_ids if train_ds.targets[i] != a.forget_class]
    full_train = loader(Subset(train_ds, train_ids), True)
    retain_train = loader(Subset(train_ds, retain_ids), True)
    relabel_train = loader(Relabeled(Subset(train_ds, train_ids)), True)
    select_loader = loader(Subset(eval_ds, select_ids))
    probe_ids = []
    for c in range(10):
        probe_ids.extend([i for i in search_ids if eval_ds.targets[i] == c][:a.probe_per_class])
    xb, yb = zip(*(eval_ds[i] for i in probe_ids))
    xb, yb = torch.stack(xb).to(device), torch.tensor(yb, device=device)
    forget_mask = yb == a.forget_class
    if a.cache_eval:
        select_loader = cached_batches(select_loader, device)
    screen_loader = None
    if a.screen_per_class:
        screen_ids = []
        for c in range(10):
            available = [i for i in search_ids if eval_ds.targets[i] == c and i not in probe_ids]
            # Tiny synthetic smoke sets have no extra discovery examples.
            if not available and a.stage == 'smoke':
                available = [i for i in search_ids if eval_ds.targets[i] == c]
            if len(available) < a.screen_per_class and a.stage != 'smoke':
                p.error('Not enough discovery examples for the requested independent screening subset.')
            screen_ids.extend(available[:a.screen_per_class])
        screen_loader = cached_batches(loader(Subset(eval_ds, screen_ids)), device)

    @torch.no_grad()
    def metrics(m, dl):
        m.eval()
        cm = torch.zeros(10, 10, dtype=torch.int64, device=device)
        valid = torch.ones((), dtype=torch.bool, device=device)
        for x, y in dl:
            z = m(x.to(device))
            valid &= torch.isfinite(z).all()
            pred = z.argmax(1)
            cm += torch.bincount(y.to(device) * 10 + pred, minlength=100).reshape(10, 10)
        if not bool(valid):
            return {'valid': False}
        cm = cm.cpu()
        per = cm.diag().float() / cm.sum(1).clamp_min(1)
        c = a.forget_class
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum()) - tp
        others = [j for j in range(10) if j != c]
        return {'valid': True, 'forget_accuracy': float(per[c]),
                'retain_accuracy': float(per[others].mean()),
                'forget_precision': tp / max(tp + fp, 1),
                'forget_false_positive_rate': fp / max(int(cm[others].sum()), 1),
                'per_class_accuracy': per.tolist(), 'confusion_matrix': cm.tolist()}

    def clone_state(m):
        return {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}

    def save_checkpoint(m, name, met):
        torch.save({'state': clone_state(m), 'validation': met, 'seed': a.seed,
                    'forget_class': a.forget_class}, a.root / (name + '.pt'))

    def load_checkpoint(m, name):
        ck = torch.load(a.root / (name + '.pt'), map_location='cpu', weights_only=True)
        if ck['seed'] != a.seed or ck['forget_class'] != a.forget_class:
            raise ValueError('Checkpoint configuration differs. Use a separate --root per seed/class.')
        m.load_state_dict(ck['state'])
        return ck

    def train_model(m, dl, epochs, lr, name, original=None):
        opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        best = -math.inf
        history = []
        for epoch in range(epochs):
            start = time.monotonic()
            m.train()
            for x, y in dl:
                opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(m(x.to(device)), y.to(device))
                loss.backward()
                opt.step()
            scheduler.step()
            met = metrics(m, select_loader)
            history.append(met)
            if not met['valid']:
                raise RuntimeError('Nonfinite training output.')
            if original is None:
                score = (met['retain_accuracy'] if name == 'retrained' else
                         sum(met['per_class_accuracy']) / 10)
            else:
                viable = (met['forget_accuracy'] <= 0.10 and
                          met['retain_accuracy'] >= original['retain_accuracy'] - 0.05)
                score = met['retain_accuracy'] if viable else -math.inf
            if score > best:
                best = score
                save_checkpoint(m, name, met)
            print(f'{name} epoch {epoch+1}/{epochs}: '
                  f'forget={met["forget_accuracy"]:.3f} retain={met["retain_accuracy"]:.3f} '
                  f'{time.monotonic()-start:.1f}s', flush=True)
        write_json(a.root / (name + '-training.json'), history)
        return best > -math.inf

    if a.stage in ['all', 'train', 'smoke']:
        # Refuse stale-run reuse; files describe one coherent experiment.
        if (a.root / 'original.pt').exists():
            raise FileExistsError('Use a new --root for a new training run; --stage search reuses checkpoints.')
        write_json(a.root / 'split.json', {'train': train_ids, 'search': search_ids, 'select': select_ids})
        write_json(a.root / 'train-config.json', {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()})
        torch.manual_seed(a.seed)
        m = model_new()
        train_model(m, full_train, a.epochs, 0.1, 'original')
        original = load_checkpoint(m, 'original')['validation']
        if a.stage != 'smoke' and min(original['forget_accuracy'], original['retain_accuracy']) < 0.70:
            raise RuntimeError('Original validation accuracy below 70%; improve training before fault experiments.')
        viable = train_model(m, relabel_train, a.unlearn_epochs, a.unlearn_lr, 'unlearned', original)
        if not viable:
            if a.stage == 'smoke':
                save_checkpoint(m, 'unlearned', metrics(m, select_loader))
            else:
                raise RuntimeError('No viable unlearning checkpoint: see unlearned-training.json. '
                                   'Adjust unlearning on validation only and use a new run directory.')
    if a.stage == 'retrain':
        saved_config = json.loads((a.root / 'train-config.json').read_text())
        for key in ['seed', 'forget_class', 'epochs', 'batch_size']:
            if saved_config[key] != getattr(a, key):
                raise ValueError(f'Retraining configuration differs: {key}.')
        saved_split = json.loads((a.root / 'split.json').read_text())
        if saved_split != {'train': train_ids, 'search': search_ids, 'select': select_ids}:
            raise ValueError('Saved split differs from reconstructed split.')
        if (a.root / 'retrained-training.json').exists() or (a.root / 'search').exists() or (a.root / 'search-no-head').exists():
            raise FileExistsError('Retraining or search already completed/started; use --stage search or a fresh run copy.')
        for name, epochs in [('original', saved_config['epochs']),
                             ('unlearned', saved_config['unlearn_epochs'])]:
            if len(json.loads((a.root / (name + '-training.json')).read_text())) != epochs:
                raise ValueError(f'{name} training is incomplete.')
        m = model_new()
        original = load_checkpoint(m, 'original')['validation']
        unlearned = load_checkpoint(m, 'unlearned')['validation']
        del m
        if (not original['valid'] or not unlearned['valid']
                or min(original['forget_accuracy'], original['retain_accuracy']) < 0.70
                or unlearned['forget_accuracy'] > 0.10
                or unlearned['retain_accuracy'] < original['retain_accuracy'] - 0.05):
            raise ValueError('Saved checkpoints do not pass the training/unlearning gates.')
        partial = a.root / 'retrained.pt'
        if partial.exists():
            partial.rename(a.root / f'retrained-interrupted-{time.time_ns()}.pt')
        write_json(a.root / 'retrain-config.json', {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()})

    if a.stage in ['all', 'train', 'smoke', 'retrain']:
        # Same initialization seed, same 10-output architecture, no forgotten-class training examples.
        torch.manual_seed(a.seed)
        m = model_new()
        train_model(m, retain_train, a.epochs, 0.1, 'retrained')
        del m
        if a.stage == 'train':
            return

    if a.stage != 'smoke':
        saved_split = json.loads((a.root / 'split.json').read_text(encoding='utf-8'))
        if saved_split != {'train': train_ids, 'search': search_ids, 'select': select_ids}:
            raise ValueError('Saved split differs from reconstructed split.')
    search_dir = a.root / (a.search_name or ('search-no-head' if a.exclude_head else 'search'))
    if a.stage == 'evaluate':
        if (search_dir / 'frozen-test-manifest.json').exists():
            raise FileExistsError('Frozen test evaluation already started; inspect its manifest/results.')
        paths = [a.root / (n + '.pt') for n in ['original', 'unlearned', 'retrained']]
        paths += [search_dir / (n + '-selected.pt') for n in ['unlearned', 'retrained']]
        paths += [search_dir / 'config.json', a.root / 'split.json', Path(__file__)]
        write_json(search_dir / 'frozen-test-manifest.json', {
            'created_unix': time.time(), 'seed': a.seed, 'forget_class': a.forget_class,
            'selection': 'Reuse selected checkpoints exactly; no search or selection during evaluation.',
            'source_config': frozen_config,
            'sha256': {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
            'prior_test_results_in_run': (a.root / 'search' / 'results.json').exists()})
    elif (search_dir / 'config.json').exists():
        raise FileExistsError('Search results exist; use a fresh run copy to change search settings.')
    search_dir.mkdir(parents=True, exist_ok=True)
    if a.stage != 'evaluate':
        write_json(search_dir / 'config.json', {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()})
    env = {'torch': torch.__version__, 'torchvision': torchvision.__version__,
           'python': platform.python_version(), 'cuda_runtime': torch.version.cuda,
           'device': str(device), 'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
           'smoke_only': synthetic}
    write_json(search_dir / ('test-environment.json' if a.stage == 'evaluate' else 'environment.json'), env)

    def eligible(m):
        return [(n, v) for n, v in m.named_parameters()
                if v.ndim >= 2 and not (a.exclude_head and n.startswith('fc.'))]

    def apply_bit(param, index, bit):
        # Toggle the actual FP32 storage. No float assignment, optimizer, or model cast.
        mask = (1 << bit) if bit < 31 else -(1 << 31)
        with torch.no_grad():
            param.view(torch.int32).flatten()[index].bitwise_xor_(mask)

    def objective(m):
        z = m(xb)
        return (a.forget_weight * F.cross_entropy(z[forget_mask], yb[forget_mask]) +
                F.cross_entropy(z[~forget_mask], yb[~forget_mask]))

    def acceptable(met, base):
        return (met['valid'] and met['retain_accuracy'] >= base['retain_accuracy'] - a.retain_drop
                and met['forget_false_positive_rate'] <= base['forget_false_positive_rate'] + a.fpr_rise)

    def limits(m):
        return {n: max(float(v.detach().abs().max()) * 16, 1e-8) for n, v in eligible(m)}

    def audit(before, m, trace):
        # Check every state entry: only logged single-bit changes are allowed, including buffers.
        expected = {n: t.clone() for n, t in before.items()}
        for t in trace:
            param = expected[t['parameter']]
            mask = (1 << t['bit']) if t['bit'] < 31 else -(1 << 31)
            param.view(torch.int32).flatten()[t['index']].bitwise_xor_(mask)
        now = clone_state(m)
        if not all(torch.equal(expected[n], now[n]) for n in expected):
            raise AssertionError('Unexpected model mutation during fault search.')

    def budget_reason(stats, candidate=False, full=False):
        if a.max_search_seconds and time.monotonic() - stats['started'] >= a.max_search_seconds:
            return 'time_limit'
        if candidate and a.max_candidate_evals and stats['candidate_evals'] >= a.max_candidate_evals:
            return 'candidate_evaluation_limit'
        if full and a.max_full_validations and stats['full_validations'] >= a.max_full_validations:
            return 'full_validation_limit'
        return None

    def extensions(m, params, lim, trace, base, screen_base, branches, stats):
        reason = budget_reason(stats, candidate=True, full=True)
        if reason:
            stats['limit'] = reason
            return []
        used = {(t['parameter'], t['index'], t['bit']) for t in trace}
        m.eval()
        m.zero_grad(set_to_none=True)
        objective(m).backward()
        pool = []
        # Global ranking of scalar weights by sensitivity. No original weights are consulted.
        for n, v in params.items():
            g = v.grad.detach().flatten()
            scores = (bit_change_scores(v, g, lim[n]) if a.ranking == 'bit-change' else
                      g.abs() * v.detach().flatten().abs().clamp_min(1e-8))
            ranked, inds = scores.topk(min(a.candidate_weights, v.numel()))
            vals = v.detach().flatten()[inds].cpu().tolist()
            grads = g[inds].cpu().tolist()
            for score, i, w, grad in zip(ranked.cpu().tolist(), inds.cpu().tolist(), vals, grads):
                pool.append((score, n, i, w, grad))
        pool.sort(reverse=True)
        candidates = []
        for _, n, i, w, grad in pool[:a.candidate_weights]:
            for bit in range(32):
                if (n, i, bit) in used:
                    continue
                new = flipped_float(w, bit)
                if math.isfinite(new) and abs(new) <= lim[n]:
                    predicted = grad * (new - w)
                    if predicted < 0:
                        candidates.append((predicted, n, i, bit, w))
        candidates.sort()
        # Measure candidate effects on the discovery split, then validate the shortlist.
        with torch.no_grad():
            current = float(objective(m))
            measured = []
            for _, n, i, bit, old in candidates[:a.candidate_tests]:
                reason = budget_reason(stats, candidate=True)
                if reason:
                    stats['limit'] = reason
                    break
                apply_bit(params[n], i, bit)
                value = float(objective(m))
                apply_bit(params[n], i, bit)
                stats['candidate_evals'] += 1
                if math.isfinite(value) and value < current:
                    measured.append((value, n, i, bit, old))
            measured.sort()
            if screen_loader is not None:
                screened = []
                for value, n, i, bit, old in measured:
                    if budget_reason(stats):
                        stats['limit'] = 'time_limit'
                        break
                    apply_bit(params[n], i, bit)
                    screen = metrics(m, screen_loader)
                    apply_bit(params[n], i, bit)
                    stats['screen_evals'] += 1
                    if (not screen['valid']
                            or screen['retain_accuracy'] < screen_base['retain_accuracy'] - a.retain_drop - a.screen_slack
                            or screen['forget_false_positive_rate'] > screen_base['forget_false_positive_rate'] + a.fpr_rise + a.screen_slack):
                        continue
                    utility = ((current - value) / max(abs(current), 1e-8)
                               - max(0, screen_base['retain_accuracy'] - screen['retain_accuracy'])
                               - max(0, screen['forget_false_positive_rate'] - screen_base['forget_false_positive_rate']))
                    screened.append((-utility, value, n, i, bit, old))
                measured = [item[1:] for item in sorted(screened)]
            if a.validation_candidates:
                measured = measured[:a.validation_candidates]
            children = []
            for value, n, i, bit, old in measured:
                reason = budget_reason(stats, full=True)
                if reason:
                    stats['limit'] = reason
                    break
                apply_bit(params[n], i, bit)
                val = metrics(m, select_loader)
                stats['full_validations'] += 1
                if acceptable(val, base):
                    flip = {'parameter': n, 'index': i, 'bit': bit, 'before': old,
                            'after': float(params[n].flatten()[i]), 'probe_objective': value}
                    children.append({'trace': trace + [flip], 'validation': val,
                                     'objective': value})
                apply_bit(params[n], i, bit)
                if len(children) >= branches:
                    break
        return children

    def targeted(name):
        m = model_new()
        load_checkpoint(m, name)
        before = clone_state(m)
        base = metrics(m, select_loader)
        screen_base = metrics(m, screen_loader) if screen_loader is not None else None
        lim, params = limits(m), dict(eligible(m))
        m.eval()
        with torch.no_grad():
            initial_objective = float(objective(m))
        beam = [{'trace': [], 'validation': base, 'objective': initial_objective}]
        history = [{'bits': 0, 'validation': base, 'objective': initial_objective}]
        best_state, best_trace, best_val = before, [], base
        width = a.beam_width if a.algorithm == 'beam' else 1
        branches = a.branch_factor if a.algorithm == 'beam' else 1
        stop_reason = 'budget_exhausted'
        stats = {'started': time.monotonic(), 'candidate_evals': 0,
                 'screen_evals': 0, 'full_validations': 0, 'limit': None}
        for step in range(a.budget):
            children = []
            for parent in beam:
                m.load_state_dict(before)
                for flip in parent['trace']:
                    apply_bit(params[flip['parameter']], flip['index'], flip['bit'])
                children.extend(extensions(m, params, lim, parent['trace'], base, screen_base, branches, stats))
                if stats['limit']:
                    break
            if not children:
                stop_reason = (stats['limit'] or budget_reason(stats, candidate=True, full=True)
                               or 'no_acceptable_extensions')
                break
            beam = select_beam(children, width)
            # Every acceptable child is considered for final selection, even if pruned.
            for child in children:
                if child['validation']['forget_accuracy'] > best_val['forget_accuracy']:
                    best_trace = copy.deepcopy(child['trace'])
                    best_val = child['validation']
                    m.load_state_dict(before)
                    for flip in best_trace:
                        apply_bit(params[flip['parameter']], flip['index'], flip['bit'])
                    audit(before, m, best_trace)
                    best_state = clone_state(m)
            leader = beam[0]
            history.append({'bits': step + 1, 'validation': leader['validation'],
                            'objective': leader['objective'], 'trace': leader['trace'],
                            'beam_size': len(beam), 'acceptable_children': len(children),
                            'elapsed_seconds': time.monotonic() - stats['started'],
                            'candidate_evals': stats['candidate_evals'],
                            'full_validations': stats['full_validations']})
            write_json(search_dir / (name + '-search.json'), history)
            print(f'{name} {a.algorithm} depth={step+1} beam={len(beam)} '
                  f'best_bits={len(best_trace)} forget={best_val["forget_accuracy"]:.3f} '
                  f'retain={best_val["retain_accuracy"]:.3f} '
                  f'elapsed={time.monotonic()-stats["started"]:.1f}s', flush=True)
            if stats['limit']:
                stop_reason = stats['limit']
                break
        write_json(search_dir / (name + '-search.json'), history)
        write_json(search_dir / (name + '-diagnostics.json'), {
            'algorithm': a.algorithm, 'stop_reason': stop_reason,
            'depth_reached': len(history) - 1, 'selected_bits': len(best_trace),
            'elapsed_seconds': time.monotonic() - stats['started'],
            'candidate_evals': stats['candidate_evals'], 'screen_evals': stats['screen_evals'],
            'full_validations': stats['full_validations'],
            'search_incomplete': stop_reason.endswith('_limit')})
        m.load_state_dict(best_state)
        audit(before, m, best_trace)
        torch.save({'state': best_state, 'trace': best_trace, 'validation': best_val},
                   search_dir / (name + '-selected.pt'))
        return m, best_trace, best_val

    def random_faults(name, count, trial):
        m = model_new()
        load_checkpoint(m, name)
        before, lim = clone_state(m), limits(m)
        entries = eligible(m)
        total = sum(v.numel() for _, v in entries)
        rng = random.Random(a.seed * 10000 + trial + 12345)
        trace, used = [], set()
        for _ in range(count):
            for attempt in range(10000):
                pos = rng.randrange(total)
                for n, v in entries:
                    if pos < v.numel():
                        break
                    pos -= v.numel()
                bit = rng.randrange(32)
                if (n, pos, bit) in used:
                    continue
                old = float(v.detach().flatten()[pos])
                new = flipped_float(old, bit)
                if math.isfinite(new) and abs(new) <= lim[n]:
                    apply_bit(v, pos, bit)
                    used.add((n, pos, bit))
                    trace.append({'parameter': n, 'index': pos, 'bit': bit,
                                  'before': old, 'after': new})
                    break
            else:
                raise RuntimeError('Could not sample a valid random bit.')
        audit(before, m, trace)
        return m, trace

    def frozen_model(name):
        m = model_new()
        load_checkpoint(m, name)
        before = clone_state(m)
        ck = torch.load(search_dir / (name + '-selected.pt'), map_location='cpu', weights_only=True)
        allowed = dict(eligible(m))
        if any(t['parameter'] not in allowed for t in ck['trace']):
            raise ValueError('Frozen trace violates the saved parameter eligibility.')
        m.load_state_dict(ck['state'])
        audit(before, m, ck['trace'])
        return m, ck['trace'], ck['validation']

    search_or_load = frozen_model if a.stage == 'evaluate' else targeted
    unlearned, un_trace, un_val = search_or_load('unlearned')
    retrained, rt_trace, rt_val = search_or_load('retrained')
    # Only now open the official test data. Search/selection above never see it.
    if a.validation_only:
        test_loader = select_loader
    else:
        if not synthetic:
            test_ds = datasets.CIFAR10(a.data, train=False, download=True, transform=plain)
        test_loader = loader(test_ds)
        if a.cache_eval:
            test_loader = cached_batches(test_loader, device)
    evaluation_key = 'validation' if a.validation_only else 'test'
    result = {'environment': env, 'evaluation_split': 'selection' if a.validation_only else 'test',
              'targeted': {}, 'baseline': {}, 'random': [],
              'note': 'Software fault simulation; ideal arbitrary-address bit access. No physical feasibility claim.'}
    for name, m, trace, val in [('unlearned', unlearned, un_trace, un_val),
                               ('retrained', retrained, rt_trace, rt_val)]:
        result['targeted'][name] = {'bits': len(trace), 'validation': val,
                                    evaluation_key: metrics(m, test_loader), 'trace': trace}
    del unlearned, retrained
    for name in ['original', 'unlearned', 'retrained']:
        m = model_new()
        load_checkpoint(m, name)
        result['baseline'][name] = metrics(m, test_loader)
        del m
    # Same bit budget and finite/layer-magnitude constraints as selected targeted model.
    for trial in range(a.random_trials):
        m, trace = random_faults('unlearned', len(un_trace), trial)
        result['random'].append({'trial': trial, 'bits': len(trace),
                                  evaluation_key: metrics(m, test_loader), 'trace': trace})
        del m
    output_path = search_dir / ('frozen-test-results.json' if a.stage == 'evaluate' else 'results.json')
    write_json(output_path, result)
    print(f'Saved {output_path}', flush=True)


if __name__ == '__main__':
    main()
