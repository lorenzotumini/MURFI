"""Benign CIFAR-10 class-unlearning robustness pilot. See README.md.

Only modifies owned model tensors in software; no physical fault injection.
"""
import argparse
import copy
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


def main():
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, Dataset, Subset
    import torchvision
    from torchvision import datasets, transforms

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['train', 'search', 'all', 'smoke'], default='all')
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
    a = p.parse_args()
    if min(a.epochs, a.unlearn_epochs, a.batch_size, a.candidate_weights,
           a.candidate_tests, a.probe_per_class, a.random_trials, a.budget) < 1:
        p.error('Epoch counts, sizes, trials and budget must be positive.')
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
        if a.stage == 'smoke':
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

    if a.stage == 'smoke':
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

    @torch.no_grad()
    def metrics(m, dl):
        m.eval()
        cm = torch.zeros(10, 10, dtype=torch.int64)
        invalid = False
        for x, y in dl:
            z = m(x.to(device))
            if not torch.isfinite(z).all():
                invalid = True
                break
            pred = z.argmax(1).cpu()
            cm += torch.bincount(y * 10 + pred, minlength=100).reshape(10, 10)
        if invalid:
            return {'valid': False}
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
    search_dir = a.root / ('search-no-head' if a.exclude_head else 'search')
    if (search_dir / 'config.json').exists():
        raise FileExistsError('Search results exist; use a fresh run copy to change search settings.')
    search_dir.mkdir(parents=True, exist_ok=True)
    write_json(search_dir / 'config.json', {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()})
    env = {'torch': torch.__version__, 'torchvision': torchvision.__version__,
           'python': platform.python_version(), 'cuda_runtime': torch.version.cuda,
           'device': str(device), 'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
           'smoke_only': a.stage == 'smoke'}
    write_json(search_dir / 'environment.json', env)

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
        return (F.cross_entropy(z[forget_mask], yb[forget_mask]) +
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

    def targeted(name):
        m = model_new()
        ck = load_checkpoint(m, name)
        before = clone_state(m)
        base = metrics(m, select_loader)
        lim = limits(m)
        params = dict(eligible(m))
        used, trace, history = set(), [], [{'bits': 0, 'validation': base}]
        best_state, best_trace, best_val = before, [], base
        for step in range(a.budget):
            m.eval()
            m.zero_grad(set_to_none=True)
            objective(m).backward()
            pool = []
            # Global ranking of scalar weights by sensitivity. No original weights are consulted.
            for n, v in params.items():
                g = v.grad.detach().flatten()
                scores = g.abs() * v.detach().flatten().abs().clamp_min(1e-8)
                _, inds = scores.topk(min(a.candidate_weights, v.numel()))
                vals = v.detach().flatten()[inds].cpu().tolist()
                grads = g[inds].cpu().tolist()
                for i, w, grad in zip(inds.cpu().tolist(), vals, grads):
                    pool.append((abs(grad) * max(abs(w), 1e-8), n, i, w, grad))
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
                    apply_bit(params[n], i, bit)
                    value = float(objective(m))
                    apply_bit(params[n], i, bit)
                    if math.isfinite(value) and value < current:
                        measured.append((value, n, i, bit, old))
                measured.sort()
                accepted = None
                for value, n, i, bit, old in measured:
                    apply_bit(params[n], i, bit)
                    val = metrics(m, select_loader)
                    if acceptable(val, base):
                        accepted = {'parameter': n, 'index': i, 'bit': bit, 'before': old,
                                    'after': float(params[n].flatten()[i]), 'probe_objective': value}
                        break
                    apply_bit(params[n], i, bit)
            if accepted is None:
                break
            trace.append(accepted)
            used.add((accepted['parameter'], accepted['index'], accepted['bit']))
            history.append({'bits': len(trace), 'validation': val, 'flip': accepted})
            # Select k using validation only; a smaller budget wins ties.
            if val['forget_accuracy'] > best_val['forget_accuracy']:
                best_state, best_trace, best_val = clone_state(m), copy.deepcopy(trace), val
            print(f'{name} target bits={len(trace)} forget={val["forget_accuracy"]:.3f} '
                  f'retain={val["retain_accuracy"]:.3f}', flush=True)
        audit(before, m, trace)
        write_json(search_dir / (name + '-search.json'), history)
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

    unlearned, un_trace, un_val = targeted('unlearned')
    retrained, rt_trace, rt_val = targeted('retrained')
    # Only now open the official test data. Search/selection above never see it.
    if a.stage != 'smoke':
        test_ds = datasets.CIFAR10(a.data, train=False, download=True, transform=plain)
    test_loader = loader(test_ds)
    result = {'environment': env, 'targeted': {}, 'baseline': {}, 'random': [],
              'note': 'Software fault simulation; ideal arbitrary-address bit access. No physical feasibility claim.'}
    for name, m, trace, val in [('unlearned', unlearned, un_trace, un_val),
                               ('retrained', retrained, rt_trace, rt_val)]:
        result['targeted'][name] = {'bits': len(trace), 'validation': val,
                                    'test': metrics(m, test_loader), 'trace': trace}
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
                                  'test': metrics(m, test_loader), 'trace': trace})
        del m
    write_json(search_dir / 'results.json', result)
    print(f'Saved {search_dir / "results.json"}', flush=True)


if __name__ == '__main__':
    main()
