#!/usr/bin/env python3
"""
Clean Step 4: Self-Supervised Pretraining (SSL) — fixed argparse indentation and flag names.
"""
import os, math, time, json, random, argparse, pathlib
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import yaml
except Exception:
    yaml = None


# --------------------------
# Utilities
# --------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EMA:
    def __init__(self, model: nn.Module, momentum: float = 0.996):
        self.m = momentum
        self.shadow = {n: p.data.clone().detach() for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.m).add_(p.data, alpha=1.0 - self.m)


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x, lambd: float):
    return GradReverse.apply(x, lambd)


# --------------------------
# Dataset
# --------------------------
class EEGNPZDataset(Dataset):
    def __init__(self, roots: List[str], drop_artifacts=True, sample_rate=256, subjects_per_dataset=None,
                 subjects_allowlist=None):
        self.files = []
        tmp_files = []
        for root in roots:
            dname = os.path.basename(root)
            for p in pathlib.Path(root).rglob('*.npz'):
                try:
                    d = np.load(p, allow_pickle=True)
                    subj = str(d.get('subject', 'unknown'))
                except Exception:
                    subj = 'unknown'
                tmp_files.append((str(p), dname, subj))
        if len(tmp_files) == 0:
            raise AssertionError('No .npz files found.')
        if subjects_allowlist:
            allow = {ds: set(subjs) for ds, subjs in subjects_allowlist.items()}
            self.files = [(p, ds, s) for (p, ds, s) in tmp_files if ds in allow and (not allow[ds] or s in allow[ds])]
        elif subjects_per_dataset:
            picked = {}
            for p, ds, s in tmp_files:
                picked.setdefault(ds, set())
                if len(picked[ds]) < subjects_per_dataset or s in picked[ds]:
                    self.files.append((p, ds, s))
                    picked[ds].add(s)
        else:
            self.files = tmp_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, dataset_id, subject = self.files[idx]
        d = np.load(path, allow_pickle=True)
        x = d['x'].astype(np.float32)
        mask_missing = d['mask_missing'].astype(np.float32)
        return {'x_raw': x, 'mask_missing': mask_missing, 'dataset_id': dataset_id, 'subject': subject}


# --------------------------
# Model components (dummy encoders)
# --------------------------
class DummyEFMRawEncoder(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(19, 64, 7, padding=3), nn.GELU(),
            nn.Conv1d(64, 128, 7, padding=3), nn.GELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(128, out_dim)
        )

    def forward(self, x): return self.net(x)


class DummyEFMSpecEncoder(nn.Module):
    def __init__(self, in_ch=19, out_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, out_dim)
        )

    def forward(self, x): return self.net(x)


class DummyFusion(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.fc = nn.Linear(dim * 2, dim)

    def forward(self, r, s): return self.fc(torch.cat([r, s], dim=-1))


# --------------------------
# SSLConfig
# --------------------------
@dataclass
class SSLConfig:
    data_roots: List[str] = None
    subjects_per_dataset: Optional[int] = None
    subjects_allowlist_path: Optional[str] = None
    batch_size: int = 32
    num_workers: int = 8
    lr: float = 3e-4
    total_steps: int = 200000
    out_dir: str = 'runs/ssl/efm_ssl_base'


# --------------------------
# Train
# --------------------------
def train(cfg: SSLConfig):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(42)
    allow = None
    if cfg.subjects_allowlist_path and os.path.exists(cfg.subjects_allowlist_path):
        with open(cfg.subjects_allowlist_path) as f:
            allow = json.load(f)
    ds = EEGNPZDataset(cfg.data_roots, subjects_per_dataset=cfg.subjects_per_dataset, subjects_allowlist=allow)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    print(f"Dataset loaded with {len(ds)} samples.")


# --------------------------
# Argparse
# --------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default=None)
    p.add_argument('--data-roots', nargs='+', dest='data_roots', required=True)
    p.add_argument('--limit-subjects-per-dataset', dest='limit_subjects_per_dataset', type=int, default=None)
    p.add_argument('--subjects-allowlist', dest='subjects_allowlist', type=str, default=None)
    p.add_argument('--batch-size', dest='batch_size', type=int, default=32)
    p.add_argument('--out', dest='out', type=str, default='runs/ssl/efm_ssl_base')
    args = p.parse_args()
    cfg = SSLConfig()
    cfg.data_roots = args.data_roots
    cfg.subjects_per_dataset = args.limit_subjects_per_dataset
    cfg.subjects_allowlist_path = args.subjects_allowlist
    cfg.batch_size = args.batch_size
    cfg.out_dir = args.out
    return cfg


if __name__ == '__main__':
    cfg = parse_args()
    print('Config:', cfg)
    train(cfg)
