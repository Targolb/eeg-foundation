#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 4 — Self-Supervised Pretraining (SSL) for EFM
--------------------------------------------------
Objectives combined:
  • MEM proxy (latent regression)
  • Multi-view contrastive (raw <-> spectro)
  • Temporal order / jigsaw
  • Dataset invariance via GRL
  • Subject-prototype alignment
  • (Optional) subject-level SupCon

Upgrades:
  (1) Adaptive weights: --adaptive-weights {none,uncertainty,cosine}
  (2) Cross-subject contrast: --loss-w-subj (SupCon over same-subject samples)
  (3) Seizure-aware scaling (minutes_to_seizure/phase if present in .npz)
  (4) Hardened saves, NaN guards, device-safe EMA
  (5) Auto-import real encoders from efm/models/models_transformer.py (fallback to dummy)

CLI: hyphenated flags, e.g. --data-roots, --batch-size
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


# ======================================================
# Utils
# ======================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6):
    denom = x.norm(dim=dim, keepdim=True).clamp_min(eps)
    return x / denom


def safe_save(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        torch.save(obj, path)
    except Exception as e:
        # Fallback directory in case of transient FS issues
        alt_dir = "/workspace/tmp_ckpts"
        try:
            os.makedirs(alt_dir, exist_ok=True)
            alt = os.path.join(alt_dir, os.path.basename(path))
            torch.save(obj, alt)
            print(f'WARN: checkpoint fallback to {alt} due to: {e}')
        except Exception as e2:
            print(f'ERROR: failed to save checkpoint to {path} and fallback dir: {e2}')


class EMA:
    def __init__(self, model: nn.Module, momentum: float = 0.996):
        self.m = momentum
        self.shadow = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.data.clone().detach().to(p.device)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if n not in self.shadow:
                self.shadow[n] = p.data.clone().detach().to(p.device)
            if self.shadow[n].device != p.device:
                self.shadow[n] = self.shadow[n].to(p.device)
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


# ======================================================
# Dataset
# ======================================================

class EEGNPZDataset(Dataset):
    def __init__(
            self,
            roots: List[str],
            drop_artifacts: bool = False,
            sample_rate: int = 256,
            subjects_per_dataset: Optional[int] = None,
            subjects_allowlist: Optional[Dict[str, List[str]]] = None
    ):
        self.files = []
        tmp_files = []
        for root in roots:
            dname = os.path.basename(root.rstrip("/"))
            for p in pathlib.Path(root).rglob('*.npz'):
                subj = 'unknown'
                try:
                    d = np.load(p, allow_pickle=True)
                    subj = str(d.get('subject', 'unknown'))
                except Exception:
                    pass
                tmp_files.append((str(p), dname, subj))

        if len(tmp_files) == 0:
            raise AssertionError('No .npz files found under provided roots.')

        if subjects_allowlist:
            allow = {ds: set(subjs) for ds, subjs in subjects_allowlist.items()}
            self.files = [(p, ds, s) for (p, ds, s) in tmp_files if ds in allow and (not allow[ds] or s in allow[ds])]
        elif subjects_per_dataset:
            picked: Dict[str, set] = {}
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
        x = d['x'].astype(np.float32)  # (C, T)
        mask_missing = d['mask_missing'].astype(np.float32)  # (C,) or (C,1)
        # Optional seizure-aware meta
        tts = float(d.get('minutes_to_seizure', np.nan)) if 'minutes_to_seizure' in d else np.nan
        phase = str(d.get('phase', 'unknown')) if 'phase' in d else 'unknown'
        return {
            'x_raw': x,
            'mask_missing': mask_missing,
            'dataset_id': dataset_id,
            'subject': subject,
            'tts': tts,
            'phase': phase
        }


# ======================================================
# Encoders (auto-import real ones or fallback to dummy)
# ======================================================

USE_REAL = False
TemporalTransformerEncoder = None
MaxViTSpectrogramEncoder = None
FusionHead = None

try:
    from efm.models.models_transformer import (
        TemporalTransformerEncoder,  # expects forward(x: (B,C,T)) -> (B,D)
        MaxViTSpectrogramEncoder,  # expects forward(spec: (B,C,F,T)) -> (B,D)
        FusionHead  # expects forward(r: (B,D), s: (B,D)) -> (B,D)
    )

    USE_REAL = True
except Exception:
    USE_REAL = False


class DummyEFMRawEncoder(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(19, 64, 7, padding=3), nn.GELU(),
            nn.Conv1d(64, 128, 7, padding=3), nn.GELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(128, out_dim)
        )

    def forward(self, x):  # x: (B,C,T)
        return self.net(x)


class DummyEFMSpecEncoder(nn.Module):
    def __init__(self, in_ch=19, out_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, out_dim)
        )

    def forward(self, x):  # x: (B,C,F,T)
        return self.net(x)


class DummyFusion(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.fc = nn.Linear(dim * 2, dim)

    def forward(self, r, s):
        return self.fc(torch.cat([r, s], dim=-1))


# ======================================================
# STFT + Augmentations
# ======================================================

class SimpleLogSTFT(nn.Module):
    def __init__(self, n_fft: int = 256, hop: int = 128, eps: float = 1e-6):
        super().__init__()
        self.n_fft, self.hop, self.eps = n_fft, hop, eps

    def forward(self, x: torch.Tensor):
        # x: (B,C,T)
        B, C, T = x.shape
        specs = []
        for c in range(C):
            S = torch.stft(x[:, c, :], n_fft=self.n_fft, hop_length=self.hop, return_complex=True)
            P = (S.real ** 2 + S.imag ** 2).clamp_min(self.eps)
            specs.append(torch.log(P + self.eps))
        spec = torch.stack(specs, dim=1)  # (B,C,F,TT)
        return spec


class RawAug(nn.Module):
    def __init__(self, time_jitter: int = 64, noise_sigma: float = 0.02,
                 channel_drop_p: float = 0.1, time_warp_pct: float = 0.05):
        super().__init__()
        self.time_jitter = time_jitter
        self.noise_sigma = noise_sigma
        self.channel_drop_p = channel_drop_p
        self.time_warp_pct = time_warp_pct

    def forward(self, x: torch.Tensor, mask_missing: torch.Tensor):
        # x: (B,C,T)
        B, C, T = x.shape
        tj = int(self.time_jitter)
        shift = torch.randint(low=-tj, high=tj + 1, size=(B,), device=x.device)
        x2 = torch.zeros_like(x)
        for b in range(B):
            s = shift[b].item()
            if s >= 0:
                x2[b, :, s:] = x[b, :, :T - s]
            else:
                x2[b, :, :T + s] = x[b, :, -s:]
        x2 = x2 + self.noise_sigma * torch.randn_like(x2)  # noise
        drop = (torch.rand(B, C, 1, device=x.device) < self.channel_drop_p).float()
        mm = mask_missing.unsqueeze(-1) if mask_missing.ndim == 2 else mask_missing
        x2 = x2 * (1.0 - drop) * (1.0 - mm) + x2 * (1.0 - mm)
        warp = 1.0 + (2 * torch.rand(B, device=x.device) - 1.0) * self.time_warp_pct
        out = torch.zeros_like(x2)
        grid = torch.linspace(0, 1, T, device=x.device)
        for b in range(B):
            t_new = (grid * warp[b]).clamp(0, 1)
            idx = (t_new * (T - 1)).round().long()
            out[b] = x2[b, :, idx]
        return out


class SpecAug(nn.Module):
    def __init__(self, time_mask_pct: float = 0.1, freq_mask_pct: float = 0.1):
        super().__init__()
        self.tmp = time_mask_pct
        self.fmp = freq_mask_pct

    def forward(self, spec: torch.Tensor):
        # spec: (B,C,F,T)
        B, C, F, T = spec.shape
        out = spec.clone()
        tlen = int(T * self.tmp)
        if tlen > 0:
            t0 = torch.randint(0, max(1, T - tlen + 1), (B,), device=spec.device)
            for b in range(B):
                out[b, :, :, t0[b]:t0[b] + tlen] = 0
        flen = int(F * self.fmp)
        if flen > 0:
            f0 = torch.randint(0, max(1, F - flen + 1), (B,), device=spec.device)
            for b in range(B):
                out[b, :, f0[b]:f0[b] + flen, :] = 0
        return out


# ======================================================
# Heads & Queues
# ======================================================

class ProjHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, in_dim), nn.GELU(), nn.Linear(in_dim, proj_dim))

    def forward(self, x):
        return safe_normalize(self.net(x), dim=-1)


class JigsawHead(nn.Module):
    def __init__(self, in_dim: int, num_perm: int = 24):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_perm)

    def forward(self, x):
        return self.fc(x)


class DomainHead(nn.Module):
    def __init__(self, in_dim: int, num_domains: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_domains)

    def forward(self, x, lambd: float):
        return self.fc(grad_reverse(x, lambd))


class MemoryQueue:
    def __init__(self, dim: int, size: int = 65536, device: str = 'cuda'):
        self.size = size
        self.ptr = 0
        q = torch.randn(size, dim, device=device)
        self.queue = safe_normalize(q, dim=-1)

    @torch.no_grad()
    def enqueue(self, feats: torch.Tensor):
        n = feats.shape[0]
        if n >= self.size:
            self.queue = feats[-self.size:].detach()
            self.ptr = 0
            return
        end = self.ptr + n
        if end <= self.size:
            self.queue[self.ptr:end] = feats.detach()
        else:
            first = self.size - self.ptr
            self.queue[self.ptr:] = feats[:first].detach()
            self.queue[:end - self.size] = feats[first:].detach()
        self.ptr = (self.ptr + n) % self.size


def info_nce(q: torch.Tensor, k: torch.Tensor, queue: MemoryQueue, temperature: float = 0.07):
    q = safe_normalize(q, dim=-1)
    k = safe_normalize(k, dim=-1)
    pos = torch.sum(q * k, dim=-1, keepdim=True)
    neg = q @ queue.queue.t()
    logits = torch.cat([pos, neg], dim=1) / temperature
    labels = torch.zeros(q.size(0), dtype=torch.long, device=q.device)
    return F.cross_entropy(logits, labels)


class SubjectProtoBank(nn.Module):
    def __init__(self, dim: int = 512, momentum: float = 0.9):
        super().__init__()
        self.dim = dim
        self.m = momentum
        self.register_buffer('keys', torch.empty(0, dim))
        self.subject_to_idx: Dict[str, int] = {}

    @torch.no_grad()
    def _ensure_subject(self, subject: List[str], device):
        for s in subject:
            if s not in self.subject_to_idx:
                self.subject_to_idx[s] = len(self.subject_to_idx)
                new_key = safe_normalize(torch.randn(1, self.dim, device=device), dim=-1)
                self.keys = new_key if self.keys.numel() == 0 else torch.cat([self.keys, new_key], dim=0)

    @torch.no_grad()
    def update(self, subjects: List[str], feats: torch.Tensor):
        self._ensure_subject(subjects, feats.device)
        for i, s in enumerate(subjects):
            idx = self.subject_to_idx[s]
            k = self.keys[idx]
            k.mul_(self.m).add_(safe_normalize(feats[i], dim=-1), alpha=1.0 - self.m)
            self.keys[idx] = safe_normalize(k, dim=-1)

    def loss(self, subjects: List[str], feats: torch.Tensor, temperature: float = 0.07):
        self._ensure_subject(subjects, feats.device)
        feats = safe_normalize(feats, dim=-1)
        protos = safe_normalize(self.keys, dim=-1)
        logits = feats @ protos.t() / temperature
        idxs = torch.tensor([self.subject_to_idx[s] for s in subjects], device=feats.device, dtype=torch.long)
        return F.cross_entropy(logits, idxs)


# ======================================================
# Extra losses / helpers
# ======================================================

def subject_supcon(z: torch.Tensor, subjects: List[str], temperature: float = 0.07):
    """
    Multi-positive (same-subject) contrastive loss on embeddings z.
    """
    z = safe_normalize(z, dim=-1)
    B = z.size(0)
    sim = z @ z.t() / temperature  # (B,B)
    eye = torch.eye(B, device=z.device, dtype=torch.bool)
    subj_arr = np.array(subjects)
    pos = torch.tensor(subj_arr[:, None] == subj_arr[None, :], device=z.device, dtype=torch.bool)
    pos = pos & (~eye)
    denom_mask = ~eye
    log_denom = torch.logsumexp(sim.masked_fill(~denom_mask, float('-inf')), dim=1, keepdim=True)
    pos_logsum = torch.where(
        pos.any(dim=1, keepdim=True),
        torch.logsumexp(sim.masked_fill(~pos, float('-inf')), dim=1, keepdim=True),
        torch.full((B, 1), 0.0, device=z.device)
    )
    valid = pos.any(dim=1)
    if valid.any():
        loss = -(pos_logsum[valid] - log_denom[valid]).mean()
    else:
        loss = sim.new_tensor(0.0)
    return loss


def seizure_weight(batch, tau_min=10.0, tau_max=60.0):
    """
    If 'tts' exists: w = exp(-tts/tau). If 'phase'=='preictal': *1.5.
    Returns a tensor (B,) or None if metadata missing.
    """
    tts = batch.get('tts', None)
    phase = batch.get('phase', None)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    w = None
    if tts is not None:
        tts_arr = np.array(tts, dtype=np.float32) if isinstance(tts, (list, tuple, np.ndarray)) else np.array([tts],
                                                                                                              dtype=np.float32)
        tts_t = torch.tensor(tts_arr, device=device)
        tau = torch.clamp(tts_t.new_full(tts_t.shape, 30.0), tau_min, tau_max)
        w = torch.exp(-tts_t / tau)
        w = torch.nan_to_num(w, nan=1.0, posinf=1.0, neginf=1.0)
    if phase is not None:
        phases = phase if isinstance(phase, list) else [phase] * (len(w) if w is not None else 1)
        boost = torch.tensor([1.5 if str(p).lower() == 'preictal' else 1.0 for p in phases], device=device)
        if w is None:
            w = boost
        else:
            w = w * boost
    return w


def cosine_lr(step, total, base_lr, warmup):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    s = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1 + math.cos(math.pi * s))


def cosine_weight(step, total, base):
    s = step / max(1, total)
    return base * (0.3 + 0.7 * 0.5 * (1 + math.cos(math.pi * s)))


# ======================================================
# EFM SSL wrapper
# ======================================================

class EFM_SSL(nn.Module):
    def __init__(self, proj_dim=256, num_domains=3, jigsaw_perms=24, ema_m=0.996, queue_size=65536, proto_m=0.9,
                 real_in_ch: int = 19, real_hidden_dim: int = 512):
        super().__init__()

        if USE_REAL and (TemporalTransformerEncoder is not None) and (MaxViTSpectrogramEncoder is not None):
            self.raw_enc = TemporalTransformerEncoder(embed_dim=real_hidden_dim)
            self.spec_enc = MaxViTSpectrogramEncoder(in_channels=real_in_ch, embed_dim=real_hidden_dim)
            self.fusion = FusionHead(in_dim=real_hidden_dim) if (FusionHead is not None) else DummyFusion(
                real_hidden_dim)
        else:
            self.raw_enc = DummyEFMRawEncoder(real_hidden_dim)
            self.spec_enc = DummyEFMSpecEncoder(real_in_ch, real_hidden_dim)
            self.fusion = DummyFusion(real_hidden_dim)

        self.raw_proj = ProjHead(real_hidden_dim, proj_dim)
        self.spec_proj = ProjHead(real_hidden_dim, proj_dim)

        self.jigsaw_head = JigsawHead(real_hidden_dim, jigsaw_perms)
        self.domain_head = DomainHead(real_hidden_dim, num_domains)

        self.raw_ema = EMA(self.raw_enc, ema_m)
        self.spec_ema = EMA(self.spec_enc, ema_m)

        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.queue = MemoryQueue(proj_dim, size=queue_size, device=dev)
        self.proto = SubjectProtoBank(dim=real_hidden_dim, momentum=proto_m)

    def forward_raw_latent(self, x):
        return self.raw_enc(x)

    def forward_spec_latent(self, spec):
        return self.spec_enc(spec)


# ======================================================
# Config
# ======================================================

@dataclass
class SSLConfig:
    data_roots: List[str] = None
    subjects_per_dataset: Optional[int] = None
    subjects_allowlist_path: Optional[str] = None
    batch_size: int = 32
    num_workers: int = 8
    lr: float = 3e-4
    weight_decay: float = 0.05
    temperature: float = 0.07
    total_steps: int = 200_000
    out_dir: str = 'runs/ssl/efm_ssl_base'
    # loss weights
    loss_w_mem: float = 1.0
    loss_w_contrast: float = 1.0
    loss_w_jigsaw: float = 0.5
    loss_w_grl: float = 0.2
    loss_w_proto: float = 0.5
    loss_w_subj: float = 0.5
    # adaptive
    adaptive_weights: str = 'none'  # none|uncertainty|cosine


# ======================================================
# MEM mask helper
# ======================================================

def masked_eeg_targets(x: torch.Tensor,
                       mask_ratio_min: float,
                       mask_ratio_max: float,
                       tmin: int,
                       tmax: int) -> torch.Tensor:
    """
    Create a boolean mask over (B, C, T) that covers ~[mask_ratio_min, mask_ratio_max] of entries,
    using contiguous spans along time on random channels.
    """
    B, C, T = x.shape
    ratio = torch.empty(B, device=x.device).uniform_(mask_ratio_min, mask_ratio_max)
    mask = torch.zeros(B, C, T, device=x.device, dtype=torch.bool)
    for b in range(B):
        total = int(ratio[b].item() * C * T)
        covered = 0
        while covered < total:
            span = random.randint(tmin, tmax)
            c = random.randint(0, C - 1)
            t0 = random.randint(0, max(0, T - span))
            t1 = min(T, t0 + span)
            mask[b, c, t0:t1] = True
            covered += (t1 - t0)
    return mask


# ======================================================
# Train
# ======================================================

def make_domain_index(names: List[str]) -> Dict[str, int]:
    uniq = sorted(set(names))
    return {n: i for i, n in enumerate(uniq)}


def train(cfg: SSLConfig):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(42)

    # Data
    allow = None
    if cfg.subjects_allowlist_path and os.path.exists(cfg.subjects_allowlist_path):
        try:
            with open(cfg.subjects_allowlist_path) as f:
                allow = json.load(f)
        except Exception:
            allow = None

    ds = EEGNPZDataset(cfg.data_roots, subjects_per_dataset=cfg.subjects_per_dataset, subjects_allowlist=allow)
    domain_map = make_domain_index([dsname for (_, dsname, _) in ds.files])
    loader = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True
    )
    print(f"Dataset loaded with {len(ds)} samples.")

    # Model & optim
    model = EFM_SSL(proj_dim=256, num_domains=len(domain_map), ema_m=0.996, queue_size=65536, proto_m=0.9).to(device)
    raw_aug, stft, spec_aug = RawAug(), SimpleLogSTFT(), SpecAug()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # Adaptive loss (uncertainty) params
    adaptive = cfg.adaptive_weights
    log_sig = None
    if adaptive == 'uncertainty':
        log_sig = {
            'mem': nn.Parameter(torch.zeros(1, device=device)),
            'contrast': nn.Parameter(torch.zeros(1, device=device)),
            'jig': nn.Parameter(torch.zeros(1, device=device)),
            'grl': nn.Parameter(torch.zeros(1, device=device)),
            'proto': nn.Parameter(torch.zeros(1, device=device)),
        }
        for v in log_sig.values():
            v.requires_grad_(True)
        opt.add_param_group({'params': list(log_sig.values()), 'lr': cfg.lr})

    def UW(name, L):
        # Kendall & Gal: L/(2*sigma^2) + log sigma
        s2 = torch.exp(2 * log_sig[name])
        return (L / (2.0 * s2)).sum() + log_sig[name].sum()

    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, 'ssl_config.json'), 'w') as f:
        json.dump(cfg.__dict__, f, indent=2)

    step, t0 = 0, time.time()
    total_steps = cfg.total_steps
    warmup_steps = max(1, int(0.01 * total_steps))

    while step < total_steps:
        for batch in loader:
            if step >= total_steps:
                break

            x = torch.as_tensor(batch['x_raw']).to(device)  # (B,C,T)
            mm = torch.as_tensor(batch['mask_missing']).to(device)  # (B,C) or (B,C,1)
            subjects = batch['subject']
            dnames = batch['dataset_id']
            d_idx = torch.tensor([domain_map[n] for n in dnames], device=device)

            # Multi-view generation
            x_v1 = raw_aug(x, mm)
            x_v2 = raw_aug(x, mm)
            x_v1 = torch.nan_to_num(x_v1, nan=0.0, posinf=0.0, neginf=0.0)
            x_v2 = torch.nan_to_num(x_v2, nan=0.0, posinf=0.0, neginf=0.0)

            spec = spec_aug(stft(x))
            spec = torch.nan_to_num(spec, nan=0.0, posinf=0.0, neginf=0.0)

            # MEM mask (placeholder; used with latent proxy)
            _ = masked_eeg_targets(x_v1, 0.30, 0.60, 64, 256)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                # Student latents
                r_lat_v1 = model.forward_raw_latent(x_v1)  # (B,D)
                r_lat_v2 = model.forward_raw_latent(x_v2)  # (B,D)
                s_lat = model.forward_spec_latent(spec)  # (B,D)

                q_raw = model.raw_proj(r_lat_v1)  # (B,P)
                q_spec = model.spec_proj(s_lat)  # (B,P)

                # EMA teachers
                model.raw_ema.update(model.raw_enc)
                model.spec_ema.update(model.spec_enc)
                with torch.no_grad():
                    r_lat_v2_t = model.forward_raw_latent(x_v2).detach()
                    s_lat_t = model.forward_spec_latent(spec).detach()
                    k_raw = safe_normalize(model.raw_proj(r_lat_v2_t), dim=-1)
                    k_spec = safe_normalize(model.spec_proj(s_lat_t), dim=-1)

                # Losses
                L_contrast = info_nce(q_raw, k_spec, model.queue, temperature=cfg.temperature) \
                             + info_nce(q_spec, k_raw, model.queue, temperature=cfg.temperature)
                L_mem = F.mse_loss(r_lat_v1, r_lat_v2_t)  # latent regression proxy
                perm_logits = model.jigsaw_head(r_lat_v1)
                perm_labels = torch.randint(0, perm_logits.size(1), (x.size(0),), device=device)
                L_jig = F.cross_entropy(perm_logits, perm_labels)
                fusion_lat = model.fusion(r_lat_v1, s_lat)
                lambd = min(1.0, step / max(1, int(total_steps * 0.3)))
                dom_logits = model.domain_head(fusion_lat.detach(), lambd)
                L_grl = F.cross_entropy(dom_logits, d_idx)
                L_proto = model.proto.loss(subjects, fusion_lat)
                L_subj = subject_supcon(fusion_lat, subjects, temperature=cfg.temperature)

                # Seizure-aware scaling (if metadata exists)
                w_sz = seizure_weight(batch)
                if w_sz is not None:
                    scale = float(torch.mean(w_sz).item())
                    L_mem *= scale
                    L_contrast *= scale
                    L_jig *= scale
                    L_grl *= scale
                    L_proto *= scale
                    L_subj *= scale

                # Combine with adaptive weights if requested
                if adaptive == 'cosine':
                    w_mem = cosine_weight(step, total_steps, cfg.loss_w_mem)
                    w_con = cosine_weight(step, total_steps, cfg.loss_w_contrast)
                    w_jig = cosine_weight(step, total_steps, cfg.loss_w_jigsaw)
                    w_grl = cosine_weight(step, total_steps, cfg.loss_w_grl)
                    w_pro = cosine_weight(step, total_steps, cfg.loss_w_proto)
                    w_sub = cosine_weight(step, total_steps, cfg.loss_w_subj)
                    loss = w_mem * L_mem + w_con * L_contrast + w_jig * L_jig + w_grl * L_grl + w_pro * L_proto + w_sub * L_subj
                elif adaptive == 'uncertainty':
                    # Kendall & Gal uncertainty weighting
                    loss = (UW('mem', L_mem) + UW('contrast', L_contrast) +
                            UW('jig', L_jig) + UW('grl', L_grl) + UW('proto', L_proto)) \
                           + cfg.loss_w_subj * L_subj
                else:
                    loss = (cfg.loss_w_mem * L_mem +
                            cfg.loss_w_contrast * L_contrast +
                            cfg.loss_w_jigsaw * L_jig +
                            cfg.loss_w_grl * L_grl +
                            cfg.loss_w_proto * L_proto +
                            cfg.loss_w_subj * L_subj)

            # Guard against non-finite
            if not torch.isfinite(loss):
                print(json.dumps({
                    'step': step, 'warn': 'nonfinite_loss',
                    'L_mem': float(L_mem.detach().nan_to_num().item()),
                    'L_contrast': float(L_contrast.detach().nan_to_num().item()),
                    'L_jig': float(L_jig.detach().nan_to_num().item()),
                    'L_grl': float(L_grl.detach().nan_to_num().item()),
                    'L_proto': float(L_proto.detach().nan_to_num().item()),
                    'L_subj': float(L_subj.detach().nan_to_num().item())
                }))
                with torch.no_grad():
                    reinit_count = min(1024, model.queue.size)
                    reinit = safe_normalize(
                        torch.randn(reinit_count, model.queue.queue.size(1), device=device), dim=-1
                    )
                    model.queue.queue[:reinit_count] = reinit
                step += 1
                continue

            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            lr_now = cosine_lr(step, total_steps, cfg.lr, warmup_steps)
            for g in opt.param_groups:
                g['lr'] = lr_now
            scaler.step(opt)
            scaler.update()

            with torch.no_grad():
                model.queue.enqueue(k_raw)
                model.queue.enqueue(k_spec)
                model.proto.update(subjects, fusion_lat)

            if (step % 100) == 0:
                print(json.dumps({'step': step, 'lr': lr_now, 'loss': float(loss.item()),
                                  'time_s': round(time.time() - t0, 2)}))

            if (step % 10000) == 0 and step > 0:
                safe_save({'model': model.state_dict()}, os.path.join(cfg.out_dir, f'ssl_step{step}.ckpt'))

            step += 1

    safe_save({'model': model.state_dict()}, os.path.join(cfg.out_dir, 'efm_base.ckpt'))


# ======================================================
# Argparse
# ======================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default=None)
    p.add_argument('--data-roots', nargs='+', dest='data_roots', required=True)
    p.add_argument('--limit-subjects-per-dataset', dest='limit_subjects_per_dataset', type=int, default=None)
    p.add_argument('--subjects-allowlist', dest='subjects_allowlist', type=str, default=None)
    p.add_argument('--batch-size', dest='batch_size', type=int, default=32)
    p.add_argument('--num-workers', dest='num_workers', type=int, default=8)
    p.add_argument('--lr', dest='lr', type=float, default=None)
    p.add_argument('--weight-decay', dest='weight_decay', type=float, default=None)
    p.add_argument('--temperature', dest='temperature', type=float, default=None)
    p.add_argument('--adaptive-weights', dest='adaptive_weights', type=str, default='none',
                   choices=['none', 'uncertainty', 'cosine'])
    p.add_argument('--loss-w-subj', dest='loss_w_subj', type=float, default=0.5)
    p.add_argument('--out', dest='out', type=str, default='runs/ssl/efm_ssl_base')
    args = p.parse_args()

    cfg = SSLConfig()
    # Optional YAML load
    if args.config and yaml is not None and os.path.exists(args.config):
        try:
            with open(args.config, 'r') as f:
                y = yaml.safe_load(f)
            if isinstance(y, dict):
                t = y.get('train', {})
                cfg.lr = t.get('base_lr', cfg.lr)
                cfg.total_steps = t.get('total_steps', cfg.total_steps)
                cfg.weight_decay = t.get('weight_decay', cfg.weight_decay)
                cfg.temperature = t.get('temperature', cfg.temperature)
        except Exception:
            pass

    cfg.data_roots = args.data_roots
    cfg.subjects_per_dataset = args.limit_subjects_per_dataset
    cfg.subjects_allowlist_path = args.subjects_allowlist
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    if args.lr is not None:
        cfg.lr = args.lr
    if args.weight_decay is not None:
        cfg.weight_decay = args.weight_decay
    if args.temperature is not None:
        cfg.temperature = args.temperature
    cfg.adaptive_weights = args.adaptive_weights
    cfg.loss_w_subj = args.loss_w_subj
    cfg.out_dir = args.out
    return cfg


# ======================================================
# Main
# ======================================================

if __name__ == '__main__':
    cfg = parse_args()
    print('Config:', cfg)
    train(cfg)
