#!/usr/bin/env python3
"""
Minimal, runnable skeleton for Step 4: Self-Supervised Pretraining (SSL) on EFM.
- Objectives: MEM (masked EEG modeling), Multi-view Contrastive (raw<->spec), Jigsaw, GRL domain invariance, Subject Prototype alignment.
- Expects your EFM encoders to be importable: EFMRawEncoder, EFMSpecEncoder, EFMFusion.
- Reads harmonized .npz windows from Step 2 (channels x time), ignoring labels for SSL.

NOTE: This is a skeleton with sensible defaults and clear TODOs to bind into your repo.
"""
from __future__ import annotations
import os, math, time, json, random, argparse, pathlib
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import yaml
except Exception:
    yaml = None


# ==========================
# Utilities
# ==========================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EMA:
    def __init__(self, model: nn.Module, momentum: float = 0.996):
        self.m = momentum
        self.shadow = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.data.clone().detach()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.m).add_(p.data, alpha=1.0 - self.m)

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                p.data.copy_(self.shadow[n])


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


# ==========================
# Data
# ==========================
class EEGNPZDataset(Dataset):
    def __init__(self, roots: List[str], drop_artifacts: bool = True, sample_rate: int = 256,
                 subjects_per_dataset: Optional[int] = None, subjects_allowlist: Optional[Dict[str, List[str]]] = None):
        self.files = []  # list[(path, dataset_id, subject)]
        self.drop_artifacts = drop_artifacts
        self.sample_rate = sample_rate
        self.subjects_per_dataset = subjects_per_dataset
        self.subjects_allowlist = subjects_allowlist or {}
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
            raise AssertionError("No .npz files found under provided roots.")
        # Filter by allowlist or cap per dataset
        if self.subjects_allowlist:
            allow = {ds: set(subjs) for ds, subjs in self.subjects_allowlist.items()}
            self.files = [(p, ds, s) for (p, ds, s) in tmp_files if ds in allow and (not allow[ds] or s in allow[ds])]
        elif subjects_per_dataset is not None:
            picked: Dict[str, set] = {}
            seen_counts: Dict[str, Dict[str, int]] = {}
            for (p, ds, s) in tmp_files:
                picked.setdefault(ds, set())
                seen_counts.setdefault(ds, {})
                if len(picked[ds]) < subjects_per_dataset or s in picked[ds]:
                    self.files.append((p, ds, s))
                    picked[ds].add(s)
                    seen_counts[ds][s] = seen_counts[ds].get(s, 0) + 1
        else:
            self.files = tmp_files
        assert len(self.files) > 0, "No files left after subject filtering. Check allowlist or cap size."

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, dataset_id, subject = self.files[idx]
        d = np.load(path, allow_pickle=True)
        x = d['x'].astype(np.float32)  # (C, T)
        mask_artifact = d['mask_artifact'].item() if np.ndim(d['mask_artifact']) == 0 else int(d['mask_artifact'])
        mask_missing = d['mask_missing'].astype(np.float32)
        # filter artifact windows if requested
        if self.drop_artifacts and mask_artifact == 1:
            return self.__getitem__((idx + 1) % len(self))
        sample = {
            'x_raw': x,
            'mask_missing': mask_missing,
            'dataset_id': dataset_id,
            'subject': subject,
        }
        return sample
        path, dataset_id = self.files[idx]
        d = np.load(path, allow_pickle=True)
        x = d['x'].astype(np.float32)  # (C, T)
        mask_artifact = d['mask_artifact'].item() if np.ndim(d['mask_artifact']) == 0 else int(d['mask_artifact'])
        mask_missing = d['mask_missing'].astype(np.float32)
        subject = str(d.get('subject', 'unknown'))
        # filter artifact windows if requested
        if self.drop_artifacts and mask_artifact == 1:
            # simple re-sample
            return self.__getitem__((idx + 1) % len(self))
        sample = {
            'x_raw': x,  # (C,T)
            'mask_missing': mask_missing,  # (C,) or (C,1)
            'dataset_id': dataset_id,  # string key
            'subject': subject,
        }
        return sample


# ==========================
# Simple STFT (log-spectrogram) helper
# ==========================
class SimpleLogSTFT(nn.Module):
    def __init__(self, n_fft: int = 256, hop: int = 128, eps: float = 1e-6):
        super().__init__()
        self.n_fft, self.hop, self.eps = n_fft, hop, eps

    def forward(self, x: torch.Tensor):
        # x: (B,C,T)
        B, C, T = x.shape
        # compute STFT per channel, then stack as 2D image (C channels as batch within)
        specs = []
        for c in range(C):
            S = torch.stft(x[:, c, :], n_fft=self.n_fft, hop_length=self.hop, return_complex=True)
            P = (S.real ** 2 + S.imag ** 2).clamp_min(self.eps)
            specs.append(torch.log(P))  # (B, F, TT)
        spec = torch.stack(specs, dim=1)  # (B, C, F, TT)
        return spec


# ==========================
# Augmentations
# ==========================
class RawAug(nn.Module):
    def __init__(self, time_jitter: int = 64, noise_sigma: float = 0.02, channel_drop_p: float = 0.1,
                 time_warp_pct: float = 0.05):
        super().__init__()
        self.time_jitter = time_jitter
        self.noise_sigma = noise_sigma
        self.channel_drop_p = channel_drop_p
        self.time_warp_pct = time_warp_pct

    def forward(self, x: torch.Tensor, mask_missing: torch.Tensor):
        # x: (B,C,T)
        B, C, T = x.shape
        # small jitter
        shift = torch.randint(low=-self.time_jitter, high=self.time_jitter + 1, size=(B,), device=x.device)
        x2 = torch.zeros_like(x)
        for b in range(B):
            s = shift[b].item()
            if s >= 0:
                x2[b, :, s:] = x[b, :, :T - s]
            else:
                x2[b, :, :T + s] = x[b, :, -s:]
        # gaussian noise
        x2 = x2 + self.noise_sigma * torch.randn_like(x2)
        # channel dropout (respect missing)
        drop = (torch.rand(B, C, 1, device=x.device) < self.channel_drop_p).float()
        if mask_missing.ndim == 2:
            mm = mask_missing.unsqueeze(-1)  # (B,C,1)
        else:
            mm = mask_missing
        x2 = x2 * (1.0 - drop) * (1.0 - mm) + x2 * (1.0 - mm)  # keep missing channels 0 if provided as 1s
        # simple time-warp by resampling (nearest)
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
        # time mask
        tlen = int(T * self.tmp)
        if tlen > 0:
            t0 = torch.randint(0, max(1, T - tlen + 1), (B,), device=spec.device)
            for b in range(B):
                out[b, :, :, t0[b]:t0[b] + tlen] = 0
        # freq mask
        flen = int(F * self.fmp)
        if flen > 0:
            f0 = torch.randint(0, max(1, F - flen + 1), (B,), device=spec.device)
            for b in range(B):
                out[b, :, f0[b]:f0[b] + flen, :] = 0
        return out


# ==========================
# Heads & Losses
# ==========================
class ProjHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.GELU(), nn.Linear(in_dim, proj_dim)
        )

    def forward(self, x):
        x = self.net(x)
        return F.normalize(x, dim=-1)


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


# InfoNCE with memory queue
class MemoryQueue:
    def __init__(self, dim: int, size: int = 65536, device: str = 'cuda'):
        self.size = size
        self.ptr = 0
        self.device = device
        self.queue = F.normalize(torch.randn(size, dim, device=device), dim=-1)

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
    # q: (B,D) student from view A, k: (B,D) teacher/key from view B
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)
    # positives: q·k
    pos = torch.sum(q * k, dim=-1, keepdim=True)
    # negatives: q·Q
    neg = q @ queue.queue.t()  # (B, K)
    logits = torch.cat([pos, neg], dim=1) / temperature
    labels = torch.zeros(q.size(0), dtype=torch.long, device=q.device)
    loss = F.cross_entropy(logits, labels)
    return loss


# Simple proto alignment (subject prototypes)
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
                new_key = F.normalize(torch.randn(1, self.dim, device=device), dim=-1)
                if self.keys.numel() == 0:
                    self.keys = new_key
                else:
                    self.keys = torch.cat([self.keys, new_key], dim=0)

    @torch.no_grad()
    def update(self, subjects: List[str], feats: torch.Tensor):
        # feats: (B, D)
        self._ensure_subject(subjects, feats.device)
        for i, s in enumerate(subjects):
            idx = self.subject_to_idx[s]
            k = self.keys[idx]
            k.mul_(self.m).add_(F.normalize(feats[i], dim=-1), alpha=1.0 - self.m)
            self.keys[idx] = F.normalize(k, dim=-1)

    def loss(self, subjects: List[str], feats: torch.Tensor, temperature: float = 0.07):
        self._ensure_subject(subjects, feats.device)
        feats = F.normalize(feats, dim=-1)
        protos = F.normalize(self.keys, dim=-1)  # (S,D)
        logits = feats @ protos.t() / temperature
        idxs = torch.tensor([self.subject_to_idx[s] for s in subjects], device=feats.device, dtype=torch.long)
        return F.cross_entropy(logits, idxs)


# ==========================
# EFM wrappers (TODO: import your actual modules)
# ==========================
class DummyEFMRawEncoder(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(19, 64, 7, padding=3), nn.GELU(),
            nn.Conv1d(64, 128, 7, padding=3), nn.GELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x):  # x: (B,C,T)
        return self.net(x)


class DummyEFMSpecEncoder(nn.Module):
    def __init__(self, in_ch=19, out_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x):  # x: (B,C,F,T)
        return self.net(x)


class DummyFusion(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.fc = nn.Linear(dim * 2, dim)

    def forward(self, r, s):  # (B,D),(B,D)
        return self.fc(torch.cat([r, s], dim=-1))


# ==========================
# Main SSL model
# ==========================
class EFM_SSL(nn.Module):
    def __init__(self, proj_dim=256, num_domains=3, jigsaw_perms=24, ema_m=0.996, queue_size=65536, proto_m=0.9):
        super().__init__()
        # TODO: replace Dummy* with your encoders
        self.raw_enc = DummyEFMRawEncoder(512)
        self.spec_enc = DummyEFMSpecEncoder(19, 512)
        self.fusion = DummyFusion(512)

        self.raw_proj = ProjHead(512, proj_dim)
        self.spec_proj = ProjHead(512, proj_dim)

        self.jigsaw_head = JigsawHead(512, jigsaw_perms)
        self.domain_head = DomainHead(512, num_domains)

        # EMA encoders for teacher
        self.raw_ema = EMA(self.raw_enc, ema_m)
        self.spec_ema = EMA(self.spec_enc, ema_m)

        self.queue = MemoryQueue(proj_dim, size=queue_size, device='cuda' if torch.cuda.is_available() else 'cpu')
        self.proto = SubjectProtoBank(dim=512, momentum=proto_m)

    def forward_raw_latent(self, x):
        return self.raw_enc(x)

    def forward_spec_latent(self, spec):
        return self.spec_enc(spec)


# ==========================
# Training loop
# ==========================
@dataclass
class SSLConfig:
    exp_name: str = 'efm_ssl_base'
    seed: int = 42
    data_roots: List[str] = None
    sample_rate: int = 256
    drop_artifacts: bool = True
    # subject filtering
    subjects_per_dataset: Optional[int] = None
    subjects_allowlist_path: Optional[str] = None
    # objectives
    mask_ratio_min: float = 0.30
    mask_ratio_max: float = 0.60
    time_span_min: int = 64
    time_span_max: int = 256
    channel_drop_p: float = 0.10
    proj_dim: int = 256
    temperature: float = 0.07
    queue_size: int = 65536
    ema_m: float = 0.996
    proto_m: float = 0.9
    loss_w_mem: float = 1.0
    loss_w_contrast: float = 1.0
    loss_w_jigsaw: float = 0.5
    loss_w_grl: float = 0.2
    loss_w_proto: float = 0.5
    grl_ramp_pct: float = 0.3
    # train
    lr: float = 3e-4
    weight_decay: float = 0.05
    total_steps: int = 200_000
    warmup_steps: int = 2000
    batch_size: int = 64
    num_workers: int = 8
    ckpt_every: int = 10_000
    log_every: int = 100
    out_dir: str = 'runs/ssl/efm_ssl_base'


def cosine_lr(step, total, base_lr, warmup):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    s = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1 + math.cos(math.pi * s))


def masked_eeg_targets(x: torch.Tensor, mask_ratio_min: float, mask_ratio_max: float, tmin: int, tmax: int):
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
            mask[b, c, t0:t0 + span] = True
            covered += span
    return mask  # (B,C,T)


def make_domain_index(names: List[str]) -> Dict[str, int]:
    uniq = sorted(set(names))
    return {n: i for i, n in enumerate(uniq)}


def train(cfg: SSLConfig):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(cfg.seed)

    # Data
    # optional subjects allowlist


allow: Optional[Dict[str, List[str]]] = None
if cfg.subjects_allowlist_path and os.path.exists(cfg.subjects_allowlist_path):
    with open(cfg.subjects_allowlist_path, 'r') as f:
        allow = json.load(f)  # {"chbmit": ["chb01","chb02"], ...}

ds = EEGNPZDataset(
    cfg.data_roots,
    drop_artifacts=cfg.drop_artifacts,
    sample_rate=cfg.sample_rate,
    subjects_per_dataset=cfg.subjects_per_dataset,
    subjects_allowlist=allow
)
# domain map from folder names
domain_map = make_domain_index([d for _, d in ds.files])

loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True,
                    drop_last=True)

# Model
model = EFM_SSL(proj_dim=cfg.proj_dim, num_domains=len(domain_map), ema_m=cfg.ema_m, queue_size=cfg.queue_size,
                proto_m=cfg.proto_m).to(device)

raw_aug = RawAug(channel_drop_p=cfg.channel_drop_p).to(device)
stft = SimpleLogSTFT().to(device)
spec_aug = SpecAug().to(device)

opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

os.makedirs(cfg.out_dir, exist_ok=True)
step = 0
t0 = time.time()

while step < cfg.total_steps:
    for batch in loader:
        if step >= cfg.total_steps:
            break
        x = torch.tensor(batch['x_raw']).to(device)  # (B,C,T)
        mm = torch.tensor(batch['mask_missing']).to(device)  # (B,C) or (B,C,1)
        subjects = batch['subject']
        dnames = batch['dataset_id']
        d_idx = torch.tensor([domain_map[n] for n in dnames], device=device)

        # Views for contrastive
        with torch.no_grad():
            # Teacher spec/ raw will use EMA encoders
            pass

        x_v1 = raw_aug(x, mm)
        x_v2 = raw_aug(x, mm)

        # Spectrogram view
        spec = stft(x)
        spec = spec_aug(spec)

        # MEM mask on v1
        mem_mask = masked_eeg_targets(x_v1, cfg.mask_ratio_min, cfg.mask_ratio_max, cfg.time_span_min,
                                      cfg.time_span_max)  # (B,C,T)

        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            # Raw latents (student)
            r_lat_v1 = model.forward_raw_latent(x_v1)  # (B,512)
            r_lat_v2 = model.forward_raw_latent(x_v2)
            # Spec latents (student)
            s_lat = model.forward_spec_latent(spec)

            # Projections
            q_raw = model.raw_proj(r_lat_v1)  # (B,D)
            q_spec = model.spec_proj(s_lat)  # (B,D)

            # EMA teacher latents (stop-grad) for MEM and contrast
            model.raw_ema.update(model.raw_enc)
            model.spec_ema.update(model.spec_enc)
            with torch.no_grad():
                r_lat_v2_t = model.forward_raw_latent(x_v2).detach()
                s_lat_t = model.forward_spec_latent(spec).detach()
                k_raw = F.normalize(model.raw_proj(r_lat_v2_t), dim=-1)
                k_spec = F.normalize(model.spec_proj(s_lat_t), dim=-1)

            # ========== Losses ==========
            # 1) InfoNCE (raw<->spec)
            L_contrast = info_nce(q_raw, k_spec, model.queue, temperature=cfg.temperature)
            L_contrast = L_contrast + info_nce(q_spec, k_raw, model.queue, temperature=cfg.temperature)

            # 2) MEM: predict teacher latents from masked positions (latent regression)
            # Here: simple latent L2 between v1 and v2 teacher as proxy (skeleton)
            L_mem = F.mse_loss(r_lat_v1, r_lat_v2_t)  # TODO: token-level decoder if using tokenized latents

            # 3) Jigsaw: split raw latent into 4 segments proxy (skeleton uses whole-latent with dummy permutation)
            perm_logits = model.jigsaw_head(r_lat_v1)
            perm_labels = torch.randint(0, perm_logits.size(1), (x.size(0),), device=device)
            L_jig = F.cross_entropy(perm_logits, perm_labels)

            # 4) GRL domain invariance on fusion embedding
            fusion_lat = model.fusion(r_lat_v1, s_lat)
            lambd = min(1.0, step / max(1, int(cfg.total_steps * cfg.grl_ramp_pct)))
            dom_logits = model.domain_head(fusion_lat.detach(), lambd)  # detach to stabilize (optional)
            L_grl = F.cross_entropy(dom_logits, d_idx)

            # 5) Subject-prototype alignment
            L_proto = model.proto.loss(subjects, fusion_lat)

            loss = (
                    cfg.loss_w_mem * L_mem +
                    cfg.loss_w_contrast * L_contrast +
                    cfg.loss_w_jigsaw * L_jig +
                    cfg.loss_w_grl * L_grl +
                    cfg.loss_w_proto * L_proto
            )

        scaler.scale(loss).backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # LR schedule
        lr_now = cosine_lr(step, cfg.total_steps, cfg.lr, cfg.warmup_steps)
        for g in opt.param_groups:
            g['lr'] = lr_now
        scaler.step(opt)
        scaler.update()

        # update memory queue with latest keys
        with torch.no_grad():
            model.queue.enqueue(k_raw)
            model.queue.enqueue(k_spec)
            model.proto.update(subjects, fusion_lat)

        if (step % cfg.log_every) == 0:
            print(json.dumps({
                'step': step,
                'lr': lr_now,
                'loss': float(loss.item()),
                'L_mem': float(L_mem.item()),
                'L_contrast': float(L_contrast.item()),
                'L_jig': float(L_jig.item()),
                'L_grl': float(L_grl.item()),
                'L_proto': float(L_proto.item()),
                'time_s': round(time.time() - t0, 2)
            }))

        if (step % cfg.ckpt_every) == 0 and step > 0:
            save_path = os.path.join(cfg.out_dir, f'ssl_step{step}.ckpt')
            torch.save({'model': model.state_dict()}, save_path)
            # also save EMA teacher copies for potential distill
            teacher_path = os.path.join(cfg.out_dir, f'teacher_step{step}.ckpt')
            # snapshot current student weights into teacher file as placeholder
            torch.save({'raw_ema': model.raw_enc.state_dict(), 'spec_ema': model.spec_enc.state_dict()}, teacher_path)

        step += 1

# final save
torch.save({'model': model.state_dict()}, os.path.join(cfg.out_dir, 'efm_base.ckpt'))
torch.save({'raw_ema': model.raw_enc.state_dict(), 'spec_ema': model.spec_enc.state_dict()},
           os.path.join(cfg.out_dir, 'teacher_ema.ckpt'))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default=None, help='Path to YAML config (optional)')
    p.add_argument('--data.roots', nargs='+', dest='data_roots', required=False)
    p.add_argument('--out', type=str, default=None)
    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--limit.subjects_per_dataset', type=int, default=None)
    p.add_argument('--subjects.allowlist', type=str, default=None,
                   help='JSON file mapping dataset->list of subject IDs')
    args = p.parse_args()
    cfg = SSLConfig()
    if args.config and yaml is not None:
        with open(args.config, 'r') as f:
            y = yaml.safe_load(f)
        cfg.exp_name = y.get('exp_name', cfg.exp_name)
        d = y.get('datasets', {})
        roots = d.get('roots', None)
        if roots is not None:
            cfg.data_roots = roots
        cfg.subjects_per_dataset = d.get('max_subjects_per_dataset', cfg.subjects_per_dataset)
        cfg.subjects_allowlist_path = d.get('subjects_allowlist_path', cfg.subjects_allowlist_path)
        m = y.get('ssl_heads', {})
        if 'mem' in m:
            mem = m['mem']
            cfg.mask_ratio_min = mem.get('mask_ratio_min', cfg.mask_ratio_min)
            cfg.mask_ratio_max = mem.get('mask_ratio_max', cfg.mask_ratio_max)
        if 'contrastive' in m:
            con = m['contrastive']
            cfg.proj_dim = con.get('proj_dim', cfg.proj_dim)
            cfg.temperature = con.get('temperature', cfg.temperature)
            cfg.queue_size = con.get('queue_size', cfg.queue_size)
        if 'grl' in m:
            grl = m['grl']
            cfg.grl_ramp_pct = grl.get('lambda_ramp_pct', cfg.grl_ramp_pct)
        if 'subject_proto' in m:
            sp = m['subject_proto']
            cfg.proto_m = sp.get('momentum', cfg.proto_m)
        t = y.get('train', {})
        cfg.lr = t.get('base_lr', cfg.lr)
        cfg.weight_decay = t.get('weight_decay', cfg.weight_decay)
        cfg.total_steps = t.get('total_steps', cfg.total_steps)
        cfg.warmup_steps = t.get('warmup_steps', cfg.warmup_steps)
        cfg.batch_size = t.get('batch_size', cfg.batch_size)  # <-- use actual batch_size if provided
        cfg.num_workers = t.get('num_workers', cfg.num_workers)
        s = y.get('save', {})
        cfg.out_dir = s.get('out_dir', cfg.out_dir)
    if args.data_roots:
        cfg.data_roots = args.data_roots
    if args.out:
        cfg.out_dir = args.out
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.limit.subjects_per_dataset is not None:
        cfg.subjects_per_dataset = args.limit.subjects_per_dataset
    if args.subjects.allowlist is not None:
        cfg.subjects_allowlist_path = args.subjects.allowlist
    assert cfg.data_roots is not None and len(cfg.data_roots) > 0, 'Provide dataset roots via --data.roots or YAML.'
    return cfg


if __name__ == '__main__':
    cfg = parse_args()
    print('Config:', cfg)
    train(cfg)
