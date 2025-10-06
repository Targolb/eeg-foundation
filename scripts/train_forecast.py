# scripts/train_forecast.py
import os
import csv
import json
import argparse
import random
from typing import Dict, List, Optional, Iterable, Set

import torch

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

try:
    import yaml
except Exception:
    yaml = None


# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml(path: str) -> Dict:
    if yaml is None:
        raise RuntimeError("PyYAML not installed but --config was provided.")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def to_device(batch, device):
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    return batch


def forward_safe(model, *args, **kwargs):
    try:
        return model(*args, **kwargs)
    except RuntimeError as e:
        msg = str(e)
        if "CUDNN_STATUS_INTERNAL_ERROR" in msg or "CUDNN_STATUS_EXECUTION_FAILED" in msg:
            import torch.backends.cudnn as cudnn
            with cudnn.flags(enabled=False):
                return model(*args, **kwargs)
        raise


# ----------------------------
# Subject discovery (from filenames)
# ----------------------------
def _extract_sid_from_name(fname: str) -> str:
    base = os.path.basename(fname)
    if base.startswith("chb"):
        # chb04_15__... -> chb04
        return base.split("_")[0]
    if "_s" in base:
        # aaaaapwd_s001_t000__... -> s001
        try:
            return "s" + base.split("_s")[1].split("_")[0]
        except Exception:
            return "unknown"
    return "unknown"


def discover_subjects(root: str, datasets: Optional[Iterable[str]] = None,
                      max_files: int = 1_000_000) -> List[str]:
    """
    Walk the processed tree(s) and infer subject IDs from .npz filenames.
    """
    roots = []
    if datasets:
        roots = [os.path.join(root, d) for d in datasets]
    else:
        roots = [root]

    subs: Set[str] = set()
    seen = 0
    for r in roots:
        if not os.path.isdir(r):
            continue
        for dirpath, _, files in os.walk(r):
            for f in files:
                if not f.endswith(".npz"):
                    continue
                sid = _extract_sid_from_name(f)
                if sid != "unknown":
                    subs.add(sid)
                seen += 1
                if seen >= max_files:
                    break
            if seen >= max_files:
                break
        if seen >= max_files:
            break
    out = sorted(list(subs))
    if not out:
        raise RuntimeError(f"No subjects discovered under {roots}. Are your .npz files mounted?")
    return out


# ----------------------------
# Losses & helpers
# ----------------------------
class FocalLoss(nn.Module):
    """
    Binary focal loss with logits.
    """

    def __init__(self, alpha: float = 0.5, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        if logits.ndim > 1:
            logits = logits.squeeze(-1)
        targets = targets.float()
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)  # p_t
        focal = (1 - pt).pow(self.gamma) * bce
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal = alpha_t * focal
        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


def make_loss(cfg_train: Dict):
    loss_name = str(cfg_train.get("loss", "bce")).lower()
    if loss_name == "focal":
        alpha = float(cfg_train.get("focal_alpha", 0.75))
        gamma = float(cfg_train.get("focal_gamma", 2.0))
        return FocalLoss(alpha=alpha, gamma=gamma)
    # default BCE with optional pos_weight
    pos_weight = float(cfg_train.get("pos_weight", 1.0))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pw = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pw)


def add_hazard_loss_if_available(raw_out, batch, loss_total, device):
    """
    If model returns 'hazard_logits' (B, H) and batch has 'hazard_target' (B, H in {0,1}),
    add BCE loss over hazard bins. Scaled by optional batch['hazard_lambda'] or cfg.
    """
    if not isinstance(raw_out, dict):
        return loss_total
    hz = raw_out.get("hazard_logits", None)
    tgt = batch.get("hazard_target", None)
    if hz is None or tgt is None:
        return loss_total
    if hz.ndim != 2 or tgt.ndim != 2:
        return loss_total
    hz = hz.to(device)
    tgt = tgt.float().to(device)
    hz_loss = torch.nn.functional.binary_cross_entropy_with_logits(hz, tgt)
    lam = float(batch.get("hazard_lambda", 1.0))
    return loss_total + lam * hz_loss


# ----------------------------
# Logit resolver (dict outputs & embeddings)
# ----------------------------
def _get_base(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def extract_logits(model, raw_out, device, opt: Optional[torch.optim.Optimizer] = None):
    """Return a 1-D tensor of logits (B,), creating an aux head if needed."""
    if isinstance(raw_out, dict):
        for k in ("forecast_logit", "logits", "y_pred", "pred"):
            if k in raw_out and torch.is_tensor(raw_out[k]):
                t = raw_out[k]
                return t.squeeze(-1) if t.ndim > 1 else t
        for k in ("fused_emb", "emb", "features"):
            if k in raw_out and torch.is_tensor(raw_out[k]):
                emb = raw_out[k]
                if emb.ndim == 2 and emb.size(1) > 1:
                    base = _get_base(model)
                    head = getattr(base, "_aux_head", None)
                    if head is None:
                        head = nn.Linear(emb.size(1), 1).to(device)
                        setattr(base, "_aux_head", head)
                        if opt is not None:
                            opt.add_param_group({"params": head.parameters()})
                    return head(emb).squeeze(-1)
        # fallback: first tensor in dict
        for v in raw_out.values():
            if torch.is_tensor(v):
                t = v
                return t.squeeze(-1) if t.ndim > 1 else t
        raise ValueError("Model output dict contains no tensors.")
    else:
        t = raw_out
        if torch.is_tensor(t) and t.ndim == 2 and t.size(1) > 1:
            base = _get_base(model)
            head = getattr(base, "_aux_head", None)
            if head is None:
                head = nn.Linear(t.size(1), 1).to(device)
                setattr(base, "_aux_head", head)
                if opt is not None:
                    opt.add_param_group({"params": head.parameters()})
            t = head(t)
        return t.squeeze(-1) if t.ndim > 1 else t


# ----------------------------
# Data builders
# ----------------------------
def _make_loader(ds, batch_size: int, shuffle: bool, workers: int, drop_last: bool,
                 sampler: Optional[WeightedRandomSampler] = None) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(sampler is None) and shuffle,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=(workers > 0),
    )


def build_dataloaders(cfg: Dict, test_subject: str, eval_subject: Optional[str]):
    data_cfg = cfg.get("data", {}) or {}
    data_cfg.setdefault("root", "/data/processed")

    # If subjects are not provided, discover and inject (so LOSODataset can pass them down)
    if "subjects" not in data_cfg or not data_cfg["subjects"]:
        discovered = discover_subjects(data_cfg["root"], data_cfg.get("datasets"))
        data_cfg["subjects"] = discovered
        print(f"[Auto] Discovered {len(discovered)} subjects.")

    from efm.data.dataset import LOSODataset
    ds_tr = LOSODataset(split="train", test_subject=test_subject, eval_subject=eval_subject, **data_cfg)
    ds_ev = LOSODataset(split="eval", test_subject=test_subject, eval_subject=eval_subject, **data_cfg)
    ds_te = LOSODataset(split="test", test_subject=test_subject, eval_subject=eval_subject, **data_cfg)

    world_size = max(1, torch.cuda.device_count())
    bs_cfg = int(cfg["train"]["batch_size"])
    batch_size = max(world_size, (bs_cfg // world_size) * world_size)
    if batch_size != bs_cfg:
        print(f"[Info] Adjusted batch_size {bs_cfg} -> {batch_size} for {world_size} GPUs.")
        cfg["train"]["batch_size"] = batch_size

    workers = int(cfg.get("io", {}).get("workers", 8))

    # Optional oversampling (preictal minority)
    sampler = None
    if bool(cfg["train"].get("oversample_preictal", False)):
        labels = _collect_labels_fast(ds_tr)
        pos = max(1, sum(labels))
        neg = max(1, len(labels) - pos)
        w_pos = neg / (pos + neg)
        w_neg = pos / (pos + neg)
        weights = [w_pos if y == 1 else w_neg for y in labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        print(f"[Oversample] pos={pos}, neg={neg}, w_pos={w_pos:.3f}, w_neg={w_neg:.3f}")

    dl_tr = _make_loader(ds_tr, batch_size, True, workers, drop_last=True, sampler=sampler)
    dl_ev = _make_loader(ds_ev, batch_size, False, workers, drop_last=True)
    dl_te = _make_loader(ds_te, batch_size, False, workers, drop_last=True)
    return dl_tr, dl_ev, dl_te


def _collect_labels_fast(ds) -> List[int]:
    """
    Try to collect labels from dataset file list without fully loading tensors.
    Works if ds exposes `files` list of .npz windows.
    """
    ys = []
    if hasattr(ds, "files"):
        import numpy as _np
        for p in ds.files:
            with _np.load(p, allow_pickle=True) as d:
                ys.append(int(d["label"]))
    else:
        # Fallback: sample first N items (slower)
        N = min(len(ds), 10000)
        for i in range(N):
            ys.append(int(ds[i]["y"]))
    return ys


# ----------------------------
# Loss / Metrics
# ----------------------------
def compute_metrics(logits: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    if logits.ndim > 1:
        logits = logits.squeeze(-1)
    probs = torch.sigmoid(logits)
    pred = (probs >= 0.5).long()
    y = y.long()

    tp = ((pred == 1) & (y == 1)).sum().item()
    tn = ((pred == 0) & (y == 0)).sum().item()
    fp = ((pred == 1) & (y == 0)).sum().item()
    fn = ((pred == 0) & (y == 1)).sum().item()

    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)

    return {"acc": acc, "precision": prec, "recall": rec, "f1": f1}


# ----------------------------
# Train / Eval loops
# ----------------------------
def train_one_epoch(model, dl, opt, loss_fn, device, hazard_lambda: float):
    model.train()
    total, n = 0.0, 0
    for batch in dl:
        batch = to_device(batch, device)
        x, mm, y = batch["x"], batch.get("mm", None), batch["y"].float()
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6).contiguous()

        opt.zero_grad(set_to_none=True)
        raw_out = forward_safe(model, x, mask_missing=mm)
        logits = extract_logits(model, raw_out, device, opt=opt)
        loss = loss_fn(logits, y)

        # optional hazard loss if present
        batch["hazard_lambda"] = hazard_lambda
        loss = add_hazard_loss_if_available(raw_out, batch, loss, device)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        base = _get_base(model)
        if hasattr(base, "_aux_head"):
            nn.utils.clip_grad_norm_(base._aux_head.parameters(), 1.0)
        opt.step()

        total += loss.item()
        n += 1
    return total / max(1, n)


@torch.no_grad()
def evaluate(model, dl, device):
    model.eval()
    all_logits, all_y = [], []
    for batch in dl:
        batch = to_device(batch, device)
        x, mm, y = batch["x"], batch.get("mm", None), batch["y"].long()
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6).contiguous()

        raw_out = forward_safe(model, x, mask_missing=mm)
        # prefer direct forecast logit if provided
        if isinstance(raw_out, dict) and "forecast_logit" in raw_out and torch.is_tensor(raw_out["forecast_logit"]):
            logits = raw_out["forecast_logit"]
            logits = logits.squeeze(-1) if logits.ndim > 1 else logits
        else:
            logits = extract_logits(model, raw_out, device, opt=None)

        all_logits.append(logits.detach().cpu())
        all_y.append(y.detach().cpu())

    if len(all_logits) == 0:
        return {"acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    logits = torch.cat(all_logits, dim=0)
    y = torch.cat(all_y, dim=0)
    return compute_metrics(logits, y)


# ----------------------------
# Early stopping
# ----------------------------
class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.0, mode="max"):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.best = None
        self.bad = 0

    def step(self, val):
        if self.best is None:
            self.best = val
            return False
        improve = (val - self.best) > self.min_delta if self.mode == "max" else (self.best - val) > self.min_delta
        if improve:
            self.best = val
            self.bad = 0
            return False
        self.bad += 1
        return self.bad >= self.patience


# ----------------------------
# LOSO runner
# ----------------------------
def subject_list_from_cfg(cfg):
    data_cfg = cfg.get("data", {}) or {}
    subs = data_cfg.get("subjects", None)
    if subs:
        return subs
    # auto-discover
    subs = discover_subjects(data_cfg.get("root", "/data/processed"), data_cfg.get("datasets"))
    return subs


def loso_eval(cfg, out_dir, subjects_limit=None):
    ensure_dir(out_dir)
    results_csv = os.path.join(out_dir, "results_loso.csv")
    print(f"[Info] Results will be written to: {results_csv}")

    # import model + config
    from efm.models.efm_model import EFM, EFMConfig

    # Train knobs
    lr = float(cfg["train"].get("lr", 3e-4))
    weight_decay = float(cfg["train"].get("weight_decay", 0.05))
    max_epochs = int(cfg["train"].get("epochs", 80))
    patience = int(cfg["train"].get("early_stop_patience", 15))
    min_delta = float(cfg["train"].get("early_stop_min_delta", 0.001))
    hazard_lambda = float(cfg["train"].get("hazard_lambda", 1.0))

    mcfg = cfg.get("model", {})
    in_ch = int(cfg.get("in_ch", mcfg.get("in_ch", 19)))

    # GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"[CUDA] Using {torch.cuda.device_count()} GPU(s): "
              f"{[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")

    # Subjects
    subjects = subject_list_from_cfg(cfg)
    if subjects_limit:
        subjects = subjects[:subjects_limit]

    rows = []
    for i, test_subject in enumerate(subjects, start=1):
        eval_subject = cfg["train"].get("eval_subject", None)
        if eval_subject is None and len(subjects) > 1:
            eval_subject = subjects[(i) % len(subjects)]

        print(f"\nFold {i}/{len(subjects)} — test={test_subject}, eval={eval_subject}, train={len(subjects) - 1} subs")
        dl_tr, dl_ev, dl_te = build_dataloaders(cfg, test_subject, eval_subject)

        efmcfg = EFMConfig(
            in_ch=in_ch,
            dim=mcfg.get("dim", 256),
            raw_depth=mcfg.get("raw_depth", 4),
            spec_depth=mcfg.get("spec_depth", 4),
            heads=mcfg.get("heads", 4),
            drop=mcfg.get("drop", 0.1),
            patch_len=mcfg.get("patch_len", 64),
            n_fft=mcfg.get("n_fft", 256),
            hop=mcfg.get("hop", 128),
            spec_ch=mcfg.get("spec_ch", 4),
            patch2d=tuple(mcfg.get("patch2d", (8, 8))),
            proj_dim=mcfg.get("proj_dim", 128),
            cls_hidden=mcfg.get("cls_hidden", 256),
            use_hazard=mcfg.get("use_hazard", False),
            hz_bins=mcfg.get("hz_bins", 12),
        )

        model = EFM(efmcfg).to(device)

        # Optional: load pretrained
        pretrain_path = cfg.get("train", {}).get("load_pretrained", "")
        if pretrain_path and os.path.exists(pretrain_path):
            state = torch.load(pretrain_path, map_location="cpu")
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            try:
                model.load_state_dict(state, strict=False)
            except RuntimeError:
                state = {k.replace("module.", ""): v for k, v in state.items()}
                model.load_state_dict(state, strict=False)
            print(f"[Init] Loaded pretrained weights from {pretrain_path}")

        if torch.cuda.device_count() > 1:
            print(f"[Info] Using DataParallel over {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)

        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = make_loss(cfg["train"])
        stopper = EarlyStopper(patience=patience, min_delta=min_delta, mode="max")

        best_ev, best_state = -1.0, None
        for epoch in range(1, max_epochs + 1):
            tr_loss = train_one_epoch(model, dl_tr, opt, loss_fn, device, hazard_lambda)
            ev = evaluate(model, dl_ev, device)
            if ev["f1"] > best_ev:
                best_ev = ev["f1"]
                state = model.state_dict()
                best_state = {k: v.cpu() for k, v in state.items()}

            print(f"Epoch {epoch}/{max_epochs} | train_loss={tr_loss:.4f} | "
                  f"eval_f1={ev['f1']:.3f} | acc={ev['acc']:.3f} | prec={ev['precision']:.3f} | rec={ev['recall']:.3f}")

            if stopper.step(ev["f1"]):
                print(f"[EarlyStop] patience reached at epoch {epoch}")
                break

        # Restore best for test (handle DP prefixes)
        if best_state is not None:
            try:
                model.load_state_dict(best_state)
            except RuntimeError:
                best_state = {k.replace("module.", ""): v for k, v in best_state.items()}
                model.load_state_dict(best_state, strict=False)

        te = evaluate(model, dl_te, device)
        print(f"[TEST] {test_subject}: acc={te['acc']:.3f}, f1={te['f1']:.3f}")
        rows.append({
            "subject": test_subject,
            "acc": f"{te['acc']:.4f}",
            "precision": f"{te['precision']:.4f}",
            "recall": f"{te['recall']:.4f}",
            "f1": f"{te['f1']:.4f}",
        })
        del model
        torch.cuda.empty_cache()

    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["subject", "acc", "precision", "recall", "f1"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "avg_acc": float(np.mean([float(r["acc"]) for r in rows])) if rows else 0.0,
        "avg_f1": float(np.mean([float(r["f1"]) for r in rows])) if rows else 0.0,
        "avg_precision": float(np.mean([float(r["precision"]) for r in rows])) if rows else 0.0,
        "avg_recall": float(np.mean([float(r["recall"]) for r in rows])) if rows else 0.0,
        "n_folds": len(rows)
    }
    with open(os.path.join(out_dir, "results_summary.json"), "w") as jf:
        json.dump(summary, jf, indent=2)
    print("[Summary]", summary)


# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--subjects_limit", type=int, default=0)
    args = ap.parse_args()

    set_seed(42)
    cfg = load_yaml(args.config)
    ensure_dir(args.out)

    nlim = args.subjects_limit if args.subjects_limit and args.subjects_limit > 0 else None
    loso_eval(cfg, args.out, nlim)


if __name__ == "__main__":
    main()
