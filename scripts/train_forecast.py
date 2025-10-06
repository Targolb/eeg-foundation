# scripts/train_forecast.py
import os
import csv
import json
import argparse
import random
from typing import Dict, List, Optional

import torch

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

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
# Logit resolver (handles dict outputs & embeddings)
# ----------------------------
def _get_base(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def extract_logits(model, raw_out, device, opt: Optional[torch.optim.Optimizer] = None):
    """Return a 1-D tensor of logits (B,), creating an aux head if needed."""
    # 1) If dict, try common keys
    if isinstance(raw_out, dict):
        for k in ("forecast_logit", "logits", "y_pred", "pred"):
            if k in raw_out and torch.is_tensor(raw_out[k]):
                t = raw_out[k]
                return t.squeeze(-1) if t.ndim > 1 else t
        # 2) Try embeddings -> attach/use aux head
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
        # 3) Fallback to first tensor in dict
        for v in raw_out.values():
            if torch.is_tensor(v):
                t = v
                break
        else:
            raise ValueError("Model output dict contains no tensors.")
    else:
        t = raw_out

    # If still 2D features, attach/use aux head
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
def _make_loader(ds, batch_size: int, shuffle: bool, workers: int, drop_last: bool) -> DataLoader:
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=workers, pin_memory=True, drop_last=drop_last
    )


def build_dataloaders(cfg: Dict, test_subject: str, eval_subject: Optional[str]):
    data_cfg = cfg.get("data", {}) or {}
    data_cfg.setdefault("root", "/data/processed")

    from efm.data.dataset import LOSODataset
    ds_tr = LOSODataset(split="train", test_subject=test_subject, eval_subject=eval_subject, **data_cfg)
    ds_ev = LOSODataset(split="eval", test_subject=test_subject, eval_subject=eval_subject, **data_cfg)
    ds_te = LOSODataset(split="test", test_subject=test_subject, eval_subject=eval_subject, **data_cfg)

    world_size = max(1, torch.cuda.device_count())
    bs_cfg = cfg["train"]["batch_size"]
    batch_size = max(world_size, (bs_cfg // world_size) * world_size)
    if batch_size != bs_cfg:
        print(f"[Info] Adjusted batch_size {bs_cfg} -> {batch_size} for {world_size} GPUs.")
        cfg["train"]["batch_size"] = batch_size

    workers = cfg.get("io", {}).get("workers", 4)

    dl_tr = _make_loader(ds_tr, batch_size, True, workers, drop_last=True)
    dl_ev = _make_loader(ds_ev, batch_size, False, workers, drop_last=True)
    dl_te = _make_loader(ds_te, batch_size, False, workers, drop_last=True)

    return dl_tr, dl_ev, dl_te


# ----------------------------
# Loss / Metrics
# ----------------------------
def bce_logits_loss(pos_weight: float):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pw = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pw)


@torch.no_grad()
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
def train_one_epoch(model, dl, opt, pos_weight, device):
    model.train()
    loss_fn = bce_logits_loss(pos_weight)
    total, n = 0.0, 0
    for batch in dl:
        batch = to_device(batch, device)
        x, mm, y = batch["x"], batch.get("mm", None), batch["y"].float()
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6).contiguous()

        opt.zero_grad(set_to_none=True)
        raw_out = forward_safe(model, x, mask_missing=mm)
        logits = extract_logits(model, raw_out, device, opt=opt)
        loss = loss_fn(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # also clip aux head if present
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
    base = _get_base(model)
    for batch in dl:
        batch = to_device(batch, device)
        x, mm, y = batch["x"], batch.get("mm", None), batch["y"].long()
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6).contiguous()

        raw_out = forward_safe(model, x, mask_missing=mm)
        # use same aux head if it exists
        if isinstance(raw_out, dict) and "forecast_logit" in raw_out:
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
# LOSO runner
# ----------------------------
def subject_list_from_cfg(cfg):
    data_cfg = cfg.get("data", {}) or {}
    subs = data_cfg.get("subjects", None)
    if subs is not None:
        return subs
    raise RuntimeError("Please define cfg['data']['subjects'].")


def loso_eval(cfg, out_dir, subjects_limit=None):
    ensure_dir(out_dir)
    results_csv = os.path.join(out_dir, "results_loso.csv")
    print(f"[Info] Results will be written to: {results_csv}")

    subjects = subject_list_from_cfg(cfg)
    if subjects_limit:
        subjects = subjects[:subjects_limit]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from efm.models.efm_model import EFM, EFMConfig

    lr = cfg["train"].get("lr", 1e-3)
    weight_decay = cfg["train"].get("weight_decay", 0.0)
    max_epochs = cfg["train"].get("epochs", 10)
    pos_weight = cfg["train"].get("pos_weight", 1.0)
    mcfg = cfg.get("model", {})
    in_ch = cfg.get("in_ch", mcfg.get("in_ch", 19))

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
        if torch.cuda.device_count() > 1:
            print(f"[Info] Using DataParallel over {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)

        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        best_ev, best_state = -1.0, None

        for epoch in range(1, max_epochs + 1):
            tr_loss = train_one_epoch(model, dl_tr, opt, pos_weight, device)
            ev = evaluate(model, dl_ev, device)
            if ev["f1"] > best_ev:
                best_ev = ev["f1"]
                # Save DP-safe state
                state = model.state_dict()
                best_state = {k: v.cpu() for k, v in state.items()}
            print(f"Epoch {epoch}/{max_epochs} | train_loss={tr_loss:.4f} | eval_f1={ev['f1']:.3f}")

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
        "avg_acc": np.mean([float(r["acc"]) for r in rows]),
        "avg_f1": np.mean([float(r["f1"]) for r in rows]),
        "avg_precision": np.mean([float(r["precision"]) for r in rows]),
        "avg_recall": np.mean([float(r["recall"]) for r in rows]),
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
