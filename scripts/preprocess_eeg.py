#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Harmonized EEG preprocessing for transfer learning / seizure forecasting.

Implements:
• Sampling: resample to target fs (default 256 Hz).
• Referencing: common average reference (CAR); remove DC; 0.5–40 Hz bandpass; optional 60 Hz notch.
• Channel map: map into 10–20 template; missing channels masked (no imputation). Provide indices for learnable embeddings.
• Windows: fixed 30 s with 50% overlap by default.
• Normalization: robust per-channel z-score (median / IQR).
• Artifacts: simple amplitude + line-noise checks; optional fast ICA for light cleanup.
• Forecasting labels: SPH/SOP framing relative to seizure onsets.

Outputs .npz files per window with keys:
  'x'                : (C, T) float32 normalized signal
  'mask_missing'     : (C,) uint8   1 = missing channel in source
  'mask_artifact'    : ()   uint8   1 = window flagged as artifact
  'ch_names'         : list[str]    target montage channel order
  'fs'               : float        sampling rate Hz
  't0','t1'          : float        window start/end (seconds from EDF start)
  'label'            : int          1 = preictal (within SPH before onset) or within SOP after onset; else 0
  'missing_indices'  : list[int]    indices of missing channels (for learnable embeddings in the model)

Author: ChatGPT (GPT-5 Thinking)
"""
from __future__ import annotations

import os
import argparse
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from scipy.signal import welch
import mne


# ---------------------------
# Configuration dataclass
# ---------------------------

@dataclass
class PreprocConfig:
    fs: int = 256
    l_freq: float = 0.5
    h_freq: float = 40.0
    notch: Optional[float] = 60.0
    template: str = "standard_1020"
    win_sec: float = 30.0
    overlap: float = 0.5
    amp_thresh_uv: float = 300.0  # amplitude threshold (µV) for artifact flag
    linenoise_band: Tuple[float, float] = (57.0, 63.0)  # around 60 Hz
    linenoise_ratio_thresh: float = 0.5  # power60 / total_bandpower over (0.5–40) -> artifact if > thresh
    do_ica: bool = False
    ica_var: float = 0.95  # keep 95% variance in ICA
    sph_min: float = 60.0  # seizure prediction horizon (minutes)
    sop_min: float = 30.0  # seizure occurrence period (minutes)


# Common 10–20 subset to standardize across datasets (19-ch + midline)
TARGET_1020 = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3", "C3", "Cz", "C4", "T4",
    "T5", "P3", "Pz", "P4", "T6",
    "O1", "O2",
]

# Some datasets use T7/T8 and P7/P8 naming; map synonyms
CANONICAL_SYNONYMS = {
    "T7": "T3",
    "T8": "T4",
    "P7": "T5",
    "P8": "T6",
    "FP1": "Fp1",
    "FP2": "Fp2",
    "CZ": "Cz", "FZ": "Fz", "PZ": "Pz",
    "OZ": "Oz",  # not in target, but normalize
}


def canonicalize(name: str) -> str:
    n = name.strip().replace(' ', '')
    n = n.replace('-REF', '').replace('.', '')
    n = CANONICAL_SYNONYMS.get(n.upper(), n)
    # ensure capitalization like Fp1 not FP1
    repl = {"FP": "Fp", "FZ": "Fz", "CZ": "Cz", "PZ": "Pz", "OZ": "Oz"}
    for k, v in repl.items():
        if n.upper().startswith(k):
            return v + n[len(k):]
    return n


# ---------------------------
# IO helpers
# ---------------------------

def read_annotations(ann_csv: Optional[str]) -> pd.DataFrame:
    if ann_csv is None or not os.path.exists(ann_csv):
        return pd.DataFrame(columns=["edf_path", "start_time_sec"])
    df = pd.read_csv(ann_csv)
    if "edf_path" not in df.columns or "start_time_sec" not in df.columns:
        raise ValueError("Annotation CSV must have columns: edf_path,start_time_sec")
    return df


def seizures_for_file(df: pd.DataFrame, edf_path: str) -> List[float]:
    if df is None or df.empty:
        return []
    hits = df[df["edf_path"] == edf_path]
    return sorted(hits["start_time_sec"].tolist())


# ---------------------------
# Core preprocessing
# ---------------------------

def load_and_preprocess(edf_path: str, cfg: PreprocConfig) -> mne.io.BaseRaw:
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    # DC removal (demean over short segments is implicit in filtering, but also center)
    raw._data = raw.get_data() - np.mean(raw.get_data(), axis=1, keepdims=True)

    # Resample
    if int(raw.info["sfreq"]) != cfg.fs:
        raw.resample(cfg.fs)

    # Filtering
    raw.filter(l_freq=cfg.l_freq, h_freq=cfg.h_freq, method="iir", verbose="ERROR")
    if cfg.notch:
        # Some datasets already notch; safe to apply once
        raw.notch_filter(freqs=[cfg.notch], verbose="ERROR")

    # Reference: Common Average Reference (CAR)
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    return raw


def map_to_template(raw: mne.io.BaseRaw, target: List[str]) -> Tuple[np.ndarray, List[str], np.ndarray, List[int]]:
    """Return data as (C,T), ch_names, mask_missing(C,), missing_indices list"""
    ch_names_src = [canonicalize(ch) for ch in raw.info["ch_names"]]
    name_to_idx = {canonicalize(ch): i for i, ch in enumerate(raw.info["ch_names"])}

    X = np.zeros((len(target), raw.n_times), dtype=np.float32)
    mask_missing = np.zeros((len(target),), dtype=np.uint8)
    missing_indices: List[int] = []

    for i, ch in enumerate(target):
        src_idx = name_to_idx.get(ch, None)
        if src_idx is None:
            mask_missing[i] = 1
            missing_indices.append(i)
            # leave zeros; model can learn an embedding for missing channel
        else:
            X[i] = raw.get_data(picks=[src_idx]).astype(np.float32).squeeze(0)

    return X, target, mask_missing, missing_indices


def robust_zscore_inplace(X: np.ndarray, mask_missing: np.ndarray, eps: float = 1e-6) -> None:
    """Median/IQR per channel; ignore missing channels (keep zeros) but still standardize scale to ~1 where present."""
    C, T = X.shape
    for c in range(C):
        if mask_missing[c]:
            continue
        x = X[c]
        med = np.median(x)
        q1 = np.percentile(x, 25)
        q3 = np.percentile(x, 75)
        iqr = max(q3 - q1, eps)
        X[c] = (x - med) / (iqr / 1.349)  # IQR-to-sigma approx


def linenoise_ratio(x: np.ndarray, fs: float, band: Tuple[float, float] = (57, 63),
                    total_band: Tuple[float, float] = (0.5, 40)) -> float:
    f, Pxx = welch(x, fs=fs, nperseg=min(len(x), 4 * int(fs)))

    def band_power(lo, hi):
        idx = (f >= lo) & (f <= hi)
        return np.trapz(Pxx[idx], f[idx]) if np.any(idx) else 0.0

    p60 = band_power(*band)
    ptotal = band_power(*total_band)
    return (p60 / (ptotal + 1e-12))


def artifact_flag(Xw: np.ndarray, fs: float, amp_thresh_uv: float, linenoise_thresh: float) -> int:
    """Return 1 if window is artifact-heavy, else 0. Xw is (C,T) in volts after MNE; convert to µV for amplitude check."""
    # MNE raw.get_data is in Volts; convert to microvolts
    amp_uv = np.max(np.abs(Xw)) * 1e6
    if amp_uv > amp_thresh_uv:
        return 1
    # line-noise check on the (robust) average channel to be fast
    avg = np.median(Xw, axis=0)
    if linenoise_ratio(avg, fs) > linenoise_thresh:
        return 1
    return 0


def maybe_do_ica(raw: mne.io.BaseRaw, cfg: PreprocConfig) -> mne.io.BaseRaw:
    if not cfg.do_ica:
        return raw
    ica = mne.preprocessing.ICA(n_components=cfg.ica_var, method="fastica", max_iter="auto", random_state=97,
                                verbose="ERROR")
    ica.fit(raw.copy().filter(1., None, verbose="ERROR"))
    raw = ica.apply(raw, verbose="ERROR")
    return raw


def window_signal(X: np.ndarray, fs: int, win_sec: float, overlap: float) -> List[Tuple[int, int]]:
    step = int(win_sec * fs * (1.0 - overlap))
    size = int(win_sec * fs)
    idx = []
    start = 0
    while start + size <= X.shape[1]:
        idx.append((start, start + size))
        start += step if step > 0 else size
    return idx


def label_windows(wins: List[Tuple[int, int]], fs: int, seizure_onsets_sec: List[float], sph_min: float,
                  sop_min: float) -> List[int]:
    labels = []
    sph = sph_min * 60.0
    sop = sop_min * 60.0
    for (a, b) in wins:
        t0 = a / fs
        t1 = b / fs
        y = 0
        for onset in seizure_onsets_sec:
            # preictal if window END is within [onset - sph, onset)
            # post-onset SOP positive windows counted as positive to avoid leakage around onset
            if (onset - sph) <= t1 < onset or (onset <= t0 <= onset + sop):
                y = 1
                break
        labels.append(y)
    return labels


def export_window(out_path: str, Xw: np.ndarray, mask_missing: np.ndarray, artifact_mask: int, fs: float, t0: float,
                  t1: float, label: int, ch_names: List[str], missing_indices: List[int]):
    np.savez_compressed(
        out_path,
        x=Xw.astype(np.float32),
        mask_missing=mask_missing.astype(np.uint8),
        mask_artifact=np.uint8(artifact_mask),
        fs=np.float32(fs),
        t0=np.float32(t0),
        t1=np.float32(t1),
        label=np.int8(label),
        ch_names=np.array(ch_names, dtype=object),
        missing_indices=np.array(missing_indices, dtype=np.int16),
    )


def process_file(edf_path: str, out_dir: str, seizures_df: pd.DataFrame, cfg: PreprocConfig) -> int:
    raw = load_and_preprocess(edf_path, cfg)
    raw = maybe_do_ica(raw, cfg)

    X, ch_names, mask_missing, missing_indices = map_to_template(raw, TARGET_1020)
    robust_zscore_inplace(X, mask_missing)

    wins = window_signal(X, cfg.fs, cfg.win_sec, cfg.overlap)
    ann = seizures_for_file(seizures_df, edf_path)
    labels = label_windows(wins, cfg.fs, ann, cfg.sph_min, cfg.sop_min)

    base = os.path.splitext(os.path.basename(edf_path))[0]
    os.makedirs(out_dir, exist_ok=True)

    n_written = 0
    for (a, b), y in zip(wins, labels):
        Xw = X[:, a:b]
        art = artifact_flag(Xw, cfg.fs, cfg.amp_thresh_uv, cfg.linenoise_ratio_thresh)
        t0 = a / cfg.fs
        t1 = b / cfg.fs
        out_name = f"{base}__{int(t0)}_{int(t1)}s__y{y}.npz"
        export_window(os.path.join(out_dir, out_name), Xw, mask_missing, art, cfg.fs, t0, t1, y, ch_names,
                      missing_indices)
        n_written += 1
    return n_written


def discover_edfs(input_dir: str) -> List[str]:
    edfs = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(".edf"):
                edfs.append(os.path.join(root, f))
    edfs.sort()
    return edfs


def main():
    ap = argparse.ArgumentParser(description="Harmonized EEG preprocessing -> NPZ windows")
    ap.add_argument("--input_dir", required=True, help="Directory with EDF files (recursively searched).")
    ap.add_argument("--ann_csv", default=None,
                    help="CSV with columns: edf_path,start_time_sec (seizure onsets in sec).")
    ap.add_argument("--out_dir", required=True, help="Output directory for NPZ windows.")
    ap.add_argument("--fs", type=int, default=256)
    ap.add_argument("--band", nargs=2, type=float, default=[0.5, 40.0], help="Bandpass [l h] Hz")
    ap.add_argument("--notch", type=float, default=60.0, help="Notch frequency Hz (set 0 to disable).")
    ap.add_argument("--template", type=str, default="standard_1020")
    ap.add_argument("--win_sec", type=float, default=30.0)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--amp_thresh_uv", type=float, default=300.0)
    ap.add_argument("--linenoise_ratio_thresh", type=float, default=0.5)
    ap.add_argument("--do_ica", action="store_true")
    ap.add_argument("--ica_var", type=float, default=0.95)
    ap.add_argument("--sph_min", type=float, default=60.0)
    ap.add_argument("--sop_min", type=float, default=30.0)
    args = ap.parse_args()

    cfg = PreprocConfig(
        fs=args.fs,
        l_freq=args.band[0],
        h_freq=args.band[1],
        notch=None if args.notch <= 0 else args.notch,
        template=args.template,
        win_sec=args.win_sec,
        overlap=args.overlap,
        amp_thresh_uv=args.amp_thresh_uv,
        linenoise_ratio_thresh=args.linenoise_ratio_thresh,
        do_ica=args.do_ica,
        ica_var=args.ica_var,
        sph_min=args.sph_min,
        sop_min=args.sop_min,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    seizures_df = read_annotations(args.ann_csv)
    edfs = discover_edfs(args.input_dir)
    if not edfs:
        print("No EDF files found.", flush=True)
        return

    total = 0
    for i, edf_path in enumerate(edfs, 1):
        try:
            n = process_file(edf_path, args.out_dir, seizures_df, cfg)
            print(f"[{i}/{len(edfs)}] {os.path.basename(edf_path)} -> {n} windows")
            total += n
        except Exception as e:
            print(f"ERROR processing {edf_path}: {e}")
    print(f"Done. Wrote {total} windows to {args.out_dir}")


if __name__ == "__main__":
    main()
