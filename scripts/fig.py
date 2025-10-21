#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CHB-MIT Spectral Parameters — phases + band breakdown (two Y-scale variants)
----------------------------------------------------------------------------
- Panel A: EEG phases with shading, blue phase labels, bold dividers
- Panel B: Interictal spectrum with δ θ α β γ band shading
- Panel C: Spectrum comparison with same band shading
- Exports TWO figures:
    1) ..._indY.*  -> Panel B keeps its own y-scale (independent)
    2) ..._sameY.* -> Panel B's y-scale matched to Panel C's
"""

import os
import numpy as np
import mne
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from fooof import FOOOF
from scipy.ndimage import gaussian_filter1d

# ---------------- Config ----------------
edf_path = "/data/chb01/chb01_03.edf"
channel = "Fp1"
baseline = (0, 10)  # Interictal
preictal = (290, 300)  # Preictal
ictal = (300, 310)  # Ictal
fmin, fmax = 1, 40

# Phase shading colors (you set these warm tones)
phase_shades = {
    "Interictal": "#FFD580",  # warm sand
    "Preictal": "#eecc16",  # golden
    "Ictal": "#ff474c",  # coral red
}

# Frequency bands (δ, θ, α, β, γ) for Panels B & C
bands = {
    "δ": (1, 3, "#ebe3ff"),
    "θ": (3, 8, "#a8c5ff"),
    "α": (8, 13, "#bcecc2"),
    "β": (13, 30, "#f6c8c8"),
    "γ": (30, 40, "#ffe0b5"),
}

# Output base
out_dir = "/workspace/figures"
os.makedirs(out_dir, exist_ok=True)
out_base = os.path.join(out_dir, "chb01_03_spectral_phases_bands")


# ---------------- Helpers ----------------
def segment(raw, ch, t0, t1):
    sf = raw.info["sfreq"]
    x = raw.get_data(picks=[ch])[0]
    return x[int(t0 * sf):int(t1 * sf)], sf


def psd(x, sf, fmin=1, fmax=40):
    from mne.time_frequency import psd_array_welch
    p, f = psd_array_welch(x, sf, fmin=fmin, fmax=fmax, n_fft=2048, average="median")
    return f, p.squeeze()


def smooth(x):
    return np.clip(gaussian_filter1d(x, 1), 1e-20, None)


def make_figure(log_i, log_p, log_s, f, x_inter, sf, match_y=False):
    """Build the figure; if match_y=True, set Panel B y-lims = Panel C y-lims."""
    fig = plt.figure(figsize=(13.5, 4.5), constrained_layout=True)
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[2.0, 2.5, 2.5])

    # ===== Panel A: EEG with phase shading + blue labels + bold dividers =====
    axA = fig.add_subplot(gs[0])
    t = np.arange(len(x_inter)) / sf
    axA.plot(t, x_inter * 1e6, color="black", lw=0.9)
    axA.set_title("A  EEG Time Series", loc="left")
    axA.set_xlabel("Time (s)")
    axA.set_ylabel("Amplitude (µV)")
    axA.set_xlim(0, 5)
    axA.grid(alpha=0.2)

    # three phases across the 5 s pane (same as before)
    phase_order = [("Interictal", 0.0, 1.5),
                   ("Preictal", 1.5, 3.0),
                   ("Ictal", 3.0, 4.5)]
    for name, x0, x1 in phase_order:
        axA.axvspan(x0, x1, facecolor=phase_shades[name], alpha=0.6, lw=0)
    for _, _, x1 in phase_order[:-1]:
        axA.axvline(x1, color="black", lw=2.5, alpha=0.85)
    for name, x0, x1 in phase_order:
        x_mid = 0.5 * (x0 + x1)
        axA.text(x_mid, 0.93, name, color="blue", fontweight="bold", fontsize=11,
                 ha="center", va="top", transform=axA.get_xaxis_transform())

    # ===== Panel C first (to know its y-lims if matching) =====
    axC = fig.add_subplot(gs[2])
    # band shading & labels a tad below top
    for label, (lo, hi, shade) in bands.items():
        axC.axvspan(lo, hi, color=shade, alpha=0.3, lw=0)
        axC.text((lo + hi) / 2, np.max([log_i.max(), log_p.max(), log_s.max()]) + 0.05,
                 label, ha="center", va="bottom", fontsize=11, fontweight="bold", color="black")
    axC.plot(f, log_i, lw=1.0, label="Interictal", color="#4063D8")
    axC.plot(f, log_p, lw=1.0, label="Preictal", color="#F18F01")
    axC.plot(f, log_s, lw=1.0, label="Ictal", color="#CB3C33")
    axC.set_title("C  Spectrum Comparison", loc="left")
    axC.set_xlabel("Frequency (Hz)")
    axC.set_ylabel("log10 Power")
    axC.set_xlim(fmin, fmax)
    axC.legend(frameon=False, loc="upper right")

    # ===== Panel B: Interictal spectrum + band shading =====
    axB = fig.add_subplot(gs[1])
    for label, (lo, hi, shade) in bands.items():
        axB.axvspan(lo, hi, color=shade, alpha=0.3, lw=0)
        axB.text((lo + hi) / 2, np.max(log_i) + 0.05, label, ha="center", va="bottom",
                 fontsize=11, fontweight="bold", color="black")
    axB.plot(f, log_i, color="black", lw=1.0, label="Power Spectrum")
    fm = FOOOF(peak_width_limits=[1, 12], max_n_peaks=5, verbose=False)
    fm.fit(f, 10 ** log_i)  # fit expects linear power
    axB.plot(f, np.log10(np.clip(fm._ap_fit, 1e-20, None)), "b--", lw=1.0, label="Aperiodic fit")
    axB.set_title("B  Spectral Parameters", loc="left")
    axB.set_xlabel("Frequency (Hz)")
    axB.set_ylabel("log10 Power")
    axB.set_xlim(fmin, fmax)
    axB.legend(frameon=False, loc="upper right")

    # If requested, match Panel B's y-axis to Panel C's range
    if match_y:
        y0, y1 = axC.get_ylim()
        axB.set_ylim(y0, y1)

    return fig


# ---------------- Load & compute ----------------
print(f"Loading EDF: {edf_path}")
raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
raw.pick_types(eeg=True)
raw.filter(1., 40., verbose="ERROR")
if channel not in raw.ch_names:
    channel = raw.ch_names[0]
print(f"Using channel: {channel}")

x_inter, sf = segment(raw, channel, *baseline)
x_prei, _ = segment(raw, channel, *preictal)
x_ictal, _ = segment(raw, channel, *ictal)

f, p_inter = psd(x_inter, sf, fmin, fmax)
_, p_prei = psd(x_prei, sf, fmin, fmax)
_, p_ictal = psd(x_ictal, sf, fmin, fmax)

log_i = np.log10(smooth(p_inter))
log_p = np.log10(smooth(p_prei))
log_s = np.log10(smooth(p_ictal))

# ---------------- Build & Save BOTH variants ----------------
# 1) Independent y-scales
fig1 = make_figure(log_i, log_p, log_s, f, x_inter, sf, match_y=False)
fig1.savefig(f"{out_base}_indY.png", dpi=300, bbox_inches="tight")
fig1.savefig(f"{out_base}_indY.pdf", dpi=300, bbox_inches="tight")
plt.close(fig1)

# 2) Matched y-scales (Panel B matches Panel C)
fig2 = make_figure(log_i, log_p, log_s, f, x_inter, sf, match_y=True)
fig2.savefig(f"{out_base}_sameY.png", dpi=300, bbox_inches="tight")
fig2.savefig(f"{out_base}_sameY.pdf", dpi=300, bbox_inches="tight")
plt.close(fig2)

print(
    "\n✅ Saved:\n"
    f"  {out_base}_indY.png\n  {out_base}_indY.pdf\n"
    f"  {out_base}_sameY.png\n  {out_base}_sameY.pdf\n"
)
