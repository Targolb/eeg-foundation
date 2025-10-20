#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CHB-MIT Spectral Parameters — With Interictal / Preictal / Ictal Partition Overlay
-------------------------------------------------------------------------------
Adds colored partitions (Interictal, Preictal, Ictal) across all 3 panels (A–C),
making the temporal phases visually consistent with the EEG waveform.

Outputs:
  /workspace/figures/chb01_03_spectral_phases.png
  /workspace/figures/chb01_03_spectral_phases.pdf
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
baseline = (0, 10)  # interictal
preictal = (290, 300)  # preictal
ictal = (300, 310)  # ictal
fmin, fmax = 1, 40

# Phase colors (UCCS-style)
phase_colors = {
    "Interictal": "#7F00FF",  # violet
    "Preictal": "#FFCCCB",  # light red/pink
    "Ictal": "#FFD580",  # light orange
}

# Output
out_dir = "/workspace/figures"
os.makedirs(out_dir, exist_ok=True)
png_path = os.path.join(out_dir, "chb01_03_spectral_phases.png")
pdf_path = os.path.join(out_dir, "chb01_03_spectral_phases.pdf")


# ---------------- Helpers ----------------
def band_mask(f, lo, hi):
    return (f >= lo) & (f < hi)


def segment(raw, ch, t0, t1):
    sf = raw.info["sfreq"]
    x = raw.get_data(picks=[ch])[0]
    return x[int(t0 * sf):int(t1 * sf)], sf


def psd(x, sf, fmin=1, fmax=40):
    from mne.time_frequency import psd_array_welch
    p, f = psd_array_welch(x, sf, fmin=fmin, fmax=fmax, n_fft=2048, average="median")
    return f, p.squeeze()


# ---------------- Load EEG ----------------
print(f"Loading EDF: {edf_path}")
raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
raw.pick_types(eeg=True)
raw.filter(1., 40., verbose="ERROR")
if channel not in raw.ch_names:
    channel = raw.ch_names[0]
print(f"Using channel: {channel}")

# Segments
x_base, sf = segment(raw, channel, *baseline)
x_prei, _ = segment(raw, channel, *preictal)
x_ictal, _ = segment(raw, channel, *ictal)

# PSD
f, psd_base = psd(x_base, sf, fmin, fmax)
_, psd_prei = psd(x_prei, sf, fmin, fmax)
_, psd_ictal = psd(x_ictal, sf, fmin, fmax)


# Smooth for plotting
def smooth(x): return np.clip(gaussian_filter1d(x, 1), 1e-20, None)


log_b = np.log10(smooth(psd_base))
log_p = np.log10(smooth(psd_prei))
log_i = np.log10(smooth(psd_ictal))

# ---------------- Figure Layout ----------------
fig = plt.figure(figsize=(13.5, 4.5), constrained_layout=True)
gs = gridspec.GridSpec(1, 4, figure=fig, width_ratios=[2.0, 2.5, 2.5, 1.5])

# ----- Panel A: EEG Time Series -----
axA = fig.add_subplot(gs[0])
t = np.arange(len(x_base)) / sf
axA.plot(t, x_base * 1e6, color="black", lw=0.9)
axA.set_title("A  EEG Time Series", loc="left")
axA.set_xlabel("Time (s)")
axA.set_ylabel("Amplitude (µV)")
axA.set_xlim(0, 5)
axA.grid(alpha=0.2)

# Add shaded partitions (Interictal, Preictal, Ictal)
for i, (name, color) in enumerate(phase_colors.items()):
    axA.axvspan(i * 1.5, (i + 1) * 1.5, facecolor=color, alpha=0.25, lw=0)
    axA.text((i * 1.5 + (i + 1) * 1.5) / 2, np.max(x_base * 1e6) * 1.05, name,
             color=color.replace("f", ""), ha="center", va="bottom", fontweight="bold")

# ----- Panel B: Spectral Parameters (Interictal) -----
axB = fig.add_subplot(gs[1])
axB.plot(f, log_b, color="black", lw=1.0, label="Power Spectrum")
fm = FOOOF(peak_width_limits=[1, 12], max_n_peaks=5, verbose=False)
fm.fit(f, psd_base)
axB.plot(f, np.log10(np.clip(fm._ap_fit, 1e-20, None)), "b--", lw=1.0, label="Aperiodic fit")
axB.set_title("B  Spectral Parameters", loc="left")
axB.set_xlabel("Frequency (Hz)")
axB.set_ylabel("log10 Power")
axB.set_xlim(fmin, fmax)
axB.legend(frameon=False, loc="upper right")

# ----- Panel C: Spectrum Comparison -----
axC = fig.add_subplot(gs[2])
axC.plot(f, log_b, lw=1.0, label="Interictal", color="#4063D8")
axC.plot(f, log_p, lw=1.0, label="Preictal", color="#F18F01")
axC.plot(f, log_i, lw=1.0, label="Ictal", color="#CB3C33")
axC.set_title("C  Spectrum Comparison", loc="left")
axC.set_xlabel("Frequency (Hz)")
axC.set_ylabel("log10 Power")
axC.set_xlim(fmin, fmax)
axC.legend(frameon=False, loc="upper right")

# ----- Panel D: Parameter Comparison -----
# axD = fig.add_subplot(gs[3])
# axD.axis("off")
# axD.text(0.0, 0.98, "D  Parameter\nComparison", transform=axD.transAxes,
#          fontsize=16, fontweight="bold", va="top")
# y = 0.78
# axD.text(0.0, y, "Δ vs Interictal", fontweight="bold", transform=axD.transAxes)
# y -= 0.08
# for lbl, dp, di in zip(["θ (3–8 Hz)", "α (8–13 Hz)", "β (13–35 Hz)"],
#                        [np.nanmean(log_p[band_mask(f, 3, 8)]) - np.nanmean(log_b[band_mask(f, 3, 8)]),
#                         np.nanmean(log_p[band_mask(f, 8, 13)]) - np.nanmean(log_b[band_mask(f, 8, 13)]),
#                         np.nanmean(log_p[band_mask(f, 13, 35)]) - np.nanmean(log_b[band_mask(f, 13, 35)])],
#                        [np.nanmean(log_i[band_mask(f, 3, 8)]) - np.nanmean(log_b[band_mask(f, 3, 8)]),
#                         np.nanmean(log_i[band_mask(f, 8, 13)]) - np.nanmean(log_b[band_mask(f, 8, 13)]),
#                         np.nanmean(log_i[band_mask(f, 13, 35)]) - np.nanmean(log_b[band_mask(f, 13, 35)])]):
#     axD.text(0.0, y, lbl, transform=axD.transAxes, ha="left", va="center")
#     axD.text(0.8, y, f"Preictal: {dp:+.2f}", ha="right", transform=axD.transAxes)
#     axD.text(1.0, y, f"Ictal: {di:+.2f}", ha="right", transform=axD.transAxes)
#     y -= 0.08

# ----- Title -----
# fig.suptitle(
#     f"CHB-MIT Spectral Parameters with Seizure Phases — {edf_path.split('/')[-1]} — {channel}",
#     fontsize=10, y=1.02
# )

# Save
plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"\n✅ Saved:\n  {png_path}\n  {pdf_path}\n")
