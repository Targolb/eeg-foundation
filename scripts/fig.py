#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CHB-MIT spectral parameters figure (A–D), tidied:
- Panel A in µV (no scientific 1e-5 offset)
- Smoothed PSD curves for readability (Gaussian)
- Consistent typography and spacing
- Cleaner band shading + greek labels (θ, α, β)
- Compact "Panel D" using text (no table layout warnings)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mne
from fooof import FOOOF
from scipy.ndimage import gaussian_filter1d

# ------------------------- Styling -------------------------
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 12,
    "axes.titlesize": 10,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ------------------------- Helpers -------------------------
def parse_range(s, default):
    if s is None:
        return default
    a, b = [float(x.strip()) for x in s.split(",")]
    return a, b


def pick_channel(raw, want="Fp1"):
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    ch_names = np.array(raw.info["ch_names"])[eeg_picks]
    return want if want in ch_names else ch_names[0]


def segment_data(raw, ch_name, t0, t1):
    sf = raw.info["sfreq"]
    i0, i1 = int(round(t0 * sf)), int(round(t1 * sf))
    x = raw.get_data(picks=[ch_name])[0]
    i0 = max(0, min(len(x) - 1, i0))
    i1 = max(i0 + 1, min(len(x), i1))
    return x[i0:i1], sf


def psd_welch(x, sf, fmin=1, fmax=40, n_fft=2048, ovlp=0.5):
    from mne.time_frequency import psd_array_welch
    psd, f = psd_array_welch(x, sf, fmin=fmin, fmax=fmax, n_fft=n_fft,
                             n_overlap=int(n_fft * ovlp), average="median")
    return f, psd.squeeze()


def band_mask(f, lo, hi):
    return (f >= lo) & (f < hi)


def alpha_peak_from_fooof(fm, lo=8.0, hi=13.0):
    res = fm.get_results()
    peaks = getattr(res, "peak_params", None)  # FOOOF 1.1
    if peaks is None or len(peaks) == 0:
        return None, None
    peaks = np.array(peaks)  # [CF, PW, BW]
    m = (peaks[:, 0] >= lo) & (peaks[:, 0] <= hi)
    if not np.any(m):
        return None, None
    p = peaks[m]
    best = p[np.argmax(p[:, 1])]
    return float(best[0]), float(best[1])


def shade_band(ax, lo, hi, color, label=None):
    ax.axvspan(lo, hi, facecolor=color, alpha=0.22, lw=0)


# ------------------------- Main -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edf", required=True)
    ap.add_argument("--channel", default="Fp1")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--comparison", default=None)
    ap.add_argument("--fmin", type=float, default=1.0)
    ap.add_argument("--fmax", type=float, default=40.0)
    ap.add_argument("--out", default="chbmit_fooof_fig")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--smooth_sigma", type=float, default=1.0,
                    help="Gaussian sigma (in index units) to smooth PSD curves")
    args = ap.parse_args()

    # Load & preprocess
    raw = mne.io.read_raw_edf(args.edf, preload=True, verbose="ERROR")
    raw.pick_types(eeg=True)
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    raw.filter(1., 40., verbose="ERROR")

    ch = pick_channel(raw, args.channel)
    base_t = parse_range(args.baseline, (0.0, 10.0))
    comp_t = parse_range(args.comparison, (60.0, 70.0))

    x_base, sf = segment_data(raw, ch, *base_t)
    x_comp, _ = segment_data(raw, ch, *comp_t)

    # PSDs
    f, psd_base = psd_welch(x_base, sf, fmin=args.fmin, fmax=args.fmax)
    _, psd_comp = psd_welch(x_comp, sf, fmin=args.fmin, fmax=args.fmax)

    # Smooth PSDs (for visual clarity only)
    if args.smooth_sigma > 0:
        psd_base_plot = np.clip(gaussian_filter1d(psd_base, args.smooth_sigma), 1e-20, None)
        psd_comp_plot = np.clip(gaussian_filter1d(psd_comp, args.smooth_sigma), 1e-20, None)
    else:
        psd_base_plot = np.clip(psd_base, 1e-20, None)
        psd_comp_plot = np.clip(psd_comp, 1e-20, None)

    log_base = np.log10(psd_base_plot)
    log_comp = np.log10(psd_comp_plot)

    # FOOOF on the *unsmoothed* spectra (use original power for fitting)
    fm_base = FOOOF(peak_width_limits=[1, 12], max_n_peaks=6, verbose=False)
    fm_base.fit(f, psd_base)
    ap_off_b, ap_exp_b = fm_base.aperiodic_params_

    fm_comp = FOOOF(peak_width_limits=[1, 12], max_n_peaks=6, verbose=False)
    fm_comp.fit(f, psd_comp)
    ap_off_c, ap_exp_c = fm_comp.aperiodic_params_

    # Aperiodic fit (positive) → log10
    ap_fit = np.clip(fm_base._ap_fit, 1e-20, None)
    log_ap = np.log10(ap_fit)

    # Alpha marker
    a_cf, _ = alpha_peak_from_fooof(fm_base, 8, 13)

    # Band deltas (log10 power)
    bands = {"θ (3–8 Hz)": (3, 8), "α (8–13 Hz)": (8, 13), "β (13–35 Hz)": (13, 35)}
    deltas = {}
    for k, (lo, hi) in bands.items():
        m = band_mask(f, lo, hi)
        deltas[k] = float(np.nanmean(log_comp[m]) - np.nanmean(log_base[m])) if np.any(m) else np.nan
    d_exp = ap_exp_c - ap_exp_b

    # ---------------- Figure layout ----------------
    fig = plt.figure(figsize=(13.5, 4.2), constrained_layout=True)
    gs = gridspec.GridSpec(1, 4, figure=fig, width_ratios=[2.1, 2.7, 2.7, 1.5])

    # A: time series in µV (first 5 s of baseline)
    ax0 = fig.add_subplot(gs[0])
    n_show = int(min(len(x_base), 5 * sf))
    t = np.arange(n_show) / sf
    ax0.plot(t, x_base[:n_show] * 1e6, color="black", lw=0.9)
    ax0.set_title("A  EEG Time Series", loc="left")
    ax0.set_xlabel("Time (s)")
    ax0.set_ylabel("Amplitude (µV)")
    if len(t) > 0:
        ax0.set_xlim(t[0], t[-1])

    # B: spectrum + aperiodic fit + bands
    ax1 = fig.add_subplot(gs[1])
    shade_band(ax1, 3, 8, "#a8c5ff")  # theta
    shade_band(ax1, 8, 13, "#bcecc2")  # alpha
    shade_band(ax1, 13, 35, "#f6c8c8")  # beta
    ax1.plot(f, log_base, color="black", lw=1.0, label="Power Spectrum")
    ax1.plot(f, log_ap, "b--", lw=1.1, label="Aperiodic fit")
    if a_cf is not None:
        ax1.axvline(a_cf, ls=":", lw=1.0, color="0.4")
        ax1.text(a_cf + 0.8, np.nanmax(log_base) - 0.2, "oscillatory\npeak (α)",
                 ha="left", va="top", fontsize=10, color="0.25")
    for x, s in [(5.5, "θ"), (10.5, "α"), (23, "β")]:
        ax1.text(x, np.nanmin(log_base) + 0.15, s, fontsize=13, alpha=0.75)
    ax1.set_title("B  Spectral Parameters", loc="left")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("log10 Power")
    ax1.set_xlim(args.fmin, args.fmax)
    ax1.legend(frameon=False, loc="upper right")

    # C: comparison
    ax2 = fig.add_subplot(gs[2])
    shade_band(ax2, 3, 8, "#a8c5ff")
    shade_band(ax2, 8, 13, "#bcecc2")
    shade_band(ax2, 13, 35, "#f6c8c8")
    ax2.plot(f, log_base, lw=1.0, label="Baseline", color="#4063D8")
    ax2.plot(f, log_comp, lw=1.0, label="Comparison", color="#CB3C33")
    ax2.set_title("C  Spectrum Comparison", loc="left")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("log10 Power")
    ax2.set_xlim(args.fmin, args.fmax)
    ax2.legend(frameon=False, loc="upper right")

    # D: compact parameter summary (no table widget)
    ax3 = fig.add_subplot(gs[3])
    ax3.axis("off")
    y0 = 0.95;
    dy = 0.12
    ax3.text(0.0, y0, "D  Parameter\nComparison", transform=ax3.transAxes,
             fontsize=18, fontweight="bold", va="top")
    y = y0 - 0.30

    def row(label, val):
        nonlocal y
        ax3.text(0.0, y, label, transform=ax3.transAxes, ha="left", va="center")
        ax3.text(0.98, y, f"{val:+.2f}", transform=ax3.transAxes, ha="right", va="center")
        y -= dy

    ax3.text(0.0, y, "Band-by-Band", transform=ax3.transAxes,
             fontweight="bold", va="center");
    y -= dy * 0.8
    row("Δθ (3–8 Hz)", deltas["θ (3–8 Hz)"])
    row("Δα (8–13 Hz)", deltas["α (8–13 Hz)"])
    row("Δβ (13–35 Hz)", deltas["β (13–35 Hz)"])
    y -= dy * 0.4
    ax3.text(0.0, y, "Aperiodic", transform=ax3.transAxes,
             fontweight="bold", va="center");
    y -= dy * 0.8
    row("Δ exponent", d_exp)

    # Title & save
    fig.suptitle(
        f"CHB-MIT Spectral Parameters  —  {args.edf.split('/')[-1]}  —  channel {ch}\n",
        # f"Baseline {base_t[0]:.1f}–{base_t[1]:.1f}s   |   Comparison {comp_t[0]:.1f}–{comp_t[1]:.1f}s",
        fontsize=8, y=1.03
    )

    png_path = f"{args.out}.png"
    pdf_path = f"{args.out}.pdf"
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved:\n  {png_path}\n  {pdf_path}")


if __name__ == "__main__":
    main()
