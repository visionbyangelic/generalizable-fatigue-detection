"""PSD verification of the temporal-noise diagnosis.

Why this check exists: the Amendment-1 retention table says how much
of the temporal signal was masked, but not what the masking was
hiding, and the interpretation depends on the answer. Mains hum,
muscle activity and contact noise are three different problems with
three different consequences for the frontal-only comparison, so I
look at the spectrum directly instead of guessing from the mask.

Case: P03/S01, a session whose temporal channels fail almost everywhere
(97% masking in the Amendment-1 retention table) while frontal channels are
clean. For the M3 block we compute, per channel:
  - Welch PSD of the RAW signal (0.5-60 Hz shown; a narrow 50 Hz mains spike
    would be visible here even though the pipeline bandpass ends at 40 Hz),
  - band powers (raw): 1-4, 4-8, 8-12, 12-20, 20-30, 30-40, 45-55 Hz,
  - high-frequency share: (20-40 Hz power) / (1-40 Hz power),
  - instability: SD and coefficient of variation of 1-s RMS over the block
    (contact noise is characteristically UNSTABLE over time; EMG elevation
    is comparatively steady while the muscle is active),
  - spectral flatness in 1-40 Hz (white/broadband noise -> high flatness).

Classification logic:
  - narrow spike at ~50 Hz dominating -> mains;
  - power concentrated >20-30 Hz with modest low-band elevation -> EMG;
  - broadband elevation across ALL bands + high flatness + high RMS
    instability -> contact noise.

Outputs: results/figures/psd_p03_s01_temporal_vs_frontal.png and
results/tables/psd_p03_s01_summary.csv.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.load_session import CHANNELS, block_intervals, load_eeg

ROOT = Path(__file__).resolve().parents[2]
P, S, BLOCK = "03", "01", "M3"
BANDS = {"1-4": (1, 4), "4-8": (4, 8), "8-12": (8, 12), "12-20": (12, 20),
         "20-30": (20, 30), "30-40": (30, 40), "45-55": (45, 55)}


def main():
    ts, x = load_eeg(P, S)
    t0, t1 = block_intervals(P, S)[BLOCK]
    m = (ts >= t0) & (ts < t1)
    seg = x[m]
    print(f"P{P}/S{S} {BLOCK}: {len(seg)} samples ({len(seg)/256:.1f} s)")
    rows, psds = [], {}
    for ci, c in enumerate(CHANNELS):
        sig = seg[:, ci]
        sig = sig[~np.isnan(sig)]
        f, pxx = welch(sig, fs=256, nperseg=1024, noverlap=512)
        psds[c] = (f, pxx)
        row = {"channel": c, "site": "temporal" if c in ("TP9", "TP10") else "frontal"}
        for b, (lo, hi) in BANDS.items():
            mm = (f >= lo) & (f < hi)
            row[f"power_{b}"] = float(np.trapz(pxx[mm], f[mm]))
        tot = sum(row[f"power_{b}"] for b in ["1-4", "4-8", "8-12", "12-20", "20-30", "30-40"])
        row["hf_share_20_40_of_1_40"] = (row["power_20-30"] + row["power_30-40"]) / tot
        row["mains_share_45_55_of_1_40"] = row["power_45-55"] / tot
        mm = (f >= 1) & (f <= 40)
        row["spectral_flatness_1_40"] = float(
            np.exp(np.mean(np.log(pxx[mm] + 1e-20))) / np.mean(pxx[mm]))
        rms1 = np.sqrt(np.mean(sig[:len(sig) // 256 * 256].reshape(-1, 256) ** 2, axis=1))
        row["rms_1s_mean"] = float(rms1.mean())
        row["rms_1s_sd"] = float(rms1.std())
        row["rms_1s_cv"] = float(rms1.std() / rms1.mean())
        rows.append(row)
    df = pd.DataFrame(rows)
    (ROOT / "results" / "tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(ROOT / "results" / "tables" / "psd_p03_s01_summary.csv", index=False)
    print(df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 5))
    for c in CHANNELS:
        f, pxx = psds[c]
        mm = (f >= 0.5) & (f <= 60)
        style = "-" if c in ("AF7", "AF8") else "--"
        ax.semilogy(f[mm], pxx[mm], style, label=c)
    ax.axvline(50, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD (uV^2/Hz, log)")
    ax.set_title(f"P{P}/S{S} {BLOCK} block — raw-signal Welch PSD by channel")
    ax.legend(); ax.grid(alpha=0.3)
    (ROOT / "results" / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "results" / "figures" / "psd_p03_s01_temporal_vs_frontal.png",
                dpi=150, bbox_inches="tight")
    print("figure saved")


if __name__ == "__main__":
    main()
