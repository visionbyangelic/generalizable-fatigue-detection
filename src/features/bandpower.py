"""Band-power features per docs/how-others-did-it.md section 6, adapted to
Amendment 1 channel masking (docs/label_rules.md).

PSD: Welch, 2-s Hamming segments, 50% overlap (nperseg=256 @128 Hz) within
each 4-s window, per channel. Bands: delta 1-4, theta 4-8, alpha 8-12,
beta 12-30 Hz (gamma excluded per section 6).

Feature layout: FIXED 43 columns for every window:
  [0:16]  log10 band power, channel-major [TP9, AF7, AF8, TP10] x [d,t,a,b]
  [16:32] relative band power (band / sum of the 4 band powers), same order
  [32:36] frontal ratios from AF7/AF8 mean band powers:
          (theta+alpha)/beta, alpha/beta, (theta+alpha)/(alpha+beta), theta/beta
  [36:40] temporal ratios from the mean band powers of the VALID temporal
          channels only (Amendment 1: features aggregate over valid channels)
  [40]    valid_TP9 (0/1), [41] valid_TP10 (0/1),
  [42]    both_temporals_masked (0/1)

Masking rules (no leakage by construction: no statistics used at all here):
  - A masked temporal channel's log/relative features are NaN in the output.
  - Temporal ratios are NaN iff BOTH temporals are masked.
  - Frontal channels are always valid in retained windows (Amendment 1
    gating), so frontal features/ratios are never NaN.
  - NaNs are imputed downstream with TRAINING-FOLD medians only
    (see src/models/baseline.py); the indicator columns make the masking
    explicit to the model, never silent.
"""
import numpy as np
import pandas as pd
from scipy.signal import welch

FS = 128
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 12), "beta": (12, 30)}
CHANNELS = ["TP9", "AF7", "AF8", "TP10"]
TEMPORAL = ["TP9", "TP10"]

FEATURE_NAMES = (
    [f"logpow_{c}_{b}" for c in CHANNELS for b in BANDS]
    + [f"relpow_{c}_{b}" for c in CHANNELS for b in BANDS]
    + [f"frontal_ratio_{r}" for r in ["th_al_over_beta", "al_over_beta",
                                     "th_al_over_al_beta", "th_over_beta"]]
    + [f"temporal_ratio_{r}" for r in ["th_al_over_beta", "al_over_beta",
                                       "th_al_over_al_beta", "th_over_beta"]]
    + ["valid_TP9", "valid_TP10", "both_temporals_masked"]
)


def _band_powers(x):
    """x: (512,) one channel -> dict band -> power (uV^2)."""
    f, pxx = welch(x, fs=FS, window="hamming", nperseg=256, noverlap=128)
    out = {}
    for b, (lo, hi) in BANDS.items():
        m = (f >= lo) & (f < hi) if b != "beta" else (f >= lo) & (f <= hi)
        out[b] = float(np.trapz(pxx[m], f[m]))
    return out


def _ratios(bp):
    d, t, a, be = bp["delta"], bp["theta"], bp["alpha"], bp["beta"]
    eps = 1e-12
    return [(t + a) / (be + eps), a / (be + eps),
            (t + a) / (a + be + eps), t / (be + eps)]


def extract_features(windows, index):
    """windows: (n,4,512) float array; index: DataFrame with valid_<ch> cols.
    Returns DataFrame (n x 43) aligned with index rows."""
    rows = []
    for i in range(len(windows)):
        bp = {c: _band_powers(windows[i, ci]) for ci, c in enumerate(CHANNELS)}
        valid = {c: bool(index[f"valid_{c}"].iloc[i]) for c in CHANNELS}
        feats = []
        for c in CHANNELS:  # log power
            feats += [np.log10(bp[c][b] + 1e-12) if valid[c] else np.nan
                      for b in BANDS]
        for c in CHANNELS:  # relative power
            tot = sum(bp[c].values())
            feats += [bp[c][b] / tot if valid[c] and tot > 0 else np.nan
                      for b in BANDS]
        frontal_bp = {b: np.mean([bp["AF7"][b], bp["AF8"][b]]) for b in BANDS}
        feats += _ratios(frontal_bp)
        vt = [c for c in TEMPORAL if valid[c]]
        if vt:
            temporal_bp = {b: np.mean([bp[c][b] for c in vt]) for b in BANDS}
            feats += _ratios(temporal_bp)
        else:
            feats += [np.nan] * 4
        feats += [float(valid["TP9"]), float(valid["TP10"]),
                  float(not vt)]
        rows.append(feats)
    return pd.DataFrame(rows, columns=FEATURE_NAMES, index=index.index)
