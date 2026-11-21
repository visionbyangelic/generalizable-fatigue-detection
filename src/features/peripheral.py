"""RQ3 peripheral features: Empatica E4 heart rate and EDA per primary window.

Implements ONLY the feature set declared in docs/rq3_preregistration.md
(as amended by Amendment 1, 21 Nov 2026). Six columns per window of the
Amendment-1 primary pool (data/processed/windows_a1_uncapped_index.csv):

  hr_mean        mean of the in-span wrist_hr samples with a positive
                 value. wrist_hr is the E4 firmware's own 1 Hz product:
                 a smoothed trailing average over the preceding 10 s,
                 not an instantaneous rate. This study's HR arm is mean
                 heart rate only; nothing here touches beat-to-beat
                 variability (the IBI stream cannot support it at
                 window level, docs/RQ3_INVENTORY.md Section 4).
  eda_mean       mean of the in-span wrist_eda samples (microS).
  eda_slope      least-squares slope of wrist_eda against time over the
                 in-span samples (microS per second).
  eda_rise_count number of rises of at least 0.01 microS from a local
                 minimum to the next local maximum in the raw in-span
                 series, counted by the fixed scanning rule in
                 rise_count() below. Named "rise count" (Amendment 1):
                 in a 4 s window, at a threshold near the E4's
                 quantisation floor, these counted rises are not
                 claimed to be skin conductance responses.
  hr_valid       1 if the window holds at least 2 usable HR samples
                 (in-span, positive value), else 0.
  eda_valid      1 if the window holds at least 8 EDA samples and their
                 maximum exceeds 0.002 microS, else 0.

Window span: the closed epoch-ms interval from the window's first to
its last EEG timestamp. A window is 1,024 raw EEG samples
(build_windows_a1.py: centre_ms = 0.5 * (ts[start] + ts[start + 1023])),
so the span is [ts[start_sample], ts[start_sample + 1023]] read off the
session's own EEG timestamps: the same axis the wrist streams publish
on (docs/RQ3_INVENTORY.md Section 5). A wrist sample belongs to the
window iff its timestamp falls in that span.

Features are NaN exactly where the matching validity indicator is 0.
Nothing is imputed or filled here: the locked pipeline's training-fold
median imputation handles NaNs downstream, and the indicators carry
the absence explicitly, never silently.

Streams are read through the project's loader (src/data/load_session),
so *2.csv continuations are concatenated in timestamp order (this
affects only P02/S01's wrist data).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.load_session import load_concat, load_eeg

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"

WIN_RAW = 1024          # raw EEG samples per window (256 Hz x 4 s)
HR_MIN_SAMPLES = 2      # hr_valid threshold (Amendment 1, stated verbatim)
EDA_MIN_SAMPLES = 8     # eda_valid threshold: count clause
EDA_MAX_FLOOR = 0.002   # eda_valid threshold: trace clause (microS)
RISE_THRESHOLD = 0.01   # minimum rise, microS (declared feature rule)

FEATURE_COLUMNS = ["hr_mean", "eda_mean", "eda_slope", "eda_rise_count",
                   "hr_valid", "eda_valid"]


def rise_count(x, thresh=RISE_THRESHOLD):
    """Count rises of at least `thresh` from a local minimum to the next
    local maximum, scanning the series in time order.

    The scan holds a current trough (the lowest value since the last
    peak) and, once the series rises above that trough, a current peak.
    When the series turns down from a peak, the rise trough->peak is
    complete: it counts once if it reached `thresh`, and the turn-down
    point becomes the new trough. A rise still in progress at the end
    of the window counts if it has reached `thresh`. Plateaus change
    nothing (strict comparisons), so a flat stretch is neither a new
    trough nor a new peak. No smoothing or filtering is applied.
    """
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return 0
    count = 0
    trough = peak = x[0]
    rising = False
    for v in x[1:]:
        if not rising:
            if v < trough:
                trough = peak = v
            elif v > trough:
                rising, peak = True, v
        else:
            if v > peak:
                peak = v
            elif v < peak:
                if peak - trough >= thresh:
                    count += 1
                trough = peak = v
                rising = False
    if rising and peak - trough >= thresh:
        count += 1
    return count


def _in_span(ts, lo, hi):
    """Indices of samples with lo <= ts <= hi (ts sorted ascending)."""
    a = np.searchsorted(ts, lo, side="left")
    b = np.searchsorted(ts, hi, side="right")
    return a, b


def extract_peripheral(idx):
    """idx: the pool index DataFrame (participant 'P01', session 'S01',
    start_sample, ...). Returns a DataFrame aligned to idx's rows with
    the six peripheral columns."""
    out = {c: np.full(len(idx), np.nan) for c in FEATURE_COLUMNS}
    pos = idx.reset_index(drop=True)
    for (part, sess), g in pos.groupby(["participant", "session"]):
        p, s = part[1:], sess[1:]
        eeg_ts, _ = load_eeg(p, s)
        hr_df = load_concat(p, s, "wrist_hr.csv")
        eda_df = load_concat(p, s, "wrist_eda.csv")
        hr_ts = hr_df["timestamp"].to_numpy(dtype=float)
        hr_v = pd.to_numeric(hr_df["hr"], errors="coerce").to_numpy(dtype=float)
        eda_ts = eda_df["timestamp"].to_numpy(dtype=float)
        eda_v = pd.to_numeric(eda_df["eda"], errors="coerce").to_numpy(dtype=float)
        for i in g.index:
            lo = float(eeg_ts[pos.at[i, "start_sample"]])
            hi = float(eeg_ts[pos.at[i, "start_sample"] + WIN_RAW - 1])
            # --- heart rate: usable samples are in-span and positive
            a, b = _in_span(hr_ts, lo, hi)
            pos_hr = hr_v[a:b]
            pos_hr = pos_hr[np.isfinite(pos_hr) & (pos_hr > 0)]
            out["hr_valid"][i] = 1.0 if len(pos_hr) >= HR_MIN_SAMPLES else 0.0
            if out["hr_valid"][i]:
                out["hr_mean"][i] = float(np.mean(pos_hr))
            # --- EDA: in-span samples, finite values
            a, b = _in_span(eda_ts, lo, hi)
            eda = eda_v[a:b]
            eda = eda[np.isfinite(eda)]
            ok = len(eda) >= EDA_MIN_SAMPLES and len(eda) > 0 \
                and float(np.max(eda)) > EDA_MAX_FLOOR
            out["eda_valid"][i] = 1.0 if ok else 0.0
            if ok:
                t_s = (eda_ts[a:b][np.isfinite(eda_v[a:b])] - lo) / 1000.0
                out["eda_mean"][i] = float(np.mean(eda))
                # least-squares slope against time in seconds
                tc = t_s - t_s.mean()
                denom = float(np.sum(tc ** 2))
                out["eda_slope"][i] = float(
                    np.sum(tc * (eda - eda.mean())) / denom) if denom > 0 else 0.0
                out["eda_rise_count"][i] = float(rise_count(eda))
    return pd.DataFrame(out, index=idx.index)


def main():
    idx = pd.read_csv(PROC / "windows_a1_uncapped_index.csv")
    feats = extract_peripheral(idx)
    table = pd.concat([
        idx[["participant", "session", "block", "run_id", "start_sample",
             "label"]].reset_index(drop=True),
        feats.reset_index(drop=True)], axis=1)
    table.to_csv(PROC / "peripheral_features_a1.csv", index=False)
    n = len(table)
    print(f"windows: {n}")
    for c in ["hr_valid", "eda_valid"]:
        print(f"{c}: {table[c].mean() * 100:.1f}% valid "
              f"({int(table[c].sum())}/{n})")
    both = (table.hr_valid == 1) & (table.eda_valid == 1)
    print(f"both valid: {both.sum()}/{n}")
    either_bad = ~both
    print(f"either invalid: {either_bad.sum()}, of which fatigued: "
          f"{int(table.label[either_bad].sum())}")


if __name__ == "__main__":
    main()
