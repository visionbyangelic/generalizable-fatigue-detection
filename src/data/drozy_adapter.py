"""DROZY adapter: EDF recordings -> windowed arrays, band-power features
and label tables, following docs/dataset2_preregistration.md.

Preprocessing mirrors the FatigueSet pipeline: 1-40 Hz zero-phase
bandpass at the native rate, downsample to 128 Hz, 4 s non-overlapping
windows. A window is kept if Fz passes the artefact checks (peak-to-peak
<= 150 uV in every 1 s sub-window, SD >= 0.5 uV); a failing non-frontal
channel is masked for that window and its features are NaN, with the
validity stored in the index.

Features mirror src/features/bandpower.py: Welch PSD (2 s Hamming
segments, 50% overlap), log10 and relative power in delta/theta/alpha/
beta per channel, ratios from the frontal channel (Fz) and from the
mean of the valid non-frontal channels, plus validity indicators.
The Fz-only variant is the Fz log/relative powers + frontal ratios.
"""
import sys
from datetime import datetime
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import butter, resample_poly, sosfiltfilt, welch

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "drozy" / "extracted" / "DROZY"
PROC = ROOT / "data" / "processed"
TAB = ROOT / "results" / "tables"

CHANNELS = ["Fz", "Cz", "Pz", "C3", "C4"]
FRONTAL = "Fz"
NONFRONTAL = ["Cz", "Pz", "C3", "C4"]
FS_IN, FS_OUT = 512, 128
WIN = 512  # 4 s at 128 Hz
P2P_LIMIT = 150.0
FLAT_SD = 0.5
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 12), "beta": (12, 30)}
SOS = butter(4, [1.0, 40.0], btype="bandpass", fs=FS_IN, output="sos")

FEATURE_NAMES = (
    [f"logpow_{c}_{b}" for c in CHANNELS for b in BANDS]
    + [f"relpow_{c}_{b}" for c in CHANNELS for b in BANDS]
    + [f"frontal_ratio_{r}" for r in ["th_al_over_beta", "al_over_beta",
                                      "th_al_over_al_beta", "th_over_beta"]]
    + [f"nonfrontal_ratio_{r}" for r in ["th_al_over_beta", "al_over_beta",
                                         "th_al_over_al_beta", "th_over_beta"]]
    + [f"valid_{c}" for c in NONFRONTAL]
    + ["all_nonfrontal_masked"]
)
FZ_FEATURES = (
    [f"logpow_Fz_{b}" for b in BANDS]
    + [f"relpow_Fz_{b}" for b in BANDS]
    + [f"frontal_ratio_{r}" for r in ["th_al_over_beta", "al_over_beta",
                                      "th_al_over_al_beta", "th_over_beta"]]
)


def load_recording(participant, test):
    """One EDF -> (76800, 5) uV at 128 Hz, channels in CHANNELS order."""
    path = RAW / "psg" / f"{participant}-{test}.edf"
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    raw.pick(CHANNELS)
    x = raw.get_data(units="uV")  # (5, 307200) at 512 Hz
    filt = sosfiltfilt(SOS, x, axis=1)
    down = resample_poly(filt, up=1, down=FS_IN // FS_OUT, axis=1)
    return np.ascontiguousarray(down.T)


def session_tests(participant):
    out = []
    for t in (1, 2, 3):
        if (RAW / "psg" / f"{participant}-{t}.edf").exists():
            out.append(t)
    return out


def build_windows():
    """All 36 sessions -> retained windows + full candidate audit table."""
    pool, cands = [], []
    for p in range(1, 15):
        for t in session_tests(p):
            sig = load_recording(p, t)
            n_win = len(sig) // WIN
            for w in range(n_win):
                win = sig[w * WIN:(w + 1) * WIN]  # (512, 5)
                sub = win.reshape(4, 128, 5)
                p2p = (sub.max(axis=1) - sub.min(axis=1)).max(axis=0)
                sd = win.std(axis=0)
                ok = (p2p <= P2P_LIMIT) & (sd >= FLAT_SD)
                row = {"participant": f"P{p:02d}", "session": f"S{t}",
                       "test": t, "window": w, "start_sample": w * WIN,
                       "retained": bool(ok[0]),
                       **{f"valid_{c}": bool(ok[i])
                          for i, c in enumerate(CHANNELS)},
                       **{f"max_p2p_{c}": float(p2p[i])
                          for i, c in enumerate(CHANNELS)}}
                cands.append(row)
                if ok[0]:
                    win_out = np.ascontiguousarray(win.T, dtype=np.float32)
                    for i, c in enumerate(CHANNELS):
                        if not ok[i]:
                            win_out[i] = 0.0
                    row["_win"] = win_out
                    pool.append(row)
            kept = sum(1 for r in cands if r["participant"] == f"P{p:02d}"
                       and r["session"] == f"S{t}" and r["retained"])
            print(f"P{p:02d}/S{t}: {kept}/{n_win} windows kept", flush=True)
    cand_df = pd.DataFrame([{k: v for k, v in r.items() if k != "_win"}
                            for r in cands])
    arr = np.stack([r.pop("_win") for r in pool])
    idx = pd.DataFrame(pool)
    return arr, idx, cand_df


def _band_powers(x):
    f, pxx = welch(x, fs=FS_OUT, window="hamming", nperseg=256, noverlap=128)
    out = {}
    for b, (lo, hi) in BANDS.items():
        m = (f >= lo) & (f < hi) if b != "beta" else (f >= lo) & (f <= hi)
        out[b] = float(np.trapezoid(pxx[m], f[m]))
    return out


def _ratios(bp):
    d, t, a, be = bp["delta"], bp["theta"], bp["alpha"], bp["beta"]
    eps = 1e-12
    return [(t + a) / (be + eps), a / (be + eps),
            (t + a) / (a + be + eps), t / (be + eps)]


def extract_features(windows, index):
    rows = []
    for i in range(len(windows)):
        bp = {c: _band_powers(windows[i, ci]) for ci, c in enumerate(CHANNELS)}
        valid = {c: bool(index[f"valid_{c}"].iloc[i]) for c in CHANNELS}
        feats = []
        for c in CHANNELS:
            feats += [np.log10(bp[c][b] + 1e-12) if valid[c] else np.nan
                      for b in BANDS]
        for c in CHANNELS:
            tot = sum(bp[c].values())
            feats += [bp[c][b] / tot if valid[c] and tot > 0 else np.nan
                      for b in BANDS]
        feats += _ratios(bp["Fz"])
        vn = [c for c in NONFRONTAL if valid[c]]
        if vn:
            mean_bp = {b: np.mean([bp[c][b] for c in vn]) for b in BANDS}
            feats += _ratios(mean_bp)
        else:
            feats += [np.nan] * 4
        feats += [float(valid[c]) for c in NONFRONTAL]
        feats += [float(not vn)]
        rows.append(feats)
    return pd.DataFrame(rows, columns=FEATURE_NAMES, index=index.index)


# --------------------------------------------------------------- labels
def parse_kss():
    vals = np.loadtxt(RAW / "KSS.txt", dtype=int)
    return {p + 1: {t + 1: int(vals[p, t]) for t in range(3)}
            for p in range(14)}


def parse_pvt():
    """Per-session PVT stats: trials, lapses (>500 ms), lapse rate."""
    fmt = "%Y-%m-%d_%H.%M.%S.%f"
    rows = []
    for p in range(1, 15):
        for t in session_tests(p):
            lines = (RAW / "pvt-rt" / f"{p}-{t}.csv").read_text().splitlines()
            rts = []
            for line in lines[1:]:
                if not line.strip():
                    continue
                stim, react = line.split(";")
                dt = (datetime.strptime(react, fmt)
                      - datetime.strptime(stim, fmt)).total_seconds() * 1000
                rts.append(dt)
            rts = np.array(rts)
            rows.append({"participant": f"P{p:02d}", "test": t,
                         "n_trials": len(rts),
                         "n_lapses": int((rts > 500).sum()),
                         "lapse_rate": float((rts > 500).mean()),
                         "rt_min_ms": float(rts.min()),
                         "rt_max_ms": float(rts.max())})
    return pd.DataFrame(rows)


def select_sessions(values):
    """values: {test: label value} for the sessions usable in one arm.
    Returns (alert_test, fatigued_test, middle_test or None, note).
    Ties at the top go to the later test, ties at the bottom to the
    earlier test; equal top and bottom means no contrast."""
    if len(values) < 2:
        return None, None, None, "fewer than two usable sessions"
    lo, hi = min(values.values()), max(values.values())
    if lo == hi:
        return None, None, None, "no contrast (all values equal)"
    alert = min(t for t, v in values.items() if v == lo)
    fat = max(t for t, v in values.items() if v == hi)
    middle = [t for t in values if t not in (alert, fat)]
    notes = []
    if sum(1 for v in values.values() if v == hi) > 1:
        notes.append("tie at the top, later test taken")
    if sum(1 for v in values.values() if v == lo) > 1:
        notes.append("tie at the bottom, earlier test taken")
    return alert, fat, (middle[0] if middle else None), "; ".join(notes)


def build_label_tables():
    kss = parse_kss()
    pvt = parse_pvt()
    tables = {}
    for arm in ("kss", "rt"):
        rows = []
        for p in range(1, 15):
            tests = session_tests(p)
            if arm == "kss":
                # invalid scores (outside 1-9) are missing before anything
                # else; a session also needs a recording to be usable
                values = {t: kss[p][t] for t in tests
                          if 1 <= kss[p][t] <= 9}
            else:
                sub = pvt[pvt.participant == f"P{p:02d}"]
                values = {int(r.test): float(r.lapse_rate)
                          for r in sub.itertuples() if int(r.test) in tests}
            alert, fat, middle, note = select_sessions(values)
            row = {"participant": f"P{p:02d}",
                   "sessions_with_data": ",".join(str(t) for t in tests),
                   "alert_test": alert, "fatigued_test": fat,
                   "middle_test": middle,
                   "alert_value": values.get(alert) if alert else None,
                   "fatigued_value": values.get(fat) if fat else None,
                   "included": alert is not None, "note": note}
            if p == 7 and arm == "kss":
                row["note"] = (note + "; " if note else "") + \
                    "test 1 never recorded (score 0 = missing); " \
                    "contrast built on tests 2-3"
            if p == 12:
                row["note"] = (note + "; " if note else "") + \
                    "excluded: one recorded session, no within-person " \
                    "contrast exists"
            rows.append(row)
        tables[arm] = pd.DataFrame(rows)
    return tables, pvt


def arm_index(idx, table):
    """Retained-window index for one arm, with labels attached."""
    parts = []
    for r in table[table.included].itertuples():
        for test, lab in ((r.alert_test, 0), (r.fatigued_test, 1)):
            sel = idx[(idx.participant == r.participant)
                      & (idx.test == test)].copy()
            sel["label"] = lab
            parts.append(sel)
    out = pd.concat(parts).sort_values(["participant", "test", "window"])
    return out.reset_index(drop=True)


def main():
    PROC.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    arr, idx, cand = build_windows()
    cand.to_csv(PROC / "drozy_candidates.csv", index=False)
    np.save(PROC / "drozy_windows.npy", arr)
    idx.to_csv(PROC / "drozy_windows_index.csv", index=False)
    print(f"kept {len(idx)} of {len(cand)} candidate windows", flush=True)

    feats = extract_features(arr, idx)
    feats.to_csv(PROC / "drozy_features.csv", index=False)
    print(f"features: {feats.shape}", flush=True)

    tables, pvt = build_label_tables()
    tables["kss"].to_csv(TAB / "drozy_labels_kss.csv", index=False)
    tables["rt"].to_csv(TAB / "drozy_labels_rt.csv", index=False)
    pvt.to_csv(TAB / "drozy_pvt_sessions.csv", index=False)

    rej = cand.groupby(["participant", "session"]).agg(
        candidate_windows=("window", "count"),
        retained_windows=("retained", "sum")).reset_index()
    masked = idx.groupby(["participant", "session"]).agg(
        masked_Cz=("valid_Cz", lambda s: int((~s).sum())),
        masked_Pz=("valid_Pz", lambda s: int((~s).sum())),
        masked_C3=("valid_C3", lambda s: int((~s).sum())),
        masked_C4=("valid_C4", lambda s: int((~s).sum()))).reset_index()
    rej = rej.merge(masked, on=["participant", "session"], how="left")
    rej["retention_pct"] = (100 * rej.retained_windows
                            / rej.candidate_windows).round(1)
    rej.to_csv(TAB / "drozy_rejection_rates.csv", index=False)
    print(rej.to_string(index=False), flush=True)


if __name__ == "__main__":
    sys.exit(main())
