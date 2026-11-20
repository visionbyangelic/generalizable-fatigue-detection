"""Phase A — build the window dataset exactly per docs/label_rules.md (LOCKED).

Pipeline per participant/session:
  load EEG (sample index = time base) -> split into continuous runs at
  timestamp gaps > 100 ms (and at NaN samples) -> bandpass 1-40 Hz zero-phase
  per run (4th-order Butterworth, sosfiltfilt) -> downsample 256->128 Hz
  (resample_poly) -> 4 s non-overlapping windows (512 samples) within runs
  only -> artefact rejection:
    (a) any 1-s sub-window with peak-to-peak > 150 uV on any channel
        (the locked "+/-75 uV peak-to-peak" rule, implemented as p2p > 2*75);
    (b) any channel flatline (SD < 0.5 uV over the window);
    (c) window overlaps a muse_device_fit == Bad (4) sample on any channel
        (label_rules section 0(c), implemented as written; per-channel fit-bad
        fractions are recorded for every candidate window).
  Block membership: window centre inside the block's own marker interval
  (per-block anchoring). Primary labels: M1 -> 0 (alert), M3 -> 1 (fatigued),
  only in sessions with delta-mental > 0. The 4 non-increasing sessions are
  saved separately as the negative-control set.

Outputs (data/processed/):
  windows.npy / windows_index.csv             seed-0 M1-capped primary set
  windows_uncapped.npy / windows_index_uncapped.csv   full retained pool
  windows_control.npy / windows_control_index.csv     negative-control set
  preprocessing_log.csv                       per session/block accounting
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, resample_poly, sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.load_session import (CHANNELS, CLOCK_STEP, NONINCREASING,
                                   PARTICIPANTS, SESSIONS, block_intervals,
                                   delta_mental, load_eeg, load_fit)

FS_IN, FS_OUT = 256, 128
WIN = 512                      # 4 s at 128 Hz
WIN_IN = WIN * 2               # 1024 samples at 256 Hz
GAP_MS = 100
P2P_LIMIT = 150.0              # uV, = 2 x 75 uV (see module docstring)
FLAT_SD = 0.5                  # uV
CAP_SEED = 0                   # canonical M1-cap seed for windows.npy

OUT = Path(__file__).resolve().parents[2] / "data" / "processed"
SOS = butter(4, [1.0, 40.0], btype="bandpass", fs=FS_IN, output="sos")


def fit_per_sample(ts, fit_ts, fit):
    """Nearest fit sample (in time) for every EEG sample; NaN if > 1 s away."""
    idx = np.searchsorted(fit_ts, ts)
    idx = np.clip(idx, 1, len(fit_ts) - 1)
    left, right = fit_ts[idx - 1], fit_ts[idx]
    choose_right = (ts - left) > (right - ts)
    near = np.where(choose_right, idx, idx - 1)
    dist = np.abs(fit_ts[near] - ts)
    out = fit[near].astype(float)
    out[dist > 1000] = np.nan
    return out


def process_session(p, s):
    ts, x = load_eeg(p, s)
    fit_ts, fit = load_fit(p, s)
    fit_s = fit_per_sample(ts, fit_ts, fit)
    blocks = block_intervals(p, s)
    valid = ~np.isnan(x).any(axis=1)
    dt = np.diff(ts)
    breaks = np.where((dt > GAP_MS) | (dt < 0))[0]
    run_start = np.concatenate([[0], breaks + 1])
    run_end = np.concatenate([breaks + 1, [len(ts)]])  # exclusive

    cands = []   # candidate windows (dicts), pre-rejection
    for rid, (a, b) in enumerate(zip(run_start, run_end)):
        if not valid[a:b].all():
            # split further at invalid samples
            sub = np.where(~valid[a:b])[0] + a
            edges = np.concatenate([[a], sub, [b]])
            spans = [(int(edges[i] + (1 if i else 0)), int(edges[i + 1]))
                     for i in range(len(edges) - 1)]
        else:
            spans = [(int(a), int(b))]
        for (a2, b2) in spans:
            n_in = b2 - a2
            if n_in < WIN_IN:
                continue
            seg = x[a2:b2]
            padlen = 3 * (2 * len(SOS) + 1)
            if n_in <= padlen:
                continue
            filt = sosfiltfilt(SOS, seg, axis=0)
            down = resample_poly(filt, up=1, down=2, axis=0)  # (ceil(n/2), 4)
            n_win = len(down) // WIN
            for w in range(n_win):
                lo_in, hi_in = a2 + w * WIN_IN, a2 + (w + 1) * WIN_IN
                if hi_in > b2:
                    break
                win = down[w * WIN:(w + 1) * WIN]
                centre_ms = 0.5 * (ts[lo_in] + ts[hi_in - 1])
                block = None
                for bn, (t0, t1) in blocks.items():
                    if np.isfinite(t0) and np.isfinite(t1) and t0 <= centre_ms < t1:
                        block = bn
                        break
                fseg = fit_s[lo_in:hi_in]
                fit_bad_frac = np.nanmean(fseg == 4, axis=0)
                sub = win.reshape(4, 128, 4)
                p2p_ch = sub.max(axis=1).max(axis=0) - sub.min(axis=1).min(axis=0)
                p2p_sub_max = (sub.max(axis=1) - sub.min(axis=1)).max()
                sd_ch = win.std(axis=0)
                rej = []
                if p2p_sub_max > P2P_LIMIT:
                    rej.append("amplitude")
                if (sd_ch < FLAT_SD).any():
                    rej.append("flatline")
                if np.nanmax(fit_bad_frac) > 0:
                    rej.append("fit")
                cands.append({
                    "participant": f"P{p}", "session": f"S{s}", "block": block,
                    "run_id": rid, "start_sample": int(lo_in),
                    "centre_ms": float(centre_ms),
                    "kept": len(rej) == 0, "rej_reasons": ";".join(rej),
                    "max_p2p_uv": float(p2p_sub_max),
                    **{f"fit_bad_frac_{c}": float(fit_bad_frac[i])
                       for i, c in enumerate(CHANNELS)},
                    **{f"sd_{c}": float(sd_ch[i]) for i, c in enumerate(CHANNELS)},
                    "_win": np.ascontiguousarray(win.T, dtype=np.float32),  # (4, 512)
                })
    return cands, blocks


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log_rows, pool, control = [], [], []
    for p in PARTICIPANTS:
        for s in SESSIONS:
            cands, blocks = process_session(p, s)
            dm = delta_mental(p, s)
            is_control = (p, s) in NONINCREASING
            step = CLOCK_STEP.get((p, s))
            for bn in ["M1", "M2", "M3"]:
                bc = [c for c in cands if c["block"] == bn]
                kept = [c for c in bc if c["kept"]]
                t0, t1 = blocks[bn]
                nominal = int((t1 - t0) / 4000) if np.isfinite(t0) else 0
                reanchored = 0
                if step:
                    after = {"M1->M2": {"M2", "M3"}, "M2->M3": {"M3"}}[step]
                    if bn in after:
                        reanchored = len(bc)
                log_rows.append({
                    "participant": f"P{p}", "session": f"S{s}", "block": bn,
                    "delta_mental": round(dm, 2), "control_session": is_control,
                    "nominal_windows_from_duration": nominal,
                    "candidate_windows": len(bc),
                    "lost_to_gaps": nominal - len(bc),
                    "windows_reanchored": reanchored,
                    "windows_lost_clockstep": 0,
                    "kept_windows": len(kept),
                    "rejected": len(bc) - len(kept),
                    "rejection_pct": round(100 * (len(bc) - len(kept)) / len(bc), 1) if bc else 0.0,
                    "rej_amplitude": sum("amplitude" in c["rej_reasons"] for c in bc),
                    "rej_flatline": sum("flatline" in c["rej_reasons"] for c in bc),
                    "rej_fit": sum("fit" in c["rej_reasons"] for c in bc),
                })
            kept13 = [c for c in cands if c["kept"] and c["block"] in ("M1", "M3")]
            for c in kept13:
                c["label"] = 0 if c["block"] == "M1" else 1
                (control if is_control else pool).append(c)
            print(f"P{p}/S{s}: candidates={len(cands)} kept M1/M3={len(kept13)} "
                  f"delta_mental={dm:+.1f} control={is_control}")

    pd.DataFrame(log_rows).to_csv(OUT / "preprocessing_log.csv", index=False)

    def save(cands, name):
        if not cands:
            return
        arr = np.stack([c.pop("_win") for c in cands])
        np.save(OUT / f"{name}.npy", arr)
        idx = pd.DataFrame(cands)
        idx.to_csv(OUT / f"{name}_index.csv", index=False)
        return arr, idx

    # Uncapped pool first (pops _win), then rebuild capped + control views.
    pool_rows = [dict(c) for c in pool]
    arr_u, idx_u = save(pool, "windows_uncapped")
    print(f"\nUncapped pool: {arr_u.shape}, per class:",
          idx_u["label"].value_counts().to_dict())
    print("Per-participant window counts (uncapped):")
    print(idx_u.groupby(["participant", "label"]).size().unstack(fill_value=0))

    # Seed-0 capped primary set: cap M1 per session to retained M3 count.
    capped = []
    for (p, s), g in idx_u.groupby(["participant", "session"]):
        m1 = g[g.label == 0]
        m3 = g[g.label == 1]
        if len(m1) > len(m3):
            rng = np.random.default_rng(CAP_SEED)
            keep = rng.choice(m1.index.to_numpy(), size=len(m3), replace=False)
            g = pd.concat([m1.loc[keep], m3])
        capped.append(g)
    capped_idx = pd.concat(capped).sort_values(["participant", "session", "start_sample"])
    pos = idx_u.reset_index().set_index(["participant", "session", "start_sample"])
    sel = [pos.index.get_loc((r.participant, r.session, r.start_sample))
           for r in capped_idx.itertuples()]
    np.save(OUT / "windows.npy", arr_u[sel])
    capped_idx.to_csv(OUT / "windows_index.csv", index=False)
    print(f"\nCapped (seed {CAP_SEED}) primary set: {len(capped_idx)} windows, "
          f"per class: {capped_idx['label'].value_counts().to_dict()}")

    if control:
        arr_c, idx_c = save(control, "windows_control")
        print(f"Negative-control set: {arr_c.shape}, blocks:",
              idx_c["block"].value_counts().to_dict())


if __name__ == "__main__":
    main()
