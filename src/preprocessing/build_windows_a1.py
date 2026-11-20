"""Phase A re-run under AMENDMENT 1 to docs/label_rules.md (LOCKED 20 Nov 2026).

Amendment 1 replaces section 0(c)'s whole-window rejection:
  - A window is RETAINED iff BOTH frontal channels (AF7, AF8) pass
    (a) amplitude, (b) flatline, (c) fit on those channels.
  - A temporal channel (TP9/TP10) failing any criterion in that window is
    MASKED for that window: its samples in the saved array are zero-filled
    ONLY alongside the explicit per-window per-channel validity mask stored
    in the index CSV (columns valid_TP9 / valid_AF7 / valid_AF8 / valid_TP10).
    Consumers MUST consult the mask; the zeros are never silent.
  - Primary configuration (locked): all-valid-channels with temporal masking.
    Pre-registered variant: frontal-only (AF7/AF8).

Everything else is unchanged from build_windows.py: per-run 1-40 Hz
zero-phase bandpass, 256->128 Hz, 4 s tumbling windows within continuous
runs only, per-block marker anchoring, subjective primary labels
(M1 -> 0 alert, M3 -> 1 fatigued, only in the 32 delta-mental > 0 sessions),
seed-0 M1 cap for the primary set, and the 4 non-increasing sessions saved
separately as the drift/sensitivity check set under the same masking rules.

Amplitude criterion: as previously implemented and recorded in
docs/week2_deviation.md, "+/-75 uV peak-to-peak" = per-channel peak-to-peak
> 150 uV in any 1-s sub-window fails that channel. Per-channel max p2p is
stored in the index so any threshold can be re-audited.

Outputs (data/processed/):
  windows_a1_uncapped.npy / windows_a1_uncapped_index.csv  full retained pool
  windows_a1.npy / windows_a1_index.csv                    seed-0 capped primary
  windows_a1_control.npy / windows_a1_control_index.csv    check set (uncapped)
  candidates_a1.csv        every candidate window + per-channel pass/fail
                           (audit base for MNAR + threshold sensitivity)
  preprocessing_log_a1.csv per session/block accounting under Amendment 1
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
from src.preprocessing.build_windows import fit_per_sample

FS_IN, FS_OUT = 256, 128
WIN = 512                      # 4 s at 128 Hz
WIN_IN = WIN * 2               # 1024 samples at 256 Hz
GAP_MS = 100
P2P_LIMIT = 150.0              # uV p2p per channel = 2 x 75 uV (see docstring)
FLAT_SD = 0.5                  # uV
CAP_SEED = 0
FRONTAL = [CHANNELS.index("AF7"), CHANNELS.index("AF8")]   # [1, 2]
TEMPORAL = [CHANNELS.index("TP9"), CHANNELS.index("TP10")]  # [0, 3]

OUT = Path(__file__).resolve().parents[2] / "data" / "processed"
SOS = butter(4, [1.0, 40.0], btype="bandpass", fs=FS_IN, output="sos")


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

    cands = []
    for rid, (a, b) in enumerate(zip(run_start, run_end)):
        if not valid[a:b].all():
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
            down = resample_poly(filt, up=1, down=2, axis=0)
            n_win = len(down) // WIN
            for w in range(n_win):
                lo_in, hi_in = a2 + w * WIN_IN, a2 + (w + 1) * WIN_IN
                if hi_in > b2:
                    break
                win = down[w * WIN:(w + 1) * WIN]          # (512, 4)
                centre_ms = 0.5 * (ts[lo_in] + ts[hi_in - 1])
                block = None
                for bn, (t0, t1) in blocks.items():
                    if np.isfinite(t0) and np.isfinite(t1) and t0 <= centre_ms < t1:
                        block = bn
                        break
                fseg = fit_s[lo_in:hi_in]
                fit_bad_frac = np.nanmean(fseg == 4, axis=0)   # (4,)
                sub = win.reshape(4, 128, 4)                   # (subwin, samp, ch)
                p2p_ch = (sub.max(axis=1) - sub.min(axis=1)).max(axis=0)  # (4,)
                sd_ch = win.std(axis=0)                        # (4,)
                # Per-channel pass/fail under Amendment 1 criteria.
                fail_amp = p2p_ch > P2P_LIMIT
                fail_flat = sd_ch < FLAT_SD
                fail_fit = fit_bad_frac > 0                    # NaN -> False
                pass_ch = ~(fail_amp | fail_flat | fail_fit)
                retained = bool(pass_ch[FRONTAL].all())
                win_out = np.ascontiguousarray(win.T, dtype=np.float32)  # (4, 512)
                if retained:
                    # Zero-fill masked temporal channels ONLY; validity mask
                    # is stored in the index (never silent zeros).
                    for ci in TEMPORAL:
                        if not pass_ch[ci]:
                            win_out[ci] = 0.0
                cands.append({
                    "participant": f"P{p}", "session": f"S{s}", "block": block,
                    "run_id": rid, "start_sample": int(lo_in),
                    "centre_ms": float(centre_ms),
                    "retained_a1": retained,
                    **{f"valid_{c}": bool(pass_ch[i]) for i, c in enumerate(CHANNELS)},
                    **{f"max_p2p_{c}": float(p2p_ch[i]) for i, c in enumerate(CHANNELS)},
                    **{f"fail_amp_{c}": bool(fail_amp[i]) for i, c in enumerate(CHANNELS)},
                    **{f"fail_flat_{c}": bool(fail_flat[i]) for i, c in enumerate(CHANNELS)},
                    **{f"fail_fit_{c}": bool(fail_fit[i]) for i, c in enumerate(CHANNELS)},
                    **{f"fit_bad_frac_{c}": float(fit_bad_frac[i])
                       for i, c in enumerate(CHANNELS)},
                    **{f"sd_{c}": float(sd_ch[i]) for i, c in enumerate(CHANNELS)},
                    "_win": win_out,
                })
    return cands, blocks


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log_rows, pool, control, all_cands = [], [], [], []
    for p in PARTICIPANTS:
        for s in SESSIONS:
            cands, blocks = process_session(p, s)
            dm = delta_mental(p, s)
            is_control = (p, s) in NONINCREASING
            step = CLOCK_STEP.get((p, s))
            for c in cands:
                cc = {k: v for k, v in c.items() if k != "_win"}
                cc["delta_mental"] = round(dm, 2)
                cc["control_session"] = is_control
                all_cands.append(cc)
            for bn in ["M1", "M2", "M3"]:
                bc = [c for c in cands if c["block"] == bn]
                kept = [c for c in bc if c["retained_a1"]]
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
                    "retained_windows_a1": len(kept),
                    "rejected_frontal_fail": len(bc) - len(kept),
                    "retention_pct": round(100 * len(kept) / len(bc), 1) if bc else 0.0,
                    "masked_TP9": sum(1 for c in kept if not c["valid_TP9"]),
                    "masked_TP10": sum(1 for c in kept if not c["valid_TP10"]),
                })
            kept13 = [c for c in cands if c["retained_a1"] and c["block"] in ("M1", "M3")]
            for c in kept13:
                c["label"] = 0 if c["block"] == "M1" else 1
                (control if is_control else pool).append(c)
            print(f"P{p}/S{s}: candidates={len(cands)} retained M1/M3={len(kept13)} "
                  f"delta_mental={dm:+.1f} control={is_control}")

    pd.DataFrame(log_rows).to_csv(OUT / "preprocessing_log_a1.csv", index=False)
    pd.DataFrame(all_cands).to_csv(OUT / "candidates_a1.csv", index=False)

    def save(cands, name):
        arr = np.stack([c.pop("_win") for c in cands])
        np.save(OUT / f"{name}.npy", arr)
        idx = pd.DataFrame(cands)
        idx.to_csv(OUT / f"{name}_index.csv", index=False)
        return arr, idx

    pool_rows = [dict(c) for c in pool]
    arr_u, idx_u = save(pool, "windows_a1_uncapped")
    print(f"\nUncapped pool (A1): {arr_u.shape}, per class:",
          idx_u["label"].value_counts().to_dict())
    print("Per-participant window counts (uncapped, A1):")
    print(idx_u.groupby(["participant", "label"]).size().unstack(fill_value=0))

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
    np.save(OUT / "windows_a1.npy", arr_u[sel])
    capped_idx.to_csv(OUT / "windows_a1_index.csv", index=False)
    print(f"\nCapped (seed {CAP_SEED}) primary set (A1): {len(capped_idx)} windows, "
          f"per class: {capped_idx['label'].value_counts().to_dict()}")

    if control:
        arr_c, idx_c = save(control, "windows_a1_control")
        print(f"Drift/sensitivity check set (A1): {arr_c.shape}, blocks:",
              idx_c["block"].value_counts().to_dict())
        print("Control per participant/session:")
        print(idx_c.groupby(["participant", "session", "label"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
