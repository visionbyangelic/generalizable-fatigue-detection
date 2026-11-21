"""Week 1 data inspection for FatigueSet.

Runs the full inspection described in PLAN.md Week 1 and
docs/fatigueset-research.md section 8: structure, Muse EEG sampling/fit,
subjective + performance labels, clock alignment, continuation files,
quality flags. Writes results/tables/inspection_summary.csv and
results/tables/inspection_key_results.json, and prints a full report.

Usage (from project root):  python3 src/data/inspect_dataset.py
"""

# @@SECTION: 0. Setup: paths, session/intensity map
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "fatigueset"
PARTICIPANTS = [f"{i:02d}" for i in range(1, 13)]
SESSIONS = ["01", "02", "03"]
BLOCKS = ["M1", "M2", "M3"]

meta = pd.read_csv(RAW / "metadata.csv", encoding="utf-8-sig", dtype=str)
meta["participant_id"] = meta["participant_id"].str.zfill(2)
INTENSITY = {}  # (participant, session) -> low/medium/high
for _, r in meta.iterrows():
    for level in ["low", "medium", "high"]:
        INTENSITY[(r["participant_id"], r[f"{level}_session"].zfill(2))] = level
print(f"metadata.csv: {len(meta)} participants; intensity map built for {len(INTENSITY)} sessions")
print("P01 mapping check:", {s: INTENSITY[('01', s)] for s in SESSIONS})

# @@SECTION: 1. Structure: 39 core files per session
CANONICAL = sorted(p.name for p in (RAW / "01" / "01").iterdir() if p.is_file())
print(f"Canonical core-file list derived from P01/S01: {len(CANONICAL)} files")
structure_rows, continuation_files = [], []
for p in PARTICIPANTS:
    for s in SESSIONS:
        d = RAW / p / s
        names = {f.name for f in d.iterdir() if f.is_file()}
        cont = sorted(n for n in names if n.endswith("2.csv"))
        continuation_files += [(p, s, n) for n in cont]
        structure_rows.append({
            "participant": p, "session": s, "intensity": INTENSITY[(p, s)],
            "n_files": len(names), "n_core_present": len(set(CANONICAL) & names),
            "missing_core": ";".join(sorted(set(CANONICAL) - names)),
            "n_continuations": len(cont),
        })
structure = pd.DataFrame(structure_rows)
print(f"Sessions with all {len(CANONICAL)} core files: {(structure.n_core_present == len(CANONICAL)).sum()}/36")
print(f"Total *2.csv continuation files: {len(continuation_files)}")
print("Sessions with continuations:", sorted({f'{p}/S{s}' for p, s, _ in continuation_files}))

# @@SECTION: 2. Markers: phase durations + missing timestamps
PHASES = [("baseline", "start_baseline", "end_baseline"),
          ("M1_block", "end_baseline", "start_activity"),
          ("activity", "start_activity", "end_activity"),
          ("M2_block", "end_activity", "start_fatigue"),
          ("task_switch", "start_fatigue", "end_fatigue"),
          ("M3_block", "end_fatigue", "end_session")]


def load_markers(p, s):
    df = pd.read_csv(RAW / p / s / "exp_markers.csv", dtype={"eventMarker": str})
    df["utcTime"] = pd.to_numeric(df["utcTime"], errors="coerce")
    return df


def marker_time(df, name, occ=0):
    vals = df.loc[df["eventMarker"] == name, "utcTime"].tolist()
    return vals[occ] if len(vals) > occ else np.nan


marker_rows, missing_markers = [], []
for p in PARTICIPANTS:
    for s in SESSIONS:
        df = load_markers(p, s)
        for i, row in df.iterrows():
            if pd.isna(row["utcTime"]):
                missing_markers.append((p, s, row["eventMarker"], int(df.index.get_loc(i))))
        row = {"participant": p, "session": s, "n_events": len(df)}
        t0, t1 = marker_time(df, "start_experiment"), marker_time(df, "end_session")
        row["total_min"] = (t1 - t0) / 60000.0
        for phase, a, b in PHASES:
            ta, tb = marker_time(df, a), marker_time(df, b)
            row[f"{phase}_min"] = (tb - ta) / 60000.0
        marker_rows.append(row)
markers = pd.DataFrame(marker_rows)
print(f"Marker events per session: min={markers.n_events.min()}, max={markers.n_events.max()} (expected 17)")
print(f"Missing marker timestamps found: {len(missing_markers)} -> {missing_markers}")
print(f"Session total duration (min): mean={markers.total_min.mean():.2f} sd={markers.total_min.std():.2f} "
      f"min={markers.total_min.min():.2f} max={markers.total_min.max():.2f}")
for ph, _, _ in PHASES:
    c = markers[f"{ph}_min"]
    print(f"  {ph:12s}: mean={c.mean():6.2f} min sd={c.std():.2f}  range {c.min():.2f}-{c.max():.2f} min  (NaN in {c.isna().sum()} sessions)")

# @@SECTION: 3. Muse EEG: sampling, channels, duplicates, gaps (base + *2 concatenated)
def read_sensor_csv(path):
    """Read a sensor CSV, dropping a stray leading index column if present."""
    df = pd.read_csv(path)
    if df.columns[0].startswith("Unnamed") or df.columns[0] == "":
        df = df.iloc[:, 1:]
    return df


def load_concat(p, s, base):
    """Load base file plus its *2.csv continuation (if any), sorted by timestamp."""
    frames, parts = [], []
    for cand in [base, base.replace(".csv", "2.csv")]:
        fp = RAW / p / s / cand
        if fp.exists():
            d = read_sensor_csv(fp)
            frames.append(d)
            parts.append((cand, len(d)))
    out = pd.concat(frames, ignore_index=True)
    if "timestamp" in out.columns:
        out = out.sort_values("timestamp", kind="stable").reset_index(drop=True)
    return out, parts


eeg_rows = []
for p in PARTICIPANTS:
    for s in SESSIONS:
        eeg, parts = load_concat(p, s, "forehead_eeg_raw.csv")
        ts = eeg["timestamp"].to_numpy(dtype=np.int64)
        dur_s = (ts[-1] - ts[0]) / 1000.0
        diffs = np.diff(ts)
        bins = (ts - ts[0]) // 1000  # wall-clock second bins
        per_sec = pd.Series(ts).groupby(bins).count()
        mk = load_markers(p, s)
        eeg_rows.append({
            "participant": p, "session": s,
            "channels": ";".join(c for c in eeg.columns if c != "timestamp"),
            "n_samples": len(ts), "duration_min": dur_s / 60.0,
            "eff_fs_hz": (len(ts) - 1) / dur_s,
            "samples_per_sec_median": float(per_sec[per_sec > 0].median()),
            "dup_ts_frac": float((diffs == 0).mean()),
            "n_gaps_gt100ms": int((diffs > 100).sum()),
            "n_gaps_gt1s": int((diffs > 1000).sum()),
            "total_gap_s": float(diffs[diffs > 100].sum() / 1000.0),
            "max_gap_ms": int(diffs.max()),
            "start_offset_s": (ts[0] - marker_time(mk, "start_baseline")) / 1000.0,
            "end_offset_s": (marker_time(mk, "end_session") - ts[-1]) / 1000.0,
            "nan_cells": int(eeg.isna().sum().sum()),
            "parts": ";".join(f"{n}:{c}" for n, c in parts),
        })
eeg = pd.DataFrame(eeg_rows)
print(f"EEG channels (all sessions identical): {eeg.channels.unique()}")
print(f"EEG duration (min): mean={eeg.duration_min.mean():.2f} sd={eeg.duration_min.std():.2f} range {eeg.duration_min.min():.2f}-{eeg.duration_min.max():.2f}")
print(f"Effective sampling rate over full span (Hz): mean={eeg.eff_fs_hz.mean():.2f} range {eeg.eff_fs_hz.min():.2f}-{eeg.eff_fs_hz.max():.2f}  (depressed by gaps)")
print(f"Median samples per wall-clock second: mean={eeg.samples_per_sec_median.mean():.1f} range {eeg.samples_per_sec_median.min():.0f}-{eeg.samples_per_sec_median.max():.0f} (nominal 256 Hz; timestamps are BLE-packet-jittered, so sample INDEX is the time base)")
print(f"Gap burden: total gap time (>100 ms intervals) mean={eeg.total_gap_s.mean():.0f}s/session, worst={eeg.total_gap_s.max():.0f}s; gaps >1 s in {(eeg.n_gaps_gt1s > 0).sum()}/36 sessions")
_worst_gap = eeg.loc[eeg.max_gap_ms.idxmax()]
print(f"Largest single gap: {_worst_gap.max_gap_ms/1000:.1f} s in P{_worst_gap.participant}/S{_worst_gap.session}")
print(f"EEG coverage vs markers: start offset (EEG start - start_baseline) median={eeg.start_offset_s.median():.1f}s range {eeg.start_offset_s.min():.1f}-{eeg.start_offset_s.max():.1f}s; end offset (end_session - EEG end) median={eeg.end_offset_s.median():.1f}s range {eeg.end_offset_s.min():.1f}-{eeg.end_offset_s.max():.1f}s")
print(f"Duplicated-timestamp fraction: mean={eeg.dup_ts_frac.mean():.3f} range {eeg.dup_ts_frac.min():.3f}-{eeg.dup_ts_frac.max():.3f}")
print(f"Sessions with gaps >100 ms: {(eeg.n_gaps_gt100ms > 0).sum()}/36; worst max-gap = {eeg.max_gap_ms.max()} ms")
print(f"NaN cells in EEG: total {eeg.nan_cells.sum()}")
print("EEG file parts (only sessions where parts are unusual):")
for _, r in eeg[eeg.parts.str.contains("2.csv") | (eeg.n_samples < 1000)].iterrows():
    print(f"  P{r.participant}/S{r.session}: {r.parts} -> n={r.n_samples}")

# @@SECTION: 4. Muse fit quality + blink/jaw event counts
fit_rows = []
for p in PARTICIPANTS:
    for s in SESSIONS:
        fit, _ = load_concat(p, s, "muse_device_fit.csv")
        ch = ["TP9", "AF7", "AF8", "TP10"]
        row = {"participant": p, "session": s, "fit_rows": len(fit)}
        for c in ch:
            row[f"good_{c}"] = float((fit[c] == 1).mean() * 100)
            row[f"bad_{c}"] = float((fit[c] == 4).mean() * 100)
        row["good_all_pct"] = float((fit[ch] == 1).all(axis=1).mean() * 100)
        row["bad_any_pct"] = float((fit[ch] == 4).any(axis=1).mean() * 100)
        bl, _ = load_concat(p, s, "muse_blinks.csv")
        jw, _ = load_concat(p, s, "muse_jaw_clenches.csv")
        row["blinks"] = int((bl["is_blink"] == 1).sum())
        row["jaw_clenches"] = int((jw["is_clench"] == 1).sum())
        fit_rows.append(row)
fit = pd.DataFrame(fit_rows)
print(f"Fit: % time ALL channels Good — mean={fit.good_all_pct.mean():.1f} sd={fit.good_all_pct.std():.1f} range {fit.good_all_pct.min():.1f}-{fit.good_all_pct.max():.1f}")
for c in ["TP9", "AF7", "AF8", "TP10"]:
    print(f"  {c}: Good mean={fit[f'good_{c}'].mean():.1f}% (worst session {fit[f'good_{c}'].min():.1f}%)  Bad mean={fit[f'bad_{c}'].mean():.1f}%")
print(f"Blinks/session: mean={fit.blinks.mean():.0f} range {fit.blinks.min()}-{fit.blinks.max()};  Jaw clenches/session: mean={fit.jaw_clenches.mean():.0f} range {fit.jaw_clenches.min()}-{fit.jaw_clenches.max()}")
print("Worst 5 sessions by all-channel Good fit:")
for _, r in fit.nsmallest(5, "good_all_pct").iterrows():
    print(f"  P{r.participant}/S{r.session}: good_all={r.good_all_pct:.1f}% bad_any={r.bad_any_pct:.1f}% blinks={r.blinks} jaw={r.jaw_clenches}")

# @@SECTION: 5. Subjective labels: exp_fatigue VAS per block
fat_rows = []
for p in PARTICIPANTS:
    for s in SESSIONS:
        f = pd.read_csv(RAW / p / s / "exp_fatigue.csv")
        assert len(f) == 3, (p, s, len(f))
        row = {"participant": p, "session": s, "intensity": INTENSITY[(p, s)]}
        for i, b in enumerate(BLOCKS):
            row[f"mental_{b}"] = float(f.loc[f.measurementNumber == i, "mentalFatigueScore"].iloc[0])
            row[f"physical_{b}"] = float(f.loc[f.measurementNumber == i, "physicalFatigueScore"].iloc[0])
        row["delta_mental"] = row["mental_M3"] - row["mental_M1"]
        row["delta_physical"] = row["physical_M3"] - row["physical_M1"]
        fat_rows.append(row)
fat = pd.DataFrame(fat_rows)
allm = fat[[f"mental_{b}" for b in BLOCKS]].to_numpy().ravel()
allp = fat[[f"physical_{b}" for b in BLOCKS]].to_numpy().ravel()
print(f"Ratings: {len(allm)} mental + {len(allp)} physical (108 each expected)")
print(f"Mental VAS:   mean={allm.mean():.1f} sd={allm.std():.1f} median={np.median(allm):.1f} range {allm.min():.1f}-{allm.max():.1f}")
print(f"Physical VAS: mean={allp.mean():.1f} sd={allp.std():.1f} median={np.median(allp):.1f} range {allp.min():.1f}-{allp.max():.1f}")
print(f"Pearson r(mental, physical) across 108 paired ratings: r={np.corrcoef(allm, allp)[0, 1]:.3f}")
n_pos = int((fat.delta_mental > 0).sum())
print(f"delta_mental (M3-M1): mean={fat.delta_mental.mean():.1f} sd={fat.delta_mental.std():.1f} range {fat.delta_mental.min():.1f}-{fat.delta_mental.max():.1f}")
print(f"*** Sessions with delta_mental > 0: {n_pos}/36 ***")
print(f"Sessions with delta_mental <= 0: {sorted(f'P{r.participant}/S{r.session} ({r.intensity}, d={r.delta_mental:.1f})' for _, r in fat[fat.delta_mental <= 0].iterrows())}")
print(f"delta_physical: mean={fat.delta_physical.mean():.1f} range {fat.delta_physical.min():.1f}-{fat.delta_physical.max():.1f}; >0 in {(fat.delta_physical > 0).sum()}/36")
print(f"Sign agreement delta_mental vs delta_physical: {int((np.sign(fat.delta_mental) == np.sign(fat.delta_physical)).sum())}/36 sessions")
for inten in ["low", "medium", "high"]:
    sub = fat[fat.intensity == inten]
    print(f"  {inten:6s}: delta_mental mean={sub.delta_mental.mean():.1f}, >0 in {(sub.delta_mental > 0).sum()}/12; delta_physical mean={sub.delta_physical.mean():.1f}")

# @@SECTION: 6. Performance labels: CRT and 2-back per block
def block_stats(p, s, fname, rt_col="responseTime"):
    df = pd.read_csv(RAW / p / s / fname)
    out = {}
    for i, b in enumerate(BLOCKS):
        blk = df[df.measurementBlock == i]
        rt = pd.to_numeric(blk[rt_col], errors="coerce")
        out[f"n_{b}"] = len(blk)
        out[f"rt_{b}"] = float(rt.mean())
        out[f"rt_n_{b}"] = int(rt.notna().sum())
        out[f"acc_{b}"] = float(blk["isCorrectResponse"].mean())
    out["d_rt"] = out["rt_M3"] - out["rt_M1"]
    out["d_acc"] = out["acc_M3"] - out["acc_M1"]
    return out


crt_rows, nb_rows = [], []
for p in PARTICIPANTS:
    for s in SESSIONS:
        crt_rows.append({"participant": p, "session": s, **block_stats(p, s, "exp_crt.csv")})
        nb_rows.append({"participant": p, "session": s, **block_stats(p, s, "exp_nback.csv")})
crt, nb = pd.DataFrame(crt_rows), pd.DataFrame(nb_rows)
for name, d, exp_n in [("CRT", crt, 36), ("N-back", nb, 20)]:
    print(f"{name}: trials/block — values {sorted(d.n_M1.unique())}/{sorted(d.n_M2.unique())}/{sorted(d.n_M3.unique())} (expected {exp_n})")
    print(f"  RT (s, responded trials): M1 mean={d.rt_M1.mean():.3f} M2 mean={d.rt_M2.mean():.3f} M3 mean={d.rt_M3.mean():.3f}")
    print(f"  Accuracy: M1 mean={d.acc_M1.mean():.3f} M2 mean={d.acc_M2.mean():.3f} M3 mean={d.acc_M3.mean():.3f}")
    print(f"  dRT (M3-M1, s): mean={d.d_rt.mean():+.3f} sd={d.d_rt.std():.3f} range {d.d_rt.min():+.3f}-{d.d_rt.max():+.3f}; slower in {(d.d_rt > 0).sum()}/36")
    print(f"  dAcc (M3-M1): mean={d.d_acc.mean():+.3f} sd={d.d_acc.std():.3f} range {d.d_acc.min():+.3f}-{d.d_acc.max():+.3f}; worse in {(d.d_acc < 0).sum()}/36")
print(f"N-back responded-trial fraction: M1={nb.rt_n_M1.sum() / nb.n_M1.sum():.2f} M3={nb.rt_n_M3.sum() / nb.n_M3.sum():.2f} (empty responseTime = correct rejections / no-response)")
# CRT responseTime appears to be measured from trial onset; the stimulus follows later.
_crt_all = pd.concat([pd.read_csv(RAW / p / s / "exp_crt.csv").assign(p=p, s=s)
                      for p in PARTICIPANTS for s in SESSIONS])
_crt_all["delay"] = _crt_all.stimulusStartTime - _crt_all.trialStartTime
_ok = _crt_all.delay.between(0, 5)
print(f"CRT stimulus delay after trial onset: median={_crt_all.loc[_ok, 'delay'].median():.3f}s; "
      f"trials dropped as corrupt/NaN delay: {int((~_ok).sum())}/{len(_crt_all)}")
_crt_ok = _crt_all[_ok].copy()
_crt_ok["rt_corr"] = _crt_ok.responseTime - _crt_ok.delay
_g = _crt_ok.groupby(["p", "s", "measurementBlock"]).rt_corr.mean().unstack()
_dcrt = _g[2] - _g[0]
print(f"CRT stimulus-locked RT (responseTime - stimulus delay): block means M1={_g[0].mean():.3f}s M2={_g[1].mean():.3f}s M3={_g[2].mean():.3f}s; dRT M3-M1 mean={_dcrt.mean():+.3f}s sd={_dcrt.std():.3f} range {_dcrt.min():+.3f}-{_dcrt.max():+.3f}")
_bad_base = crt[crt.acc_M1 < 0.5]
print(f"CRT sessions with broken M1 baseline (accuracy < 0.5): {[(f'P{r.participant}/S{r.session}', round(r.acc_M1, 2)) for _, r in _bad_base.iterrows()]}")

# @@SECTION: 7. Alignment: two clocks, missing markers, continuation quirks
off_rows = []
for p in PARTICIPANTS:
    for s in SESSIONS:
        mk = load_markers(p, s)
        t0 = marker_time(mk, "start_experiment")
        f = pd.read_csv(RAW / p / s / "exp_fatigue.csv")
        subs = mk.loc[mk.eventMarker == "submit_survey", "utcTime"].tolist()
        offs = [(subs[k] - t0) / 1000.0 - f.fatigueSurveySubmissionTime.iloc[k] for k in range(min(3, len(subs)))]
        crt_df = pd.read_csv(RAW / p / s / "exp_crt.csv")
        crt_starts = mk.loc[mk.eventMarker == "start_crt", "utcTime"].tolist()
        crt_offs = []
        for b in range(3):
            first_trial = crt_df.loc[crt_df.measurementBlock == b, "trialStartTime"].min()
            if b < len(crt_starts) and pd.notna(crt_starts[b]):
                crt_offs.append((crt_starts[b] - t0) / 1000.0 - first_trial)
        off_rows.append({"participant": p, "session": s,
                         "fatigue_offsets_s": offs, "crt_offsets_s": crt_offs})
fat_off = np.array([o for r in off_rows for o in r["fatigue_offsets_s"]])
crt_off = np.array([o for r in off_rows for o in r["crt_offsets_s"]])
print("Clock model: epoch_ms ~= start_experiment + relative_s * 1000")
print(f"Fatigue submit offset (marker - relative): n={len(fat_off)} median={np.median(fat_off):.2f}s range {fat_off.min():.2f}-{fat_off.max():.2f}s")
print(f"CRT start offset (marker - first trial):  n={len(crt_off)} median={np.median(crt_off):.2f}s range {crt_off.min():.2f}-{crt_off.max():.2f}s")
print(f"Missing markers (all sessions): {missing_markers}")
# Per-block offset analysis: in 8 sessions the relative clock steps by ~180 s at one
# block boundary (= the 3-min activity period is excluded from the relative clock).
_off_by_block = np.array([r["fatigue_offsets_s"] for r in off_rows])  # 36 x 3
print(f"Per-block fatigue offsets (marker - relative), median per block: "
      f"M1={np.median(_off_by_block[:, 0]):.2f}s M2={np.median(_off_by_block[:, 1]):.2f}s M3={np.median(_off_by_block[:, 2]):.2f}s")
_stepped = []
for i, r in enumerate(off_rows):
    o = _off_by_block[i]
    steps = [o[1] - o[0], o[2] - o[1]]
    if max(abs(steps[0]), abs(steps[1])) > 5:
        _stepped.append((f"P{r['participant']}/S{r['session']}", [round(x, 1) for x in steps]))
print(f"Sessions with a >5 s offset step at a block boundary: {len(_stepped)} -> {_stepped}")
print("VERDICT: anchor EACH BLOCK to its own markers (submit_survey/start_crt for that block); "
      "with per-block anchoring every offset is <1 s. A single global origin would misalign later blocks by ~180 s in 8/36 sessions.")
eeg_cont = [(p, s, n) for p, s, n in continuation_files if "forehead_eeg" in n]
print(f"Continuation files by type: EEG files affected -> {eeg_cont}")
hdr_quirks = []
for p, s, n in continuation_files:
    with open(RAW / p / s / n) as fh:
        first = fh.readline().strip()
    if first.startswith(",") or "Unnamed" in first:
        hdr_quirks.append((p, s, n))
print(f"*2.csv files with a stray leading index column: {len(hdr_quirks)} -> {sorted(set((p, s) for p, s, _ in hdr_quirks))}")

# @@SECTION: 8. Quality flags + summary table
summary = structure[["participant", "session", "intensity"]].copy()
summary = summary.merge(markers[["participant", "session", "total_min"]], on=["participant", "session"])
summary = summary.merge(eeg[["participant", "session", "duration_min", "eff_fs_hz", "samples_per_sec_median",
                             "n_gaps_gt100ms", "total_gap_s", "start_offset_s"]],
                        on=["participant", "session"]).rename(columns={"duration_min": "eeg_duration_min"})
summary = summary.merge(fit[["participant", "session", "good_all_pct", "blinks", "jaw_clenches"]], on=["participant", "session"])
summary = summary.merge(fat[["participant", "session", "mental_M1", "mental_M2", "mental_M3", "delta_mental",
                              "physical_M1", "physical_M2", "physical_M3", "delta_physical"]], on=["participant", "session"])
summary = summary.merge(crt[["participant", "session", "d_rt", "d_acc", "acc_M1"]].rename(
                        columns={"d_rt": "crt_drt_s", "d_acc": "crt_dacc", "acc_M1": "crt_acc_m1"}),
                        on=["participant", "session"])
summary = summary.merge(nb[["participant", "session", "d_rt", "d_acc"]].rename(columns={"d_rt": "nback_drt_s", "d_acc": "nback_dacc"}),
                        on=["participant", "session"])
out_csv = ROOT / "results" / "tables" / "inspection_summary.csv"
summary.round(3).to_csv(out_csv, index=False)
print(f"Wrote {out_csv} ({len(summary)} rows)")

flags = []
for _, r in summary.iterrows():
    f = []
    if r.good_all_pct < 50: f.append(f"poor fit (good_all={r.good_all_pct:.0f}%)")
    if r.delta_mental <= 0: f.append(f"delta_mental<=0 ({r.delta_mental:.1f}) — excluded from primary subjective set")
    if r.total_gap_s > 60: f.append(f"EEG gap burden {r.total_gap_s:.0f}s total in >100ms gaps")
    if r.crt_acc_m1 < 0.5: f.append(f"CRT M1 baseline broken (acc={r.crt_acc_m1:.2f}) — performance baseline unusable")
    if r.total_min < 15: f.append(f"short session ({r.total_min:.1f} min)")
    if f: flags.append((f"P{r.participant}/S{r.session}", "; ".join(f)))
print(f"\nQuality flags ({len(flags)} sessions flagged):")
for k, v in flags: print(f"  {k}: {v}")

key = {
    "n_sessions_delta_mental_pos": n_pos,
    "mental_vas_mean": float(allm.mean()), "mental_vas_range": [float(allm.min()), float(allm.max())],
    "r_mental_physical": float(np.corrcoef(allm, allp)[0, 1]),
    "eeg_fs_mean": float(eeg.eff_fs_hz.mean()), "eeg_duration_min_mean": float(eeg.duration_min.mean()),
    "fit_good_all_mean": float(fit.good_all_pct.mean()), "fit_good_all_min": float(fit.good_all_pct.min()),
    "crt_drt_mean_s": float(crt.d_rt.mean()), "nback_dacc_mean": float(nb.d_acc.mean()),
    "missing_markers": [list(m) for m in missing_markers],
    "fatigue_offset_median_s": float(np.median(fat_off)), "crt_offset_median_s": float(np.median(crt_off)),
    "flags": flags,
}
(ROOT / "results" / "tables" / "inspection_key_results.json").write_text(json.dumps(key, indent=2))
print("\nWrote results/tables/inspection_key_results.json")
print(summary.round(2).to_string(index=False))
