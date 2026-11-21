"""RQ4 validity checks for the prototype-faithful replication (21 Nov 2026).

Why these checks exist: the prototype-faithful replication
(src/evaluation/rq4_replication_proto.py) reproduced the preliminary
experiment's 99.18% fatigued recall in its own configuration, and its
mask cut showed the fatigued class sitting almost entirely in windows
whose AF7 channel fails the pre-specified Amendment-1 amplitude
criterion (Session 03 retention 2.5% vs Session 01 retention 97.1%).
The result stands softened to non-identifiability, and three checks
are owed before any stronger wording ("recording-quality failure")
is allowed. I hold that line deliberately: the mask cut shows where
the fatigued class sits, not why, and each check below is aimed at
one of the explanations the soft wording leaves open:

  Check 1 - AF7 validity audit. The amplitude criterion is a rule
    about a number, not a diagnosis: drowsiness itself raises frontal
    slow-wave and slow-eye-movement amplitude, so a threshold sweep,
    the failure-mode breakdown, the other channels' behaviour, the
    failure onset timing, and the device streams that do not depend
    on the amplitude rule (fit/contact quality, accelerometer, gyro,
    blinks, jaw clenches, touch) together decide whether Session 03's
    AF7 amplitudes are a wholesale different regime or a marginal
    call straddling the threshold.

  Check 2 - route location. In the replication neither model ever
    received the mask, so "the features carry the information" was an
    inference. A mask-only logistic regression (validity indicators
    as the only features), plus AF7 and frontal-pair ablations of
    both models in the same configuration, locate the route.

  Check 3 - per-participant retention table. Subject 01's retention
    asymmetry (97.1% vs 2.5%) is extreme; this check prices it
    against all 36 sessions (Amendment-1 retention overall and by
    block phase, M1 vs M3, from the pipeline's own preprocessing
    log), and confirms from the Week 2 records which participant the
    mask-driven below-chance 0.225 belonged to.

Everything here is post hoc and diagnostic. No frozen file is edited;
the replication's split, bootstrap and metric helpers are imported
and reused so the numbers are computed the same way. Outputs:
results/tables/rq4_check1_*.csv, rq4_check2_route_location.csv,
rq4_check3_*.csv, and one run folder per Check 2 model run under
experiments/runs/. The write-up lives in docs/rq4_session_labels.md
("Validity checks" subsection of the replication section).
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.load_session import CHANNELS, load_concat, load_eeg, load_fit
from src.evaluation.metrics import binary_metrics
from src.evaluation.rq4_replication_proto import (block_bootstrap,
                                                  carve_groups,
                                                  chronological_masks,
                                                  subset_metrics)
from src.features.bandpower import FEATURE_NAMES, extract_features
from src.models.baseline import make_model
from src.models.cnn import build_inputs, train_fold
from src.preprocessing.build_windows import fit_per_sample

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
RUNS = ROOT / "experiments" / "runs"
TAB = ROOT / "results" / "tables"
DATE = "2026-11-21"
P_NUM = "01"
P2P_LIMIT, FLAT_SD = 150.0, 0.5     # Amendment-1 criteria (locked)
MULTIPLIERS = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
MODEL_SEEDS = [0, 1, 2]
AF7, AF8 = CHANNELS.index("AF7"), CHANNELS.index("AF8")
FRONTAL = [AF7, AF8]


def git_commit():
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()


# ---------------------------------------------------------------------
# Check 1 - AF7 validity audit
# ---------------------------------------------------------------------

def recompute_mask_components(arr, idx):
    """Per-window, per-channel Amendment-1 components for the proto
    window set, recomputed exactly as the replication computed them:
    largest 1 s sub-window peak-to-peak and window SD from the saved
    (filtered, 128 Hz) windows, Bad-fit fraction from the raw fit
    stream over the window's raw-rate span. Returns a DataFrame
    aligned with idx plus the recomputed pass flags, so the saved
    index can be verified rather than trusted."""
    n = len(idx)
    sub = arr.reshape(n, 4, 4, 128)          # (win, ch, 1 s subwin, samp)
    p2p = (sub.max(axis=3) - sub.min(axis=3)).max(axis=2)   # (n, 4)
    sd = arr.std(axis=2)                                   # (n, 4)
    fit_bad = np.zeros((n, 4))
    for folder, sname in (("01", "S01"), ("03", "S03")):
        ts, _x = load_eeg(P_NUM, folder)
        fit_ts, fit = load_fit(P_NUM, folder)
        fit_s = fit_per_sample(ts, fit_ts, fit)
        rows = np.where(idx.session.to_numpy() == sname)[0]
        for r in rows:
            k = int(idx.stream_window.iloc[r])
            seg = fit_s[k * 512:k * 512 + 1024]
            fit_bad[r] = np.nanmean(seg == 4, axis=0)
    comp = pd.DataFrame({
        "participant": idx.participant.values, "session": idx.session.values,
        "stream_window": idx.stream_window.values, "label": idx.label.values})
    for i, c in enumerate(CHANNELS):
        comp[f"p2p_{c}"] = p2p[:, i]
        comp[f"sd_{c}"] = sd[:, i]
        comp[f"fit_bad_frac_{c}"] = fit_bad[:, i]
        comp[f"fail_amp_{c}"] = p2p[:, i] > P2P_LIMIT
        comp[f"fail_flat_{c}"] = sd[:, i] < FLAT_SD
        comp[f"fail_fit_{c}"] = fit_bad[:, i] > 0
        comp[f"pass_{c}"] = ~(comp[f"fail_amp_{c}"] | comp[f"fail_flat_{c}"]
                              | comp[f"fail_fit_{c}"])
    return comp


def check1(arr, idx):
    comp = recompute_mask_components(arr, idx)
    # Verification: the recomputed flags must reproduce the saved index.
    mism = {}
    for c in CHANNELS:
        mism[c] = int((comp[f"pass_{c}"].to_numpy()
                       != idx[f"valid_{c}"].to_numpy()).sum())
    ret_re = (comp["pass_AF7"] & comp["pass_AF8"]).to_numpy()
    mism["a1_retained"] = int((ret_re != idx.a1_retained.to_numpy()).sum())
    print(f"CHECK 1 mask recomputation vs saved index, mismatches: {mism}")

    # --- amplitude distributions (the numbers the threshold judges)
    dist_rows = []
    for sname, g in comp.groupby("session"):
        for c in CHANNELS:
            v = g[f"p2p_{c}"]
            dist_rows.append({
                "session": sname, "channel": c, "n_windows": len(g),
                **{f"p2p_p{q}": float(np.percentile(v, q))
                   for q in (1, 5, 25, 50, 75, 95, 99)},
                "p2p_max": float(v.max()),
                "sd_median": float(g[f"sd_{c}"].median()),
                "frac_amp_fail": float(g[f"fail_amp_{c}"].mean())})
    pd.DataFrame(dist_rows).to_csv(
        TAB / "rq4_check1_amplitude_distributions.csv", index=False)

    # --- threshold sweep: amplitude pass rate per channel, and the
    # full-rule frontal retention with only the amplitude limit swept
    sweep_rows = []
    for sname, g in comp.groupby("session"):
        for m in MULTIPLIERS:
            lim = P2P_LIMIT * m
            row = {"session": sname, "multiplier": m, "limit_uV": lim}
            for c in CHANNELS:
                row[f"amp_pass_{c}"] = float((g[f"p2p_{c}"] <= lim).mean())
            frontal_ok = np.ones(len(g), dtype=bool)
            for ci in FRONTAL:
                c = CHANNELS[ci]
                frontal_ok &= ((g[f"p2p_{c}"] <= lim)
                               & ~g[f"fail_flat_{c}"] & ~g[f"fail_fit_{c}"])
            row["retention_full_rule"] = float(frontal_ok.mean())
            sweep_rows.append(row)
    pd.DataFrame(sweep_rows).to_csv(
        TAB / "rq4_check1_threshold_sweep.csv", index=False)

    # --- failure modes per channel (which rule, alone or combined)
    mode_rows = []
    for sname, g in comp.groupby("session"):
        for c in CHANNELS:
            a, f, t = (g[f"fail_amp_{c}"], g[f"fail_flat_{c}"],
                       g[f"fail_fit_{c}"])
            cats = {
                "pass": ~a & ~f & ~t, "amp_only": a & ~f & ~t,
                "flat_only": ~a & f & ~t, "fit_only": ~a & ~f & t,
                "amp_flat": a & f & ~t, "amp_fit": a & ~f & t,
                "flat_fit": ~a & f & t, "amp_flat_fit": a & f & t}
            mode_rows.append({"session": sname, "channel": c,
                              "n_windows": len(g),
                              **{k: int(v.sum()) for k, v in cats.items()}})
    pd.DataFrame(mode_rows).to_csv(
        TAB / "rq4_check1_failure_modes.csv", index=False)

    # --- channel retention (valid fraction) per session
    ret_rows = []
    for sname, g in comp.groupby("session"):
        ret_rows.append({"session": sname, "n_windows": len(g),
                         **{f"valid_{c}": float(g[f"pass_{c}"].mean())
                            for c in CHANNELS},
                         "a1_retained": float(
                             (g["pass_AF7"] & g["pass_AF8"]).mean())})
    pd.DataFrame(ret_rows).to_csv(
        TAB / "rq4_check1_channel_retention.csv", index=False)

    # --- onset timing: AF7 failure by stream decile, both sessions
    onset_rows, onset_summary = [], {}
    for folder, sname in (("01", "S01"), ("03", "S03")):
        ts, _x = load_eeg(P_NUM, folder)
        g = comp[comp.session == sname].sort_values("stream_window")
        n = len(g)
        dec = np.minimum(9, (np.arange(n) * 10) // n)
        for d in range(10):
            sel = dec == d
            onset_rows.append({
                "session": sname, "decile": d + 1, "n_windows": int(sel.sum()),
                "af7_amp_fail_frac": float(g.fail_amp_AF7.to_numpy()[sel].mean()),
                "af7_any_fail_frac": float(
                    (g.fail_amp_AF7 | g.fail_flat_AF7
                     | g.fail_fit_AF7).to_numpy()[sel].mean())})
        fail_amp = np.where(g.fail_amp_AF7.to_numpy())[0]
        first_k = int(g.stream_window.iloc[fail_amp[0]]) if len(fail_amp) else None
        onset_summary[sname] = {
            "n_windows": n, "first_af7_amp_fail_window": first_k,
            "first_fail_elapsed_min": (
                float((ts[first_k * 512] - ts[0]) / 60000.0)
                if first_k is not None else None),
            "frac_fail_first10pct": float(
                g.fail_amp_AF7.to_numpy()[: max(1, n // 10)].mean()),
            "frac_fail_last10pct": float(
                g.fail_amp_AF7.to_numpy()[-max(1, n // 10):].mean()),
            "frac_fail_overall": float(g.fail_amp_AF7.mean())}
    pd.DataFrame(onset_rows).to_csv(TAB / "rq4_check1_onset.csv", index=False)
    for sname, s in onset_summary.items():
        print(f"CHECK 1 onset {sname}: {s}")

    independent_streams()
    return comp


def _num(df, col):
    return pd.to_numeric(df[col], errors="coerce")


def independent_streams():
    """Device streams that do not depend on the amplitude rule, per
    session: the Muse fit (contact-quality) stream's value
    distribution, forehead accelerometer and gyroscope motion
    summaries, blink and jaw-clench event rates, and the touch
    stream. These say whether anything independent of my threshold
    also saw Session 03 differently."""
    rows, fit_rows = [], []
    for folder, sname in (("01", "S01"), ("03", "S03")):
        ts, _x = load_eeg(P_NUM, folder)
        hours = (ts[-1] - ts[0]) / 3_600_000.0
        row = {"session": sname, "duration_hours": float(hours)}

        fit_df = load_concat(P_NUM, folder, "muse_device_fit.csv")
        for c in CHANNELS:
            v = _num(fit_df, c)
            for val, lab in ((1, "good"), (2, "medium"), (4, "bad")):
                frac = float((v == val).mean())
                fit_rows.append({"session": sname, "channel": c,
                                 "fit_value": lab, "frac_samples": frac})
            row[f"fit_bad_frac_{c}"] = float((v == 4).mean())

        acc = load_concat(P_NUM, folder, "forehead_acc.csv")
        mag = np.sqrt(sum(_num(acc, a) ** 2 for a in ("ax", "ay", "az")))
        dyn = (mag - 1.0).abs()
        row.update({"acc_mag_mean": float(mag.mean()),
                    "acc_mag_median": float(mag.median()),
                    "acc_mag_p95": float(np.percentile(mag, 95)),
                    "acc_dyn_mean": float(dyn.mean()),
                    "acc_frac_dyn_gt_0.1g": float((dyn > 0.1).mean())})
        gyro = load_concat(P_NUM, folder, "forehead_gyro.csv")
        gmag = np.sqrt(sum(_num(gyro, a) ** 2 for a in ("gx", "gy", "gz")))
        row.update({"gyro_mag_mean": float(gmag.mean()),
                    "gyro_mag_median": float(gmag.median()),
                    "gyro_mag_p95": float(np.percentile(gmag, 95))})

        for base, col, key in (("muse_blinks.csv", "is_blink", "blinks"),
                               ("muse_jaw_clenches.csv", "is_clench",
                                "jaw_clenches")):
            ev = load_concat(P_NUM, folder, base)
            n_ev = int((_num(ev, col) == 1).sum())
            row[f"{key}_n"] = n_ev
            row[f"{key}_per_hour"] = float(n_ev / hours)
        touch = load_concat(P_NUM, folder, "muse_device_touch.csv")
        row["touch_frac"] = float((_num(touch, "is_touching") == 1).mean())
        rows.append(row)
        print(f"CHECK 1 streams {sname}: duration {hours:.2f} h, "
              f"blinks/h {row['blinks_per_hour']:.1f}, "
              f"jaw/h {row['jaw_clenches_per_hour']:.1f}, "
              f"acc dyn mean {row['acc_dyn_mean']:.3f} g, "
              f"touch {row['touch_frac']:.3f}, "
              f"fit Bad AF7 {row['fit_bad_frac_AF7']:.3f}")
    pd.DataFrame(rows).to_csv(
        TAB / "rq4_check1_independent_streams.csv", index=False)
    pd.DataFrame(fit_rows).to_csv(
        TAB / "rq4_check1_fit_distribution.csv", index=False)


# ---------------------------------------------------------------------
# Check 3 - per-participant retention table
# ---------------------------------------------------------------------

def check3():
    """Amendment-1 retention for all 36 sessions, overall and by
    block phase (M1 vs M3), from the pipeline's own preprocessing log
    (docs/label_rules.md Amendment 1 made this table a mandatory
    diagnostic), cross-checked against the candidate-window file."""
    log = pd.read_csv(PROC / "preprocessing_log_a1.csv")
    log = log[log.block.isin(["M1", "M3"])]

    def pct(ret, cand):
        """Retention percent, NaN when the block produced no candidate
        windows at all (that happens: a block can be lost entirely to
        recording gaps, which is itself recorded rather than hidden)."""
        return round(100 * ret / cand, 1) if cand else float("nan")

    rows = []
    for (p, s), g in log.groupby(["participant", "session"]):
        d = {r.block: r for r in g.itertuples()}
        cand = int(g.candidate_windows.sum())
        ret = int(g.retained_windows_a1.sum())
        rows.append({
            "participant": p, "session": s,
            "is_subject01": p == "P01",
            "delta_mental": float(g.delta_mental.iloc[0]),
            "control_session": bool(g.control_session.iloc[0]),
            "m1_candidates": int(d["M1"].candidate_windows),
            "m1_retained": int(d["M1"].retained_windows_a1),
            "m1_retention_pct": pct(int(d["M1"].retained_windows_a1),
                                    int(d["M1"].candidate_windows)),
            "m3_candidates": int(d["M3"].candidate_windows),
            "m3_retained": int(d["M3"].retained_windows_a1),
            "m3_retention_pct": pct(int(d["M3"].retained_windows_a1),
                                    int(d["M3"].candidate_windows)),
            "overall_candidates_m1m3": cand,
            "overall_retained_m1m3": ret,
            "overall_retention_pct": pct(ret, cand)})
    tab = pd.DataFrame(rows).sort_values(["participant", "session"])
    tab.to_csv(TAB / "rq4_check3_retention_by_session.csv", index=False)

    # Cross-check the log against the candidate-window audit file.
    candf = pd.read_csv(PROC / "candidates_a1.csv")
    candf = candf[candf.block.isin(["M1", "M3"])]
    chk = candf.groupby(["participant", "session"]).agg(
        n=("retained_a1", "size"), ret=("retained_a1", "sum")).reset_index()
    merged = tab.merge(chk, on=["participant", "session"], how="left")
    agree = bool((merged.overall_candidates_m1m3 == merged.n).all()
                 and (merged.overall_retained_m1m3 == merged.ret).all())
    print(f"CHECK 3 log vs candidates_a1 cross-check, all 36 sessions "
          f"agree: {agree}")

    # Asymmetry per participant: latest vs earliest session (the
    # day-level contrast) and fatigued-phase vs alert-phase blocks.
    asym = []
    for p, g in tab.groupby("participant"):
        by_sess = g.set_index("session")
        m1r = 100 * g.m1_retained.sum() / g.m1_candidates.sum()
        m3r = 100 * g.m3_retained.sum() / g.m3_candidates.sum()
        asym.append({
            "participant": p, "is_subject01": p == "P01",
            "ret_s01_pct": float(by_sess.loc["S01", "overall_retention_pct"]),
            "ret_s03_pct": float(by_sess.loc["S03", "overall_retention_pct"]),
            "gap_day_pp": float(by_sess.loc["S03", "overall_retention_pct"]
                                - by_sess.loc["S01", "overall_retention_pct"]),
            "ret_m1_pct": float(m1r), "ret_m3_pct": float(m3r),
            "gap_phase_pp": float(m3r - m1r)})
    asym = pd.DataFrame(asym).sort_values("gap_day_pp")
    asym.to_csv(TAB / "rq4_check3_retention_asymmetry.csv", index=False)
    print("CHECK 3 asymmetry table (sorted by day gap, percentage points):")
    print(asym.to_string(index=False))
    return tab, asym


# ---------------------------------------------------------------------
# Check 2 - route location (prototype configuration, chronological 80/20)
# ---------------------------------------------------------------------

MASKONLY_COLS = ["valid_TP9", "valid_AF7", "valid_AF8", "valid_TP10",
                 "n_valid_channels", "n_valid_temporals",
                 "both_temporals_masked", "a1_retained"]

CHECK2_CHECKLIST = """Leakage checklist (honest version):
- [x] Preprocessing statistics fit on training windows only (LR imputation/scaling; CNN normalisation and class weights from the subtraining windows)
- [ ] NO session sharing between train and test: VIOLATED BY DESIGN (the prototype's configuration, inherited from the replication; the violation is the object of the run)
- [ ] NO overlapping windows across the split: VIOLATED BY DESIGN (50% overlap is the prototype's windowing; the test window after the cut overlaps the last training window by 2 s)
- [x] No tuning on test data (both models are the frozen pipeline's; the ablation removes inputs, it does not tune anything)
- [x] Specification written before the run (panel review 28, docs/methods_notes.md, owed these checks by name)
"""


def maskonly_features(idx):
    """The per-window validity information and nothing else: the four
    channel pass flags from the TRUE Amendment-1 mask (not the
    all-valid modelling copy), plus coverage summaries of the same
    flags. This is the Check 2 analogue of the Week 2 mask-only
    baseline (docs/week2_diagnostics.md section 7), widened to all
    four channels because in this configuration the frontal flags are
    not constant: no window was retained or rejected before the run,
    so the frontal indicators vary and are exactly what is being
    priced."""
    v = {c: idx[f"valid_{c}"].to_numpy(dtype=float) for c in CHANNELS}
    out = pd.DataFrame({
        "valid_TP9": v["TP9"], "valid_AF7": v["AF7"],
        "valid_AF8": v["AF8"], "valid_TP10": v["TP10"],
        "n_valid_channels": sum(v.values()),
        "n_valid_temporals": v["TP9"] + v["TP10"],
        "both_temporals_masked": ((v["TP9"] == 0)
                                  & (v["TP10"] == 0)).astype(float),
        "a1_retained": idx.a1_retained.to_numpy(dtype=float)})
    return out[MASKONLY_COLS]


def write_check2_run(name, model_desc, feat_desc, seed_note, idx_te, y_te,
                     pred, proba, cis, extra_cfg, extra_notes):
    """One run folder in the experiments/ convention: config.yaml,
    metrics.csv (all / mask-retained / mask-rejected slices of the
    same test predictions), predictions.csv, notes.md."""
    d = RUNS / name
    d.mkdir(parents=True, exist_ok=True)
    ret = idx_te.a1_retained.values
    rows = []
    for sname, mask in (("all", np.ones(len(idx_te), bool)),
                        ("a1_retained", ret), ("a1_rejected", ~ret)):
        m = subset_metrics(y_te[mask], pred[mask], proba[mask])
        key = sname.replace("a1_", "")
        rows.append({"subset": sname, **m,
                     "ba_ci95_low": cis[(key, "balanced_accuracy")][0],
                     "ba_ci95_high": cis[(key, "balanced_accuracy")][1],
                     "recall_fatigued_ci95_low":
                         cis[(key, "recall_fatigued")][0],
                     "recall_fatigued_ci95_high":
                         cis[(key, "recall_fatigued")][1]})
    pd.DataFrame(rows).to_csv(d / "metrics.csv", index=False)
    preds = idx_te[["participant", "session", "stream_window",
                    "start_sample", "a1_retained"]].copy()
    preds["y_true"] = y_te
    preds["y_pred"] = pred
    preds["y_proba"] = proba
    preds.to_csv(d / "predictions.csv", index=False)
    cfg = {
        "model": model_desc,
        "features/input": feat_desc,
        "protocol": "RQ4 validity checks, Check 2 route location "
                    "(POST HOC, panel review 28): Subject 01 only, "
                    "prototype window set (1,353 windows, whole-stream "
                    "epoching), day-level labels, chronological 80/20 "
                    "split, identical to the replication",
        "split": "chronological 80/20 within each session (first "
                 "floor(0.8*n) windows in stream order = train, "
                 "remainder = test, no purge at the cut)",
        "seed": seed_note,
        "data": "windows_proto_p01 (the replication's window build)",
        "git_commit": git_commit(), **extra_cfg}
    (d / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (d / "notes.md").write_text(f"""# RQ4 validity Check 2, {model_desc}, chronological split, {seed_note}

Post hoc check owed by panel review 28 (docs/methods_notes.md). Same
window set, labels and split as the prototype-faithful replication
(src/evaluation/rq4_replication_proto.py); only the model's inputs
change, which is the point: the replication's models never received
the mask, so where the score's information lives was an inference.
This run prices one input change at a time.
{extra_notes}
{CHECK2_CHECKLIST}
Environment: pinned current stack (docs/environment_pin.md).
""")


def run_lr_cols(feats, cols, y, tr, te):
    m = make_model()
    m.fit(feats[cols].values[tr], y[tr])
    return (m.predict(feats[cols].values[te]),
            m.predict_proba(feats[cols].values[te])[:, 1])


def run_cnn_ablated(X, y, idx, tr, te, seed):
    """One EEGNet fit through the frozen cnn.train_fold on an input
    whose ablated EEG channel plane(s) were zeroed before training
    (see check2 for the adaptation note). Mirrors the replication's
    run_cnn."""
    groups = carve_groups(idx.iloc[np.where(tr)[0]].reset_index(drop=True))
    model, mean, std, note = train_fold(X[tr], y[tr], groups, seed)
    if model is None:
        raise RuntimeError(f"CNN carve failed: {note}")
    Xn = X[te].copy()
    Xn[:, 0, :4] = (Xn[:, 0, :4] - mean[None, :, None]) / std[None, :, None]
    model.eval()
    with torch.no_grad():
        proba = torch.softmax(model(torch.tensor(Xn)), dim=1).numpy()
    return proba.argmax(1), proba[:, 1], note


def check2(arr, idx):
    y = idx.label.values
    retained_all = idx.a1_retained.values
    tr, te = chronological_masks(idx)
    idx_te = idx.iloc[np.where(te)[0]].reset_index(drop=True)
    y_te, ret_te = y[te], retained_all[te]

    # Modelling copy with every channel treated valid, exactly as the
    # replication built its features and CNN inputs.
    idx_m = idx.copy()
    for c in CHANNELS:
        idx_m[f"valid_{c}"] = True
    feats = extract_features(arr, idx_m)
    summary = []

    def add_row(model, variant, source, sname, m, cis_row):
        summary.append({
            "model": model, "variant": variant, "source": source,
            "subset": sname, **m,
            "ba_ci95_low": cis_row[0], "ba_ci95_high": cis_row[1],
            "recall_fatigued_ci95_low": cis_row[2],
            "recall_fatigued_ci95_high": cis_row[3]})

    def slices(pred, proba):
        return (("all", np.ones(len(idx_te), bool)),
                ("a1_retained", ret_te), ("a1_rejected", ~ret_te))

    # ---- reference rows from the replication's stored summary, so
    # the ablations sit next to the numbers they are compared against
    ref = pd.read_csv(TAB / "rq4_replication_proto_summary.csv")
    ref = ref[ref.split == "chronological"]
    for _, r in ref.iterrows():
        add_row(r.model, "full (replication)", "replication run",
                r.subset,
                {k: r[k] for k in ("n", "n_alert", "n_fatigued",
                                   "accuracy", "balanced_accuracy",
                                   "recall_alert", "recall_fatigued",
                                   "roc_auc")},
                (r.ba_ci95_low, r.ba_ci95_high,
                 r.recall_fatigued_ci95_low, r.recall_fatigued_ci95_high))

    # ---- LR full, recomputed as a wiring check (deterministic; must
    # match the replication's stored row)
    pred, proba = run_lr_cols(feats, list(FEATURE_NAMES), y, tr, te)
    m_all = subset_metrics(y_te, pred, proba)
    print(f"CHECK 2 LR full recompute: BA {m_all['balanced_accuracy']:.3f} "
          f"(replication stored 1.000)")

    # ---- LR mask-only + the trivial mask rule
    mfeats = maskonly_features(idx)
    pred, proba = run_lr_cols(mfeats, MASKONLY_COLS, y, tr, te)
    cis = block_bootstrap(idx_te, y_te, ret_te, [(pred, proba)])
    write_check2_run(
        f"{DATE}_validity-check2_lr-maskonly_chronological_s0",
        "band-power logistic regression (frozen baseline)",
        "MASK ONLY: 8 validity features (4 channel pass flags, "
        "n_valid_channels, n_valid_temporals, both_temporals_masked, "
        "a1_retained); no EEG features",
        "deterministic (s0 by convention)", idx_te, y_te, pred, proba,
        cis, {},
        "The model sees the Amendment-1 validity pattern of each "
        "window and nothing else. The trivial rule (predict fatigued "
        "iff the window is not mask-retained) is reported alongside "
        "in results/tables/rq4_check2_route_location.csv; it needs no "
        "fitting, so it has no run folder of its own.\n")
    for sname, mask in slices(pred, proba):
        key = sname.replace("a1_", "")
        add_row("lr", "mask-only", "this run", sname,
                subset_metrics(y_te[mask], pred[mask], proba[mask]),
                (*cis[(key, "balanced_accuracy")],
                 *cis[(key, "recall_fatigued")]))
    rule_pred = (~ret_te).astype(int)
    m_rule = binary_metrics(y_te, rule_pred, rule_pred.astype(float))
    add_row("rule", "predict fatigued iff not mask-retained",
            "this run (no fitting)", "all",
            {k: m_rule[k] for k in ("n", "n_alert", "n_fatigued",
                                    "balanced_accuracy", "recall_alert",
                                    "recall_fatigued")},
            (np.nan, np.nan, np.nan, np.nan))
    print(f"CHECK 2 mask-only LR: BA "
          f"{subset_metrics(y_te, pred, proba)['balanced_accuracy']:.3f}; "
          f"trivial rule BA {m_rule['balanced_accuracy']:.3f}")

    # ---- LR ablations: drop every feature derived from the ablated
    # channel(s). The frontal ratios are functions of AF7 and AF8
    # jointly, so they go in both ablations; what remains is stated
    # in each run's config.
    drop_sets = {
        "af7abl": [c for c in FEATURE_NAMES
                   if c.startswith(("logpow_AF7_", "relpow_AF7_",
                                    "frontal_ratio_"))],
        "frontalabl": [c for c in FEATURE_NAMES
                       if c.startswith(("logpow_AF7_", "relpow_AF7_",
                                        "logpow_AF8_", "relpow_AF8_",
                                        "frontal_ratio_"))],
    }
    for variant, drops in drop_sets.items():
        cols = [c for c in FEATURE_NAMES if c not in drops]
        pred, proba = run_lr_cols(feats, cols, y, tr, te)
        cis = block_bootstrap(idx_te, y_te, ret_te, [(pred, proba)])
        write_check2_run(
            f"{DATE}_validity-check2_lr-{variant}_chronological_s0",
            "band-power logistic regression (frozen baseline)",
            f"band-power features minus {len(drops)} dropped columns: "
            + ", ".join(drops),
            "deterministic (s0 by convention)", idx_te, y_te, pred,
            proba, cis, {"dropped_features": drops,
                         "n_features_used": len(cols)},
            f"Ablation: every feature derived from "
            f"{'AF7' if variant == 'af7abl' else 'the frontal pair AF7+AF8'} "
            f"is removed before fitting ({len(drops)} of 43 columns, "
            f"listed in config.yaml). The mask indicators were constant "
            f"in this configuration and are unchanged.\n")
        for sname, mask in slices(pred, proba):
            key = sname.replace("a1_", "")
            add_row("lr", variant, "this run", sname,
                    subset_metrics(y_te[mask], pred[mask], proba[mask]),
                    (*cis[(key, "balanced_accuracy")],
                     *cis[(key, "recall_fatigued")]))
        m = subset_metrics(y_te, pred, proba)
        print(f"CHECK 2 LR {variant}: BA {m['balanced_accuracy']:.3f}, "
              f"fatigued recall {m['recall_fatigued']:.3f}")

    # ---- CNN ablations. The frozen EEGNet architecture fixes the
    # input at 6 planes (4 EEG + 2 temporal mask-indicator planes),
    # and its spatial filter spans all 6, so a channel cannot be
    # dropped without changing the architecture the replication's
    # numbers came from. The adaptation, stated here and in every
    # run note: the ablated channel's EEG plane is set to zero for
    # every window, train and test alike, keeping the frozen layout;
    # the network then receives no signal from that electrode (its
    # normalisation statistics are 0/0 on that plane, so it stays
    # zero), while every other part of the pipeline is untouched.
    X = build_inputs(arr, idx_m)
    cnn_variants = {"af7abl": [AF7], "frontalabl": [AF7, AF8]}
    for variant, chans in cnn_variants.items():
        Xa = X.copy()
        for ci in chans:
            Xa[:, 0, ci, :] = 0.0
        preds, notes = [], []
        for seed in MODEL_SEEDS:
            p_s, pr_s, note = run_cnn_ablated(Xa, y, idx, tr, te, seed)
            preds.append((p_s, pr_s))
            notes.append(note)
            print(f"CHECK 2 CNN {variant} seed {seed}: {note}")
        cis_cnn = block_bootstrap(idx_te, y_te, ret_te, preds)
        nan_cis = {k: (np.nan, np.nan) for k in cis_cnn}
        names = ", ".join(CHANNELS[ci] for ci in chans)
        for j, (seed, (p_s, pr_s)) in enumerate(zip(MODEL_SEEDS, preds)):
            write_check2_run(
                f"{DATE}_validity-check2_cnn-{variant}_chronological_s{seed}",
                "EEGNet CNN (frozen cnn.py train_fold)",
                f"raw stream windows, frozen 6-plane input with the "
                f"{names} EEG plane(s) zeroed for every window "
                f"(channel ablation; see notes)",
                f"model seed {seed}", idx_te, y_te, p_s, pr_s, nan_cis,
                {"ablated_channels": [CHANNELS[ci] for ci in chans]},
                f"Ablation adaptation: the frozen architecture fixes "
                f"6 input planes, so the {names} EEG plane(s) are "
                f"zeroed rather than removed; the network receives no "
                f"signal from that electrode/those electrodes. "
                f"Interval columns are empty in this per-seed folder: "
                f"intervals are computed for the seed-mean statistic "
                f"and reported in results/tables/"
                f"rq4_check2_route_location.csv. CNN carve (frozen "
                f"train_fold, tail-quintile groups): {notes[j]}.\n")
        for sname, mask in slices(None, None):
            per_seed = [subset_metrics(y_te[mask], p[mask], pr[mask])
                        for p, pr in preds]
            agg = {c: float(np.mean([mm[c] for mm in per_seed]))
                   for c in per_seed[0]}
            key = sname.replace("a1_", "")
            add_row("cnn", variant, "this run (seed mean)", sname, agg,
                    (*cis_cnn[(key, "balanced_accuracy")],
                     *cis_cnn[(key, "recall_fatigued")]))
        m_all = subset_metrics(y_te, preds[0][0], preds[0][1])
        print(f"CHECK 2 CNN {variant}: seed-mean metrics in summary "
              f"table (seed 0 BA {m_all['balanced_accuracy']:.3f})")

    out = pd.DataFrame(summary)
    out.to_csv(TAB / "rq4_check2_route_location.csv", index=False)
    print("CHECK 2 summary written to "
          "results/tables/rq4_check2_route_location.csv")
    return out


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    arr = np.load(PROC / "windows_proto_p01.npy")
    idx = pd.read_csv(PROC / "windows_proto_p01_index.csv").reset_index(drop=True)
    print(f"proto window set: {arr.shape}, index rows {len(idx)}")
    if which in ("all", "check1"):
        check1(arr, idx)
    if which in ("all", "check3"):
        check3()
    if which in ("all", "check2"):
        check2(arr, idx)
    if which == "check2cnn":
        check2_cnn_recovery(arr, idx)





# ---------------------------------------------------------------------
# Check 2 recovery (environment restart during the CNN stage)
# ---------------------------------------------------------------------

def check2_cnn_recovery(arr, idx):
    """Recovery run for Check 2's CNN stage. The first execution of
    the validity checks was interrupted by an environment restart
    after the three LR run folders were written and before any CNN
    fit had finished. This function completes the stage with the
    specification unchanged: it reuses the LR stage's stored run
    folders for the summary table (their metrics.csv files are the
    LR results), recomputes the trivial mask rule (no fitting
    involved), and runs the two CNN ablations, writing each seed's
    run folder immediately after its fit so a second interruption
    cannot take finished fits with it. The incident is recorded in
    the CNN run notes and in docs/rq4_session_labels.md, the same
    treatment the replication's own restart received."""
    y = idx.label.values
    retained_all = idx.a1_retained.values
    tr, te = chronological_masks(idx)
    idx_te = idx.iloc[np.where(te)[0]].reset_index(drop=True)
    y_te, ret_te = y[te], retained_all[te]
    summary = []

    def add_row(model, variant, source, sname, m, cis_row):
        summary.append({
            "model": model, "variant": variant, "source": source,
            "subset": sname, **m,
            "ba_ci95_low": cis_row[0], "ba_ci95_high": cis_row[1],
            "recall_fatigued_ci95_low": cis_row[2],
            "recall_fatigued_ci95_high": cis_row[3]})

    # Reference rows from the replication's stored summary.
    ref = pd.read_csv(TAB / "rq4_replication_proto_summary.csv")
    ref = ref[ref.split == "chronological"]
    for _, r in ref.iterrows():
        add_row(r.model, "full (replication)", "replication run",
                r.subset,
                {k: r[k] for k in ("n", "n_alert", "n_fatigued",
                                   "accuracy", "balanced_accuracy",
                                   "recall_alert", "recall_fatigued",
                                   "roc_auc")},
                (r.ba_ci95_low, r.ba_ci95_high,
                 r.recall_fatigued_ci95_low, r.recall_fatigued_ci95_high))

    # LR rows from the first execution's stored run folders.
    mcols = ("n", "n_alert", "n_fatigued", "accuracy",
             "balanced_accuracy", "macro_f1", "mcc", "recall_alert",
             "recall_fatigued", "roc_auc")
    for variant, folder in (
            ("mask-only", "lr-maskonly"), ("af7abl", "lr-af7abl"),
            ("frontalabl", "lr-frontalabl")):
        met = pd.read_csv(RUNS / f"{DATE}_validity-check2_{folder}"
                          "_chronological_s0" / "metrics.csv")
        for _, r in met.iterrows():
            add_row("lr", variant, "this run (LR stage, stored folder)",
                    r.subset, {k: r[k] for k in mcols},
                    (r.ba_ci95_low, r.ba_ci95_high,
                     r.recall_fatigued_ci95_low,
                     r.recall_fatigued_ci95_high))
    print("CHECK 2 recovery: LR rows loaded from stored run folders",
          flush=True)

    # Trivial mask rule (deterministic, no fitting).
    rule_pred = (~ret_te).astype(int)
    m_rule = binary_metrics(y_te, rule_pred, rule_pred.astype(float))
    add_row("rule", "predict fatigued iff not mask-retained",
            "this run (no fitting)", "all",
            {k: m_rule[k] for k in ("n", "n_alert", "n_fatigued",
                                    "balanced_accuracy", "recall_alert",
                                    "recall_fatigued")},
            (np.nan, np.nan, np.nan, np.nan))
    print(f"CHECK 2 recovery: trivial rule BA "
          f"{m_rule['balanced_accuracy']:.3f}", flush=True)

    # CNN ablations, same adaptation as the original stage.
    idx_m = idx.copy()
    for c in CHANNELS:
        idx_m[f"valid_{c}"] = True
    X = build_inputs(arr, idx_m)
    nan_cis = {(s, m): (np.nan, np.nan)
               for s in ("all", "retained", "rejected")
               for m in ("balanced_accuracy", "recall_fatigued")}
    for variant, chans in {"af7abl": [AF7],
                           "frontalabl": [AF7, AF8]}.items():
        Xa = X.copy()
        for ci in chans:
            Xa[:, 0, ci, :] = 0.0
        names = ", ".join(CHANNELS[ci] for ci in chans)
        preds = []
        for seed in MODEL_SEEDS:
            p_s, pr_s, note = run_cnn_ablated(Xa, y, idx, tr, te, seed)
            preds.append((p_s, pr_s))
            print(f"CHECK 2 recovery CNN {variant} seed {seed}: {note}",
                  flush=True)
            write_check2_run(
                f"{DATE}_validity-check2_cnn-{variant}"
                f"_chronological_s{seed}",
                "EEGNet CNN (frozen cnn.py train_fold)",
                f"raw stream windows, frozen 6-plane input with the "
                f"{names} EEG plane(s) zeroed for every window "
                f"(channel ablation; see notes)",
                f"model seed {seed}", idx_te, y_te, p_s, pr_s, nan_cis,
                {"ablated_channels": [CHANNELS[ci] for ci in chans],
                 "recovery_run": True},
                f"Ablation adaptation: the frozen architecture fixes "
                f"6 input planes, so the {names} EEG plane(s) are "
                f"zeroed rather than removed; the network receives no "
                f"signal from that electrode/those electrodes. "
                f"Recovery run: the first execution of these checks "
                f"was interrupted by an environment restart during "
                f"this CNN stage, after the LR stage's folders were "
                f"written; this stage was completed afterwards with "
                f"the specification unchanged, reusing the driver's "
                f"own functions. Interval columns are empty in this "
                f"per-seed folder: intervals are computed for the "
                f"seed-mean statistic and reported in results/tables/"
                f"rq4_check2_route_location.csv. CNN carve (frozen "
                f"train_fold, tail-quintile groups): {note}.\n")
        cis_cnn = block_bootstrap(idx_te, y_te, ret_te, preds)
        for sname, mask in (("all", np.ones(len(idx_te), bool)),
                            ("a1_retained", ret_te),
                            ("a1_rejected", ~ret_te)):
            per_seed = [subset_metrics(y_te[mask], p[mask], pr[mask])
                        for p, pr in preds]
            agg = {c: float(np.mean([mm[c] for mm in per_seed]))
                   for c in per_seed[0]}
            key = sname.replace("a1_", "")
            add_row("cnn", variant, "this run (seed mean, recovery)",
                    sname, agg,
                    (*cis_cnn[(key, "balanced_accuracy")],
                     *cis_cnn[(key, "recall_fatigued")]))
        print(f"CHECK 2 recovery CNN {variant}: done", flush=True)

    out = pd.DataFrame(summary)
    out.to_csv(TAB / "rq4_check2_route_location.csv", index=False)
    print("CHECK 2 summary written to "
          "results/tables/rq4_check2_route_location.csv", flush=True)
    return out


if __name__ == "__main__":
    main()
