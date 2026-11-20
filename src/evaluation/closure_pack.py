"""Week 2 CLOSURE PACK (Reviewer 1 review 11 - docs/methods_notes.md).

Fills the gaps Reviewer 1 required before Week 2 can close, under the
docs/decision_table_lock.md rules. Same discipline throughout:
Amendment-1 windows, cap seeds 0-4, logistic regression (C=1.0,
class_weight='balanced'), imputation/scaling/class weights fit on the
training fold only. Every analysis gets a run folder per
experiments/README.md.

Subcommands:
  frontal     (task 1) frontal-only (F) baseline, within + session-out
  lopo        (task 2) leave-one-PARTICIPANT-out, P (43 feats) and F
  poscontrol  (task 3) heterogeneous positive control + MDE dose curve
  gap_p08     (tasks 4-5) leakage-gap bootstrap CI; with/without P08
  gate        (task 6) feasibility-gate INPUTS (no verdict declared)
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from numpy.fft import irfft, rfft, rfftfreq
from scipy import stats
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.diagnostics import git_commit, write_run
from src.evaluation.metrics import binary_metrics
from src.evaluation.panel_additions import aggregate_sessionout, inject
from src.features.bandpower import FEATURE_NAMES, extract_features
from src.models.baseline import (METRIC_COLS, cap_pool, make_model,
                                 run_folds, sessionout_folds, summarise,
                                 within_folds)

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
RUNS = ROOT / "experiments" / "runs"
TAB = ROOT / "results" / "tables"
DATE = "2026-11-20"
SEEDS = [0, 1, 2, 3, 4]
FS = 128
CHANS = ["TP9", "AF7", "AF8", "TP10"]

# Frontal-only (F) feature set - EXACT list (20 features). Frontal log and
# relative band powers for AF7/AF8 plus the four frontal ratios, which are
# computed from AF7/AF8 mean band powers only (no temporal channel enters
# any of them - see src/features/bandpower.py). NO mask/validity
# indicators: those describe temporal masking and are not frontal data.
FRONTAL_FEATURES = (
    [f"logpow_{c}_{b}" for c in ["AF7", "AF8"]
     for b in ["delta", "theta", "alpha", "beta"]]
    + [f"relpow_{c}_{b}" for c in ["AF7", "AF8"]
       for b in ["delta", "theta", "alpha", "beta"]]
    + [f"frontal_ratio_{r}" for r in ["th_al_over_beta", "al_over_beta",
                                      "th_al_over_al_beta", "th_over_beta"]])


def load_uncapped():
    arr = np.load(PROC / "windows_a1_uncapped.npy")
    idx = pd.read_csv(PROC / "windows_a1_uncapped_index.csv").reset_index(drop=True)
    return arr, idx


def capped_feats(idx_u, feats_u, seed):
    capped = cap_pool(idx_u, seed)
    pos = capped.index.to_numpy()
    return (capped.reset_index(drop=True),
            feats_u.iloc[pos].reset_index(drop=True))


def write_protocol_run(name, protocol, seed, per_group, preds, skipped,
                       group_col, feature_desc, split_desc):
    """Like baseline.write_run but with closure-pack naming/config."""
    d = RUNS / name
    d.mkdir(parents=True, exist_ok=True)
    metrics = per_group.copy()
    agg = {c: per_group[c].mean() for c in METRIC_COLS if c in per_group}
    metrics.loc[len(metrics)] = {group_col: "AGGREGATE_mean_over_groups", **agg}
    metrics.to_csv(d / "metrics.csv", index=False)
    preds.to_csv(d / "predictions.csv", index=False)
    cfg = {"model": "logistic_regression", "C": 1.0,
           "class_weight": "balanced", "features": feature_desc,
           "imputation": "training-fold median",
           "scaling": "training-fold StandardScaler",
           "protocol": protocol, "labels": "subjective (Amendment 1 windows)",
           "cap_seed": seed, "data": "windows_a1_uncapped, M1 capped per session",
           "git_commit": git_commit()}
    (d / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (d / "notes.md").write_text(f"""# {name}

Split definition: {split_desc}. A fold counts only if train and test
both contain both classes; skipped folds: {skipped if skipped else 'none'}.

Leakage checklist:
- [x] Normalisation (StandardScaler) fit on training fold only (Pipeline, per fold)
- [x] Median imputation fit on training fold only
- [x] Class weights from training fold only (class_weight='balanced')
- [x] No overlapping windows across splits (tumbling windows; splits at session/participant level)
- [x] No tuning on test data (C=1.0 locked default)
- [x] Label rules locked (docs/label_rules.md incl. Amendment 1) before any training

Environment: local VM (CPU), scikit-learn. Seed = M1-cap subsample seed {seed};
logistic regression itself is deterministic given the data.
""")
    return agg


def cross_seed_summary(parts, gcol, out_name):
    allp = pd.concat(parts)
    summ = allp.groupby(gcol)[METRIC_COLS].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.reset_index().to_csv(TAB / out_name, index=False)
    return allp


# ------------------------------------------------------------------ task 1
def frontal(feats_u, idx_u):
    fidx = [FEATURE_NAMES.index(c) for c in FRONTAL_FEATURES]
    feats_f = feats_u.iloc[:, fidx]
    assert list(feats_f.columns) == FRONTAL_FEATURES
    desc = ("FRONTAL-ONLY (F): 20 features = logpow + relpow for AF7/AF8 "
            "(4 bands each) + 4 frontal ratios (AF7/AF8 means only). "
            "No temporal features, no mask indicators.")
    summaries = {"within": [], "session-out": []}
    for seed in SEEDS:
        df, fc = capped_feats(idx_u, feats_f, seed)
        for protocol, fold_fn, gcol, split_desc in [
                ("within", within_folds, "participant",
                 "per participant, rotate held-out session (train = other "
                 "sessions of the same participant)"),
                ("session-out", sessionout_folds, "heldout_session",
                 "pooled: hold out one (participant, session); train on all "
                 "remaining sessions of all participants")]:
            preds, skipped = run_folds(df, fc, fold_fn(df))
            if protocol == "within":
                per = summarise(preds, "participant")
            else:
                per = summarise(preds, "fold").rename(
                    columns={"fold": "heldout_session"})
            write_protocol_run(
                f"{DATE}_baseline_{protocol}_subjective_frontal_s{seed}",
                protocol, seed, per, preds, skipped, gcol, desc, split_desc)
            per["seed"] = seed
            summaries[protocol].append(per)
            print(f"F seed {seed} {protocol}: groups={len(per)} "
                  f"skipped={skipped} mean BA={per.balanced_accuracy.mean():.3f} "
                  f"AUC={per.roc_auc.mean():.3f} MCC={per.mcc.mean():.3f}",
                  flush=True)
    for protocol, parts in summaries.items():
        gcol = "participant" if protocol == "within" else "heldout_session"
        cross_seed_summary(parts, gcol,
                           f"baseline_{protocol}_subjective_frontal_summary.csv")


# ------------------------------------------------------------------ task 2
def lopo_folds(df):
    keys = []
    for part in df.participant.unique():
        te = (df.participant == part).values
        keys.append((part, ~te, te))
    return keys


def participant_boot_ci(per_participant_ba, n_boot=10_000, seed=42):
    vals = np.asarray(per_participant_ba, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    means = vals[idx].mean(axis=1)
    return float(vals.mean()), float(np.percentile(means, 2.5)), \
        float(np.percentile(means, 97.5)), len(vals)


def lopo(feats_u, idx_u):
    fidx = [FEATURE_NAMES.index(c) for c in FRONTAL_FEATURES]
    variants = {"P_masked_allchannel": feats_u,
                "F_frontal_only": feats_u.iloc[:, fidx]}
    agg_rows = []
    for vname, feats_v in variants.items():
        desc = ("bandpower_43_amendment1 (P, masked all-channel)"
                if vname.startswith("P") else
                "FRONTAL-ONLY (F): 20 AF7/AF8 features (see closure_pack.py)")
        suffix = "" if vname.startswith("P") else "_frontal"
        parts = []
        for seed in SEEDS:
            df, fc = capped_feats(idx_u, feats_v, seed)
            preds, skipped = run_folds(df, fc, lopo_folds(df))
            per = summarise(preds, "participant")
            write_protocol_run(
                f"{DATE}_baseline_loso_subjective{suffix}_s{seed}",
                "loso (leave-one-participant-out)", seed, per, preds, skipped,
                "participant", desc,
                "hold out ALL sessions of one participant; train on the "
                "other 11 participants. P05 has no fold: its only retained "
                "session is single-class (5 alert / 0 fatigued), so the "
                "per-session M1 cap reduces P05 to zero windows in every "
                "capped dataset (verified in the closure pack)")
            per["seed"] = seed
            parts.append(per)
            print(f"LOPO {vname} seed {seed}: participants={len(per)} "
                  f"skipped={skipped} mean BA={per.balanced_accuracy.mean():.3f}",
                  flush=True)
        allp = cross_seed_summary(
            parts, "participant",
            f"baseline_loso_subjective{suffix}_a1_summary.csv")
        per_part = allp.groupby("participant").balanced_accuracy.mean()
        pt, lo, hi, n = participant_boot_ci(per_part.values)
        per_part_mcc = allp.groupby("participant").mcc.mean()
        ptm, lom, him, _ = participant_boot_ci(per_part_mcc.values)
        per_part_auc = allp.groupby("participant").roc_auc.mean()
        pta, loa, hia, _ = participant_boot_ci(per_part_auc.values)
        agg_rows.append({"variant": vname, "n_participants": n,
                         "balanced_accuracy": pt, "ba_ci95_low": lo,
                         "ba_ci95_high": hi, "mcc": ptm, "mcc_ci95_low": lom,
                         "mcc_ci95_high": him, "roc_auc": pta,
                         "auc_ci95_low": loa, "auc_ci95_high": hia,
                         "ci_method": "participant-level bootstrap, 10000 "
                                      "resamples over cross-seed per-"
                                      "participant means, rng 42"})
        print(f"LOPO {vname}: BA={pt:.3f} [{lo:.3f}, {hi:.3f}] n={n}",
              flush=True)
    pd.DataFrame(agg_rows).to_csv(TAB / "closurepack_lopo_aggregate.csv",
                                  index=False)


# ------------------------------------------------------------------ task 3
DOSES = [("theta_x2.0_alpha_x1.5", 2.0, 1.5),
         ("theta_x1.5_alpha_x1.25", 1.5, 1.25),
         ("theta_x1.25_alpha_x1.125", 1.25, 1.125),
         ("theta_x1.1_alpha_x1.05", 1.1, 1.05)]
HETERO_SEED = 20261120


def session_boot_ci(per_session_vals, n_boot=10_000, seed=42):
    vals = np.asarray(per_session_vals, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    means = vals[idx].mean(axis=1)
    return float(vals.mean()), float(np.percentile(means, 2.5)), \
        float(np.percentile(means, 97.5)), len(vals)


def inject_hetero(arr, idx, gains):
    """Per-session spectral-gain injection into fatigued windows only,
    valid channels only. gains: dict (participant, session) ->
    (theta_power_gain, alpha_power_gain). Phase preserved."""
    out = arr.copy()
    freqs = rfftfreq(arr.shape[2], d=1.0 / FS)
    th = (freqs >= 4) & (freqs < 8)
    al = (freqs >= 8) & (freqs < 12)
    keys = list(zip(idx.participant, idx.session))
    for i in np.where(idx.label.values == 1)[0]:
        g_th, g_al = gains[keys[i]]
        if g_th == 1.0 and g_al == 1.0:
            continue
        for ci, c in enumerate(CHANS):
            if not bool(idx[f"valid_{c}"].iloc[i]):
                continue
            X = rfft(out[i, ci].astype(np.float64))
            X[th] *= np.sqrt(g_th)
            X[al] *= np.sqrt(g_al)
            out[i, ci] = irfft(X, n=arr.shape[2]).astype(out.dtype)
    return out


def hetero_gains(idx):
    """Documented scheme (seed HETERO_SEED): per primary session, theta
    power gain ~ LogNormal(ln 1.5, 0.55), alpha ~ LogNormal(ln 1.25,
    0.45) - centred on the homogeneous half-strength dose. Then a
    uniform draw per session: r < 0.10 -> null session (gains = 1);
    0.10 <= r < 0.20 -> opposite-sign session (gains inverted). So ~20%
    of sessions are null-or-opposite BY DESIGN; realised counts are
    saved with the gains table."""
    rng = np.random.default_rng(HETERO_SEED)
    sessions = sorted(set(zip(idx.participant, idx.session)))
    rows, gains = [], {}
    for part, sess in sessions:
        a = float(np.exp(rng.normal(np.log(1.5), 0.55)))
        b = float(np.exp(rng.normal(np.log(1.25), 0.45)))
        r = float(rng.uniform())
        kind = "graded"
        if r < 0.10:
            a, b, kind = 1.0, 1.0, "null"
        elif r < 0.20:
            a, b, kind = 1.0 / a, 1.0 / b, "opposite"
        gains[(part, sess)] = (a, b)
        rows.append({"participant": part, "session": sess,
                     "theta_power_gain": a, "alpha_power_gain": b,
                     "uniform_draw": r, "session_kind": kind})
    return gains, pd.DataFrame(rows)


def poscontrol(arr_u, idx_u):
    comb = pd.read_csv(TAB / "diagnostics_permutation_combined_summary.csv")
    null_hi = float(comb[(comb.statistic == "balanced_accuracy")
                         & (comb.null_type == "block_circular_shift")]
                    .null_q975.iloc[0])
    print(f"block-permutation null band upper edge (q975): {null_hi:.4f}",
          flush=True)
    base_feats = extract_features(arr_u, idx_u)
    fat = idx_u.label.values == 1

    # ---- homogeneous dose curve (all four doses, per-fold saved)
    dose_rows, fold_rows, mde_rows = [], [], []
    for name, g_th, g_al in DOSES:
        arr_mod = inject(arr_u, idx_u, g_th, g_al)
        feats_mod = extract_features(arr_mod, idx_u)
        per_session = {}
        for seed in SEEDS:
            df, fc = capped_feats(idx_u, feats_mod, seed)
            per, agg = aggregate_sessionout(df, fc)
            dose_rows.append({"dose": name, "theta_power_gain": g_th,
                              "alpha_power_gain": g_al, "seed": seed, **agg})
            for _, r in per.iterrows():
                fold_rows.append({"dose": name, "seed": seed,
                                  "heldout_session": r.heldout_session,
                                  "balanced_accuracy": r.balanced_accuracy,
                                  "roc_auc": r.roc_auc, "mcc": r.mcc})
                per_session.setdefault(r.heldout_session, []).append(
                    r.balanced_accuracy)
            print(f"  dose {name} seed {seed}: BA={agg['balanced_accuracy']:.3f}",
                  flush=True)
        sess_means = pd.Series({k: np.mean(v) for k, v in per_session.items()})
        pt, lo, hi, n = session_boot_ci(sess_means.values)
        mde_rows.append({"dose": name, "theta_power_gain": g_th,
                         "alpha_power_gain": g_al, "n_sessions": n,
                         "balanced_accuracy": pt, "ba_ci95_low": lo,
                         "ba_ci95_high": hi,
                         "block_null_q975": null_hi,
                         "ci_clears_null_band": bool(lo > null_hi)})
    doses_df = pd.DataFrame(dose_rows)
    folds_df = pd.DataFrame(fold_rows)
    mde_df = pd.DataFrame(mde_rows)
    doses_df.to_csv(TAB / "diagnostics_positive_control_doses.csv", index=False)
    folds_df.to_csv(TAB / "diagnostics_positive_control_doses_folds.csv",
                    index=False)
    mde_df.to_csv(TAB / "diagnostics_positive_control_mde.csv", index=False)
    clearing = mde_df[mde_df.ci_clears_null_band]
    mde_name = (clearing.sort_values("theta_power_gain").dose.iloc[0]
                if len(clearing) else "none of the tested doses")
    write_run(f"{DATE}_baseline_positivecontrol-mde_subjective_s0",
              {"model": "logistic_regression", "C": 1.0,
               "class_weight": "balanced",
               "features": "bandpower_43_amendment1 (re-extracted after injection)",
               "injection": "homogeneous spectral gain, fatigued windows "
                            "only, valid channels only; doses theta/alpha "
                            "power x(2.0,1.5), (1.5,1.25), (1.25,1.125), "
                            "(1.1,1.05) - extends panel_additions.py doses "
                            "downward, re-run with per-fold outputs",
               "protocol": "session-out", "labels": "subjective (Amendment 1)",
               "cap_seeds": "0-4 (folder named s0; all seeds in metrics.csv)",
               "mde_definition": "smallest tested dose whose session-level "
                                 "bootstrap 95% CI for balanced accuracy "
                                 "(cross-seed per-session means, 10000 "
                                 "resamples, rng 42) has lower bound above "
                                 "the block-permutation null band q975",
               "git_commit": git_commit()},
              mde_df,
              f"""# Positive control - minimum detectable effect (MDE) dose curve

Reviewer 1 review 11: the homogeneous positive control was the best
case; extend the dose curve downward to the minimum detectable effect.
Four homogeneous doses (theta/alpha power gains), session-out, cap
seeds 0-4, per-fold outputs saved. Block-permutation null band upper
edge (q975, circular shifts): {null_hi:.3f}.

{ mde_df.to_string(index=False) }

MDE (smallest tested dose whose recovery CI clears the null band):
**{mde_name}**. This bounds what the pipeline could have detected in
the real labels: a homogeneous cross-session theta/alpha effect below
the MDE could exist undetected; at/above it, it would have cleared the
null band.

Leakage checklist:
- [x] Injection is label-dependent BY DESIGN (sensitivity control);
      applied window-wise before splitting, identical for all folds
- [x] Scaling/imputation fit on training folds only
- [x] No tuning on test data (locked C=1.0)

Environment: local VM (CPU), scikit-learn.
""")
    print(f"MDE: {mde_name}", flush=True)

    # ---- heterogeneous injection
    gains, gains_df = hetero_gains(idx_u)
    gains_df.to_csv(TAB / "diagnostics_positive_control_hetero_gains.csv",
                    index=False)
    arr_h = inject_hetero(arr_u, idx_u, gains)
    feats_h = extract_features(arr_h, idx_u)
    # manipulation check: observed frontal log-power shift, fatigued windows
    checks = []
    for band, col_gain in (("theta", "theta_power_gain"),
                           ("alpha", "alpha_power_gain")):
        cols = [f"logpow_{c}_{band}" for c in ["AF7", "AF8"]]
        shift = (feats_h.loc[fat, cols] - base_feats.loc[fat, cols])
        tgt = np.log10(idx_u.loc[fat].apply(
            lambda r: gains[(r.participant, r.session)][0 if band == "theta"
                                                        else 1], axis=1))
        checks.append({"band": band,
                       "mean_target_log10_shift": float(tgt.mean()),
                       "observed_mean_log10_shift_frontal":
                           float(shift.mean().mean())})
    pd.DataFrame(checks).to_csv(
        TAB / "diagnostics_positive_control_hetero_check.csv", index=False)
    het_rows, het_folds, per_session = [], [], {}
    for seed in SEEDS:
        df, fc = capped_feats(idx_u, feats_h, seed)
        per, agg = aggregate_sessionout(df, fc)
        het_rows.append({"seed": seed, **agg})
        for _, r in per.iterrows():
            het_folds.append({"seed": seed, **r.to_dict()})
            per_session.setdefault(r.heldout_session, []).append(
                r.balanced_accuracy)
        print(f"  hetero seed {seed}: BA={agg['balanced_accuracy']:.3f} "
              f"AUC={agg['roc_auc']:.3f} MCC={agg['mcc']:.3f}", flush=True)
    het_df = pd.DataFrame(het_rows)
    het_df.to_csv(TAB / "diagnostics_positive_control_hetero.csv", index=False)
    pd.DataFrame(het_folds).to_csv(
        TAB / "diagnostics_positive_control_hetero_folds.csv", index=False)
    gmap = gains_df.set_index(["participant", "session"])
    xs, ys = [], []
    for sess_key, vals in per_session.items():
        part, sess = sess_key.split("/")
        xs.append(np.log(gmap.loc[(part, sess), "theta_power_gain"]))
        ys.append(np.mean(vals))
    rho, p_rho = stats.spearmanr(xs, ys)
    kinds = gains_df.session_kind.value_counts().to_dict()
    write_run(f"{DATE}_baseline_positivecontrol-hetero_subjective_s0",
              {"model": "logistic_regression", "C": 1.0,
               "class_weight": "balanced",
               "features": "bandpower_43_amendment1 (re-extracted after injection)",
               "injection": "HETEROGENEOUS per-session gains (seed "
                            f"{HETERO_SEED}): theta gain ~ LogNormal(ln1.5, "
                            "0.55), alpha ~ LogNormal(ln1.25, 0.45); uniform "
                            "draw per session: <0.10 null (gain 1), "
                            "0.10-0.20 opposite (gains inverted); fatigued "
                            "windows only, valid channels only",
               "protocol": "session-out", "labels": "subjective (Amendment 1)",
               "cap_seeds": "0-4 (folder named s0; all seeds in metrics.csv)",
               "git_commit": git_commit()},
              het_df,
              f"""# Positive control - HETEROGENEOUS per-session injection

Reviewer 1 review 11: the homogeneous injection is the best case -
a real fatigue effect would vary in size (and possibly sign) across
sessions. This run injects per-session random-amplitude theta/alpha
effects centred on the homogeneous half-strength dose (theta x1.5,
alpha x1.25), with ~20% of sessions null or opposite-sign BY DESIGN
(seed {HETERO_SEED}; exact scheme in src/evaluation/closure_pack.py
hetero_gains(); realised per-session gains in
results/tables/diagnostics_positive_control_hetero_gains.csv;
realised session kinds: {kinds}). Manipulation check in
results/tables/diagnostics_positive_control_hetero_check.csv.

Recovered (session-out, cap seeds 0-4): cross-seed mean BA
{het_df.balanced_accuracy.mean():.3f}, AUC {het_df.roc_auc.mean():.3f},
MCC {het_df.mcc.mean():.3f} (uninjected: 0.515 / 0.549 / 0.045;
homogeneous half-strength: 0.695 / 0.768 / 0.360). Per-session recovery
vs injected log theta gain: Spearman rho = {rho:.3f} (p = {p_rho:.3f},
n = {len(xs)} sessions) - descriptive.

Leakage checklist:
- [x] Injection is label-dependent BY DESIGN (sensitivity control);
      gains drawn per session from a fixed seed, before splitting
- [x] Scaling/imputation fit on training folds only
- [x] No tuning on test data (locked C=1.0)

Environment: local VM (CPU), scikit-learn.
""")
    print(f"hetero: BA={het_df.balanced_accuracy.mean():.3f} "
          f"rho(gain, BA)={rho:.3f}", flush=True)


# ------------------------------------------------------------------ task 4
def gap_ci(idx_u, feats_u):
    from src.evaluation.diagnostics import load_capped
    df0, feats0 = load_capped(0)
    y = df0.label.values
    # leaky split, identical to panel_additions.leakage_gap, preds kept
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    preds = np.full(len(y), -1); proba = np.zeros(len(y))
    for tr, te in skf.split(feats0.values, y):
        m = make_model()
        m.fit(feats0.values[tr], y[tr])
        preds[te] = m.predict(feats0.values[te])
        proba[te] = m.predict_proba(feats0.values[te])[:, 1]
    oof = binary_metrics(y, preds, proba)
    lk = df0[["participant", "session"]].copy()
    lk["y_true"], lk["y_pred"], lk["y_proba"] = y, preds, proba
    leaky_s = {}
    for (part, sess), g in lk.groupby(["participant", "session"]):
        if len(g.y_true.unique()) == 2:
            leaky_s[f"{part}/{sess}"] = binary_metrics(
                g.y_true, g.y_pred, g.y_proba)["balanced_accuracy"]
    # honest seed-0 per-fold BA from the original session-out run
    hp = pd.read_csv(RUNS / f"{DATE}_baseline_session-out_subjective_s0"
                     / "predictions.csv")
    honest_s = {}
    for fid, g in hp.groupby("fold"):
        honest_s[fid] = binary_metrics(g.y_true, g.y_pred,
                                       g.y_proba)["balanced_accuracy"]
    paired = sorted(set(leaky_s) & set(honest_s))
    rows = [{"heldout_session": s, "ba_leaky_oof": leaky_s[s],
             "ba_honest_fold": honest_s[s],
             "gap": leaky_s[s] - honest_s[s]} for s in paired]
    pair_df = pd.DataFrame(rows)
    pair_df.to_csv(TAB / "diagnostics_leakage_gap_paired_sessions.csv",
                   index=False)
    pt, lo, hi, n = session_boot_ci(pair_df.gap.values)
    agg_gap = oof["balanced_accuracy"] - float(np.mean(list(honest_s.values())))
    summ = pd.DataFrame([
        {"quantity": "leaky_OOF_pooled_BA", "value": oof["balanced_accuracy"]},
        {"quantity": "honest_session-out_mean_of_folds_BA_seed0",
         "value": float(np.mean(list(honest_s.values())))},
        {"quantity": "headline_gap_aggregate", "value": agg_gap},
        {"quantity": "paired_mean_gap_over_sessions", "value": pt},
        {"quantity": "paired_gap_ci95_low", "value": lo},
        {"quantity": "paired_gap_ci95_high", "value": hi},
        {"quantity": "n_paired_sessions", "value": n}])
    summ.to_csv(TAB / "diagnostics_leakage_gap_ci.csv", index=False)
    write_run(f"{DATE}_baseline_leakygap-ci_subjective_s0",
              {"model": "logistic_regression", "C": 1.0,
               "class_weight": "balanced",
               "features": "bandpower_43_amendment1",
               "protocol": "leakage-gap CI: paired per-session bootstrap",
               "leaky": "StratifiedKFold(5, shuffle, random_state=0), "
                        "identical to panel_additions leakage-gap demo",
               "honest": "session-out cap seed 0 predictions (original run)",
               "bootstrap": "10000 resamples over paired held-out "
                            "sessions, rng 42",
               "labels": "subjective (Amendment 1)", "cap_seed": 0,
               "git_commit": git_commit()},
              pair_df,
              f"""# Leakage-gap confidence interval - cap seed 0

Reviewer 1 review 11: report the leakage gap WITH a confidence
interval and drop the literature-contrast framing. The gap is paired
by session: for each held-out session, leaky BA = balanced accuracy of
that session's windows under the random window-level split
(out-of-fold predictions, sessions with both classes only); honest BA
= the same session as a session-out test fold. Bootstrap over the
{n} paired sessions (10,000 resamples):

- headline aggregate gap (pooled leaky OOF BA minus honest
  mean-of-folds BA): {agg_gap:+.3f}
- paired mean gap: {pt:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}]

Per-session pairs in metrics.csv and
results/tables/diagnostics_leakage_gap_paired_sessions.csv.

Leakage checklist:
- [x] Leaky variant differs from the honest pipeline ONLY in the split
- [x] Scaling/imputation fit on training folds only in both variants
- [x] No tuning on test data (locked C=1.0)

Environment: local VM (CPU), scikit-learn.
""")
    print(f"gap: aggregate {agg_gap:+.3f}; paired {pt:+.3f} "
          f"[{lo:+.3f}, {hi:+.3f}] over {n} sessions", flush=True)


# ------------------------------------------------------------------ task 5
def without_p08():
    rows = []
    # session-out (P features), from the original per-seed run folders
    for seed in SEEDS:
        m = pd.read_csv(RUNS / f"{DATE}_baseline_session-out_subjective_s{seed}"
                        / "metrics.csv")
        m = m[m.heldout_session != "AGGREGATE_mean_over_groups"]
        for label, sub in (("with_P08", m),
                           ("without_P08", m[~m.heldout_session.str
                                             .startswith("P08")])):
            rows.append({"protocol": "session-out", "variant": "P_masked",
                         "subset": label, "seed": seed,
                         "n_units": len(sub),
                         "balanced_accuracy": sub.balanced_accuracy.mean(),
                         "roc_auc": sub.roc_auc.mean(),
                         "mcc": sub.mcc.mean()})
    # LOPO (P and F), from the closure-pack run folders
    for variant, suffix in (("P_masked", ""), ("F_frontal", "_frontal")):
        for seed in SEEDS:
            m = pd.read_csv(RUNS / f"{DATE}_baseline_loso_subjective{suffix}"
                            f"_s{seed}" / "metrics.csv")
            m = m[m.participant != "AGGREGATE_mean_over_groups"]
            for label, sub in (("with_P08", m),
                               ("without_P08", m[m.participant != "P08"])):
                rows.append({"protocol": "loso", "variant": variant,
                             "subset": label, "seed": seed,
                             "n_units": len(sub),
                             "balanced_accuracy":
                                 sub.balanced_accuracy.mean(),
                             "roc_auc": sub.roc_auc.mean(),
                             "mcc": sub.mcc.mean()})
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "closurepack_without_p08_folds.csv", index=False)
    summ = df.groupby(["protocol", "variant", "subset"]).agg(
        n_units=("n_units", "first"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_sd_across_seeds=("balanced_accuracy", "std"),
        roc_auc_mean=("roc_auc", "mean"), mcc_mean=("mcc", "mean")
        ).reset_index()
    # participant-bootstrap CI for LOPO without P08 (cross-seed means)
    ci_rows = []
    for variant, summ_file in (
            ("P_masked", "baseline_loso_subjective_a1_summary.csv"),
            ("F_frontal", "baseline_loso_subjective_frontal_a1_summary.csv")):
        s = pd.read_csv(TAB / summ_file)
        for label, sub in (("with_P08", s),
                           ("without_P08", s[s.participant != "P08"])):
            pt, lo, hi, n = participant_boot_ci(sub.balanced_accuracy_mean
                                                .values)
            ci_rows.append({"protocol": "loso", "variant": variant,
                            "subset": label, "n_units": n,
                            "balanced_accuracy_mean": pt,
                            "balanced_accuracy_sd_across_seeds": np.nan,
                            "roc_auc_mean": np.nan, "mcc_mean": np.nan,
                            "ba_ci95_low": lo, "ba_ci95_high": hi})
    summ = pd.concat([summ, pd.DataFrame(ci_rows)], ignore_index=True)
    summ.to_csv(TAB / "closurepack_without_p08.csv", index=False)
    write_run(f"{DATE}_baseline_withoutp08_subjective_s0",
              {"model": "none (re-aggregation of existing run outputs)",
               "sources": "experiments/runs/2026-11-20_baseline_session-out_"
                          "subjective_s0..s4/metrics.csv; ..._loso_subjective"
                          "(_frontal)_s0..s4/metrics.csv",
               "method": "per-seed mean over units, then cross-seed mean; "
                         "LOPO CIs by participant bootstrap over cross-seed "
                         "per-participant means (10000, rng 42)",
               "git_commit": git_commit()},
              summ,
              """# With / without P08 - sensitivity of the headline aggregates

Reviewer 1 review 11 / Week 2 closure: P08 is the participant whose
within-subject inversion (BA 0.225) was traced to a mask-carried
domain-shift artefact (docs/week2_diagnostics.md section 5). These
tables recompute the headline session-out and LOPO aggregates with
P08's folds excluded, from the existing run outputs - no refitting.
P08 stays IN the headline numbers; this is a sensitivity report, not
a replacement.

Environment: local VM (CPU).
""")
    print(summ.to_string(index=False), flush=True)


# ------------------------------------------------------------------ task 6
def _perm_stats(df0, feats0, labels_by_perm):
    """labels_by_perm: list of permuted label arrays. Returns per-perm
    session-mean BA and participant-mean BA."""
    folds = sessionout_folds(df0)
    sess_of_fold = {fid: fid for fid, _, _ in folds}
    out = []
    for yp in labels_by_perm:
        per_fold = []
        for fid, tr, te in folds:
            ytr, yte = yp[tr], yp[te]
            if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
                continue
            m = make_model()
            m.fit(feats0.values[tr], ytr)
            fm = binary_metrics(yte, m.predict(feats0.values[te]),
                                m.predict_proba(feats0.values[te])[:, 1])
            per_fold.append({"fold": fid,
                             "participant": fid.split("/")[0],
                             "balanced_accuracy": fm["balanced_accuracy"]})
        pf = pd.DataFrame(per_fold)
        out.append({"ba_session_mean": pf.balanced_accuracy.mean(),
                    "ba_participant_mean":
                        pf.groupby("participant").balanced_accuracy
                          .mean().mean(),
                    "n_folds": len(pf)})
    return pd.DataFrame(out)


def gate():
    from src.evaluation.diagnostics import load_capped
    df0, feats0 = load_capped(0)
    y = df0.label.values.copy()
    sess_key = (df0.participant + "/" + df0.session).to_numpy()

    # variant A: circular shifts (replicates panel_additions exactly)
    perms_a = []
    for p in range(200):
        rng = np.random.default_rng(20_000 + p)
        yp = y.copy()
        for s in np.unique(sess_key):
            msk = np.where(sess_key == s)[0]
            yp[msk] = np.roll(yp[msk], rng.integers(1, len(msk)))
        perms_a.append(yp)
    stats_a = _perm_stats(df0, feats0, perms_a)
    stats_a["variant"] = "circular_shift"
    print(f"gate circular: null SD session-mean "
          f"{stats_a.ba_session_mean.std():.4f}, participant-mean "
          f"{stats_a.ba_participant_mean.std():.4f}", flush=True)

    # variant B: block permutation, block length 8 windows
    perms_b = []
    for p in range(100):
        rng = np.random.default_rng(30_000 + p)
        yp = y.copy()
        for s in np.unique(sess_key):
            msk = np.where(sess_key == s)[0]
            labs = yp[msk]
            blocks = [labs[i:i + 8] for i in range(0, len(labs), 8)]
            order = rng.permutation(len(blocks))
            yp[msk] = np.concatenate([blocks[i] for i in order])
        perms_b.append(yp)
    stats_b = _perm_stats(df0, feats0, perms_b)
    stats_b["variant"] = "block_len8_shuffle"
    print(f"gate block8: null SD session-mean "
          f"{stats_b.ba_session_mean.std():.4f}, participant-mean "
          f"{stats_b.ba_participant_mean.std():.4f}", flush=True)

    nulls = pd.concat([stats_a, stats_b], ignore_index=True)
    nulls.to_csv(TAB / "diagnostics_gate_permutation_nulls.csv", index=False)

    z = stats.norm.ppf(0.95) + stats.norm.ppf(0.90)  # 2.9265
    so = pd.read_csv(TAB / "baseline_session-out_subjective_a1_summary.csv")
    sd_sess = float(so.balanced_accuracy_mean.std(ddof=1))
    lopo_s = pd.read_csv(TAB / "baseline_loso_subjective_a1_summary.csv")
    sd_part = float(lopo_s.balanced_accuracy_mean.std(ddof=1))
    bases = [
        ("perm_null_circular_session_mean_statistic",
         float(stats_a.ba_session_mean.std(ddof=1)),
         "SD of the session-mean BA statistic under 200 circular-shift "
         "permutations (cap seed 0)"),
        ("perm_null_block8_session_mean_statistic",
         float(stats_b.ba_session_mean.std(ddof=1)),
         "SD of the session-mean BA statistic under 100 block-length-8 "
         "permutations (cap seed 0)"),
        ("perm_null_circular_participant_mean_statistic",
         float(stats_a.ba_participant_mean.std(ddof=1)),
         "SD of the participant-mean BA statistic (mean over participants "
         "of per-participant fold means) under the same 200 permutations"),
        ("perm_null_block8_participant_mean_statistic",
         float(stats_b.ba_participant_mean.std(ddof=1)),
         "SD of the participant-mean BA statistic under the 100 "
         "block-length-8 permutations"),
        ("analytical_session_N24",
         sd_sess / np.sqrt(24),
         f"per-session BA SD {sd_sess:.4f} (20 held-out sessions, "
         "cross-seed means) / sqrt(24)"),
        ("analytical_session_N20_observed",
         sd_sess / np.sqrt(20),
         f"per-session BA SD {sd_sess:.4f} / sqrt(20 observed folds)"),
        ("analytical_participant_N12",
         sd_part / np.sqrt(12),
         f"per-participant LOPO BA SD {sd_part:.4f} (11 participants "
         "observed) / sqrt(12)"),
    ]
    inputs = pd.DataFrame(
        [{"basis": b, "se": se, "delta_min_90pct_power": z * se,
          "definition": d} for b, se, d in bases])
    inputs.to_csv(TAB / "diagnostics_gate_inputs.csv", index=False)

    cand = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15]
    tbl = {"candidate_delta": cand}
    for b, se, _ in bases:
        tbl[b] = ["testable" if d >= z * se else "unattainable"
                  for d in cand]
    tbl_df = pd.DataFrame(tbl)
    tbl_df.to_csv(TAB / "diagnostics_gate_delta_table.csv", index=False)
    write_run(f"{DATE}_baseline_gate-inputs_subjective_s0",
              {"model": "logistic_regression (permutation reruns)",
               "permutations": "circular shifts x200 (rng 20000+p, "
                               "replicates panel_additions) + block-"
                               "length-8 shuffles x100 (rng 30000+p); "
                               "per-fold BA saved for both statistics",
               "power_formula": "delta_min = (z_0.95 + z_0.90) * SE = "
                                "2.9265 * SE for a one-sided alpha=0.05 "
                                "equivalence test at true effect 0, "
                                "~90% power (lock doc section 2)",
               "labels": "subjective (Amendment 1)", "cap_seed": 0,
               "git_commit": git_commit()},
              inputs,
              f"""# Feasibility-gate INPUTS - NO VERDICT DECLARED

docs/decision_table_lock.md section 2: the equivalence bound Delta is
my to confirm (candidate value 0.05). This run supplies ONLY
the gate's inputs - standard errors of the headline statistics from
label-free sources - and the conditional table: IF a given Delta were
locked, WOULD equivalence be testable at ~90% power (N as stated)?
No equivalence verdict is declared here, for any Delta.

Gate inputs (SE basis -> minimum Delta for ~90% power):
{ inputs.to_string(index=False) }

Conditional table (rows: candidate Delta; columns: SE basis):
{ tbl_df.to_string(index=False) }

Robustness: the permutation null was computed under two block
structures (whole-sequence circular shifts, n=200; block-length-8
shuffles, n=100). Per-permutation statistics in
results/tables/diagnostics_gate_permutation_nulls.csv.

Environment: local VM (CPU), scikit-learn.
""")
    print(inputs.to_string(index=False), flush=True)
    print(tbl_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    arr_u, idx_u = load_uncapped()
    print("extracting features (uncapped) ...", flush=True)
    feats_u = extract_features(arr_u, idx_u)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("frontal", "all"):
        print("== task 1: frontal-only ==", flush=True)
        frontal(feats_u, idx_u)
    if cmd in ("lopo", "all"):
        print("== task 2: LOPO ==", flush=True)
        lopo(feats_u, idx_u)
    if cmd in ("poscontrol", "all"):
        print("== task 3: positive control hetero + MDE ==", flush=True)
        poscontrol(arr_u, idx_u)
    if cmd in ("gap_p08", "all"):
        print("== task 4: leakage-gap CI ==", flush=True)
        gap_ci(idx_u, feats_u)
        print("== task 5: with/without P08 ==", flush=True)
        without_p08()
    if cmd in ("gate", "all"):
        print("== task 6: feasibility-gate inputs ==", flush=True)
        gate()
    print("DONE", flush=True)
