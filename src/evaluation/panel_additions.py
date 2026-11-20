"""Panel additions to the Week 2 diagnostics (Reviewer 1 review 7,
synthesis round - docs/methods_notes.md). Same discipline as
src/evaluation/diagnostics.py: Amendment-1 features, logistic regression
(C=1.0, class_weight='balanced'), imputation/scaling fit on training
folds only. Each item is logged as its own run folder.

  1. mask-only baseline     - labels predicted from validity indicators +
                              coverage counts ONLY (no EEG), session-out,
                              cap seeds 0-4. The shortcut ceiling.
  2. positive control       - synthetic graded theta/alpha power increase
                              injected into fatigued-class windows only
                              (spectral gain on the windows' own FFT,
                              valid channels only); standard pipeline,
                              session-out, cap seeds 0-4. The pipeline
                              MUST recover the injected effect.
  3. leakage-gap demo       - identical features/model, cap seed 0, but a
                              stratified RANDOM window-level split (5-fold,
                              windows pooled across all sessions and
                              participants) vs the honest session-out.
  4. block permutation      - circular label shifts within session (label
                              block structure preserved), 200 permutations,
                              session-out, cap seed 0; reported alongside
                              the iid null from diagnostics.py.
  5. equivalence            - participant/session-level bootstrap 95% CIs
                              for the within and session-out headline
                              balanced accuracy (and MCC), tested against
                              the pre-specified SESOI of 0.58.
  6. coverage by class      - candidate -> A1-retained -> capped counts
                              and coverage % by class (alert/fatigued),
                              overall and per participant.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from numpy.fft import irfft, rfft, rfftfreq
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.diagnostics import git_commit, load_capped, write_run
from src.evaluation.metrics import binary_metrics
from src.features.bandpower import extract_features
from src.models.baseline import cap_pool, make_model, sessionout_folds

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
TAB = ROOT / "results" / "tables"
DATE = "2026-11-20"
FS = 128
SEEDS = [0, 1, 2, 3, 4]


def aggregate_sessionout(df, feats):
    """One cap-seed session-out pass; returns (per-fold DataFrame,
    aggregate dict)."""
    per = []
    for fid, tr, te in sessionout_folds(df):
        ytr, yte = df.label.values[tr], df.label.values[te]
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            continue
        m = make_model()
        m.fit(feats.values[tr], ytr)
        fm = binary_metrics(yte, m.predict(feats.values[te]),
                            m.predict_proba(feats.values[te])[:, 1])
        per.append({"heldout_session": fid, **fm})
    per = pd.DataFrame(per)
    agg = {c: per[c].mean() for c in ["balanced_accuracy", "macro_f1",
                                      "roc_auc", "mcc"]}
    return per, agg


def load_uncapped():
    arr = np.load(PROC / "windows_a1_uncapped.npy")
    idx = pd.read_csv(PROC / "windows_a1_uncapped_index.csv").reset_index(drop=True)
    return arr, idx


def capped_feats(idx_u, feats_u, seed):
    capped = cap_pool(idx_u, seed)
    pos = capped.index.to_numpy()
    return capped.reset_index(drop=True), feats_u.iloc[pos].reset_index(drop=True)


# ------------------------------------------------------------------- (1)
def mask_only(idx_u):
    vt = idx_u[["valid_TP9", "valid_TP10"]].astype(float)
    X = pd.DataFrame({
        "valid_TP9": vt.valid_TP9, "valid_TP10": vt.valid_TP10,
        "both_temporals_masked": ((vt.valid_TP9 == 0) & (vt.valid_TP10 == 0)).astype(float),
        "n_valid_temporals": vt.valid_TP9 + vt.valid_TP10,
        "n_valid_channels": vt.valid_TP9 + vt.valid_TP10
                            + idx_u.valid_AF7.astype(float)
                            + idx_u.valid_AF8.astype(float)})
    rows = []
    for seed in SEEDS:
        df, Xc = capped_feats(idx_u, X, seed)
        _, agg = aggregate_sessionout(df, Xc)
        rows.append({"seed": seed, **agg})
        print(f"  mask-only seed {seed}: BA={agg['balanced_accuracy']:.3f} "
              f"AUC={agg['roc_auc']:.3f} MCC={agg['mcc']:.3f}", flush=True)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- (2)
def inject(arr, idx, theta_gain, alpha_gain):
    """Spectral-gain injection into fatigued windows only, valid channels
    only. theta_gain/alpha_gain are POWER multipliers (amplitude gain =
    sqrt). Phase preserved (the window's own spectrum is scaled)."""
    out = arr.copy()
    freqs = rfftfreq(arr.shape[2], d=1.0 / FS)
    th = (freqs >= 4) & (freqs < 8)
    al = (freqs >= 8) & (freqs < 12)
    chans = ["TP9", "AF7", "AF8", "TP10"]
    for i in np.where(idx.label.values == 1)[0]:
        for ci, c in enumerate(chans):
            if not bool(idx[f"valid_{c}"].iloc[i]):
                continue
            X = rfft(out[i, ci].astype(np.float64))
            X[th] *= np.sqrt(theta_gain)
            X[al] *= np.sqrt(alpha_gain)
            out[i, ci] = irfft(X, n=arr.shape[2]).astype(out.dtype)
    return out


def positive_control(arr_u, idx_u):
    variants = {"full_strength_theta_x2.0_alpha_x1.5": (2.0, 1.5),
                "half_strength_theta_x1.5_alpha_x1.25": (1.5, 1.25)}
    rows, checks = [], []
    base_feats = extract_features(arr_u, idx_u)
    fat = idx_u.label.values == 1
    for name, (g_th, g_al) in variants.items():
        arr_mod = inject(arr_u, idx_u, g_th, g_al)
        feats_mod = extract_features(arr_mod, idx_u)
        # manipulation check: mean band log-power shift in fatigued windows
        for band, gain in (("theta", g_th), ("alpha", g_al)):
            cols = [f"logpow_{c}_{band}" for c in ["AF7", "AF8"]]
            shift = (feats_mod.loc[fat, cols] - base_feats.loc[fat, cols]).mean().mean()
            checks.append({"variant": name, "band": band,
                           "target_power_gain": gain,
                           "target_log10_shift": np.log10(gain),
                           "observed_mean_log10_shift_frontal": shift})
        for seed in SEEDS:
            df, fc = capped_feats(idx_u, feats_mod, seed)
            _, agg = aggregate_sessionout(df, fc)
            rows.append({"variant": name, "seed": seed, **agg})
            print(f"  pos-control {name} seed {seed}: "
                  f"BA={agg['balanced_accuracy']:.3f} AUC={agg['roc_auc']:.3f} "
                  f"MCC={agg['mcc']:.3f}", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(checks)


# ------------------------------------------------------------------- (3)
def leakage_gap(df0, feats0):
    y = df0.label.values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    preds = np.full(len(y), -1); proba = np.zeros(len(y))
    fold_rows = []
    for k, (tr, te) in enumerate(skf.split(feats0.values, y)):
        m = make_model()
        m.fit(feats0.values[tr], y[tr])
        preds[te] = m.predict(feats0.values[te])
        proba[te] = m.predict_proba(feats0.values[te])[:, 1]
        fm = binary_metrics(y[te], preds[te], proba[te])
        fold_rows.append({"fold": k, **fm})
    oof = binary_metrics(y, preds, proba)
    return pd.DataFrame(fold_rows), oof


# ------------------------------------------------------------------- (4)
def block_permutation(df0, feats0, n_perm=200):
    y = df0.label.values.copy()
    sess_key = (df0.participant + "/" + df0.session).to_numpy()
    folds = sessionout_folds(df0)

    def run_aggregate(labels):
        per = []
        for fid, tr, te in folds:
            ytr, yte = labels[tr], labels[te]
            if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
                continue
            m = make_model()
            m.fit(feats0.values[tr], ytr)
            fm = binary_metrics(yte, m.predict(feats0.values[te]),
                                m.predict_proba(feats0.values[te])[:, 1])
            per.append(fm)
        per = pd.DataFrame(per)
        return per.balanced_accuracy.mean(), per.mcc.mean(), per.roc_auc.mean()

    null = []
    for p in range(n_perm):
        rng = np.random.default_rng(20_000 + p)
        yp = y.copy()
        for s in np.unique(sess_key):
            msk = np.where(sess_key == s)[0]
            shift = rng.integers(1, len(msk))
            yp[msk] = np.roll(yp[msk], shift)
        ba, mcc, auc = run_aggregate(yp)
        null.append({"permutation": p, "balanced_accuracy": ba,
                     "mcc": mcc, "roc_auc": auc})
        if (p + 1) % 50 == 0:
            print(f"  block permutation {p + 1}/{n_perm}", flush=True)
    return pd.DataFrame(null)


# ------------------------------------------------------------------- (5)
def equivalence(n_boot=10_000):
    rng = np.random.default_rng(42)
    so = pd.read_csv(TAB / "baseline_session-out_subjective_a1_summary.csv")
    wi = pd.read_csv(TAB / "baseline_within_subjective_a1_summary.csv")

    def boot(vals):
        vals = np.asarray(vals, dtype=float)
        idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
        means = vals[idx].mean(axis=1)
        return float(vals.mean()), float(np.percentile(means, 2.5)), \
            float(np.percentile(means, 97.5)), len(vals)

    rows = []
    for protocol, vals_ba, vals_mcc, unit in [
            ("session-out", so.balanced_accuracy_mean, so.mcc_mean, "session"),
            ("within", wi.balanced_accuracy_mean, wi.mcc_mean, "participant")]:
        pt, lo, hi, n = boot(vals_ba)
        ptm, lom, him, _ = boot(vals_mcc)
        rows.append({"protocol": protocol, "resample_unit": unit, "n_units": n,
                     "balanced_accuracy": pt, "ba_ci95_low": lo,
                     "ba_ci95_high": hi, "sesoi_ba": 0.58,
                     "upper_bound_below_sesoi": bool(hi < 0.58),
                     "mcc": ptm, "mcc_ci95_low": lom, "mcc_ci95_high": him})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- (6)
def coverage():
    cand = pd.read_csv(PROC / "candidates_a1.csv")
    prim = cand[(cand.control_session == False) & (cand.block.isin(["M1", "M3"]))]
    rows = []
    for scope, g in [("ALL", prim)] + [(p, gg) for p, gg in prim.groupby("participant")]:
        for block, cls in (("M1", "alert"), ("M3", "fatigued")):
            gb = g[g.block == block]
            ret = int(gb.retained_a1.sum())
            rows.append({"scope": scope, "class": cls,
                         "candidate_windows": len(gb),
                         "retained_a1_uncapped": ret,
                         "coverage_pct_uncapped": 100 * ret / max(len(gb), 1)})
    cov = pd.DataFrame(rows)
    a1 = pd.read_csv(PROC / "windows_a1_index.csv")
    capped_counts = a1.label.value_counts().to_dict()
    allrow = cov[cov.scope == "ALL"].copy()
    allrow["retained_capped_primary"] = [capped_counts.get(0, 0),
                                         capped_counts.get(1, 0)]
    allrow["coverage_pct_capped_of_candidates"] = (
        allrow.retained_capped_primary / allrow.candidate_windows * 100)
    cov = pd.concat([cov, allrow.assign(scope="ALL_capped")], ignore_index=True)
    return cov


def main():
    commit = git_commit()
    arr_u, idx_u = load_uncapped()

    print("(1) mask-only ...", flush=True)
    mo = mask_only(idx_u)
    mo.to_csv(TAB / "diagnostics_maskonly.csv", index=False)
    mo_agg = mo[["balanced_accuracy", "macro_f1", "roc_auc", "mcc"]].mean()
    write_run(f"{DATE}_baseline_maskonly_subjective_s0",
              {"model": "logistic_regression", "C": 1.0,
               "class_weight": "balanced",
               "features": "MASK ONLY: valid_TP9, valid_TP10, "
                           "both_temporals_masked, n_valid_temporals, "
                           "n_valid_channels (no EEG features)",
               "protocol": "session-out", "labels": "subjective (Amendment 1)",
               "cap_seeds": "0-4 (folder named s0; all seeds in metrics.csv)",
               "git_commit": commit},
              mo,
              f"""# Mask-only baseline - session-out, cap seeds 0-4

Panel addition (Reviewer 1 review 7): how much label information do
the Amendment-1 validity indicators and coverage counts alone carry?
This is the shortcut ceiling for any model given the indicators.

Cross-seed aggregate: balanced accuracy {mo_agg.balanced_accuracy:.3f},
ROC-AUC {mo_agg.roc_auc:.3f}, MCC {mo_agg.mcc:.3f} (full EEG model:
0.515 / 0.549 / 0.045). Frontal validity is constant (1) in retained
windows by Amendment-1 gating, so all information here is temporal
masking pattern + coverage.

Leakage checklist:
- [x] Scaling fit on training folds only (Pipeline, per fold)
- [x] Features are protocol/quality metadata only - no EEG, no labels
- [x] No tuning on test data (locked C=1.0)

Environment: local VM (CPU), scikit-learn.
""")

    print("(2) positive control ...", flush=True)
    pc, pc_check = positive_control(arr_u, idx_u)
    pc.to_csv(TAB / "diagnostics_positive_control.csv", index=False)
    pc_check.to_csv(TAB / "diagnostics_positive_control_check.csv", index=False)
    pc_agg = pc.groupby("variant")[["balanced_accuracy", "roc_auc", "mcc"]].mean()
    write_run(f"{DATE}_baseline_positivecontrol_subjective_s0",
              {"model": "logistic_regression", "C": 1.0,
               "class_weight": "balanced",
               "features": "bandpower_43_amendment1 (re-extracted after injection)",
               "injection": "spectral gain on fatigued windows only, valid "
                            "channels only; full: theta power x2.0, alpha "
                            "power x1.5; half: theta x1.5, alpha x1.25",
               "protocol": "session-out", "labels": "subjective (Amendment 1)",
               "cap_seeds": "0-4 (folder named s0; all seeds in metrics.csv)",
               "git_commit": commit},
              pc,
              f"""# Positive control - injected theta/alpha effect, session-out

Panel addition (Reviewer 1 review 7): can this pipeline detect a
fatigue-like spectral effect when one is DEFINITELY present? Method:
for every fatigued-class window (label 1) in windows_a1_uncapped, each
VALID channel's own FFT was scaled in the theta (4-8 Hz) and alpha
(8-12 Hz) bins by sqrt(power gain) - phase preserved, alert windows
untouched, masked channels untouched (they carry no features). Two
graded strengths: full (theta power x2.0 = +100%, alpha x1.5 = +50%)
and half (theta x1.5, alpha x1.25). Features were then re-extracted
with the standard extractor and the standard session-out LR pipeline
run unchanged. Manipulation check (observed vs target mean log10
frontal band-power shift in fatigued windows) is in
results/tables/diagnostics_positive_control_check.csv.

Recovered (cross-seed means):
{pc_agg.to_string()}

Uninjected reference: BA 0.515, AUC 0.549, MCC 0.045.

Leakage checklist:
- [x] Injection is label-dependent BY DESIGN (it is a sensitivity
      control, not an experiment); it is applied window-wise before
      splitting and identical for all folds
- [x] Scaling/imputation fit on training folds only
- [x] No tuning on test data (locked C=1.0)

Environment: local VM (CPU), scikit-learn.
""")

    print("(3) leakage-gap ...", flush=True)
    df0, feats0 = load_capped(0)
    folds_df, oof = leakage_gap(df0, feats0)
    gap = pd.DataFrame([
        {"protocol": "random_window_split_5fold (LEAKY)", **oof},
        {"protocol": "session-out (honest), cap seed 0",
         "balanced_accuracy": 0.4991, "roc_auc": 0.4929, "mcc": 0.0090,
         "macro_f1": 0.4055, "n": 806}])
    gap.to_csv(TAB / "diagnostics_leakage_gap.csv", index=False)
    folds_df.to_csv(TAB / "diagnostics_leakage_gap_folds.csv", index=False)
    write_run(f"{DATE}_baseline_leakygap_subjective_s0",
              {"model": "logistic_regression", "C": 1.0,
               "class_weight": "balanced",
               "features": "bandpower_43_amendment1",
               "protocol": "random window-level split (diagnostic demo of "
                           "the leaky protocol - NOT a registered protocol)",
               "split": "StratifiedKFold(5, shuffle, random_state=0) over "
                        "all 806 windows pooled across sessions/participants",
               "labels": "subjective (Amendment 1)", "cap_seed": 0,
               "git_commit": commit},
              gap,
              f"""# Leakage-gap demonstration - cap seed 0

Panel addition (Reviewer 1 review 7): identical features, identical
model, identical data - the ONLY changed variable is the split.
Random window-level split (windows pooled across all sessions and
participants, stratified 5-fold): balanced accuracy
{oof['balanced_accuracy']:.3f}, ROC-AUC {oof['roc_auc']:.3f}, MCC
{oof['mcc']:.3f}. Honest session-out on the same seed: 0.499 / 0.493 /
0.009. The gap between these numbers is what the split discipline is
worth; the leaky number is the kind of figure the thesis's RQ1/RQ4
argue is uninterpretable as fatigue detection.

Leakage checklist (for the honest pipeline this demo contrasts with):
- [x] Scaling/imputation fit on training folds only, even in the leaky
      variant - the split is the sole changed variable
- [x] No tuning on test data (locked C=1.0)

Environment: local VM (CPU), scikit-learn.
""")
    print(f"  leaky OOF: BA={oof['balanced_accuracy']:.3f} "
          f"AUC={oof['roc_auc']:.3f} MCC={oof['mcc']:.3f}", flush=True)

    print("(4) block permutation (200) ...", flush=True)
    bnull = block_permutation(df0, feats0)
    bnull.to_csv(TAB / "diagnostics_permutation_block_null.csv", index=False)
    iid = pd.read_csv(TAB / "diagnostics_permutation_summary.csv")
    real = {"balanced_accuracy": 0.4991, "mcc": 0.0090, "roc_auc": 0.4929}
    rows = []
    for stat, col in (("balanced_accuracy", "balanced_accuracy"),
                      ("mcc", "mcc"), ("roc_auc", "roc_auc")):
        v = bnull[col]
        rows.append({"statistic": stat, "null_type": "block_circular_shift",
                     "real_seed0": real[stat], "null_mean": v.mean(),
                     "null_sd": v.std(), "null_q025": v.quantile(0.025),
                     "null_q975": v.quantile(0.975),
                     "p_value_ge_real": (1 + (v >= real[stat]).sum()) / 201})
        r = iid[iid.statistic == stat].iloc[0]
        rows.append({"statistic": stat, "null_type": "iid_within_session",
                     "real_seed0": real[stat], "null_mean": r.null_mean,
                     "null_sd": r.null_sd, "null_q025": r.null_q025,
                     "null_q975": r.null_q975,
                     "p_value_ge_real": r.p_value_ge_real_seed0})
    comb = pd.DataFrame(rows)
    comb.to_csv(TAB / "diagnostics_permutation_combined_summary.csv", index=False)
    bba = bnull.balanced_accuracy
    write_run(f"{DATE}_baseline_permutation-block_subjective_s0",
              {"model": "logistic_regression", "C": 1.0,
               "class_weight": "balanced",
               "features": "bandpower_43_amendment1",
               "protocol": "session-out with BLOCK permutation (circular "
                           "label shifts within session)",
               "labels": "subjective (Amendment 1 windows), block-permuted",
               "cap_seed": 0, "n_permutations": 200,
               "permutation_rng": "default_rng(20000 + permutation_index); "
                                  "per session an independent shift in "
                                  "1..n_session-1 applied to the time-ordered "
                                  "label sequence",
               "git_commit": commit},
              comb,
              f"""# Block-permutation test - session-out, cap seed 0

Correction required by Reviewer 1 review 7: the original permutation
test shuffled labels iid within session, which is anti-conservative
under autocorrelation. This version CIRCULARLY SHIFTS each session's
time-ordered label sequence by a random offset (block structure and
per-session class counts preserved exactly), 200 permutations,
identical folds/pipeline. Real seed-0 aggregate: BA 0.499, MCC 0.009,
AUC 0.493.

Block null: BA {bba.mean():.3f} +/- {bba.std():.3f} (95% range
{bba.quantile(0.025):.3f}-{bba.quantile(0.975):.3f}); iid null was
0.498 +/- 0.028. Full comparison in
results/tables/diagnostics_permutation_combined_summary.csv. The block
null is the one to cite.

Leakage checklist:
- [x] Identical folds/imputation/scaling discipline as the real run
- [x] Permutation restricted to within-session circular shifts
- [x] No tuning on test data (locked C=1.0)

Environment: local VM (CPU), scikit-learn.
""")
    print(f"  block null BA {bba.mean():.3f}+/-{bba.std():.3f}", flush=True)

    print("(5) equivalence ...", flush=True)
    eq = equivalence()
    eq.to_csv(TAB / "diagnostics_equivalence.csv", index=False)
    write_run(f"{DATE}_baseline_equivalence_subjective_s0",
              {"model": "bootstrap CI on existing baseline aggregates",
               "resampling": "10,000 bootstrap resamples over held-out "
                             "sessions (session-out) / participants "
                             "(within), percentile 95% CI, rng seed 42",
               "sesoi": "balanced accuracy 0.58 (pre-specified smallest "
                        "effect of interest, Reviewer 1 review 7)",
               "inputs": "results/tables/baseline_*_subjective_a1_summary.csv",
               "git_commit": commit},
              eq,
              f"""# Equivalence assessment - headline balanced accuracy

Non-significance is not equivalence (Reviewer 1 review 7). Criterion
(pre-specified): the claim "equivalent to chance within the smallest
effect of interest" is supported for a protocol iff the UPPER bound of
the 95% bootstrap CI for balanced accuracy is below the SESOI of 0.58.

{ eq.to_string(index=False) }

Resampling units: held-out sessions (n=20) for session-out, participants
(n=8) for within - the CI therefore reflects across-unit variability of
the cross-seed per-unit means, not window-level noise.

Environment: local VM (CPU).
""")
    print(eq.to_string(index=False), flush=True)

    print("(6) coverage by class ...", flush=True)
    cov = coverage()
    cov.to_csv(TAB / "diagnostics_coverage_by_class.csv", index=False)
    write_run(f"{DATE}_baseline_coverage_subjective_s0",
              {"model": "none (data accounting)",
               "source": "data/processed/candidates_a1.csv (primary, "
                         "non-control sessions; M1=alert, M3=fatigued) and "
                         "windows_a1_index.csv (capped primary set)",
               "git_commit": commit},
              cov,
              """# Coverage by class - Amendment-1 retention accounting

Panel addition (Reviewer 1 review 7): masking/rejection is
post-outcome selection, so coverage must be reported BY CLASS.
Candidates are all windows cut in M1/M3 blocks of the 32 primary
(delta-mental > 0) sessions before Amendment-1 retention; coverage =
retained / candidates. The ALL_capped rows add the final stage: the
canonical cap-seed-0 primary set (windows_a1) after the per-session M1
cap. Per-participant rows are in the metrics.csv / diagnostics CSV.

Environment: local VM (CPU).
""")
    print(cov[cov.scope.isin(["ALL", "ALL_capped"])].to_string(index=False),
          flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
