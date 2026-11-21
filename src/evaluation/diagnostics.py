"""Week 2 diagnostics on the band-power baseline.

Why these exist: a headline score at chance can mean several very
different things (no signal anywhere, signal only inside sessions,
a label construction problem, a pipeline that could not find a real
effect if one were there), and the Week 2 numbers on their own do
not separate those readings. Each diagnostic below is aimed at one
of them, so that whatever I conclude from the baseline rests on
more than the headline itself.

All diagnostics use the Amendment-1 feature set (src/features/bandpower.py,
43 features), logistic regression (C=1.0, class_weight='balanced'), and the
same train-only discipline as src/models/baseline.py (median imputation +
standardisation fit on the training fold only, via sklearn Pipeline).

Analyses (cap seed 0 unless stated):
  1. within-session ceiling check - time-blocked folds INSIDE single
     sessions (both classes, >=20 windows/class): each class's windows are
     split into 3 contiguous time-ordered chunks; fold k tests on chunk k
     of both classes, trains on the remaining chunks with a 2-window purge
     at every test-chunk boundary.
  2. label-permutation test - pooled leave-one-session-out, labels permuted
     within session (per-session class counts preserved), 200 permutations.
  3. trivial baselines - (a) normalised window position within session only,
     (b) delta log-power only (valid channels); session-out, cap seeds 0-4.
  4. per-held-out-session balanced accuracy vs the session's delta-mental
     (Spearman).
  5. P08 deep-dive - confusion matrices, per-class recall, LR coefficient
     signs, predicted-probability distributions (from the within-protocol
     run predictions + refit fold models for coefficients).
  6. feature sanity - NaN fraction / variance / near-constant flags.

Outputs: results/tables/diagnostics_*.csv, one figure in results/figures/,
run folders under experiments/runs/ per experiments/README.md.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.metrics import binary_metrics
from src.features.bandpower import FEATURE_NAMES, extract_features
from src.models.baseline import cap_pool, make_model, sessionout_folds

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
RUNS = ROOT / "experiments" / "runs"
TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"
DATE = "2026-11-20"
AGG = "AGGREGATE_mean_over_groups"


def git_commit():
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()


def write_run(name, cfg, metrics_df, notes, extra_files=None):
    d = RUNS / name
    d.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(d / "metrics.csv", index=False)
    (d / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (d / "notes.md").write_text(notes)
    for fn, content in (extra_files or {}).items():
        (d / fn).write_text(content)
    return d


def load_capped(seed):
    arr = np.load(PROC / "windows_a1_uncapped.npy")
    idx = pd.read_csv(PROC / "windows_a1_uncapped_index.csv").reset_index(drop=True)
    capped = cap_pool(idx, seed)
    pos = capped.index.to_numpy()
    df = capped.reset_index(drop=True)
    feats = extract_features(arr[pos], df)
    return df, feats


def session_position(df):
    """Normalised position (0..1) of each window within its session,
    ordered by start_sample. Computed on the (capped) set itself."""
    pos = np.zeros(len(df))
    for _, g in df.groupby(["participant", "session"]):
        order = np.argsort(g.start_sample.to_numpy())
        ranks = np.empty(len(g)); ranks[order] = np.arange(len(g))
        pos[g.index.to_numpy()] = ranks / max(len(g) - 1, 1)
    return pos


# ---------------------------------------------------------------- ceiling
def ceiling_check(df, feats):
    rows, fold_rows = [], []
    eligible = []
    for (part, sess), g in df.groupby(["participant", "session"]):
        vc = g.label.value_counts()
        if len(vc) == 2 and vc.min() >= 20:
            eligible.append((part, sess))
    for part, sess in eligible:
        g = df[(df.participant == part) & (df.session == sess)]
        g = g.sort_values("start_sample")
        gpos = g.index.to_numpy()          # positions in df/feats
        y = g.label.to_numpy()
        # per-class contiguous chunks (time order preserved within class)
        chunks = {}
        for cls in (0, 1):
            cls_pos = np.where(y == cls)[0]        # positions within g
            chunks[cls] = np.array_split(cls_pos, 3)
        preds = []
        for k in range(3):
            test_local = np.sort(np.concatenate([chunks[0][k], chunks[1][k]]))
            test_set = set(test_local.tolist())
            # purge: any non-test window within 2 positions (in the
            # session's time order) of a test window is dropped from train
            purge = set()
            for p in test_local:
                for q in range(p - 2, p + 3):
                    if 0 <= q < len(g) and q not in test_set:
                        purge.add(q)
            train_local = np.array([i for i in range(len(g))
                                    if i not in test_set and i not in purge])
            tr, te = gpos[train_local], gpos[test_local]
            m = make_model()
            m.fit(feats.values[tr], df.label.values[tr])
            rec = pd.DataFrame({
                "participant": part, "session": sess, "fold": k,
                "y_true": df.label.values[te],
                "y_pred": m.predict(feats.values[te]),
                "y_proba": m.predict_proba(feats.values[te])[:, 1]})
            preds.append(rec)
            fm = binary_metrics(rec.y_true, rec.y_pred, rec.y_proba)
            fold_rows.append({"participant": part, "session": sess, "fold": k,
                              "n_train": len(tr), "n_purged": len(purge), **fm})
        allp = pd.concat(preds)
        m_all = binary_metrics(allp.y_true, allp.y_pred, allp.y_proba)
        rows.append({"participant": part, "session": sess, **m_all})
    per = pd.DataFrame(rows)
    folds = pd.DataFrame(fold_rows)
    agg = {c: per[c].mean() for c in ["balanced_accuracy", "macro_f1", "roc_auc",
                                      "mcc", "recall_alert", "recall_fatigued",
                                      "precision_alert", "precision_fatigued",
                                      "n", "n_alert", "n_fatigued"]}
    out = pd.concat([per, pd.DataFrame([{"participant": AGG, "session": "", **agg}])],
                    ignore_index=True)
    return out, folds, [f"{p}/{s}" for p, s in eligible]


# ------------------------------------------------------------- permutation
def permutation_test(df, feats, n_perm=200):
    folds = sessionout_folds(df)
    y = df.label.values.copy()
    sess_key = (df.participant + "/" + df.session).to_numpy()

    def run_aggregate(labels):
        per = []
        for fid, tr, te in folds:
            ytr, yte = labels[tr], labels[te]
            if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
                continue
            m = make_model()
            m.fit(feats.values[tr], ytr)
            fm = binary_metrics(yte, m.predict(feats.values[te]),
                                m.predict_proba(feats.values[te])[:, 1])
            per.append(fm)
        per = pd.DataFrame(per)
        return (per.balanced_accuracy.mean(), per.mcc.mean(),
                per.roc_auc.mean(), len(per))

    real_ba, real_mcc, real_auc, n_folds = run_aggregate(y)
    null = []
    for p in range(n_perm):
        rng = np.random.default_rng(10_000 + p)
        yp = y.copy()
        for s in np.unique(sess_key):
            msk = sess_key == s
            yp[msk] = rng.permutation(yp[msk])
        ba, mcc, auc, _ = run_aggregate(yp)
        null.append({"permutation": p, "balanced_accuracy": ba,
                     "mcc": mcc, "roc_auc": auc})
        if (p + 1) % 50 == 0:
            print(f"  permutation {p + 1}/{n_perm}", flush=True)
    null = pd.DataFrame(null)
    return real_ba, real_mcc, real_auc, n_folds, null


# ---------------------------------------------------------------- trivial
def trivial_baselines(arr_idx_pairs):
    """arr_idx_pairs: list of (seed, df, feats). Returns per-seed rows."""
    delta_cols = [f"logpow_{c}_delta" for c in ["TP9", "AF7", "AF8", "TP10"]]
    rows = []
    for seed, df, feats in arr_idx_pairs:
        pos = session_position(df).reshape(-1, 1)
        variants = {
            "window_index_only": pd.DataFrame(pos, columns=["win_pos"]),
            "delta_logpower_only": feats[delta_cols],
            "full_43_features": feats,
        }
        for name, X in variants.items():
            per = []
            for fid, tr, te in sessionout_folds(df):
                ytr, yte = df.label.values[tr], df.label.values[te]
                if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
                    continue
                m = make_model()
                m.fit(X.values[tr], ytr)
                fm = binary_metrics(yte, m.predict(X.values[te]),
                                    m.predict_proba(X.values[te])[:, 1])
                per.append(fm)
            per = pd.DataFrame(per)
            rows.append({"seed": seed, "model": name, "n_folds": len(per),
                         "balanced_accuracy": per.balanced_accuracy.mean(),
                         "macro_f1": per.macro_f1.mean(),
                         "roc_auc": per.roc_auc.mean(),
                         "mcc": per.mcc.mean()})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- P08
def p08_deep_dive(df0, feats0):
    """df0/feats0: cap seed 0 (coefficients); predictions pooled from the
    existing within-protocol run folders (all 5 cap seeds)."""
    part = "P08"
    coef_rows = []
    for sess in sorted(df0[df0.participant == part].session.unique()):
        te = ((df0.participant == part) & (df0.session == sess)).values
        tr = ((df0.participant == part) & (df0.session != sess)).values
        if len(np.unique(df0.label.values[tr])) < 2 or \
           len(np.unique(df0.label.values[te])) < 2:
            continue
        m = make_model()
        m.fit(feats0.values[tr], df0.label.values[tr])
        coefs = m.named_steps["lr"].coef_[0]
        for fn, c in zip(FEATURE_NAMES, coefs):
            coef_rows.append({"heldout_session": sess, "feature": fn,
                              "coef": c})
    coef_df = pd.DataFrame(coef_rows)
    coef_df.to_csv(TAB / "diagnostics_p08_lr_coefficients.csv", index=False)
    top = (coef_df.assign(abs_coef=coef_df.coef.abs())
           .groupby("feature")
           .agg(mean_coef=("coef", "mean"), mean_abs_coef=("abs_coef", "mean"),
                sign_consistency=("coef", lambda s: max((s > 0).mean(),
                                                        (s < 0).mean())))
           .sort_values("mean_abs_coef", ascending=False).head(10)
           .reset_index())
    top.to_csv(TAB / "diagnostics_p08_top_features.csv", index=False)

    preds = []
    for s in range(5):
        p = pd.read_csv(RUNS / f"{DATE}_baseline_within_subjective_s{s}"
                        / "predictions.csv")
        p = p[p.participant == part].copy(); p["cap_seed"] = s
        preds.append(p)
    preds = pd.concat(preds)
    cm_rows = []
    for sess, g in preds.groupby("session"):
        for yt in (0, 1):
            for yp in (0, 1):
                cm_rows.append({"heldout_session": sess, "y_true": yt,
                                "y_pred": yp,
                                "count_all_seeds": int(((g.y_true == yt) &
                                                         (g.y_pred == yp)).sum())})
    pd.DataFrame(cm_rows).to_csv(TAB / "diagnostics_p08_confusion.csv",
                                  index=False)
    proba_rows = []
    for (sess, yt), g in preds.groupby(["session", "y_true"]):
        proba_rows.append({
            "heldout_session": sess, "y_true": yt, "n": len(g),
            "proba_mean": g.y_proba.mean(), "proba_sd": g.y_proba.std(),
            "proba_q25": g.y_proba.quantile(0.25),
            "proba_median": g.y_proba.median(),
            "proba_q75": g.y_proba.quantile(0.75),
            "frac_proba_0.4_0.6": float(((g.y_proba >= 0.4) &
                                         (g.y_proba <= 0.6)).mean()),
            "frac_pred_fatigued": float((g.y_pred == 1).mean())})
    for yt, g in preds.groupby("y_true"):
        proba_rows.append({
            "heldout_session": "ALL", "y_true": yt, "n": len(g),
            "proba_mean": g.y_proba.mean(), "proba_sd": g.y_proba.std(),
            "proba_q25": g.y_proba.quantile(0.25),
            "proba_median": g.y_proba.median(),
            "proba_q75": g.y_proba.quantile(0.75),
            "frac_proba_0.4_0.6": float(((g.y_proba >= 0.4) &
                                         (g.y_proba <= 0.6)).mean()),
            "frac_pred_fatigued": float((g.y_pred == 1).mean())})
    pd.DataFrame(proba_rows).to_csv(TAB / "diagnostics_p08_proba_dist.csv",
                                    index=False)
    counts = (df0[df0.participant == part]
              .groupby(["session", "label"]).size().unstack(fill_value=0)
              .rename(columns={0: "n_alert_capped_s0", 1: "n_fatigued_capped_s0"})
              .reset_index())
    idx_u = pd.read_csv(PROC / "windows_a1_uncapped_index.csv")
    counts_u = (idx_u[idx_u.participant == part]
                .groupby(["session", "label"]).size().unstack(fill_value=0)
                .rename(columns={0: "n_alert_uncapped", 1: "n_fatigued_uncapped"})
                .reset_index())
    counts = counts.merge(counts_u, on="session", how="outer").fillna(0)
    counts.insert(0, "participant", part)
    counts.to_csv(TAB / "diagnostics_p08_window_counts.csv", index=False)
    return coef_df, top, preds


def main():
    commit = git_commit()
    print("loading cap seed 0 ...", flush=True)
    df0, feats0 = load_capped(0)

    # ---- 6. feature sanity (cheap; do first)
    sanity = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "nan_frac": feats0.isna().mean().values,
        "mean": feats0.mean().values, "sd": feats0.std().values,
        "n_unique": [feats0[c].nunique() for c in feats0.columns]})
    sanity["near_constant"] = (sanity.sd < 1e-6) | (sanity.n_unique <= 2)
    sanity.to_csv(TAB / "diagnostics_feature_sanity.csv", index=False)
    print(f"feature sanity: {(sanity.nan_frac > 0).sum()} features with NaNs, "
          f"{sanity.near_constant.sum()} near-constant", flush=True)

    # ---- 1. within-session ceiling check
    print("ceiling check ...", flush=True)
    ceil, ceil_folds, eligible = ceiling_check(df0, feats0)
    ceil.to_csv(TAB / "diagnostics_within_session_ceiling.csv", index=False)
    ceil_folds.to_csv(TAB / "diagnostics_within_session_ceiling_folds.csv",
                      index=False)
    agg = ceil[ceil.participant == AGG].iloc[0]
    write_run(f"{DATE}_baseline_withinsession_subjective_s0",
              {"model": "logistic_regression", "C": 1.0,
               "class_weight": "balanced",
               "features": "bandpower_43_amendment1",
               "protocol": "within-session time-blocked diagnostic (NOT a "
                           "registered protocol - Reviewer 2 review 6)",
               "labels": "subjective (Amendment 1 windows)", "cap_seed": 0,
               "split": "per class 3 contiguous time chunks; fold k tests "
                        "chunk k of both classes; 2-window purge at chunk "
                        "boundaries",
               "eligibility": "both classes and >=20 windows/class (capped)",
               "git_commit": commit},
              ceil,
              f"""# Baseline within-session ceiling check - subjective labels, cap seed 0

Diagnostic requested by Reviewer 2 review 6 (docs/methods_notes.md):
does ANY early-vs-late signal exist inside a single session? Sessions
eligible (both classes, >=20 windows/class after cap): {eligible}.

Split: within each session, each class's windows (time-ordered) are cut
into 3 contiguous chunks; fold k tests on chunk k of BOTH classes and
trains on the remaining chunks of the same session, dropping a 2-window
purge on each side of every test chunk. Metrics per session are computed
on the concatenated out-of-fold predictions.

Aggregate (mean over sessions): balanced accuracy {agg.balanced_accuracy:.3f},
ROC-AUC {agg.roc_auc:.3f}, MCC {agg.mcc:.3f}.

Leakage checklist:
- [x] Imputation/scaling fit on the training segments only (Pipeline, per fold)
- [x] Class weights from training segments only
- [x] No window in both train and test; purge buffer at segment boundaries
- [x] No tuning on test data (locked C=1.0)
- [x] Label rules locked before any training

Environment: local VM (CPU), scikit-learn. NOTE: this is a diagnostic,
not one of the PLAN section 6 protocols; within a session the label is
perfectly confounded with block/time by construction (label_rules.md
section 1), so a high score here shows session-specific separability,
not fatigue physiology.
""")
    print(f"ceiling: sessions={eligible} agg BA={agg.balanced_accuracy:.3f} "
          f"AUC={agg.roc_auc:.3f} MCC={agg.mcc:.3f}", flush=True)

    # ---- 2. permutation test
    print("permutation test (200 perms) ...", flush=True)
    real_ba, real_mcc, real_auc, n_folds, null = permutation_test(df0, feats0)
    null.to_csv(TAB / "diagnostics_permutation_null.csv", index=False)
    def pctile(v, real):
        return float((v < real).mean())
    def pval(v, real):
        return float((1 + (v >= real).sum()) / (len(v) + 1))
    perm_summary = pd.DataFrame([{
        "statistic": "balanced_accuracy", "real_seed0": real_ba,
        "real_cross_seed": 0.5151, "null_mean": null.balanced_accuracy.mean(),
        "null_sd": null.balanced_accuracy.std(),
        "null_q025": null.balanced_accuracy.quantile(0.025),
        "null_q975": null.balanced_accuracy.quantile(0.975),
        "percentile_of_real_seed0": pctile(null.balanced_accuracy, real_ba),
        "percentile_of_real_cross_seed": pctile(null.balanced_accuracy, 0.5151),
        "p_value_ge_real_seed0": pval(null.balanced_accuracy, real_ba)},
        {"statistic": "mcc", "real_seed0": real_mcc,
         "real_cross_seed": 0.0446, "null_mean": null.mcc.mean(),
         "null_sd": null.mcc.std(), "null_q025": null.mcc.quantile(0.025),
         "null_q975": null.mcc.quantile(0.975),
         "percentile_of_real_seed0": pctile(null.mcc, real_mcc),
         "percentile_of_real_cross_seed": pctile(null.mcc, 0.0446),
         "p_value_ge_real_seed0": pval(null.mcc, real_mcc)},
        {"statistic": "roc_auc", "real_seed0": real_auc,
         "real_cross_seed": 0.5485, "null_mean": null.roc_auc.mean(),
         "null_sd": null.roc_auc.std(), "null_q025": null.roc_auc.quantile(0.025),
         "null_q975": null.roc_auc.quantile(0.975),
         "percentile_of_real_seed0": pctile(null.roc_auc, real_auc),
         "percentile_of_real_cross_seed": pctile(null.roc_auc, 0.5485),
         "p_value_ge_real_seed0": pval(null.roc_auc, real_auc)}])
    perm_summary.to_csv(TAB / "diagnostics_permutation_summary.csv", index=False)
    write_run(f"{DATE}_baseline_permutation_subjective_s0",
              {"model": "logistic_regression", "C": 1.0,
               "class_weight": "balanced",
               "features": "bandpower_43_amendment1",
               "protocol": "session-out with within-session label permutation",
               "labels": "subjective (Amendment 1 windows), permuted",
               "cap_seed": 0, "n_permutations": 200,
               "permutation_rng": "default_rng(10000 + permutation_index)",
               "aggregate": "mean over valid held-out sessions (same as the "
                            "baseline AGGREGATE row)",
               "git_commit": commit},
              perm_summary,
              f"""# Baseline label-permutation test - session-out, cap seed 0

Labels permuted WITHIN each session (per-session class counts preserved),
200 permutations, identical pipeline and folds to the real session-out run
({n_folds} valid folds). Real seed-0 aggregate: balanced accuracy
{real_ba:.3f}, MCC {real_mcc:.3f}, ROC-AUC {real_auc:.3f}. (The headline
0.515 / 0.045 figures are the cross-seed means over cap seeds 0-4; their
percentiles in this seed-0 null are also reported.)

Null (mean +/- sd): BA {null.balanced_accuracy.mean():.3f} +/- {null.balanced_accuracy.std():.3f},
MCC {null.mcc.mean():.3f} +/- {null.mcc.std():.3f}. Permutation p-values
((1 + #null >= real) / 201): BA {pval(null.balanced_accuracy, real_ba):.3f},
MCC {pval(null.mcc, real_mcc):.3f}.

Leakage checklist:
- [x] Identical folds/imputation/scaling discipline as the real run
- [x] Permutation restricted to within-session (class counts preserved)
- [x] No tuning on test data (locked C=1.0)

Environment: local VM (CPU), scikit-learn.
""")
    print(f"permutation: real BA={real_ba:.3f} MCC={real_mcc:.3f}; null BA "
          f"{null.balanced_accuracy.mean():.3f}+/-{null.balanced_accuracy.std():.3f}, "
          f"p(BA)={pval(null.balanced_accuracy, real_ba):.3f} "
          f"p(MCC)={pval(null.mcc, real_mcc):.3f}", flush=True)

    # ---- 3. trivial baselines (seeds 0-4)
    print("trivial baselines ...", flush=True)
    pairs = [(0, df0, feats0)] + [(s, *load_capped(s)) for s in range(1, 5)]
    triv = trivial_baselines(pairs)
    triv.to_csv(TAB / "diagnostics_trivial_baselines.csv", index=False)
    triv_summ = (triv.groupby("model")[["balanced_accuracy", "macro_f1",
                                        "roc_auc", "mcc"]]
                 .agg(["mean", "std"]))
    write_run(f"{DATE}_baseline_trivial_subjective_s0",
              {"model": "logistic_regression", "C": 1.0,
               "class_weight": "balanced",
               "features": "variants: window_index_only / delta_logpower_only "
                           "/ full_43_features",
               "protocol": "session-out", "labels": "subjective (Amendment 1)",
               "cap_seeds": "0-4 (folder named s0; all seeds in metrics.csv)",
               "git_commit": commit},
              triv,
              """# Trivial baselines - session-out, cap seeds 0-4

Same folds and pipeline as the main baseline. Variants: (a) a single
feature = the window's normalised position within its session (0..1 by
start_sample among the session's capped windows) - a pure time-on-task
probe; (b) delta-band log-power only (4 channels, valid channels, NaNs
median-imputed on the training fold); (c) the full 43-feature model for
reference. Aggregate per seed = mean over valid held-out sessions.

Leakage checklist:
- [x] Scaling/imputation fit on training folds only
- [x] Window position uses only protocol metadata (window order), no labels
- [x] No tuning on test data (locked C=1.0)

Environment: local VM (CPU), scikit-learn.
""")
    print(triv.groupby("model").balanced_accuracy.agg(["mean", "std"]), flush=True)

    # ---- 4. delta-mental relation
    print("delta-mental relation ...", flush=True)
    insp = pd.read_csv(ROOT / "results" / "tables" / "inspection_summary.csv",
                       dtype={"participant": str, "session": str})
    insp["heldout_session"] = ("P" + insp.participant + "/S" + insp.session)
    summ = pd.read_csv(TAB / "baseline_session-out_subjective_a1_summary.csv")
    s0 = pd.read_csv(RUNS / f"{DATE}_baseline_session-out_subjective_s0"
                     / "metrics.csv")
    s0 = s0[s0.heldout_session != AGG][["heldout_session", "balanced_accuracy",
                                        "mcc"]]
    s0.columns = ["heldout_session", "balanced_accuracy_seed0", "mcc_seed0"]
    dm = (summ[["heldout_session", "balanced_accuracy_mean", "mcc_mean",
                "roc_auc_mean", "n_mean"]]
          .merge(insp[["heldout_session", "delta_mental", "intensity"]],
                 on="heldout_session")
          .merge(s0, on="heldout_session"))
    rho, p_rho = stats.spearmanr(dm.delta_mental, dm.balanced_accuracy_mean)
    rho0, p_rho0 = stats.spearmanr(dm.delta_mental, dm.balanced_accuracy_seed0)
    dm["spearman_rho_BA_vs_delta"] = rho
    dm["spearman_p"] = p_rho
    dm.to_csv(TAB / "diagnostics_delta_mental.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(dm.delta_mental, dm.balanced_accuracy_mean, zorder=3)
    for _, r in dm.iterrows():
        ax.annotate(r.heldout_session, (r.delta_mental,
                                        r.balanced_accuracy_mean),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.axhline(0.5, color="grey", lw=1, ls="--")
    ax.set_xlabel("Session delta-mental (M3 - M1 VAS)")
    ax.set_ylabel("Session-out balanced accuracy (mean over cap seeds 0-4)")
    ax.set_title(f"Per-session accuracy vs delta-mental\n"
                 f"Spearman rho = {rho:.2f}, p = {p_rho:.3f}, n = {len(dm)} sessions")
    fig.tight_layout()
    fig.savefig(FIG / "diagnostics_delta_mental_scatter.png", dpi=150)
    plt.close(fig)
    print(f"delta-mental: Spearman rho={rho:.3f} p={p_rho:.3f} n={len(dm)} "
          f"(seed0: rho={rho0:.3f} p={p_rho0:.3f})", flush=True)

    # ---- 5. P08 deep-dive
    print("P08 deep-dive ...", flush=True)
    coef_df, top, preds = p08_deep_dive(df0, feats0)
    print("P08 top features:\n", top.to_string(index=False), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
