"""RQ4 tidy-up computations (21 Nov 2026).

Three small jobs owed to the panel review of the RQ4 write-up
(docs/rq4_session_labels.md, reviews 23 and 24 in
docs/methods_notes.md). None of them changes a headline; each one
either pins down a number the write-up quotes or closes a check the
review asked for explicitly.

  1. Solver sensitivity for the bit-identity audit. The audit traced
     the gap between the stored Week 2 primary LOPO numbers and this
     arm's re-run to solver-level differences between two builds of
     the numeric stack (attribution by elimination: the old stack was
     never reproduced and its versions were not recorded). The review
     asked for two things: the solver's own convergence record
     (n_iter_ and any ConvergenceWarning) in the primary LOPO folds,
     which the Week 2 run folders did not store, and a re-run of the
     same folds with a tight solver tolerance, to show whether the
     remaining probability movement is solver slack or something
     structural. If a tight tolerance collapses the movement to
     ~1e-6, the story is "the default tolerance stops at slightly
     different iterates under different builds"; if it does not, the
     audit's explanation is wrong and I need to know that too.
     Both variants run here, on the restricted pool (the Week 2
     primary windows), cap seeds 0 to 4, LOPO folds, identical in
     every other respect.

  2. Exact swap null for the day-level LOPO arm. The day-level
     headline (0.484 over 11 participants) was reported with a
     bootstrap interval only. With 11 fold scores, the exact
     session-level swap null is enumerable: 2^11 = 2048 sign flips of
     the per-participant deviations from 0.5. This computes it
     exactly instead of sampling it.

  3. Leaky-split accuracy. The day-level write-up reports balanced
     accuracy and fatigued recall for the deliberately leaky
     random-window split; the review asked for plain accuracy and
     BOTH class recalls next to them (the preliminary's 99.18% was a
     fatigued-class recall, so the recalls are the comparable
     quantities). Recomputed here from the stored predictions of the
     leaky run folders, per split seed.

Outputs: run folders for the two solver variants (per cap seed),
results/tables/rq4_tolerance_sensitivity.csv,
results/tables/rq4_daylevel_swapnull.csv,
results/tables/rq4_leaky_accuracy.csv. Everything is computed on the
pinned current stack (docs/environment_pin.md).
"""
import itertools
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.metrics import binary_metrics
from src.evaluation.rq4_session_labels import load_arm, lopo_folds
from src.features.bandpower import extract_features
from src.models.baseline import METRIC_COLS, cap_pool

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "experiments" / "runs"
TAB = ROOT / "results" / "tables"
DATE = "2026-11-21"
SEEDS = [0, 1, 2, 3, 4]
TIGHT_TOL, TIGHT_MAX_ITER = 1e-8, 10000


def git_commit():
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()


def make_model_tight():
    """The locked baseline pipeline with only the solver tolerance
    tightened (tol 1e-4 default -> 1e-8) and the iteration cap raised
    so the tolerance, not the cap, is what stops the solver."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, class_weight="balanced",
                                  tol=TIGHT_TOL, max_iter=TIGHT_MAX_ITER)),
    ])


def run_lopo_variant(feats, idx, variant):
    """LOPO over the restricted pool for one solver variant, cap seeds
    0-4. Returns (predictions with a seed column, solver log rows).
    The fold construction is the RQ4 driver's lopo_folds and the
    capped pools are the frozen cap_pool, so the ONLY difference from
    the RQ4 restricted arm is the model factory (and, for the tight
    variant, the tolerance inside it)."""
    from src.models.baseline import make_model
    factory = make_model if variant == "std" else make_model_tight
    all_preds, solver_rows = [], []
    for seed in SEEDS:
        capped = cap_pool(idx, seed)
        pos = capped.index.to_numpy()
        df = capped.reset_index(drop=True)
        fc = feats.iloc[pos].reset_index(drop=True)
        for fid, tr, te in lopo_folds(df):
            ytr, yte = df.label.values[tr], df.label.values[te]
            if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
                continue
            m = factory()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                m.fit(fc.values[tr], ytr)
            conv = [w for w in caught
                    if issubclass(w.category, ConvergenceWarning)]
            solver_rows.append({
                "variant": variant, "cap_seed": seed, "fold": fid,
                "n_iter": int(m.named_steps["lr"].n_iter_[0]),
                "convergence_warnings": len(conv),
                "n_train": int(tr.sum()), "n_test": int(te.sum())})
            rec = df.iloc[np.where(te)[0]][["participant",
                                            "session"]].copy()
            rec["fold"] = fid
            rec["seed"] = seed
            rec["y_true"] = yte
            rec["y_pred"] = m.predict(fc.values[te])
            rec["y_proba"] = m.predict_proba(fc.values[te])[:, 1]
            all_preds.append(rec)
    return pd.concat(all_preds), pd.DataFrame(solver_rows)


def per_participant(preds):
    rows = []
    for (seed, part), g in preds.groupby(["seed", "participant"]):
        rows.append({"seed": seed, "participant": part,
                     **binary_metrics(g.y_true, g.y_pred, g.y_proba)})
    return pd.DataFrame(rows)


def write_variant_runs(variant, preds, solver_rows):
    label = ("standard solver (locked defaults: tol=1e-4, max_iter=1000)"
             if variant == "std" else
             f"tight solver sensitivity (tol={TIGHT_TOL}, "
             f"max_iter={TIGHT_MAX_ITER})")
    for seed in SEEDS:
        p_seed = preds[preds.seed == seed]
        per = per_participant(p_seed)
        name = f"{DATE}_baseline_loso_subjective-tolcheck-{variant}_s{seed}"
        d = RUNS / name
        d.mkdir(parents=True, exist_ok=True)
        metrics = per.drop(columns=["seed"]).copy()
        agg = {c: metrics[c].mean() for c in METRIC_COLS if c in metrics}
        metrics.loc[len(metrics)] = {
            "participant": "AGGREGATE_mean_over_groups", **agg}
        metrics.to_csv(d / "metrics.csv", index=False)
        p_seed.drop(columns=["seed"]).to_csv(d / "predictions.csv",
                                             index=False)
        slog = solver_rows[solver_rows.cap_seed == seed]
        cfg = {
            "model": "logistic_regression", "C": 1.0,
            "class_weight": "balanced",
            "solver_variant": label,
            "features": "bandpower_43_amendment1",
            "imputation": "training-fold median",
            "scaling": "training-fold StandardScaler",
            "protocol": "loso (leave-one-participant-out)",
            "labels": "subjective (Week 2 primary pool, restricted32)",
            "cap_seed": seed,
            "purpose": "RQ4 audit solver sensitivity (docs/"
                       "rq4_session_labels.md); not a primary result",
            "git_commit": git_commit(),
        }
        (d / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        (d / "notes.md").write_text(f"""# Baseline LOPO, subjective labels, solver check ({variant}), cap seed {seed}

RQ4 audit support run (docs/rq4_session_labels.md, bit-identity audit).
Identical windows, features, folds and seeds to the Week 2 primary
LOPO; the only change is the solver setting: {label}. Purpose: record
the solver's convergence behaviour (n_iter_, warnings), which the
Week 2 folders did not store, and measure how much of the small
probability gap between the stored Week 2 numbers and the current
stack is solver slack.

Solver record this seed: n_iter_ per fold: {slog.n_iter.tolist()};
ConvergenceWarnings: {int(slog.convergence_warnings.sum())}.

Leakage checklist:
- [x] Normalisation (StandardScaler) fit on training fold only (Pipeline, per fold)
- [x] Median imputation fit on training fold only
- [x] Class weights computed from training fold only (class_weight='balanced')
- [x] No overlapping windows across splits (tumbling windows; split at participant level)
- [x] No tuning on test data (C=1.0 locked; tolerance is the studied variable, fixed before the run)

Environment: pinned current stack (docs/environment_pin.md).
""")


def compare_probabilities(a, b, key_cols):
    """Max/mean absolute probability difference and label flips
    between two prediction frames over the same windows. Rows are
    matched the way the audit matched them: within each
    (participant, session) group, both frames list the windows in
    the same order (both code paths emit the capped pool's row
    order), which I verify by checking the true-label sequences
    agree group by group before comparing element by element."""
    diffs, flips = [], 0
    b_groups = {k: g for k, g in b.groupby(key_cols)}
    for keys, ga in a.groupby(key_cols):
        gb = b_groups[keys]
        assert len(ga) == len(gb), f"group {keys}: {len(ga)} vs {len(gb)}"
        assert (ga.y_true.values == gb.y_true.values).all(), \
            f"group {keys}: true-label order differs between frames"
        diffs.append(np.abs(ga.y_proba.values - gb.y_proba.values))
        flips += int((ga.y_pred.values != gb.y_pred.values).sum())
    d = np.concatenate(diffs)
    return float(d.max()), float(d.mean()), flips, len(d)


def solver_sensitivity(feats, idx):
    preds, logs = {}, {}
    for variant in ["std", "tight"]:
        preds[variant], logs[variant] = run_lopo_variant(feats, idx,
                                                         variant)
        write_variant_runs(variant, preds[variant], logs[variant])
        per = per_participant(preds[variant])
        headline = per.groupby("participant") \
            .balanced_accuracy.mean().mean()
        print(f"{variant}: LOPO headline (cross-seed participant mean) "
              f"= {headline:.6f}; n_iter_ max "
              f"{logs[variant].n_iter.max()}, median "
              f"{logs[variant].n_iter.median()}; ConvergenceWarnings "
              f"{int(logs[variant].convergence_warnings.sum())}")
    # tight vs standard, matched per (seed, participant, session)
    rows = []
    for seed in SEEDS:
        a = preds["std"][preds["std"].seed == seed]
        b = preds["tight"][preds["tight"].seed == seed]
        mx, mn, fl, n = compare_probabilities(
            a, b, ["participant", "session"])
        rows.append({"comparison": "tight vs standard (current stack)",
                     "cap_seed": seed, "max_abs_proba_diff": mx,
                     "mean_abs_proba_diff": mn, "label_flips": fl,
                     "n_windows": n})
    # standard re-run vs the stored Week 2 primary predictions: the
    # environment-gap comparison the audit is about.
    for seed in SEEDS:
        stored = pd.read_csv(
            RUNS / f"2026-11-20_baseline_loso_subjective_s{seed}"
            / "predictions.csv")
        a = preds["std"][preds["std"].seed == seed]
        mx, mn, fl, n = compare_probabilities(
            a, stored, ["participant", "session"])
        rows.append({"comparison": "standard (current stack) vs stored "
                                   "Week 2 primary",
                     "cap_seed": seed, "max_abs_proba_diff": mx,
                     "mean_abs_proba_diff": mn, "label_flips": fl,
                     "n_windows": n})
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "rq4_tolerance_sensitivity.csv", index=False)
    print(out.to_string(index=False))
    logs_all = pd.concat([logs["std"], logs["tight"]])
    return out, logs_all


def daylevel_swap_null():
    """Exact swap null over the 11 day-level LOPO fold scores: every
    sign assignment of the per-participant deviations from 0.5
    (2^11 = 2048), statistic = mean balanced accuracy. Two-sided p:
    share of assignments at least as far from 0.5 as the observed
    mean. This is the session-level swap null at participant level:
    swapping a participant's two day labels negates their deviation,
    which is exactly the sign flip."""
    per = pd.read_csv(TAB / "baseline_loso_daylevel_summary.csv")
    ba = per.balanced_accuracy.values
    assert len(ba) == 11
    dev = ba - 0.5
    obs = float(ba.mean())
    stats = np.array([0.5 + float(np.dot(np.array(signs), dev)) / len(dev)
                      for signs in itertools.product([-1, 1], repeat=11)])
    p_two = float((np.abs(stats - 0.5) >= abs(obs - 0.5) - 1e-12).mean())
    out = pd.DataFrame([{
        "arm": "day-level LOPO (post hoc)",
        "n_participants": 11, "observed_mean_ba": obs,
        "null_n_assignments": len(stats),
        "null_mean": float(stats.mean()), "null_sd": float(stats.std()),
        "null_q025": float(np.percentile(stats, 2.5)),
        "null_q975": float(np.percentile(stats, 97.5)),
        "p_two_sided_exact": p_two,
        "definition": "exact enumeration of all 2^11 sign flips of "
                      "per-participant (BA - 0.5); statistic = mean BA; "
                      "two-sided p = share of assignments with "
                      "|mean - 0.5| >= |observed - 0.5|"}])
    out.to_csv(TAB / "rq4_daylevel_swapnull.csv", index=False)
    print(f"day-level swap null: observed {obs:.4f}, null mean "
          f"{stats.mean():.4f} sd {stats.std():.4f}, exact two-sided "
          f"p = {p_two:.4f} over {len(stats)} assignments")
    return out


def leaky_accuracy():
    """Plain accuracy (and both recalls, recomputed) for the leaky
    day-level split, per split seed, from the stored predictions."""
    rows = []
    for seed in SEEDS:
        p = pd.read_csv(RUNS / f"{DATE}_baseline_leaky-random_daylevel"
                        f"_s{seed}" / "predictions.csv")
        m = binary_metrics(p.y_true, p.y_pred, p.y_proba)
        rows.append({"seed": seed,
                     "accuracy": float((p.y_true == p.y_pred).mean()),
                     "balanced_accuracy": m["balanced_accuracy"],
                     "recall_alert": m["recall_alert"],
                     "recall_fatigued": m["recall_fatigued"],
                     "roc_auc": m["roc_auc"], "n": m["n"]})
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "rq4_leaky_accuracy.csv", index=False)
    print("leaky split, per-seed (pooled over folds):")
    print(out.to_string(index=False))
    print(f"means: accuracy {out.accuracy.mean():.3f}, recall_alert "
          f"{out.recall_alert.mean():.3f}, recall_fatigued "
          f"{out.recall_fatigued.mean():.3f}, AUC "
          f"{out.roc_auc.mean():.3f}")
    return out


def main():
    arr, idx = load_arm("restricted32")
    feats = extract_features(arr, idx)
    print(f"restricted pool: {len(idx)} windows")
    solver_sensitivity(feats, idx)
    daylevel_swap_null()
    leaky_accuracy()


if __name__ == "__main__":
    main()
