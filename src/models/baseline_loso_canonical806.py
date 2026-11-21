"""Band-power baseline LOPO on the canonical cap-seed-0 window set
(21 Nov 2026, panel-owed sensitivity).

Why this run exists: the RQ2 contrast in docs/cnn_loso_results.md
compares the CNN (run on the canonical cap-seed-0 set, 806 windows)
against the primary baseline headline (five cap seeds over the
1,053-window pool). That contrast therefore mixes two differences at
once: model class and window set. This run removes the second one.
It runs the frozen baseline pipeline (src/models/baseline.py:
43 band-power features, median imputation, standardisation, logistic
regression C=1.0 with balanced class weights, everything fit on the
training fold only) under leave-one-participant-out on EXACTLY the
806 windows the CNN used (data/processed/windows_a1*), same folds,
so the paired CNN-minus-baseline contrast can be recomputed on an
identical window set.

The set is the cap-seed-0 subsample by construction, so there is one
deterministic run, not five: logistic regression has no model seed,
and the cap seed is already embodied in the window set. This is a
labelled sensitivity, not a replacement headline; the primary
baseline number stays the five-cap-seed estimate over the full pool.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# The frozen feature code calls np.trapz, which NumPy 2.5 removed
# after renaming it np.trapezoid. Alias the old name here rather than
# editing frozen code (the workaround the RQ4 drivers already use;
# the audit in docs/rq4_session_labels.md verified the two are the
# same function and the features are unaffected).
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

from src.evaluation.metrics import binary_metrics
from src.features.bandpower import extract_features
from src.models.baseline import METRIC_COLS, make_model

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
RUNS = ROOT / "experiments" / "runs"
TAB = ROOT / "results" / "tables"
RUN_NAME = "2026-11-21_baseline_loso_subjective_canonical806"


def main():
    arr = np.load(PROC / "windows_a1.npy")
    idx = pd.read_csv(PROC / "windows_a1_index.csv").reset_index(drop=True)
    assert len(idx) == len(arr) == 806, (len(idx), len(arr))
    feats = extract_features(arr, idx)

    preds, skipped = [], []
    for part in sorted(idx.participant.unique()):
        te = (idx.participant == part).to_numpy()
        tr = ~te
        ytr, yte = idx.label.values[tr], idx.label.values[te]
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            skipped.append(part)
            continue
        m = make_model()
        m.fit(feats.values[tr], ytr)
        rec = idx.iloc[np.where(te)[0]][["participant", "session"]].copy()
        rec["fold"] = part
        rec["y_true"] = yte
        rec["y_pred"] = m.predict(feats.values[te])
        rec["y_proba"] = m.predict_proba(feats.values[te])[:, 1]
        preds.append(rec)
    preds = pd.concat(preds)
    per = pd.DataFrame([
        {"participant": gid, **binary_metrics(g.y_true, g.y_pred, g.y_proba)}
        for gid, g in preds.groupby("participant")])
    print(per[["participant", "balanced_accuracy", "roc_auc", "mcc", "n"]]
          .to_string(index=False))
    print(f"mean BA over participants: {per.balanced_accuracy.mean():.3f}; "
          f"skipped folds: {skipped if skipped else 'none'}")

    # Run folder (project convention: config.yaml, metrics.csv,
    # notes.md with the leakage checklist).
    d = RUNS / RUN_NAME
    d.mkdir(parents=True, exist_ok=True)
    agg = {c: per[c].mean() for c in METRIC_COLS if c in per}
    metrics = per.copy()
    metrics.loc[len(metrics)] = {"participant": "AGGREGATE_mean_over_groups",
                                 **agg}
    metrics.to_csv(d / "metrics.csv", index=False)
    preds.to_csv(d / "predictions.csv", index=False)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    cfg = {"model": "logistic_regression", "C": 1.0,
           "class_weight": "balanced",
           "features": "bandpower_43_amendment1 (see src/features/bandpower.py)",
           "imputation": "training-fold median",
           "scaling": "training-fold StandardScaler",
           "protocol": "loso (leave-one-participant-out)",
           "labels": "subjective (Amendment 1 windows)",
           "data": "windows_a1 (canonical cap-seed-0 set, 806 windows, "
                   "11 participants) - the exact window set and folds "
                   "of the CNN LOPO run",
           "run_type": "sensitivity (panel-owed): baseline on the CNN's "
                       "window set so the RQ2 contrast holds the window "
                       "set fixed",
           "git_commit": commit}
    (d / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (d / "notes.md").write_text(f"""# Baseline LOPO - subjective labels, canonical cap-seed-0 set (806 windows)

Split definition: leave-one-participant-out on the canonical
cap-seed-0 Amendment-1 set (data/processed/windows_a1*): the exact
806 windows, and the exact 11 folds, of the CNN LOPO run. One
deterministic pass: the cap seed is embodied in the set and logistic
regression has no model seed. Skipped folds: {skipped if skipped else 'none'}.

WHY (panel-owed sensitivity): the RQ2 contrast as first reported
mixed model class with window set (CNN on these 806 windows,
baseline headline over five cap seeds on the 1,053-window pool).
This run holds the window set fixed so the recomputed contrast is a
model comparison only. It is a sensitivity; the primary baseline
headline remains the five-cap-seed estimate.

Leakage checklist:
- [x] Held-out participant contributes no window to training
- [x] Normalisation (StandardScaler) fit on training fold only (Pipeline, per fold)
- [x] Median imputation for masked-channel NaNs fit on training fold only
- [x] Class weights computed from training fold only (class_weight='balanced')
- [x] No overlapping windows across splits (tumbling windows; participant-level split)
- [x] No tuning on test data (C=1.0 locked default)
- [x] Label rules locked (docs/label_rules.md incl. Amendment 1) before any training

Environment: local VM (CPU), scikit-learn (pinned stack,
docs/environment_pin.md).
""")

    # Summary table in the baseline LOPO summary layout (single run:
    # mean = the fold value, std = NaN, there is no seed axis).
    summ = per.set_index("participant")
    out = pd.DataFrame({"participant": summ.index})
    for c in METRIC_COLS:
        out[f"{c}_mean"] = summ[c].to_numpy()
        out[f"{c}_std"] = np.nan
    out.to_csv(TAB / "baseline_loso_subjective_canonical806_summary.csv",
               index=False)

    # Recomputed paired contrast on the identical window set: CNN
    # per-participant seed means minus this baseline, participant
    # bootstrap (same convention as cnn_loso_aggregate.py: 10,000
    # resamples, rng seed 0, percentile 95% CI).
    cnn = pd.read_csv(TAB / "cnn_loso_subjective_a1_summary.csv")
    cnn = cnn.set_index("participant")["balanced_accuracy_mean"]
    both = pd.DataFrame({"cnn_ba": cnn,
                         "baseline_ba": per.set_index("participant")
                         .balanced_accuracy}).dropna()
    both["diff"] = both.cnn_ba - both.baseline_ba
    rng = np.random.default_rng(0)
    v = both["diff"].to_numpy()
    reps = rng.choice(v, size=(10_000, len(v)), replace=True).mean(axis=1)
    lo, hi = float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))
    print(f"RQ2 paired contrast on canonical 806 (CNN - baseline BA): "
          f"{v.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] over {len(both)} "
          f"participants")
    outc = both.reset_index()
    outc.loc[len(outc)] = ["MEAN_DIFF", both.cnn_ba.mean(),
                           both.baseline_ba.mean(), both["diff"].mean()]
    outc.to_csv(TAB / "cnn_loso_rq2_contrast_canonical806.csv", index=False)


if __name__ == "__main__":
    main()
