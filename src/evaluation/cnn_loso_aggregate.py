"""Aggregate the CNN leave-one-participant-out folds (21 Nov 2026).

Reads the per-fold run folders written by src/models/cnn_loso_segval.py
(one folder per held-out participant per model seed) and produces the
RQ2 / RQ1 summary artefacts:

- results/tables/cnn_loso_subjective_a1_summary.csv: per-participant
  mean/SD over model seeds, same layout as the baseline LOPO summary
  so the two models sit side by side.
- results/tables/cnn_loso_rq2_contrast.csv: per-participant CNN minus
  baseline balanced accuracy (each model averaged over its own seeds
  first), with the mean difference and its participant-bootstrap CI.
- results/tables/rq1_generalisation_gap.csv: within-subject minus
  LOPO balanced accuracy for both models, from the stored Week 2
  within-subject summaries and the two LOPO summaries.

Intervals follow the closure-pack convention: participant-level
bootstrap, 10,000 resamples over per-participant cross-seed means,
percentile 95% CI. Everything is reported as estimates with intervals
(the locked feasibility-gate ruling makes this phase estimation-only:
no equivalence or null verdicts are computed here).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "experiments" / "runs"
TAB = ROOT / "results" / "tables"
DATE = "2026-11-21"
METRICS = ["balanced_accuracy", "macro_f1", "roc_auc", "mcc",
           "recall_alert", "recall_fatigued",
           "precision_alert", "precision_fatigued", "n", "n_alert",
           "n_fatigued"]
N_BOOT = 10_000


def boot_ci(values, rng):
    """Participant bootstrap percentile CI for the mean."""
    v = np.asarray(values, dtype=float)
    reps = rng.choice(v, size=(N_BOOT, len(v)), replace=True).mean(axis=1)
    return float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


def load_folds():
    rows = []
    for d in sorted(RUNS.glob(f"{DATE}_cnn_loso_subjective_*_s*")):
        mfile = d / "metrics.csv"
        if not mfile.exists():
            continue
        seed = int(d.name.rsplit("_s", 1)[1])
        m = pd.read_csv(mfile)
        m["seed"] = seed
        rows.append(m)
    if not rows:
        sys.exit("no CNN LOPO fold metrics found")
    return pd.concat(rows, ignore_index=True)


def main():
    folds = load_folds()
    seeds = sorted(folds.seed.unique())
    parts = sorted(folds.participant.unique())
    print(f"folds: {len(folds)} rows, participants={parts}, seeds={seeds}")

    # Per-participant summary over seeds (baseline summary layout).
    summ = folds.groupby("participant")[METRICS].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reset_index()
    summ.to_csv(TAB / "cnn_loso_subjective_a1_summary.csv", index=False)

    rng = np.random.default_rng(0)
    per = summ.set_index("participant")
    headline = {}
    for col in ("balanced_accuracy", "roc_auc", "mcc"):
        vals = per[f"{col}_mean"].to_numpy()
        lo, hi = boot_ci(vals, rng)
        headline[col] = (float(vals.mean()), lo, hi)
        print(f"CNN LOPO {col}: {vals.mean():.3f} [{lo:.3f}, {hi:.3f}] "
              f"(participant bootstrap, n={len(vals)})")

    # RQ2 paired contrast: CNN minus baseline per participant.
    base = pd.read_csv(TAB / "baseline_loso_subjective_a1_summary.csv")
    base = base.set_index("participant")["balanced_accuracy_mean"]
    both = pd.DataFrame({"cnn_ba": per["balanced_accuracy_mean"],
                         "baseline_ba": base}).dropna()
    both["diff"] = both.cnn_ba - both.baseline_ba
    lo, hi = boot_ci(both["diff"].to_numpy(), rng)
    print(f"RQ2 paired contrast (CNN - baseline BA): "
          f"{both['diff'].mean():+.3f} [{lo:+.3f}, {hi:+.3f}] "
          f"over {len(both)} participants")
    out = both.reset_index()
    out.loc[len(out)] = ["MEAN_DIFF", both.cnn_ba.mean(),
                         both.baseline_ba.mean(), both["diff"].mean()]
    out.to_csv(TAB / "cnn_loso_rq2_contrast.csv", index=False)

    # RQ1 generalisation gap: within-subject minus LOPO, per model.
    b_within = pd.read_csv(TAB / "baseline_within_subjective_a1_summary.csv")
    c_within = pd.read_csv(TAB / "cnn_within_subjective_a1_summary.csv")
    gap_rows = [
        {"model": "band-power baseline",
         "within_ba": float(b_within.balanced_accuracy_mean.mean()),
         "within_participants": int(len(b_within)),
         "lopo_ba": float(base.mean()),
         "lopo_participants": int(len(base))},
        {"model": "1D-CNN (EEGNet)",
         "within_ba": float(c_within.balanced_accuracy_mean.mean()),
         "within_participants": int(len(c_within)),
         "lopo_ba": float(per["balanced_accuracy_mean"].mean()),
         "lopo_participants": int(len(per))},
    ]
    gaps = pd.DataFrame(gap_rows)
    gaps["gap_within_minus_lopo"] = gaps.within_ba - gaps.lopo_ba
    gaps.to_csv(TAB / "rq1_generalisation_gap.csv", index=False)
    print(gaps.to_string(index=False))

    # Paired version: same participants in both protocols, gap per
    # participant first, then averaged (the aggregate rows above mix
    # participant sets, because the within protocols cover fewer
    # participants than LOPO does).
    paired_rows = []
    for model, wfile, lseries in (
            ("band-power baseline", "baseline_within_subjective_a1_summary.csv",
             base),
            ("1D-CNN (EEGNet)", "cnn_within_subjective_a1_summary.csv",
             per["balanced_accuracy_mean"])):
        w = pd.read_csv(TAB / wfile).set_index("participant")[
            "balanced_accuracy_mean"]
        both = pd.DataFrame({"within_ba": w, "lopo_ba": lseries}).dropna()
        both["gap"] = both.within_ba - both.lopo_ba
        lo, hi = boot_ci(both["gap"].to_numpy(), rng)
        paired_rows.append({"model": model, "participants": len(both),
                            "within_ba": float(both.within_ba.mean()),
                            "lopo_ba": float(both.lopo_ba.mean()),
                            "paired_gap": float(both.gap.mean()),
                            "paired_gap_ci_lo": lo, "paired_gap_ci_hi": hi})
        print(f"{model} paired gap (n={len(both)}): "
              f"{both.gap.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
    pd.DataFrame(paired_rows).to_csv(
        TAB / "rq1_generalisation_gap_paired.csv", index=False)


if __name__ == "__main__":
    main()
