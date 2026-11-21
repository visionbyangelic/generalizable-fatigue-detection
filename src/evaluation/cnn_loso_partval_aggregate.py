"""Aggregate the participant-carved-validation CNN LOPO sensitivity
(21 Nov 2026, POST HOC).

Reads the per-fold run folders written by
src/models/cnn_loso_partval.py and produces:

- results/tables/cnn_loso_partval_subjective_summary.csv:
  per-participant mean/SD over model seeds, same layout as the
  headline CNN LOPO summary.
- results/tables/cnn_loso_partval_vs_segval.csv: per-participant
  balanced accuracy under both validation carves (seed means), the
  paired difference (participant-carved minus segment-carved), and
  its participant-bootstrap CI.

Intervals follow the project convention: participant-level bootstrap,
10,000 resamples over per-participant cross-seed means, percentile
95% CI, rng seed 0. Estimates with intervals only; no verdicts.
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
    v = np.asarray(values, dtype=float)
    reps = rng.choice(v, size=(N_BOOT, len(v)), replace=True).mean(axis=1)
    return float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


def main():
    rows = []
    for d in sorted(RUNS.glob(f"{DATE}_cnn_loso_partval_subjective_*_s*")):
        mfile = d / "metrics.csv"
        if not mfile.exists():
            continue
        seed = int(d.name.rsplit("_s", 1)[1])
        m = pd.read_csv(mfile)
        m["seed"] = seed
        rows.append(m)
    if not rows:
        sys.exit("no partval fold metrics found")
    folds = pd.concat(rows, ignore_index=True)
    print(f"folds: {len(folds)} rows, seeds={sorted(folds.seed.unique())}")

    summ = folds.groupby("participant")[METRICS].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reset_index()
    summ.to_csv(TAB / "cnn_loso_partval_subjective_summary.csv", index=False)

    rng = np.random.default_rng(0)
    per = summ.set_index("participant")
    for col in ("balanced_accuracy", "roc_auc", "mcc"):
        vals = per[f"{col}_mean"].to_numpy()
        lo, hi = boot_ci(vals, rng)
        print(f"partval CNN LOPO {col}: {vals.mean():.3f} "
              f"[{lo:.3f}, {hi:.3f}] (participant bootstrap, "
              f"n={len(vals)})")

    seg = pd.read_csv(TAB / "cnn_loso_subjective_a1_summary.csv")
    seg = seg.set_index("participant")["balanced_accuracy_mean"]
    both = pd.DataFrame({"partval_ba": per["balanced_accuracy_mean"],
                         "segval_ba": seg}).dropna()
    both["diff_partval_minus_segval"] = both.partval_ba - both.segval_ba
    lo, hi = boot_ci(both["diff_partval_minus_segval"].to_numpy(), rng)
    print(f"paired movement (partval - segval BA): "
          f"{both['diff_partval_minus_segval'].mean():+.3f} "
          f"[{lo:+.3f}, {hi:+.3f}] over {len(both)} participants")
    out = both.reset_index()
    out.loc[len(out)] = ["MEAN_DIFF", both.partval_ba.mean(),
                         both.segval_ba.mean(),
                         both["diff_partval_minus_segval"].mean()]
    out.to_csv(TAB / "cnn_loso_partval_vs_segval.csv", index=False)


if __name__ == "__main__":
    main()
