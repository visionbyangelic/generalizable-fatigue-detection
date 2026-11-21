"""CNN LOPO degeneracy report (21 Nov 2026, panel-owed follow-up).

Extraction only: reads the stored predictions.csv and config.yaml of
the 55 CNN LOPO run folders (experiments/runs/2026-11-21_cnn_loso_
subjective_*) and reports, per fold (held-out participant x model
seed), how the predictions are distributed across the two classes and
how long training ran before early stopping stopped it.

Why this exists: the headline MCC came out at exactly 0.000 with an
AUC of 0.517, a combination that suggests many fits may have produced
near-constant predictions (the network barely learning anything that
separates the classes on an unseen person). Before the RQ2 wording
says anything about the CNN, this report counts how often that
happened, under one simple criterion fixed here:

  degenerate       one class predicted for >= 95% of the fold's
                   test windows
  near-degenerate  one class predicted for >= 90% but < 95%
  constant         one class predicted for 100% (a subset of
                   degenerate, reported separately)

The selected epoch is the epochs_run value the frozen training loop
recorded in each folder's config.yaml carve note (the epoch at which
early stopping halted, with the best-validation-loss weights
restored). Nothing is retrained and no stored number is recomputed
except the predicted-class fractions, which come straight from the
stored y_pred column.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "experiments" / "runs"
TAB = ROOT / "results" / "tables"
DATE = "2026-11-21"
DEGEN = 0.95
NEAR = 0.90


def main():
    rows = []
    for d in sorted(RUNS.glob(f"{DATE}_cnn_loso_subjective_*_s*")):
        pred_file = d / "predictions.csv"
        cfg_file = d / "config.yaml"
        if not pred_file.exists() or not cfg_file.exists():
            print(f"[skip] {d.name}: missing predictions or config",
                  flush=True)
            continue
        part = d.name.split("_subjective_")[1].rsplit("_s", 1)[0]
        seed = int(d.name.rsplit("_s", 1)[1])
        preds = pd.read_csv(pred_file)
        cfg = yaml.safe_load(cfg_file.read_text())
        m = re.search(r"epochs_run=(\d+)", str(cfg.get("carve", "")))
        epochs = int(m.group(1)) if m else None
        n = len(preds)
        frac_fat = float((preds.y_pred == 1).mean())
        majority = max(frac_fat, 1.0 - frac_fat)
        if majority >= DEGEN:
            flag = "degenerate"
        elif majority >= NEAR:
            flag = "near-degenerate"
        else:
            flag = ""
        met = pd.read_csv(d / "metrics.csv").iloc[0]
        rows.append({
            "participant": part, "seed": seed, "n": n,
            "frac_pred_fatigued": round(frac_fat, 4),
            "frac_pred_majority": round(majority, 4),
            "constant": bool(majority >= 1.0),
            "epochs_run": epochs,
            "balanced_accuracy": float(met.balanced_accuracy),
            "roc_auc": float(met.roc_auc), "mcc": float(met.mcc),
            "flag": flag,
        })
    df = pd.DataFrame(rows).sort_values(["participant", "seed"])
    df.to_csv(TAB / "cnn_loso_degeneracy.csv", index=False)
    n_deg = int((df.flag == "degenerate").sum())
    n_near = int((df.flag == "near-degenerate").sum())
    n_const = int(df.constant.sum())
    print(f"folds: {len(df)}; degenerate (>={DEGEN:.0%} one class): "
          f"{n_deg}; near-degenerate (>={NEAR:.0%}): {n_near}; "
          f"constant (100%): {n_const}")
    print(f"epochs_run: median {df.epochs_run.median():.0f}, "
          f"range {df.epochs_run.min()}-{df.epochs_run.max()}; "
          f"folds stopping by epoch 20: {int((df.epochs_run <= 20).sum())}")
    print(df.groupby("participant").flag.apply(
        lambda s: (s == "degenerate").sum()).rename(
        "degenerate_folds_of_5").to_string())


if __name__ == "__main__":
    sys.exit(main())
