"""Descriptive balanced accuracy over the non-degenerate CNN LOPO fits.

This is a small companion to cnn_loso_degeneracy.py. That report flags
16 of the 55 CNN LOPO fits as degenerate (one class predicted for at
least 95% of a fold's test windows, a criterion defined from the
predictions only, never from the labels). A natural question follows:
what do the remaining 39 fits average on their own?

I compute it here strictly as a descriptive, post hoc number. It is
not a headline, not a sensitivity and not a replacement for the
primary estimate: dropping fits because of how their predictions
came out would bias any estimate upward, which is exactly why this
number is only ever reported with that label attached, inside the
degeneracy discussion in docs/cnn_loso_results.md.

Method, mirroring the headline aggregation so the two numbers are
comparable in construction: per participant, the mean balanced
accuracy over that participant's non-degenerate seeds; then the mean
of those per-participant means over the 11 participants, with a
participant bootstrap interval (10,000 resamples over the 11
per-participant values, seed 0). Input is the per-fit balanced
accuracies in results/tables/cnn_loso_degeneracy.csv, which were
extracted from the stored predictions; nothing is retrained.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "results" / "tables"


def main() -> None:
    deg = pd.read_csv(TAB / "cnn_loso_degeneracy.csv")
    keep = deg[deg["flag"].isna()].copy()
    per_participant = (
        keep.groupby("participant")["balanced_accuracy"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "ba_nondegen_mean", "count": "n_nondegen_fits"})
        .reset_index()
    )
    values = per_participant["ba_nondegen_mean"].to_numpy()
    estimate = float(values.mean())

    rng = np.random.default_rng(0)
    boot = np.array(
        [rng.choice(values, size=len(values), replace=True).mean() for _ in range(10_000)]
    )
    lo, hi = np.percentile(boot, [2.5, 97.5])

    summary = pd.DataFrame(
        [
            {
                "scope": "non-degenerate fits only (descriptive, post hoc)",
                "n_participants": len(values),
                "n_fits": int(keep.shape[0]),
                "balanced_accuracy": estimate,
                "ci_lo": float(lo),
                "ci_hi": float(hi),
            }
        ]
    )
    out = TAB / "cnn_loso_nondegen_descriptive.csv"
    per_participant.to_csv(TAB / "cnn_loso_nondegen_per_participant.csv", index=False)
    summary.to_csv(out, index=False)
    print(summary.to_string(index=False))
    print(per_participant.to_string(index=False))


if __name__ == "__main__":
    main()
