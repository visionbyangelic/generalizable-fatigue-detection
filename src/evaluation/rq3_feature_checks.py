"""RQ3 Phase 2 sanity checks on the peripheral feature table.

Runs BEFORE any RQ3 model is fit. Verifies the table built by
src/features/peripheral.py against the Phase 1 inventory
(docs/RQ3_INVENTORY.md Section 3), which was computed independently:

  - HR valid 94.3% of pool windows (alert 99.8%, fatigued 87.9%)
  - EDA valid 94.4% (alert 100%, fatigued 87.9%)
  - both-valid (the restricted pool) = 993 windows
  - 60 windows fail at least one rule; 59 of the 60 are fatigued
    (27 = all of P08/S02's M3 pool windows, 32 = all of P10/S01's pool
    windows, 1 = an alert window in P02/S01)

Also reports the feature distributions by class (valid windows only)
and the correlation between the EEG mask indicators and the peripheral
validity indicators across the pool, which Amendment 1 to the RQ3
pre-registration requires before the combined arm is interpreted.

Outputs: results/tables/rq3_feature_sanity.csv (validity accounting),
results/tables/rq3_indicator_correlation.csv (Pearson correlations).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
TAB = ROOT / "results" / "tables"

EEG_INDICATORS = ["valid_TP9", "valid_TP10", "both_temporals_masked"]
PER_FEATURES = ["hr_mean", "eda_mean", "eda_slope", "eda_rise_count"]


def main():
    idx = pd.read_csv(PROC / "windows_a1_uncapped_index.csv").reset_index(drop=True)
    per = pd.read_csv(PROC / "peripheral_features_a1.csv")
    assert len(per) == len(idx) == 1053
    key = ["participant", "session", "start_sample"]
    assert (per[key].values == idx[key].values).all(), "window identity mismatch"
    df = pd.concat([idx[["participant", "session", "block", "label",
                         "valid_TP9", "valid_TP10"]],
                    per[["hr_mean", "eda_mean", "eda_slope", "eda_rise_count",
                         "hr_valid", "eda_valid"]]], axis=1)
    # the third EEG mask indicator is a derived band-power feature:
    # both temporal channels masked in the same window
    df["both_temporals_masked"] = (
        (~df.valid_TP9.astype(bool)) & (~df.valid_TP10.astype(bool))).astype(float)
    df["both_valid"] = ((df.hr_valid == 1) & (df.eda_valid == 1)).astype(int)

    rows = []
    def rate(mask_rows, col):
        sub = df[mask_rows]
        return len(sub), float(sub[col].mean())
    for scope, mask in [("pool", np.ones(len(df), bool)),
                        ("alert", df.label == 0),
                        ("fatigued", df.label == 1)]:
        for col in ["hr_valid", "eda_valid", "both_valid"]:
            n, r = rate(mask, col)
            rows.append({"scope": scope, "measure": f"{col}_rate",
                         "n_windows": n, "value": r})
    for part, g in df.groupby("participant"):
        rows.append({"scope": part, "measure": "both_valid_rate",
                     "n_windows": len(g), "value": float(g.both_valid.mean())})
    bad = df[df.both_valid == 0]
    rows.append({"scope": "pool", "measure": "either_invalid_count",
                 "n_windows": len(df), "value": float(len(bad))})
    rows.append({"scope": "pool", "measure": "either_invalid_fatigued",
                 "n_windows": len(df), "value": float(bad.label.sum())})
    for (part, sess), g in bad.groupby(["participant", "session"]):
        rows.append({"scope": f"{part}/{sess}", "measure": "invalid_windows",
                     "n_windows": int(len(g)), "value": float(len(g))})
    sanity = pd.DataFrame(rows)
    sanity.to_csv(TAB / "rq3_feature_sanity.csv", index=False)

    print("== validity (target: hr .943 / eda .944 / both 993; "
          "invalid 60, fatigued 59) ==")
    print(sanity[sanity.measure.str.endswith("_rate")].to_string(index=False))
    print(sanity[sanity.measure.str.contains("invalid")].to_string(index=False))

    print("\n== feature distributions on valid windows, by class ==")
    dist = (df.groupby("label")[PER_FEATURES]
              .agg(["mean", "median", "std", "count"]))
    print(dist.round(4).to_string())
    dist.to_csv(TAB / "rq3_feature_distributions.csv")

    print("\n== per-participant means (valid windows) ==")
    pp = df.groupby("participant")[PER_FEATURES + ["both_valid"]].mean()
    print(pp.round(3).to_string())

    print("\n== indicator correlation (pool, Pearson) ==")
    inds = df[EEG_INDICATORS].astype(float).copy()
    inds["hr_valid"] = df.hr_valid
    inds["eda_valid"] = df.eda_valid
    corr = inds.corr(method="pearson")
    print(corr.round(3).to_string())
    corr.round(4).to_csv(TAB / "rq3_indicator_correlation.csv")


if __name__ == "__main__":
    main()
