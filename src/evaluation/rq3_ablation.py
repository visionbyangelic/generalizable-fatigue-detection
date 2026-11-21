"""RQ3 post hoc ablation: HR-only and EDA-only arms (descriptive).

Status: POST HOC. These two arms were not in docs/rq3_preregistration.md
or its Amendment 1. I added them after the RQ3 results were in, for
one descriptive purpose: to record what each signal family scores on
its own under the identical locked design, beside the peripheral arm
that combines them. The combined arm's score on its own invites the
question of which family carries it; these arms answer that question
descriptively, and nothing stronger is claimed from them. No
feature-set selection is made from these arms, and no conclusion about
which channel matters more is drawn beyond the two estimates
themselves; both are reported whatever they show.

Design (identical to the RQ3 grid, src/evaluation/rq3_arms.py):
  hr   hr_mean + hr_valid.
  eda  eda_mean, eda_slope, eda_rise_count + eda_valid.
Keep-all pool (the 1,053 primary-pool windows, M1 capped per session
per cap seed, exactly as the Week 2 baseline), LOPO folds, cap seeds
0-4, the locked logistic regression (src/models/baseline.py: L2,
C = 1.0, class weights balanced, median imputation and standardisation
inside a pipeline fitted on each training fold only).

Aggregation (the project convention): per-participant balanced
accuracy, cap-seed means first, participant-level bootstrap intervals
(10,000 resamples over the per-participant seed means, rng 42).

Outputs: one run folder per arm/seed under experiments/runs/
(2026-11-21_rq3_keepall_{hr,eda}_lopo_s{seed}, each with config.yaml,
metrics.csv, predictions.csv, notes.md) + results/tables/
rq3_ablation_per_participant.csv, rq3_ablation_summary.csv and
rq3_ablation_indicator_coefficients.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.precision_variants import boot_ci, lopo_folds
from src.evaluation.rq3_arms import EXTRACTOR_SHA, METRICS, git_commit, load_frames
from src.models.baseline import cap_pool, make_model, run_folds, summarise

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "experiments" / "runs"
TAB = ROOT / "results" / "tables"
DATE = "2026-11-21"
SEEDS = [0, 1, 2, 3, 4]

ARMS = {
    "hr": (["hr_mean", "hr_valid"], "hr_valid"),
    "eda": (["eda_mean", "eda_slope", "eda_rise_count", "eda_valid"],
            "eda_valid"),
}


def write_run(arm, seed, per, preds, skipped, feat_cols):
    name = f"{DATE}_rq3_keepall_{arm}_lopo_s{seed}"
    d = RUNS / name
    d.mkdir(parents=True, exist_ok=True)
    metrics = per.copy()
    if len(metrics):
        agg = {c: metrics[c].mean() for c in metrics.columns
               if c not in ("participant",)
               and pd.api.types.is_numeric_dtype(metrics[c])}
        metrics.loc[len(metrics)] = {
            "participant": "AGGREGATE_mean_over_participants", **agg}
    metrics.to_csv(d / "metrics.csv", index=False)
    preds.to_csv(d / "predictions.csv", index=False)
    cfg = {
        "model": "logistic_regression", "C": 1.0, "class_weight": "balanced",
        "imputation": "training-fold median",
        "scaling": "training-fold StandardScaler",
        "arm": arm, "features": feat_cols, "pool": "keepall",
        "protocol": "lopo",
        "labels": "primary subjective (Amendment 1 windows)",
        "cap_seed": seed,
        "data": "windows_a1_uncapped + peripheral_features_a1",
        "peripheral_extractor_sha256": EXTRACTOR_SHA,
        "status": "post hoc descriptive ablation; not in the "
                  "pre-registration; added after the RQ3 results review "
                  "(docs/methods_notes.md reviews 37-38)",
        "git_commit": git_commit(),
    }
    (d / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (d / "notes.md").write_text(f"""# RQ3 ablation (post hoc) / {arm} / lopo — cap seed {seed}

POST HOC AND DESCRIPTIVE. This arm was not in docs/rq3_preregistration.md
or Amendment 1. It was added after the RQ3 results were reviewed, to
record what this signal family scores on its own under the identical
locked design. No feature-set selection is made from it, and it is
reported beside the peripheral arm as description only.

Pool: the full 1,053-window primary pool, M1 capped per session per
cap seed, exactly as the Week 2 baseline and the RQ3 grid.

Split definition: hold out ALL sessions of one participant; train on
the other participants. A fold counts only if train and test each
contain both classes; skipped folds: {skipped if skipped else 'none'}.
Metrics in metrics.csv are per participant (out-of-fold predictions
concatenated per participant), the aggregation the project fixes.

Leakage checklist:
- [x] Normalisation (StandardScaler) fit on training fold only (Pipeline, per fold)
- [x] Median imputation fit on training fold only
- [x] Class weights computed from training fold only (class_weight='balanced')
- [x] No overlapping windows across splits (tumbling windows; splits at participant level)
- [x] No tuning on test data (C=1.0 locked, identical to the Week 2 baseline)
- [x] Label rules locked (docs/label_rules.md incl. Amendment 1) before any training
- [x] Peripheral features computed by the declared extractor only (SHA-256 {EXTRACTOR_SHA[:12]}...), no feature added or tuned

Environment: local VM (CPU), scikit-learn, pinned stack
(docs/environment_pin.md). Seed = M1-cap subsample seed {seed};
logistic regression itself is deterministic given the data.
""")
    return name


def main():
    idx, per, _frames = load_frames()
    per_part_rows, coef_rows = [], []
    for arm, (cols, indicator) in ARMS.items():
        feats_master = per[cols].copy()
        for seed in SEEDS:
            capped = cap_pool(idx, seed)
            pos = capped.index.to_numpy()
            df = capped.reset_index(drop=True)
            feats = feats_master.iloc[pos].reset_index(drop=True)
            keys = lopo_folds(df)
            preds, skipped = run_folds(df, feats, keys)
            per_metrics = summarise(preds, "participant")
            write_run(arm, seed, per_metrics, preds, skipped, cols)
            # per-fold coefficient on this arm's validity indicator
            col_list = list(feats.columns)
            for fid, tr, te in keys:
                ytr = df.label.values[tr]
                if len(np.unique(ytr)) < 2:
                    continue
                m = make_model()
                m.fit(feats.values[tr], ytr)
                coef = m.named_steps["lr"].coef_[0]
                coef_rows.append({
                    "pool": "keepall", "arm": arm, "seed": seed,
                    "fold": fid, "feature": indicator,
                    "coef": float(coef[col_list.index(indicator)])})
            for _, row in per_metrics.iterrows():
                per_part_rows.append({
                    "pool": "keepall", "arm": arm, "protocol": "lopo",
                    "seed": seed, "participant": row.participant,
                    **{m: row[m] for m in METRICS}, "n": row.n})
            print(f"keepall/{arm}/lopo s{seed}: "
                  f"participants={len(per_metrics)} skipped={skipped} "
                  f"meanBA={per_metrics.balanced_accuracy.mean():.3f}")

    pp = pd.DataFrame(per_part_rows)
    pp.to_csv(TAB / "rq3_ablation_per_participant.csv", index=False)
    pd.DataFrame(coef_rows).to_csv(
        TAB / "rq3_ablation_indicator_coefficients.csv", index=False)

    seedmean = (pp.groupby(["pool", "arm", "protocol", "participant"])[METRICS]
                  .mean().reset_index())
    summ = []
    for (cell, arm, protocol), g in seedmean.groupby(
            ["pool", "arm", "protocol"]):
        row = {"pool": cell, "arm": arm, "protocol": protocol,
               "n_participants": len(g)}
        for m in METRICS:
            mean, lo, hi, _ = boot_ci(g[m].to_numpy())
            row[f"{m}_mean"], row[f"{m}_lo"], row[f"{m}_hi"] = mean, lo, hi
        summ.append(row)
    summ = pd.DataFrame(summ)
    summ.to_csv(TAB / "rq3_ablation_summary.csv", index=False)
    print(summ[["pool", "arm", "protocol", "n_participants",
                "balanced_accuracy_mean", "balanced_accuracy_lo",
                "balanced_accuracy_hi"]].to_string(index=False))
    sm = seedmean.pivot_table(index="participant", columns="arm",
                              values="balanced_accuracy")
    print(sm.round(3).to_string())


if __name__ == "__main__":
    main()
