"""Ablation: band-power baseline WITHOUT the mask/validity indicators.

Why this run exists: the mask-only baseline (BA 0.530) matching the
full 43-feature model (0.515) shows EEG adds nothing beyond the mask
across sessions, but does not by itself prove the mask CAUSES the
cross-session residue. Two matching scores are consistent with the
mask carrying the residue, but also with the spectral features
carrying the same information some other way. The causal test is to
take the mask away and see what falls. This run removes the 3
indicator features (valid_TP9, valid_TP10, both_temporals_masked)
and keeps only the 40 spectral features (log/relative band powers +
frontal/temporal ratios), everything else identical to
src/models/baseline.py: same Amendment-1 window sets, same cap seeds
0-4, same folds, same train-fold-only imputation/scaling/class
weights. The prediction under test, stated before running: if the
mask carries the residue, session-out collapses to ~0.50 and P08's
within-subject inversion (0.225 with indicators) disappears.

Note: masked temporal channels' spectral features remain NaN and are
median-imputed from the training fold, exactly as in the full model.
The ONLY change is that the model is no longer told which channels
were masked.

Outputs: run folders experiments/runs/2026-11-20_baseline_<protocol>_
subjective_ablated_s<seed>/ (config.yaml, metrics.csv, predictions.csv,
notes.md) + results/tables/baseline_<protocol>_subjective_ablated_
summary.csv.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.bandpower import FEATURE_NAMES, extract_features
from src.models.baseline import (METRIC_COLS, SEEDS, cap_pool, run_folds,
                                 sessionout_folds, summarise, within_folds)

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
RUNS = ROOT / "experiments" / "runs"
TAB = ROOT / "results" / "tables"
INDICATORS = ["valid_TP9", "valid_TP10", "both_temporals_masked"]
KEEP = [c for c in FEATURE_NAMES if c not in INDICATORS]
assert len(KEEP) == 40, KEEP


def write_run(protocol, seed, per_group, preds, skipped, group_col):
    name = f"2026-11-20_baseline_{protocol}_subjective_ablated_s{seed}"
    d = RUNS / name
    d.mkdir(parents=True, exist_ok=True)
    agg = {c: per_group[c].mean() for c in METRIC_COLS if c in per_group}
    metrics = per_group.copy()
    metrics.loc[len(metrics)] = {group_col: "AGGREGATE_mean_over_groups", **agg}
    metrics.to_csv(d / "metrics.csv", index=False)
    preds.to_csv(d / "predictions.csv", index=False)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    cfg = {
        "model": "logistic_regression", "C": 1.0, "class_weight": "balanced",
        "features": "bandpower_40_spectral_only_ablated (43-feature set minus "
                    "valid_TP9, valid_TP10, both_temporals_masked)",
        "ablation": "mask/validity-indicator features REMOVED "
                    "(Reviewer 2 review 10 causal test)",
        "imputation": "training-fold median", "scaling": "training-fold StandardScaler",
        "protocol": protocol, "labels": "subjective (Amendment 1 windows)",
        "cap_seed": seed, "data": "windows_a1_uncapped, M1 capped per session",
        "git_commit": commit,
    }
    (d / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (d / "notes.md").write_text(f"""# Baseline {protocol} ABLATED (no mask indicators) — subjective labels, cap seed {seed}

Ablation of experiments/runs/2026-11-20_baseline_{protocol}_subjective_s{seed}:
identical data, cap, folds, model and leakage discipline; the only change
is the feature set — 40 EEG spectral features (log/relative band powers +
frontal/temporal ratios), with the 3 mask/validity-indicator features
(valid_TP9, valid_TP10, both_temporals_masked) removed. Commissioned by
Reviewer 2 review 10 as the causal test of the mask-shortcut claim.

Split definition: {'per participant, rotate held-out session (train = other sessions of the same participant)' if protocol == 'within' else 'pooled: hold out one (participant, session); train on all remaining sessions of all participants'}. A fold counts only if train and test both contain both classes; skipped folds: {skipped if skipped else 'none'}.

Leakage checklist:
- [x] Normalisation (StandardScaler) fit on training fold only (Pipeline, per fold)
- [x] Median imputation for masked-channel NaNs fit on training fold only
- [x] Class weights computed from training fold only (class_weight='balanced')
- [x] No overlapping windows across splits (tumbling windows; splits at session level)
- [x] No tuning on test data (C=1.0 locked default, unchanged from full model)
- [x] Label rules locked (docs/label_rules.md incl. Amendment 1) before any training

Environment: local VM (CPU), scikit-learn. Seed = M1-cap subsample seed {seed};
logistic regression itself is deterministic given the data.
""")
    return per_group


def main():
    arr = np.load(PROC / "windows_a1_uncapped.npy")
    idx = pd.read_csv(PROC / "windows_a1_uncapped_index.csv")
    idx = idx.reset_index(drop=True)
    summaries = {"within": [], "session-out": []}
    seed_aggs = {"within": [], "session-out": []}
    for seed in SEEDS:
        capped = cap_pool(idx, seed)
        pos = capped.index.to_numpy()
        feats43 = extract_features(arr[pos], capped.reset_index(drop=True))
        feats = feats43[KEEP]  # the ablation: spectral features only
        cdf = capped.reset_index(drop=True)
        for protocol, fold_fn in [("within", within_folds),
                                  ("session-out", sessionout_folds)]:
            preds, skipped = run_folds(cdf, feats, fold_fn(cdf))
            if protocol == "within":
                per = summarise(preds, "participant")
                gcol = "participant"
            else:
                per = summarise(preds, "fold").rename(
                    columns={"fold": "heldout_session"})
                gcol = "heldout_session"
            write_run(protocol, seed, per, preds, skipped, gcol)
            per["seed"] = seed
            summaries[protocol].append(per)
            agg = per[METRIC_COLS[:4]].mean()
            seed_aggs[protocol].append({"seed": seed, **agg.to_dict()})
            print(f"seed {seed} {protocol}: groups={len(per)} skipped={skipped} "
                  f"mean bal_acc={agg.balanced_accuracy:.3f} macroF1={agg.macro_f1:.3f} "
                  f"AUC={agg.roc_auc:.3f} MCC={agg.mcc:.3f}")
            if protocol == "within":
                p08 = per[per.participant == "P08"]
                if len(p08):
                    r = p08.iloc[0]
                    print(f"    P08 within: BA={r.balanced_accuracy:.3f} "
                          f"AUC={r.roc_auc:.3f} MCC={r.mcc:.3f} n={int(r.n)}")
    for protocol, parts in summaries.items():
        allp = pd.concat(parts)
        gcol = "participant" if protocol == "within" else "heldout_session"
        summ = allp.groupby(gcol)[METRIC_COLS].agg(["mean", "std"])
        summ.columns = [f"{a}_{b}" for a, b in summ.columns]
        summ.reset_index().to_csv(
            TAB / f"baseline_{protocol}_subjective_ablated_summary.csv",
            index=False)
        sa = pd.DataFrame(seed_aggs[protocol])
        m, s = sa[METRIC_COLS[:4]].mean(), sa[METRIC_COLS[:4]].std()
        print(f"{protocol} ABLATED cross-seed (per-seed aggregates): "
              f"BA={m.balanced_accuracy:.3f}±{s.balanced_accuracy:.3f} "
              f"macroF1={m.macro_f1:.3f}±{s.macro_f1:.3f} "
              f"AUC={m.roc_auc:.3f}±{s.roc_auc:.3f} "
              f"MCC={m.mcc:.3f}±{s.mcc:.3f}")
    p08w = pd.concat(summaries["within"])
    p08 = p08w[p08w.participant == "P08"]
    print("P08 within ABLATED per seed:",
          [f"{r.balanced_accuracy:.3f}" for r in p08.itertuples()],
          f"mean BA={p08.balanced_accuracy.mean():.3f}±{p08.balanced_accuracy.std():.3f} "
          f"mean MCC={p08.mcc.mean():.3f} mean AUC={p08.roc_auc.mean():.3f}")


if __name__ == "__main__":
    main()
