"""Phase B — band-power baseline (logistic regression) under Amendment 1.

Data: data/processed/windows_a1_uncapped (npy + index). For each cap seed
0-4 the M1 class is capped per session to the retained M3 count (same rule
as the seed-0 primary set in build_windows_a1.py), features are extracted
(src/features/bandpower.py, 43 features), and two protocols run:

  within       per participant: rotate the held-out session (train on the
               participant's other sessions, test on the held-out one).
               A fold is valid only if train AND test each contain both
               classes. Per-participant metrics are computed on the
               concatenated out-of-fold predictions.
  session-out  pooled: hold out one (participant, session) at a time, train
               on every other session of every participant. Same fold
               validity rule. Metrics per held-out session + aggregate.

Model (locked, how-others-did-it.md section 6): logistic regression, L2,
C=1.0. Leakage discipline: median imputation for masked-channel NaNs,
standardisation, and class weights ('balanced') are ALL fit on the
training fold only (sklearn Pipeline, fit per fold).

Outputs: one run folder per protocol per seed under experiments/runs/
(config.yaml, metrics.csv, notes.md) + cross-seed summary tables in
results/tables/.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.metrics import binary_metrics
from src.features.bandpower import extract_features

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
RUNS = ROOT / "experiments" / "runs"
TAB = ROOT / "results" / "tables"
SEEDS = [0, 1, 2, 3, 4]
METRIC_COLS = ["balanced_accuracy", "macro_f1", "roc_auc", "mcc",
               "recall_alert", "recall_fatigued",
               "precision_alert", "precision_fatigued", "n", "n_alert", "n_fatigued"]


def cap_pool(idx, seed):
    out = []
    for (_, _), g in idx.groupby(["participant", "session"]):
        m1, m3 = g[g.label == 0], g[g.label == 1]
        if len(m1) > len(m3):
            rng = np.random.default_rng(seed)
            keep = rng.choice(m1.index.to_numpy(), size=len(m3), replace=False)
            g = pd.concat([m1.loc[keep], m3])
        out.append(g)
    return pd.concat(out).sort_values(["participant", "session", "start_sample"])


def make_model():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, class_weight="balanced",
                                  max_iter=1000)),
    ])


def run_folds(df, feats, fold_keys):
    """fold_keys: list of (fold_id, train_mask, test_mask). Returns
    per-fold prediction records + skipped-fold log."""
    preds, skipped = [], []
    for fid, tr, te in fold_keys:
        ytr, yte = df.label.values[tr], df.label.values[te]
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            skipped.append(fid)
            continue
        m = make_model()
        m.fit(feats.values[tr], ytr)
        rec = df.iloc[np.where(te)[0]][["participant", "session"]].copy()
        rec["fold"] = fid
        rec["y_true"] = yte
        rec["y_pred"] = m.predict(feats.values[te])
        rec["y_proba"] = m.predict_proba(feats.values[te])[:, 1]
        preds.append(rec)
    return (pd.concat(preds) if preds else pd.DataFrame()), skipped


def within_folds(df):
    keys = []
    for part, g in df.groupby("participant"):
        for sess in g.session.unique():
            te = ((df.participant == part) & (df.session == sess)).values
            tr = ((df.participant == part) & (df.session != sess)).values
            keys.append((f"{part}/{sess}", tr, te))
    return keys


def sessionout_folds(df):
    keys = []
    for (part, sess), _ in df.groupby(["participant", "session"]):
        te = ((df.participant == part) & (df.session == sess)).values
        keys.append((f"{part}/{sess}", ~te, te))
    return keys


def summarise(preds, group_col):
    rows = []
    for gid, g in preds.groupby(group_col):
        rows.append({group_col: gid,
                     **binary_metrics(g.y_true, g.y_pred, g.y_proba)})
    return pd.DataFrame(rows)


def write_run(protocol, seed, per_group, preds, skipped, group_col):
    name = f"2026-11-20_baseline_{protocol}_subjective_s{seed}"
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
        "features": "bandpower_43_amendment1 (see src/features/bandpower.py)",
        "imputation": "training-fold median", "scaling": "training-fold StandardScaler",
        "protocol": protocol, "labels": "subjective (Amendment 1 windows)",
        "cap_seed": seed, "data": "windows_a1_uncapped, M1 capped per session",
        "git_commit": commit,
    }
    (d / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (d / "notes.md").write_text(f"""# Baseline {protocol} — subjective labels, cap seed {seed}

Split definition: {'per participant, rotate held-out session (train = other sessions of the same participant)' if protocol == 'within' else 'pooled: hold out one (participant, session); train on all remaining sessions of all participants'}. A fold counts only if train and test both contain both classes; skipped folds: {skipped if skipped else 'none'}.

Leakage checklist:
- [x] Normalisation (StandardScaler) fit on training fold only (Pipeline, per fold)
- [x] Median imputation for masked-channel NaNs fit on training fold only
- [x] Class weights computed from training fold only (class_weight='balanced')
- [x] No overlapping windows across splits (tumbling windows; splits at session level)
- [x] No tuning on test data (C=1.0 locked default from how-others-did-it.md section 6)
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
    for seed in SEEDS:
        capped = cap_pool(idx, seed)
        pos = capped.index.to_numpy()
        feats = extract_features(arr[pos], capped.reset_index(drop=True))
        for protocol, fold_fn, gcol in [
                ("within", within_folds, "participant"),
                ("session-out", sessionout_folds, "heldout_session")]:
            preds, skipped = run_folds(capped.reset_index(drop=True), feats,
                                       fold_fn(capped.reset_index(drop=True)))
            if protocol == "within":
                per = summarise(preds, "participant")
            else:
                per = summarise(preds, "fold").rename(columns={"fold": "heldout_session"})
            write_run(protocol, seed, per, preds, skipped, gcol)
            per["seed"] = seed
            summaries[protocol].append(per)
            agg = per[METRIC_COLS[:4]].mean()
            print(f"seed {seed} {protocol}: groups={len(per)} skipped={skipped} "
                  f"mean bal_acc={agg.balanced_accuracy:.3f} macroF1={agg.macro_f1:.3f} "
                  f"AUC={agg.roc_auc:.3f} MCC={agg.mcc:.3f}")
    for protocol, parts in summaries.items():
        allp = pd.concat(parts)
        gcol = "participant" if protocol == "within" else "heldout_session"
        summ = allp.groupby(gcol)[METRIC_COLS].agg(["mean", "std"])
        summ.columns = [f"{a}_{b}" for a, b in summ.columns]
        summ.reset_index().to_csv(
            TAB / f"baseline_{protocol}_subjective_a1_summary.csv", index=False)
        overall = allp[METRIC_COLS[:4]].mean()
        print(f"{protocol} cross-seed group-mean: bal_acc={overall.balanced_accuracy:.3f} "
              f"macroF1={overall.macro_f1:.3f} AUC={overall.roc_auc:.3f} MCC={overall.mcc:.3f}")


if __name__ == "__main__":
    main()
