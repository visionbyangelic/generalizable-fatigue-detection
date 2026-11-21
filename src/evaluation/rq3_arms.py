"""RQ3 Phase 2: the multimodal arms under the locked baseline.

Design: docs/rq3_preregistration.md as amended by Amendment 1
(21 Nov 2026, committed before any modelling). In short:

  Arms (feature sets; the model never changes):
    indicators  hr_valid, eda_valid only. The shortcut ceiling.
    peripheral  the six peripheral columns (4 continuous + 2 indicators).
    combined    the 43 Amendment-1 EEG features + the six peripheral.
    eeg         the 43 EEG features. On the keep-all pool this arm is a
                VERIFICATION rerun only: the headline comparator is the
                stored Week 2 table, and this rerun exists to prove the
                harness reproduces it on identical capped sets. On the
                restricted pool and in complete-test scoring it is a
                real comparator, re-evaluated on the same windows.

  Cells (pool x protocol):
    keepall    LOPO (headline) for all arms; within-subject and
               session-out tiers for peripheral and combined.
    restricted the 993 windows with hr_valid = eda_valid = 1, capped
               inside the restricted pool; LOPO, all four arms. A
               co-gate with a conditional estimand ("when the band
               worked"), never a cleaner pool.
    completetest  keep-all training, LOPO folds, but each held-out
               participant is scored on their complete-data windows
               only; all four arms. On those windows the indicators are
               constant, so the missingness pattern carries nothing.
    norm       within-participant-normalised features (V2-style,
               labelled sensitivity); LOPO; peripheral and combined.

  Aggregation (fixed by the amendment): per-participant balanced
  accuracy, cap-seed means first (seeds 0-4), participant bootstrap
  intervals (10,000 resamples, rng 42, the project convention). The
  headline quantities are the paired increments peripheral minus
  indicators and combined minus EEG-only, not the levels.

Model: the locked Week 2 baseline exactly (src/models/baseline.py):
logistic regression, L2, C = 1.0, class weights balanced, median
imputation and standardisation inside a pipeline fitted on each
training fold only. Indicator coefficients (hr_valid, eda_valid) are
recorded per fold for the peripheral and combined arms.

Outputs: one run folder per cell/arm/seed under experiments/runs/
(config.yaml, metrics.csv, predictions.csv, notes.md with the leakage
checklist) + summary tables in results/tables/ (rq3_per_participant,
rq3_arm_summary, rq3_contrasts, rq3_indicator_coefficients).
"""
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# The frozen band-power module calls np.trapz, removed in NumPy 2.5;
# alias it rather than editing frozen code (the project's standing
# workaround, docs/environment_pin.md).
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.metrics import binary_metrics
from src.evaluation.precision_variants import boot_ci, lopo_folds
from src.features.bandpower import FEATURE_NAMES, extract_features
from src.models.baseline import (cap_pool, make_model, run_folds,
                                 sessionout_folds, summarise, within_folds)

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
RUNS = ROOT / "experiments" / "runs"
TAB = ROOT / "results" / "tables"
DATE = "2026-11-21"
SEEDS = [0, 1, 2, 3, 4]
EXTRACTOR_SHA = hashlib.sha256(
    (ROOT / "src" / "features" / "peripheral.py").read_bytes()).hexdigest()

PER_COLS = ["hr_mean", "eda_mean", "eda_slope", "eda_rise_count",
            "hr_valid", "eda_valid"]
IND_COLS = ["hr_valid", "eda_valid"]
EEG_INDICATORS = ["valid_TP9", "valid_TP10", "both_temporals_masked"]
CONT_PER = ["hr_mean", "eda_mean", "eda_slope", "eda_rise_count"]
CONT_EEG = [c for c in FEATURE_NAMES if c not in EEG_INDICATORS]
METRICS = ["balanced_accuracy", "roc_auc", "mcc", "macro_f1"]


def git_commit():
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()


# ---------------------------------------------------------------- frames
def load_frames():
    """Master frames over the 1,053 pool windows, aligned by row."""
    idx = pd.read_csv(PROC / "windows_a1_uncapped_index.csv").reset_index(drop=True)
    per = pd.read_csv(PROC / "peripheral_features_a1.csv")
    key = ["participant", "session", "start_sample"]
    assert (per[key].values == idx[key].values).all()
    cache = PROC / "eeg_features_a1_cache.csv"
    if cache.exists():
        eeg = pd.read_csv(cache)
        assert list(eeg.columns) == list(FEATURE_NAMES) and len(eeg) == len(idx)
    else:
        arr = np.load(PROC / "windows_a1_uncapped.npy")
        eeg = extract_features(arr, idx).reset_index(drop=True)
        eeg.to_csv(cache, index=False)
    frames = {
        "indicators": per[IND_COLS].copy(),
        "peripheral": per[PER_COLS].copy(),
        "eeg": eeg,
        "combined": pd.concat([eeg, per[PER_COLS]], axis=1),
    }
    return idx, per, frames


def participant_normalise(feats, participants, arm):
    """V2-style within-participant normalisation (Amendment 1): each
    continuous feature centred/scaled with that participant's own
    unlabelled windows (nanmean/nanstd, ddof=0; labels never used).
    Indicator features unchanged. Guards mirror the Week 2 V2: SD 0 or
    undefined -> centre only; all-NaN -> stays NaN for the pipeline's
    training-fold imputation."""
    cont = CONT_PER if arm == "peripheral" else CONT_PER + CONT_EEG
    out = feats.copy()
    keys = pd.Series(np.asarray(participants), index=feats.index)
    for col in cont:
        x = feats[col].astype(float)
        tmp = pd.DataFrame({"k": keys.values, "x": x.values})
        g = tmp.groupby("k")["x"]
        mean = g.transform(lambda s: np.nanmean(s) if s.notna().any() else np.nan)
        sd = g.transform(lambda s: np.nanstd(s) if s.notna().any() else np.nan)
        z = (x.values - mean.values) / np.where(
            (sd.values > 0) & np.isfinite(sd.values), sd.values, 1.0)
        out[col] = np.where(np.isnan(mean.values), np.nan, z)
    return out


# ---------------------------------------------------------------- folds
def completetest_folds(df, complete_mask):
    """LOPO folds where the test side keeps only complete-data windows
    of the held-out participant; training is the full keep-all set of
    the other participants."""
    keys = []
    for part in df.participant.unique():
        te = ((df.participant == part) & complete_mask).values
        tr = (df.participant != part).values
        keys.append((part, tr, te))
    return keys


# ---------------------------------------------------------------- runs
def evaluate(df, feats, fold_keys):
    """Run folds; return (preds, per-participant metrics, skipped)."""
    preds, skipped = run_folds(df, feats, fold_keys)
    if preds.empty:
        return preds, pd.DataFrame(), skipped
    per = summarise(preds, "participant")
    return preds, per, skipped


def fit_coefficients(df, feats, fold_keys, arm):
    """Per-fold standardised LR coefficients on the two indicators,
    for the peripheral and combined arms (Amendment 1: quasi-separation
    must be visible, not hidden inside an aggregate)."""
    rows = []
    cols = list(feats.columns)
    for fid, tr, te in fold_keys:
        ytr = df.label.values[tr]
        if len(np.unique(ytr)) < 2:
            continue
        m = make_model()
        m.fit(feats.values[tr], ytr)
        coef = m.named_steps["lr"].coef_[0]
        for ind in IND_COLS:
            rows.append({"fold": fid, "feature": ind,
                         "coef": float(coef[cols.index(ind)])})
    return rows


CELL_NOTES = {
    "keepall": "Pool: the full 1,053-window primary pool, M1 capped per "
               "session per cap seed, exactly as the Week 2 baseline.",
    "restricted": "Pool: the 993 windows with hr_valid = eda_valid = 1, "
                  "capped inside the restricted pool. This pool is a "
                  "co-gate with a conditional estimand (performance when "
                  "the band worked); it is not a cleaner pool: the "
                  "restriction removes fatigued windows at about 12% "
                  "and alert windows at about 0.2%.",
    "completetest": "Training on the full keep-all capped pool; the "
                    "held-out participant is scored on their "
                    "complete-data windows only (P08 and P10 flagged: "
                    "their stopped bands removed 27 and 32 fatigued "
                    "windows respectively). On complete test windows "
                    "the validity indicators are constant, so the "
                    "missingness pattern carries no information there.",
    "norm": "Labelled sensitivity (Amendment 1): continuous features "
            "normalised within participant using that participant's own "
            "unlabelled windows before the locked pipeline runs; "
            "indicator features unchanged. Transductive at participant "
            "level in the same declared way the Week 2 V2 was at "
            "session level.",
}


def write_run(cell, arm, protocol, seed, per, preds, skipped, feat_cols):
    name = f"{DATE}_rq3_{cell}_{arm}_{protocol}_s{seed}"
    d = RUNS / name
    d.mkdir(parents=True, exist_ok=True)
    metrics = per.copy()
    if len(metrics):
        agg = {c: metrics[c].mean() for c in metrics.columns
               if c not in ("participant",) and pd.api.types.is_numeric_dtype(metrics[c])}
        metrics.loc[len(metrics)] = {"participant": "AGGREGATE_mean_over_participants",
                                     **agg}
    metrics.to_csv(d / "metrics.csv", index=False)
    preds.to_csv(d / "predictions.csv", index=False)
    cfg = {
        "model": "logistic_regression", "C": 1.0, "class_weight": "balanced",
        "imputation": "training-fold median", "scaling": "training-fold StandardScaler",
        "arm": arm, "features": feat_cols, "pool": cell, "protocol": protocol,
        "labels": "primary subjective (Amendment 1 windows)",
        "cap_seed": seed, "data": "windows_a1_uncapped + peripheral_features_a1",
        "peripheral_extractor_sha256": EXTRACTOR_SHA,
        "git_commit": git_commit(),
    }
    (d / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    split_desc = {
        "lopo": "hold out ALL sessions of one participant; train on the other participants",
        "lopo-completetest": "hold out one participant; train on the other participants' full capped pool; score only the held-out participant's complete-data windows",
        "within": "per participant, rotate the held-out session (train = the participant's other sessions)",
        "sessionout": "pooled: hold out one (participant, session); train on all remaining sessions of all participants; metrics aggregated per participant",
    }[protocol]
    (d / "notes.md").write_text(f"""# RQ3 {cell} / {arm} / {protocol} — cap seed {seed}

Design: docs/rq3_preregistration.md incl. Amendment 1. {CELL_NOTES[cell]}

Split definition: {split_desc}. A fold counts only if train and test
each contain both classes; skipped folds: {skipped if skipped else 'none'}.
Metrics in metrics.csv are per participant (out-of-fold predictions
concatenated per participant), the aggregation the amendment fixes.

Leakage checklist:
- [x] Normalisation (StandardScaler) fit on training fold only (Pipeline, per fold)
- [x] Median imputation fit on training fold only
- [x] Class weights computed from training fold only (class_weight='balanced')
- [x] No overlapping windows across splits (tumbling windows; splits at participant/session level)
- [x] No tuning on test data (C=1.0 locked, identical to the Week 2 baseline)
- [x] Label rules locked (docs/label_rules.md incl. Amendment 1) before any training
- [x] Peripheral features computed by the declared extractor only (SHA-256 {EXTRACTOR_SHA[:12]}...), no feature added or tuned

Environment: local VM (CPU), scikit-learn, pinned stack
(docs/environment_pin.md). Seed = M1-cap subsample seed {seed};
logistic regression itself is deterministic given the data.
""")
    return name


def run_cell(cell, pool_idx, frames, per_frame, plan, coef_rows,
             per_part_rows, summary_rows):
    """plan: list of (arm, protocol). pool_idx: the (sub)pool index
    frame with original row positions preserved in its index."""
    for arm, protocol in plan:
        feats_master = frames[arm]
        for seed in SEEDS:
            capped = cap_pool(pool_idx, seed)
            pos = capped.index.to_numpy()
            df = capped.reset_index(drop=True)
            feats = feats_master.iloc[pos].reset_index(drop=True)
            if cell == "norm":
                feats = participant_normalise(feats, df.participant, arm)
            if protocol == "lopo":
                keys = lopo_folds(df)
            elif protocol == "lopo-completetest":
                complete = ((per_frame.iloc[pos]["hr_valid"] == 1)
                            & (per_frame.iloc[pos]["eda_valid"] == 1))
                keys = completetest_folds(df, complete.to_numpy())
            elif protocol == "within":
                keys = within_folds(df)
            else:
                keys = sessionout_folds(df)
            preds, per, skipped = evaluate(df, feats, keys)
            write_run(cell, arm, protocol, seed, per, preds, skipped,
                      list(feats.columns))
            if arm in ("peripheral", "combined") and protocol == "lopo":
                for r in fit_coefficients(df, feats, keys, arm):
                    coef_rows.append({"pool": cell, "arm": arm, "seed": seed,
                                      **r})
            for _, row in per.iterrows():
                per_part_rows.append({
                    "pool": cell, "arm": arm, "protocol": protocol,
                    "seed": seed, "participant": row.participant,
                    **{m: row[m] for m in METRICS}, "n": row.n})
            print(f"{cell}/{arm}/{protocol} s{seed}: "
                  f"participants={len(per)} skipped={skipped} "
                  f"meanBA={per.balanced_accuracy.mean() if len(per) else float('nan'):.3f}")


def main():
    idx, per, frames = load_frames()
    coef_rows, per_part_rows = [], []
    summary_rows = None  # built from per_part_rows afterwards

    both_valid = ((per.hr_valid == 1) & (per.eda_valid == 1)).to_numpy()
    restricted_idx = idx[both_valid]

    run_cell("keepall", idx, frames, per,
             [("indicators", "lopo"), ("peripheral", "lopo"),
              ("combined", "lopo"), ("eeg", "lopo"),
              ("peripheral", "within"), ("combined", "within"),
              ("peripheral", "sessionout"), ("combined", "sessionout")],
             coef_rows, per_part_rows, summary_rows)
    run_cell("restricted", restricted_idx, frames, per,
             [("indicators", "lopo"), ("peripheral", "lopo"),
              ("combined", "lopo"), ("eeg", "lopo")],
             coef_rows, per_part_rows, summary_rows)
    run_cell("completetest", idx, frames, per,
             [("indicators", "lopo-completetest"),
              ("peripheral", "lopo-completetest"),
              ("combined", "lopo-completetest"),
              ("eeg", "lopo-completetest")],
             coef_rows, per_part_rows, summary_rows)
    run_cell("norm", idx, frames, per,
             [("peripheral", "lopo"), ("combined", "lopo")],
             coef_rows, per_part_rows, summary_rows)

    pp = pd.DataFrame(per_part_rows)
    pp.to_csv(TAB / "rq3_per_participant.csv", index=False)
    pd.DataFrame(coef_rows).to_csv(
        TAB / "rq3_indicator_coefficients.csv", index=False)

    # ---- per-participant seed means, then bootstrap CIs on the means
    seedmean = (pp.groupby(["pool", "arm", "protocol", "participant"])[METRICS]
                  .mean().reset_index())
    seedmean.to_csv(TAB / "rq3_per_participant_seedmeans.csv", index=False)
    summ = []
    for (cell, arm, protocol), g in seedmean.groupby(["pool", "arm", "protocol"]):
        row = {"pool": cell, "arm": arm, "protocol": protocol,
               "n_participants": len(g)}
        for m in METRICS:
            mean, lo, hi, _ = boot_ci(g[m].to_numpy())
            row[f"{m}_mean"], row[f"{m}_lo"], row[f"{m}_hi"] = mean, lo, hi
        summ.append(row)
    summ = pd.DataFrame(summ)
    summ.to_csv(TAB / "rq3_arm_summary.csv", index=False)
    print(summ[["pool", "arm", "protocol", "n_participants",
                "balanced_accuracy_mean", "balanced_accuracy_lo",
                "balanced_accuracy_hi"]].to_string(index=False))

    # ---- paired contrasts (per-participant seed-mean differences)
    stored = pd.read_csv(TAB / "baseline_loso_subjective_a1_summary.csv")
    stored_eeg = stored.set_index("participant")["balanced_accuracy_mean"]
    piv = seedmean[seedmean.protocol.isin(["lopo", "lopo-completetest"])]
    piv = piv.pivot_table(index="participant",
                          columns=["pool", "arm"], values="balanced_accuracy")

    def contrast(name, a, b, note):
        d = (piv[a] - piv[b]).dropna()
        mean, lo, hi, n = boot_ci(d.to_numpy())
        return {"contrast": name, "n_participants": n, "diff_mean": mean,
                "diff_lo": lo, "diff_hi": hi, "note": note}

    contrasts = [
        contrast("peripheral - indicators (keepall LOPO)",
                 ("keepall", "peripheral"), ("keepall", "indicators"),
                 "headline increment 1: peripheral signal beyond the shortcut ceiling"),
        contrast("combined - eeg-stored (keepall LOPO)",
                 ("keepall", "combined"), ("keepall", "eeg"),
                 "computed against this harness's eeg arm; verified equal to the stored Week 2 values before use (see results doc)"),
        contrast("peripheral - indicators (restricted LOPO)",
                 ("restricted", "peripheral"), ("restricted", "indicators"),
                 "co-gate increment on the conditional pool"),
        contrast("combined - eeg (restricted LOPO)",
                 ("restricted", "combined"), ("restricted", "eeg"),
                 "co-gate: combined increment on identical restricted windows"),
        contrast("peripheral - indicators (norm sensitivity LOPO)",
                 ("norm", "peripheral"), ("keepall", "indicators"),
                 "indicators are unchanged by normalisation, so the keep-all indicators arm is the comparator"),
        contrast("combined - combined-unnormalised (norm sensitivity)",
                 ("norm", "combined"), ("keepall", "combined"),
                 "descriptive movement under within-participant normalisation"),
        contrast("peripheral - peripheral-unnormalised (norm sensitivity)",
                 ("norm", "peripheral"), ("keepall", "peripheral"),
                 "descriptive movement under within-participant normalisation"),
    ]
    # headline contrast 2 against the STORED Week 2 EEG values directly
    d = (piv[("keepall", "combined")]
         - stored_eeg.reindex(piv.index)).dropna()
    mean, lo, hi, n = boot_ci(d.to_numpy())
    contrasts.insert(1, {"contrast": "combined - EEG-only stored Week 2 (keepall LOPO)",
                         "n_participants": n, "diff_mean": mean,
                         "diff_lo": lo, "diff_hi": hi,
                         "note": "headline increment 2, against the stored Week 2 per-participant values as declared"})
    ct = pd.DataFrame(contrasts)
    ct.to_csv(TAB / "rq3_contrasts.csv", index=False)
    print(ct[["contrast", "n_participants", "diff_mean", "diff_lo",
              "diff_hi"]].to_string(index=False))

    # ---- verification: this harness's keep-all EEG LOPO vs stored Week 2
    mine = piv[("keepall", "eeg")]
    both = pd.concat([mine.rename("rerun"), stored_eeg.rename("stored")],
                     axis=1).dropna()
    print("\nEEG verification, max |rerun - stored| per-participant BA:",
          float((both.rerun - both.stored).abs().max()),
          f"over {len(both)} participants")


if __name__ == "__main__":
    main()
