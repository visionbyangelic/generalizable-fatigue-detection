"""RQ4 diagnostic: SESSION-PHASE labels under the honest protocols.

Why this run exists (PLAN.md RQ4, docs/label_rules.md section 3): my
preliminary experiment reached 99.18% fatigued-class recall with labels
that came from the recording session rather than from any fatigue
measurement. RQ4 asks how much of that kind of performance is the label
definition itself. Here the label of a window is decided ONLY by which
block of the session it sits in: M1 block = alert (0), M3 block =
fatigued (1). No rating is consulted. Session labels are a diagnostic,
never ground truth, and nothing in this file feeds the primary results.

Two arms, both on the Amendment-1 retained windows with the locked
band-power features and the locked logistic-regression baseline:

  restricted32  the same windows as the Week 2 primary pool (the 32
                sessions whose subjective rating rose, delta-mental > 0).
                On these windows the session-phase label and the
                subjective label are the same number for every window:
                the subjective construction inherits the block label and
                its only extra act is the delta filter that chose the
                sessions. So this arm must reproduce the Week 2 primary
                numbers exactly, and I check that it does (printed at
                the end, against the stored Week 2 summary tables).
  all36         restricted32 PLUS the 4 non-increasing sessions the
                subjective rules set aside (P07/S01, P07/S02, P10/S03,
                P12/S03, the drift/sensitivity check set, saved
                separately as windows_a1_control). Session-phase labels
                do not care whether the participant's rating rose, so
                these sessions get labels too. The gap between the two
                arms is therefore exactly what the delta filter was
                worth, measured on identical machinery.

Protocols (identical fold definitions to Week 2): within-subject,
pooled leave-one-session-out, and leave-one-participant-out (LOPO).
Cap seeds 0-4, same per-session M1 cap as the primary baseline.

This file is a thin driver: it reuses the frozen fold/model/metric code
in src/models/baseline.py and the frozen feature extractor, and adds
only the fold builder + bootstrap that the closure pack used for LOPO
(copied here in behaviour, so I do not have to import the whole closure
pack to get them). No frozen file is edited.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import cohen_kappa_score

# Environment note (not a pipeline change): the frozen feature code in
# src/features/bandpower.py calls np.trapz, which NumPy 2.5 removed
# after renaming it np.trapezoid in 2.0. The two names are the same
# function (a rename, not a new algorithm), so I alias the old name
# here, in the driver, instead of editing the frozen file. The DROZY
# adapter (src/data/drozy_adapter.py) already uses np.trapezoid for
# the same reason.
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.bandpower import extract_features
from src.models.baseline import (METRIC_COLS, cap_pool, run_folds,
                                 sessionout_folds, summarise, within_folds)

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
RUNS = ROOT / "experiments" / "runs"
TAB = ROOT / "results" / "tables"
DATE = "2026-11-21"
SEEDS = [0, 1, 2, 3, 4]

SPLIT_DESC = {
    "within": "per participant, rotate held-out session (train = other "
              "sessions of the same participant)",
    "session-out": "pooled: hold out one (participant, session); train on "
                   "all remaining sessions of all participants",
    "loso": "hold out ALL sessions of one participant; train on the "
            "other 11 participants",
}


def lopo_folds(df):
    """One fold per participant: test = every window of that participant.
    Same definition as the closure pack's LOPO (Week 2 headline)."""
    keys = []
    for part in df.participant.unique():
        te = (df.participant == part).values
        keys.append((part, ~te, te))
    return keys


def group_boot_ci(values, n_boot=10_000, seed=42):
    """Bootstrap CI over group-level values (participants or sessions),
    resampling GROUPS, never windows. Same construction as the closure
    pack's participant/session bootstraps: 10,000 resamples, rng 42,
    percentile interval on the resampled means."""
    vals = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    means = vals[idx].mean(axis=1)
    return (float(vals.mean()), float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)), len(vals))


def git_commit():
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()


def load_arm(arm):
    """Assemble one arm's window array + index, with session-phase labels.

    The stored `label` column in both A1 files already IS the block
    label (build_windows_a1.py sets label = 0 for M1, 1 for M3 in the
    pool and in the check set alike), so the session-phase label needs
    no new builder, but I re-derive it from the block column here and
    assert the two agree, so the identity is verified, not assumed.
    """
    idx_p = pd.read_csv(PROC / "windows_a1_uncapped_index.csv")
    arr_p = np.load(PROC / "windows_a1_uncapped.npy")
    if arm == "restricted32":
        arr, idx = arr_p, idx_p
    else:  # all36: add the drift/sensitivity check set back in
        idx_c = pd.read_csv(PROC / "windows_a1_control_index.csv")
        arr_c = np.load(PROC / "windows_a1_control.npy")
        assert list(idx_p.columns) == list(idx_c.columns), \
            "pool and check-set index columns differ"
        arr = np.concatenate([arr_p, arr_c])
        idx = pd.concat([idx_p, idx_c], ignore_index=True)
    idx = idx.reset_index(drop=True)
    session_label = (idx.block == "M3").astype(int)
    assert (session_label.values == idx.label.values).all(), \
        "stored label column is not the block label"
    idx["label"] = session_label
    return arr, idx


def write_run(arm, protocol, seed, per_group, preds, skipped, group_col):
    name = f"{DATE}_baseline_{protocol}_sessionlabels-{arm}_s{seed}"
    d = RUNS / name
    d.mkdir(parents=True, exist_ok=True)
    metrics = per_group.copy()
    agg = {c: per_group[c].mean() for c in METRIC_COLS if c in per_group}
    metrics.loc[len(metrics)] = {group_col: "AGGREGATE_mean_over_groups",
                                 **agg}
    metrics.to_csv(d / "metrics.csv", index=False)
    preds.to_csv(d / "predictions.csv", index=False)
    arm_desc = ("32 delta-mental > 0 sessions only (identical windows to "
                "the Week 2 primary pool)"
                if arm == "restricted32" else
                "all 36 sessions (primary pool + the 4 non-increasing "
                "drift/sensitivity check sessions)")
    cfg = {
        "model": "logistic_regression", "C": 1.0,
        "class_weight": "balanced",
        "features": "bandpower_43_amendment1 (see src/features/bandpower.py)",
        "imputation": "training-fold median",
        "scaling": "training-fold StandardScaler",
        "protocol": protocol,
        "labels": "SESSION-PHASE (RQ4 diagnostic, never ground truth): "
                  "M1 block = alert, M3 block = fatigued, assigned by "
                  "block only; no rating consulted",
        "arm": arm, "sessions": arm_desc,
        "cap_seed": seed,
        "data": "windows_a1_uncapped (+ windows_a1_control for all36), "
                "M1 capped per session",
        "git_commit": git_commit(),
    }
    (d / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (d / "notes.md").write_text(f"""# Baseline {protocol} — SESSION-PHASE labels (RQ4), arm {arm}, cap seed {seed}

RQ4 diagnostic run. Labels come from block position only (M1 = alert,
M3 = fatigued); no subjective rating is consulted. Arm: {arm_desc}.
Session labels are a diagnostic for label-definition artefacts
(docs/label_rules.md section 3) and are never presented as fatigue
ground truth.

Split definition: {SPLIT_DESC[protocol]}. A fold counts only if train
and test both contain both classes; skipped folds: {skipped if skipped else 'none'}.

Leakage checklist:
- [x] Normalisation (StandardScaler) fit on training fold only (Pipeline, per fold)
- [x] Median imputation for masked-channel NaNs fit on training fold only
- [x] Class weights computed from training fold only (class_weight='balanced')
- [x] No overlapping windows across splits (tumbling windows; splits at session/participant level)
- [x] No tuning on test data (C=1.0 locked default from how-others-did-it.md section 6)
- [x] Label rules locked (docs/label_rules.md section 3, RQ4 diagnostic) before any training

Environment: local VM (CPU), scikit-learn. Seed = M1-cap subsample seed {seed};
logistic regression itself is deterministic given the data.
""")


def run_arm(arm):
    arr, idx = load_arm(arm)
    feats = extract_features(arr, idx)
    print(f"\n=== arm {arm}: {len(idx)} windows, "
          f"{idx.groupby(['participant', 'session']).ngroups} sessions, "
          f"classes {idx.label.value_counts().to_dict()}")
    fold_fns = {"within": (within_folds, "participant"),
                "session-out": (sessionout_folds, "fold"),
                "loso": (lopo_folds, "participant")}
    aggregates = []
    for protocol, (fold_fn, gcol) in fold_fns.items():
        parts = []
        for seed in SEEDS:
            capped = cap_pool(idx, seed)
            pos = capped.index.to_numpy()
            df = capped.reset_index(drop=True)
            fc = feats.iloc[pos].reset_index(drop=True)
            preds, skipped = run_folds(df, fc, fold_fn(df))
            if protocol == "session-out":
                per = summarise(preds, "fold").rename(
                    columns={"fold": "heldout_session"})
                gname = "heldout_session"
            else:
                per = summarise(preds, "participant")
                gname = "participant"
            write_run(arm, protocol, seed, per, preds, skipped, gname)
            per["seed"] = seed
            parts.append(per)
            print(f"  {arm} {protocol} s{seed}: groups={len(per)} "
                  f"skipped={skipped} mean BA="
                  f"{per.balanced_accuracy.mean():.3f}", flush=True)
        allp = pd.concat(parts)
        summ = allp.groupby(gname)[METRIC_COLS].agg(["mean", "std"])
        summ.columns = [f"{a}_{b}" for a, b in summ.columns]
        summ.reset_index().to_csv(
            TAB / f"baseline_{protocol}_sessionlabels-{arm}_summary.csv",
            index=False)
        # Headline aggregate, Week 2 convention: mean over groups of the
        # per-group cross-seed means, with a group-level bootstrap CI.
        per_group = allp.groupby(gname).balanced_accuracy.mean()
        pt, lo, hi, n = group_boot_ci(per_group.values)
        per_group_mcc = allp.groupby(gname).mcc.mean()
        mt, mlo, mhi, _ = group_boot_ci(per_group_mcc.values)
        per_group_auc = allp.groupby(gname).roc_auc.mean()
        at, alo, ahi, _ = group_boot_ci(per_group_auc.values)
        aggregates.append({
            "arm": arm, "protocol": protocol, "labels": "session-phase",
            "n_groups": n, "balanced_accuracy": pt, "ba_ci95_low": lo,
            "ba_ci95_high": hi, "mcc": mt, "mcc_ci95_low": mlo,
            "mcc_ci95_high": mhi, "roc_auc": at, "auc_ci95_low": alo,
            "auc_ci95_high": ahi,
            "ci_method": "group-level bootstrap (participants for "
                         "within/loso, sessions for session-out), 10000 "
                         "resamples over cross-seed per-group means, "
                         "rng 42"})
        print(f"  {arm} {protocol}: BA={pt:.3f} [{lo:.3f}, {hi:.3f}] "
              f"over {n} groups")
    return aggregates


def label_agreement():
    """Cohen's kappa between session-phase and subjective labels.

    Subjective labels exist only on the restricted pool (the delta
    filter is part of their definition), so kappa is computed on the
    1,053 windows where both label sets are defined. On those windows
    the two constructions assign the same label to every window (both
    are the block label), so kappa must come out at 1.0, computed here
    with sklearn so the number is on record, not just asserted. The
    coverage rows record where the definitions actually differ: the 68
    check-set windows carry a session-phase label but no subjective
    label at all.
    """
    # Read the two label sets independently: the subjective label is the
    # `label` column stored by the A1 window build (block inheritance +
    # delta filter), the session-phase label is re-derived here from the
    # block column alone.
    idx_p = pd.read_csv(PROC / "windows_a1_uncapped_index.csv")
    _, idx_a = load_arm("all36")
    subjective_lab = idx_p.label.values
    session_lab = (idx_p.block == "M3").astype(int).values
    rows = [{
        "comparison": "session-phase vs subjective, shared windows "
                      "(restricted pool, uncapped)",
        "n_windows": len(idx_p),
        "observed_agreement": float((session_lab == subjective_lab).mean()),
        "cohen_kappa": float(cohen_kappa_score(subjective_lab,
                                               session_lab)),
    }, {
        "comparison": "coverage: windows with a session-phase label but "
                      "no subjective label (non-increasing sessions)",
        "n_windows": int(len(idx_a) - len(idx_p)),
        "observed_agreement": np.nan, "cohen_kappa": np.nan,
    }, {
        "comparison": "coverage: all session-labelled windows",
        "n_windows": int(len(idx_a)),
        "observed_agreement": np.nan, "cohen_kappa": np.nan,
    }]
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "rq4_label_agreement.csv", index=False)
    print("\nLabel agreement:")
    print(out.to_string(index=False))
    return out


def main():
    label_agreement()
    aggregates = []
    for arm in ["restricted32", "all36"]:
        aggregates += run_arm(arm)
    agg = pd.DataFrame(aggregates)
    agg.to_csv(TAB / "rq4_aggregates.csv", index=False)
    print("\n=== RQ4 aggregates ===")
    print(agg[["arm", "protocol", "n_groups", "balanced_accuracy",
               "ba_ci95_low", "ba_ci95_high", "mcc", "roc_auc"]]
          .to_string(index=False))
    # Reproduction check: the restricted arm uses identical windows,
    # identical labels and identical code paths as the Week 2 primary,
    # so its aggregates must match the Week 2 headlines (within 0.484,
    # session-out 0.515, LOPO 0.508). Print side by side; the results
    # doc records the comparison.
    ref = {"within": 0.484, "session-out": 0.515, "loso": 0.508}
    print("\nReproduction check (restricted32 vs Week 2 subjective):")
    for _, r in agg[agg.arm == "restricted32"].iterrows():
        print(f"  {r.protocol}: session-phase {r.balanced_accuracy:.3f} "
              f"vs Week 2 subjective {ref[r.protocol]:.3f} "
              f"(diff {r.balanced_accuracy - ref[r.protocol]:+.4f})")


if __name__ == "__main__":
    main()
