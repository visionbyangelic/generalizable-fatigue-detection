"""DROZY baseline runs: feasibility gate first, then leave-one-
participant-out (LOPO) for the two label arms.

The model and the fold machinery are the ones already used for
FatigueSet (src/models/baseline.py, src/evaluation/closure_pack.py);
this file only wires them to the DROZY adapter's outputs. Nothing here
changes how the model is fit: logistic regression, C = 1.0, balanced
class weights, median imputation and scaling fit inside each training
fold.

Subcommands:
  gate   label-free feasibility gate at N = 13 (run before any model
         sees the real labels; fits only permuted labels)
  runs   the four LOPO arm/variant combinations + the two declared
         sensitivities, reading the gate's null distributions for the
         permutation p-values
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.drozy_adapter import FEATURE_NAMES, FZ_FEATURES
from src.evaluation.closure_pack import lopo_folds, participant_boot_ci
from src.models.baseline import METRIC_COLS, make_model, run_folds, summarise

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
RUNS = ROOT / "experiments" / "runs"
TAB = ROOT / "results" / "tables"
DATE = "2026-11-20"

# (arm name, labels file tag, feature set name, feature columns)
COMBOS = [
    ("kss_all", "kss", "all53", FEATURE_NAMES),
    ("kss_fz", "kss", "fz12", FZ_FEATURES),
    ("rt_all", "rt", "all53", FEATURE_NAMES),
    ("rt_fz", "rt", "fz12", FZ_FEATURES),
]
LABELS_TAG = {"kss": "subjective", "rt": "performance"}
FEATURE_DESC = {
    "all53": "DROZY all-channel: log + relative band powers for Fz, Cz, "
             "Pz, C3, C4 (40) + Fz ratios (4) + non-frontal ratios over "
             "valid channels (4) + 5 validity indicators = 53 features "
             "(src/data/drozy_adapter.py)",
    "fz12": "DROZY Fz-only: Fz log + relative band powers (8) + Fz "
            "ratios (4) = 12 features",
}


def git_commit():
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()


def load_all():
    idx = pd.read_csv(PROC / "drozy_windows_index.csv")
    feats = pd.read_csv(PROC / "drozy_features.csv")
    tables = {arm: pd.read_csv(TAB / f"drozy_labels_{arm}.csv")
              for arm in ("kss", "rt")}
    return idx, feats, tables


def arm_frame(idx, feats, table, cols, drop=()):
    """Windows for one label arm: each included participant contributes
    their alert session (label 0) and fatigued session (label 1).
    Returns (df, feats) positionally aligned, df with the columns the
    fold machinery needs (participant, session, label)."""
    parts, fparts = [], []
    for r in table[table.included].itertuples():
        if r.participant in drop:
            continue
        for test, lab in ((int(r.alert_test), 0), (int(r.fatigued_test), 1)):
            m = (idx.participant == r.participant) & (idx.test == test)
            sel = idx[m].copy()
            sel["label"] = lab
            parts.append(sel)
            fparts.append(feats.loc[sel.index, cols])
    df = pd.concat(parts).reset_index(drop=True)
    df["session"] = "S" + df.test.astype(str)
    return df, pd.concat(fparts).reset_index(drop=True)


def lopo_statistic(df, feats, y=None):
    """Mean over participants of per-participant balanced accuracy
    under LOPO. With y given, df's labels are replaced first (used for
    permutations); the folds themselves never depend on labels."""
    if y is not None:
        df = df.copy()
        df["label"] = y
    preds, skipped = run_folds(df, feats, lopo_folds(df))
    per = summarise(preds, "participant")
    return per.balanced_accuracy.mean(), per


def swapped_labels(df, rng):
    """One null labelling: within each participant, swap the two
    session labels or keep them, coin flip per participant.

    DROZY labels live at session level (every window of a session has
    the same label), so the within-session circular shifts used for
    FatigueSet would change nothing here. Swapping whole session
    labels inside the participant is the same idea one level up: the
    label blocks are permuted, window order and autocorrelation inside
    a session are untouched, and each participant keeps exactly one
    alert and one fatigued session."""
    y = df.label.values.copy()
    for _, g in df.groupby("participant"):
        if rng.random() < 0.5:
            y[g.index.to_numpy()] = 1 - y[g.index.to_numpy()]
    return y


# ------------------------------------------------------------------ gate
def gate():
    idx, feats, tables = load_all()
    z = stats.norm.ppf(0.95) + stats.norm.ppf(0.90)  # 2.9265, as in Week 2
    null_rows, input_rows = [], []
    for name, arm, fset, cols in COMBOS:
        df, fc = arm_frame(idx, feats, tables[arm], cols)
        variants = [("swap_A", 200, 20_000)]
        if name == "kss_all":
            variants.append(("swap_B", 100, 30_000))
        for vname, n_perm, seed0 in variants:
            stats_p, sd_parts = [], []
            for p in range(n_perm):
                rng = np.random.default_rng(seed0 + p)
                yp = swapped_labels(df, rng)
                stat, per = lopo_statistic(df, fc, y=yp)
                stats_p.append(stat)
                sd_parts.append(per.balanced_accuracy.std(ddof=1))
                null_rows.append({"combo": name, "variant": vname,
                                  "perm": p, "ba_participant_mean": stat,
                                  "per_participant_ba_sd":
                                      per.balanced_accuracy.std(ddof=1)})
            sd = float(np.std(stats_p, ddof=1))
            input_rows.append({
                "combo": name, "basis": f"perm_null_{vname}_"
                "participant_mean_statistic", "se": sd,
                "delta_min_90pct_power": z * sd,
                "definition": f"SD of the LOPO participant-mean BA under "
                              f"{n_perm} within-participant session-label "
                              f"swaps (rng {seed0}+p)"})
            print(f"gate {name} {vname}: null SD {sd:.4f} -> "
                  f"delta_min {z * sd:.3f}", flush=True)
            if vname == "swap_A":
                # analytical basis on the same null: average spread of
                # per-participant BA across permutations, over sqrt(N)
                se_an = float(np.mean(sd_parts)) / np.sqrt(13)
                input_rows.append({
                    "combo": name, "basis": "analytical_participant_N13_"
                    "from_null", "se": se_an,
                    "delta_min_90pct_power": z * se_an,
                    "definition": "mean per-participant BA SD across the "
                                  "swap_A permutations / sqrt(13)"})
    pd.DataFrame(null_rows).to_csv(TAB / "drozy_gate_permutation_nulls.csv",
                                   index=False)
    inputs = pd.DataFrame(input_rows)
    inputs.to_csv(TAB / "drozy_gate_inputs.csv", index=False)
    print(inputs.to_string(index=False), flush=True)


# ------------------------------------------------------------------ runs
def write_run(name, combo_desc, per, preds, skipped, extra_cfg, notes):
    d = RUNS / name
    d.mkdir(parents=True, exist_ok=True)
    metrics = per.copy()
    agg = {c: per[c].mean() for c in METRIC_COLS if c in per}
    metrics.loc[len(metrics)] = {"participant": "AGGREGATE_mean_over_groups",
                                 **agg}
    metrics.to_csv(d / "metrics.csv", index=False)
    preds.to_csv(d / "predictions.csv", index=False)
    cfg = {"model": "logistic_regression", "C": 1.0,
           "class_weight": "balanced",
           "imputation": "training-fold median",
           "scaling": "training-fold StandardScaler",
           "protocol": "loso (leave-one-participant-out)",
           "git_commit": git_commit(), **extra_cfg}
    (d / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (d / "notes.md").write_text(notes)
    return agg


def run_combo(idx, feats, tables, name, arm, fset, cols, nulls,
              drop=(), suffix=""):
    df, fc = arm_frame(idx, feats, tables[arm], cols, drop=drop)
    preds, skipped = run_folds(df, fc, lopo_folds(df))
    per = summarise(preds, "participant")
    pt, lo, hi, n = participant_boot_ci(per.balanced_accuracy.values)
    ptm, lom, him, _ = participant_boot_ci(per.mcc.values)
    pta, loa, hia, _ = participant_boot_ci(per.roc_auc.values)
    row = {"combo": name + suffix, "arm": arm, "feature_set": fset,
           "n_participants": n, "balanced_accuracy": pt,
           "ba_ci95_low": lo, "ba_ci95_high": hi, "mcc": ptm,
           "mcc_ci95_low": lom, "mcc_ci95_high": him, "roc_auc": pta,
           "auc_ci95_low": loa, "auc_ci95_high": hia}
    pval = None
    if nulls is not None:
        nv = nulls[(nulls.combo == name) & (nulls.variant == "swap_A")] \
            .ba_participant_mean.values
        pval = (1 + (nv >= pt).sum()) / (len(nv) + 1)
        row.update({"perm_null_mean": float(nv.mean()),
                    "perm_null_sd": float(nv.std(ddof=1)),
                    "perm_null_q025": float(np.percentile(nv, 2.5)),
                    "perm_null_q975": float(np.percentile(nv, 97.5)),
                    "p_value_ge_real": float(pval)})
    folder = (f"{DATE}_baseline_loso_{LABELS_TAG[arm]}_drozy-{name}"
              f"{suffix}_s0")
    notes = f"""# {folder}

Split definition: leave-one-participant-out. Test = all windows of one
participant (their alert + fatigued sessions for the {arm.upper()} label
arm); train = the other participants. {n} folds. A fold counts only if
train and test both contain both classes; skipped folds: {skipped if skipped else 'none'}.
{f'Participants dropped for this sensitivity run: {sorted(drop)}. ' if drop else ''}Single deterministic pass: the DROZY design has no subsampling step
(no per-session cap; each participant contributes exactly one alert and
one fatigued session), so there is no seed for the data to vary with,
and logistic regression is deterministic given the data. The folder
suffix s0 marks that single pass, the same convention as the Week 2
precision variants.

Result: balanced accuracy {pt:.3f}, 95% CI [{lo:.3f}, {hi:.3f}] by
participant bootstrap (10,000 resamples over per-participant values,
rng 42).{' Permutation null (200 within-participant session-label swaps): mean ' + format(row['perm_null_mean'], '.3f') + ', SD ' + format(row['perm_null_sd'], '.3f') + ', p = ' + format(pval, '.3f') + '.' if pval is not None else ' No permutation null for sensitivity runs.'}

Leakage checklist:
- [x] Normalisation (StandardScaler) fit on training fold only (Pipeline, per fold)
- [x] Median imputation for masked-channel NaNs fit on training fold only
- [x] Class weights computed from training fold only (class_weight='balanced')
- [x] No overlapping windows across splits (tumbling windows; splits at participant level)
- [x] Held-out participant contributes nothing to fitting, imputation, scaling or tuning
- [x] No tuning on test data (C=1.0 locked default)
- [x] Label rules fixed in docs/dataset2_preregistration.md incl. Amendment 1 before any training

Environment: local VM (CPU), scikit-learn 1.9.1, MNE 1.13.2.
"""
    write_run(folder, name, per, preds, skipped,
              {"features": FEATURE_DESC[fset],
               "labels": f"{LABELS_TAG[arm]} (DROZY {arm.upper()} arm, "
                         f"docs/dataset2_preregistration.md)"},
              notes)
    per_out = per.copy()
    per_out["combo"] = name + suffix
    per_out.to_csv(TAB / f"baseline_loso_drozy_{name}{suffix}_summary.csv",
                   index=False)
    print(f"{name}{suffix}: BA {pt:.3f} [{lo:.3f}, {hi:.3f}] "
          f"AUC {pta:.3f} MCC {ptm:.3f} n={n}"
          + (f" null p={pval:.3f}" if pval is not None else ""), flush=True)
    return row


def runs():
    idx, feats, tables = load_all()
    nulls = pd.read_csv(TAB / "drozy_gate_permutation_nulls.csv")
    rows = []
    for name, arm, fset, cols in COMBOS:
        rows.append(run_combo(idx, feats, tables, name, arm, fset, cols,
                              nulls))
    # declared sensitivities, KSS arm only: S1 drops Participant 11
    # (the tie-break participant), S2 drops Participant 7 (the
    # two-session participant). Robustness checks, not new primaries.
    for suffix, drop in (("-sensS1", ("P11",)), ("-sensS2", ("P07",))):
        for name, arm, fset, cols in COMBOS:
            if arm != "kss":
                continue
            rows.append(run_combo(idx, feats, tables, name, arm, fset,
                                  cols, None, drop=drop, suffix=suffix))
    pd.DataFrame(rows).to_csv(TAB / "drozy_lopo_aggregate.csv", index=False)
    print("DONE", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "runs"
    if cmd == "gate":
        gate()
    else:
        runs()
