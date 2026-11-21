"""RQ4 diagnostic, day-level arm (POST HOC, added 21 Nov 2026).

Why this run exists: the block-phase arm of RQ4
(src/evaluation/rq4_session_labels.py) prices the rating filter, but
it cannot speak to my preliminary experiment, because the preliminary
did not label blocks. It labelled whole recording sessions: Session 01
as alert, Session 03 as fatigued, for one participant, with a split
inside the same recordings, and reported 99.18% fatigued-class recall.
The day-level form of session labels (docs/label_rules.md section 3)
is that design. This driver prices it, post hoc and labelled as such,
in three runs:

  1. LOPO with day-level labels. The honest estimate: can the model
     tell a held-out participant's earliest recorded session from
     their latest, having never seen that participant?
  2. A shared-session random-window split with day-level labels. The
     leaky member of the same design family: windows are pooled
     across participants and split at random, so every session appears
     in both train and test. This is deliberate. It is the split
     structure that lets a model harvest session-specific patterns
     instead of fatigue, and running it next to run 1 prices the
     split, with the labels held fixed.
  3. A session-identification probe. Within each participant, can the
     same 43 features tell which of the three sessions a window came
     from, under a random-window split? If sessions are easy to
     identify from the features, the fingerprinting mechanism that
     run 2 rewards is demonstrably available to a model.

Day-level label definition (exact): for each participant, every
retained Amendment-1 window in the earliest recorded session (S01)
is labelled alert (0) and every retained window in the latest
recorded session (S03) is labelled fatigued (1). The middle session
(S02) is excluded entirely. Session numbering follows recording
order. "Retained windows" means the Amendment-1 window files (the
primary pool plus the drift/sensitivity check set): those files hold
M1- and M3-block windows only, because M2 windows were set aside
upstream by the primary construction, before any label question. No
M1 cap is applied: the cap is a block-label construct (it balances
M1 against M3 inside a session), and under day-level labels there is
no block distinction left to balance.

Participants: 11 of 12. P05 is excluded from the whole arm, from
training as well as from test folds, because its latest session
retained no windows at all (the known P05 frontal-contact failure),
so it has no day-level contrast; this also keeps the participant set
identical to the primary LOPO. Its middle session is excluded by the
definition anyway and its earliest session's 5 windows go unused
here; the exclusion and the reason are stated in every run note.

Seed conventions (stated because they differ from the primary):
  - Run 1 has no stochastic element. The only randomness in the
    primary baseline was the cap subsample, and there is no cap here;
    the logistic regression is deterministic given the data. The run
    folder carries s0 by the project's naming convention only.
  - Runs 2 and 3 are run at split seeds 0 to 4 (the randomness being
    priced is the split itself), and reported as means over seeds
    with the per-seed values kept.

This file reuses the frozen feature extractor, the frozen model
pipeline and the frozen metric code. No frozen file is edited.
Session labels are a diagnostic, never ground truth, and nothing in
this file feeds the primary results.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Environment note (not a pipeline change): the frozen feature code
# in src/features/bandpower.py calls np.trapz, which NumPy 2.5
# removed after renaming it np.trapezoid in 2.0. The two names are
# the same function (a rename, not a new algorithm), so I alias the
# old name here, in the driver, instead of editing the frozen file.
# The same alias is used by the other post-restart drivers.
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.metrics import binary_metrics
from src.features.bandpower import extract_features
from src.models.baseline import METRIC_COLS, make_model, run_folds, summarise

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
RUNS = ROOT / "experiments" / "runs"
TAB = ROOT / "results" / "tables"
DATE = "2026-11-21"
SPLIT_SEEDS = [0, 1, 2, 3, 4]
EXCLUDED_PARTICIPANT = "P05"  # latest session retained no windows


def group_boot_ci(values, n_boot=10_000, seed=42):
    """Bootstrap CI over group-level values (participants), resampling
    groups, never windows. Same construction as the closure pack and
    the block-phase RQ4 driver: 10,000 resamples, rng 42, percentile
    interval on the resampled means."""
    vals = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    means = vals[idx].mean(axis=1)
    return (float(vals.mean()), float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)), len(vals))


def git_commit():
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()


def load_retained():
    """All retained Amendment-1 windows (primary pool + check set)."""
    idx_p = pd.read_csv(PROC / "windows_a1_uncapped_index.csv")
    arr_p = np.load(PROC / "windows_a1_uncapped.npy")
    idx_c = pd.read_csv(PROC / "windows_a1_control_index.csv")
    arr_c = np.load(PROC / "windows_a1_control.npy")
    assert list(idx_p.columns) == list(idx_c.columns), \
        "pool and check-set index columns differ"
    arr = np.concatenate([arr_p, arr_c])
    idx = pd.concat([idx_p, idx_c], ignore_index=True).reset_index(drop=True)
    return arr, idx


def day_level_frame(idx):
    """The day-level analysis frame: S01/S03 windows of the 11
    participants who have a day-level contrast, with day labels.
    Returns row positions into the full retained set plus the frame."""
    keep = idx.session.isin(["S01", "S03"]) \
        & (idx.participant != EXCLUDED_PARTICIPANT)
    pos = np.where(keep.values)[0]
    df = idx.iloc[pos].reset_index(drop=True).copy()
    df["label"] = (df.session == "S03").astype(int)
    return pos, df


def lopo_folds(df):
    """One fold per participant: test = every window of that
    participant. Same definition as the closure pack's LOPO."""
    keys = []
    for part in df.participant.unique():
        te = (df.participant == part).values
        keys.append((part, ~te, te))
    return keys


def base_config(protocol, labels, seed_note, seed):
    return {
        "model": "logistic_regression", "C": 1.0,
        "class_weight": "balanced",
        "features": "bandpower_43_amendment1 (see src/features/bandpower.py)",
        "imputation": "training-fold median",
        "scaling": "training-fold StandardScaler",
        "protocol": protocol,
        "labels": labels,
        "seed": seed_note,
        "cap_seed": seed,
        "data": "windows_a1_uncapped + windows_a1_control (all retained "
                "Amendment-1 windows), NO M1 cap (cap is a block-label "
                "construct)",
        "git_commit": git_commit(),
    }


DAY_LABELS = ("DAY-LEVEL session labels (RQ4 diagnostic, POST HOC, never "
              "ground truth): whole earliest recorded session (S01) = "
              "alert, whole latest recorded session (S03) = fatigued, "
              "middle session excluded; P05 excluded entirely (latest "
              "session retained no windows)")


def run_lopo(feats, idx):
    pos, df = day_level_frame(idx)
    fc = feats.iloc[pos].reset_index(drop=True)
    print(f"LOPO day-level: {len(df)} windows, "
          f"classes {df.label.value_counts().to_dict()}, "
          f"participants {df.participant.nunique()}")
    preds, skipped = run_folds(df, fc, lopo_folds(df))
    per = summarise(preds, "participant")
    name = f"{DATE}_baseline_loso_daylevel_s0"
    d = RUNS / name
    d.mkdir(parents=True, exist_ok=True)
    metrics = per.copy()
    agg = {c: per[c].mean() for c in METRIC_COLS if c in per}
    metrics.loc[len(metrics)] = {"participant": "AGGREGATE_mean_over_groups",
                                 **agg}
    metrics.to_csv(d / "metrics.csv", index=False)
    preds.to_csv(d / "predictions.csv", index=False)
    cfg = base_config(
        "loso (leave-one-participant-out)", DAY_LABELS,
        "s0 by naming convention only: no stochastic element (no cap; "
        "logistic regression is deterministic given the data)", 0)
    (d / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (d / "notes.md").write_text(f"""# Baseline LOPO, DAY-LEVEL session labels (RQ4, POST HOC), s0

RQ4 diagnostic run, post hoc arm. Labels are whole-session: earliest
recorded session (S01) = alert, latest recorded session (S03) =
fatigued, middle session excluded, every retained window in the
session carrying the session's label. P05 is excluded entirely: its
latest session retained no windows, so it has no day-level contrast.
Session labels are a diagnostic for label-definition artefacts
(docs/label_rules.md section 3) and are never presented as fatigue
ground truth.

Split definition: hold out ALL day-labelled windows of one
participant; train on the other 10 participants. A fold counts only
if train and test both contain both classes; skipped folds:
{skipped if skipped else 'none'}.

Seed convention: this run has no stochastic element. The primary
baseline's seeds were cap-subsample seeds; there is no cap under
day-level labels, and the logistic regression is deterministic given
the data. The s0 suffix is the project's naming convention only.

Leakage checklist:
- [x] Normalisation (StandardScaler) fit on training fold only (Pipeline, per fold)
- [x] Median imputation for masked-channel NaNs fit on training fold only
- [x] Class weights computed from training fold only (class_weight='balanced')
- [x] No overlapping windows across splits (tumbling windows; split at participant level)
- [x] No tuning on test data (C=1.0 locked default from how-others-did-it.md section 6)
- [x] Label definition fixed in the driver docstring before the run (post hoc arm, labelled as such)

Environment: local VM (CPU), scikit-learn.
""")
    # `per` already carries n, n_alert and n_fatigued per participant
    # from binary_metrics; the summary table is those rows as they are.
    per_out = per.sort_values("participant").reset_index(drop=True)
    per_out.to_csv(TAB / "baseline_loso_daylevel_summary.csv", index=False)
    pt, lo, hi, n = group_boot_ci(per.balanced_accuracy.values)
    print(f"  LOPO day-level: BA={pt:.3f} [{lo:.3f}, {hi:.3f}] over "
          f"{n} participants; skipped={skipped}")
    return {"run": "lopo", "balanced_accuracy": pt, "ba_ci95_low": lo,
            "ba_ci95_high": hi, "n_groups": n,
            "recall_fatigued": float(per.recall_fatigued.mean()),
            "mcc": float(per.mcc.mean()),
            "roc_auc": float(per.roc_auc.mean())}


def run_leaky(feats, idx):
    """Shared-session random-window split, day-level labels. The split
    is the leak under test; preprocessing stays fold-fitted so the
    split is the only leak, as in the Week 2 leakage-gap demo."""
    pos, df = day_level_frame(idx)
    X = feats.iloc[pos].reset_index(drop=True).values
    y = df.label.values
    rows = []
    for seed in SPLIT_SEEDS:
        skf = StratifiedKFold(n_splits=5, shuffle=True,
                              random_state=seed)
        fold_rows, all_true, all_pred, all_proba = [], [], [], []
        for k, (tr, te) in enumerate(skf.split(X, y)):
            m = make_model()
            m.fit(X[tr], y[tr])
            pred = m.predict(X[te])
            proba = m.predict_proba(X[te])[:, 1]
            fm = binary_metrics(y[te], pred, proba)
            fm["fold"] = k
            fold_rows.append(fm)
            all_true.append(y[te]); all_pred.append(pred)
            all_proba.append(proba)
        yt = np.concatenate(all_true); yp = np.concatenate(all_pred)
        ypr = np.concatenate(all_proba)
        pooled = binary_metrics(yt, yp, ypr)
        name = f"{DATE}_baseline_leaky-random_daylevel_s{seed}"
        d = RUNS / name
        d.mkdir(parents=True, exist_ok=True)
        metrics = pd.DataFrame(fold_rows)
        agg = {c: metrics[c].mean() for c in METRIC_COLS if c in metrics}
        metrics.loc[len(metrics)] = {"fold": "AGGREGATE_mean_over_folds",
                                     **agg}
        metrics.to_csv(d / "metrics.csv", index=False)
        pd.DataFrame({"participant": df.participant.values,
                      "session": df.session.values,
                      "y_true": yt, "y_pred": yp,
                      "y_proba": ypr}).to_csv(d / "predictions.csv",
                                              index=False)
        cfg = base_config(
            "leaky random window split (stratified 5-fold, pooled "
            "across participants)", DAY_LABELS,
            f"split shuffle seed {seed} (the split is the stochastic "
            "element under test)", seed)
        (d / "config.yaml").write_text(yaml.safe_dump(cfg,
                                                      sort_keys=False))
        (d / "notes.md").write_text(f"""# Baseline, DAY-LEVEL labels, LEAKY random-window split (RQ4, POST HOC), seed {seed}

RQ4 diagnostic run, post hoc arm. Same day-level labels, same
features, same model as the day-level LOPO run; only the split
changes. Windows from all 11 participants are pooled and split at
random into 5 stratified folds, so windows of every session appear
in both train and test.

THIS SPLIT IS DELIBERATELY LEAKY. It is run because it is the design
family of the preliminary experiment (whole-session labels, train
and test sharing sessions): it lets the model use session-specific
patterns (drift, contact quality, day effects) as if they were
fatigue. The number it produces is an upper bound manufactured by
the design, not a performance estimate, and it is never reported as
one. Preprocessing (imputation, scaling, class weights) is still fit
on training folds only, so the split is the single leak, matching
the Week 2 leakage-gap demo's discipline.

Leakage checklist (honest version):
- [x] Normalisation (StandardScaler) fit on training fold only (Pipeline, per fold)
- [x] Median imputation fit on training fold only
- [x] Class weights computed from training fold only (class_weight='balanced')
- [ ] NO session sharing between train and test: VIOLATED BY DESIGN (the violation is the object of the run)
- [x] No tuning on test data (C=1.0 locked default)
- [x] Label definition fixed in the driver docstring before the run (post hoc arm, labelled as such)

Environment: local VM (CPU), scikit-learn. Seed = split shuffle
seed {seed}.
""")
        rows.append({"seed": seed, **{c: pooled[c] for c in
                                      ["balanced_accuracy", "macro_f1",
                                       "roc_auc", "mcc", "recall_alert",
                                       "recall_fatigued", "n"]}})
        print(f"  leaky s{seed}: pooled BA={pooled['balanced_accuracy']:.3f} "
              f"recall_fatigued={pooled['recall_fatigued']:.3f}")
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "baseline_leaky-random_daylevel_summary.csv",
               index=False)
    return out


def make_multiclass_model():
    """Same discipline as the frozen binary pipeline (training-fold
    median imputation and scaling, logistic regression C=1.0,
    balanced class weights); the estimator handles the 3-class
    session target multinomially. Written here rather than reusing
    make_model so the probe's target encoding stays explicit."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, class_weight="balanced",
                                  max_iter=1000)),
    ])


def run_probe(feats, idx):
    """Session-identification probe: within a participant, classify
    which session (S01/S02/S03) a window came from, random-window
    split. A participant is probed only if every session has at
    least 2 retained windows (a session with 0 or 1 windows cannot
    be both trained on and tested); folds per participant are
    min(5, smallest session count) so the stratification stays
    valid, and both facts are reported per participant."""
    sessions = ["S01", "S02", "S03"]
    y_map = {s: i for i, s in enumerate(sessions)}
    per_participant = []
    pred_frames = {seed: [] for seed in SPLIT_SEEDS}
    seed_part_rows = {seed: [] for seed in SPLIT_SEEDS}
    for part, g in idx.groupby("participant"):
        counts = g.session.value_counts()
        have = [s for s in sessions if counts.get(s, 0) > 0]
        min_n = min(counts.get(s, 0) for s in sessions)
        if len(have) < 3 or min_n < 2:
            print(f"  probe: {part} skipped (sessions with windows: "
                  f"{have}, smallest session count {min_n})")
            per_participant.append({
                "participant": part, "probed": False,
                "reason": ("single session with retained windows"
                           if len(have) < 2 else
                           "a session has fewer than 2 retained windows"),
                "n_S01": int(counts.get("S01", 0)),
                "n_S02": int(counts.get("S02", 0)),
                "n_S03": int(counts.get("S03", 0))})
            continue
        n_splits = int(min(5, min_n))
        pos = g.index.to_numpy()
        X = feats.iloc[pos].values
        y = g.session.map(y_map).values
        seed_rows = []
        for seed in SPLIT_SEEDS:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                  random_state=seed)
            yt_all, yp_all = [], []
            for tr, te in skf.split(X, y):
                m = make_multiclass_model()
                m.fit(X[tr], y[tr])
                yp_all.append(m.predict(X[te]))
                yt_all.append(y[te])
                pred_frames[seed].append(pd.DataFrame({
                    "participant": part, "session": g.session.values[te],
                    "y_true": y[te], "y_pred": yp_all[-1]}))
            yt = np.concatenate(yt_all); yp = np.concatenate(yp_all)
            ba_s = balanced_accuracy_score(yt, yp)
            acc_s = accuracy_score(yt, yp)
            seed_rows.append({"seed": seed, "balanced_accuracy": ba_s,
                              "accuracy": acc_s})
            seed_part_rows[seed].append({
                "participant": part, "balanced_accuracy": ba_s,
                "accuracy": acc_s, "n_S01": int(counts["S01"]),
                "n_S02": int(counts["S02"]), "n_S03": int(counts["S03"])})
        sr = pd.DataFrame(seed_rows)
        per_participant.append({
            "participant": part, "probed": True, "reason": "",
            "n_S01": int(counts["S01"]), "n_S02": int(counts["S02"]),
            "n_S03": int(counts["S03"]), "n_splits": n_splits,
            "balanced_accuracy": sr.balanced_accuracy.mean(),
            "balanced_accuracy_std": sr.balanced_accuracy.std(),
            "accuracy": sr.accuracy.mean(),
            "majority_rate": float(counts.max() / counts.sum()),
            "chance_balanced_accuracy": 1 / 3})
        print(f"  probe {part}: BA={sr.balanced_accuracy.mean():.3f} "
              f"(acc {sr.accuracy.mean():.3f}, majority "
              f"{counts.max() / counts.sum():.3f}, folds {n_splits})")
    for seed in SPLIT_SEEDS:
        name = f"{DATE}_probe_sessionid_randomwindow_s{seed}"
        d = RUNS / name
        d.mkdir(parents=True, exist_ok=True)
        pd.concat(pred_frames[seed]).to_csv(d / "predictions.csv",
                                            index=False)
        metrics = pd.DataFrame(seed_part_rows[seed])
        metrics.loc[len(metrics)] = {
            "participant": "AGGREGATE_mean_over_participants",
            "balanced_accuracy": metrics.balanced_accuracy.mean(),
            "accuracy": metrics.accuracy.mean(),
            "n_S01": np.nan, "n_S02": np.nan, "n_S03": np.nan}
        metrics.to_csv(d / "metrics.csv", index=False)
        cfg = {
            "model": "logistic_regression_multiclass", "C": 1.0,
            "class_weight": "balanced",
            "features": "bandpower_43_amendment1 (see "
                        "src/features/bandpower.py)",
            "imputation": "training-fold median",
            "scaling": "training-fold StandardScaler",
            "protocol": "within-participant stratified random-window "
                        "CV, folds = min(5, smallest session count)",
            "labels": "PROBE target: session identity (S01/S02/S03) "
                      "within participant; not a fatigue label",
            "seed": f"split shuffle seed {seed}",
            "data": "windows_a1_uncapped + windows_a1_control (all "
                    "retained Amendment-1 windows, all 3 sessions)",
            "git_commit": git_commit(),
        }
        (d / "config.yaml").write_text(yaml.safe_dump(cfg,
                                                      sort_keys=False))
        (d / "notes.md").write_text(f"""# Session-identification probe, random-window split (RQ4, POST HOC), seed {seed}

RQ4 diagnostic probe, post hoc arm. Question: from the same 43
band-power features, can a model tell which recording session a
window came from, within the same participant? If sessions carry a
strong feature fingerprint, any evaluation that shares sessions
between train and test can reward recognising the session instead
of the labelled state.

Split definition: per participant, stratified random-window
cross-validation over that participant's retained windows (all
three sessions), folds = min(5, smallest session count) so every
session appears in every training fold. Participants are probed
only if all three sessions have at least 2 retained windows: P05
(one session with windows) and P10 (latest session has a single
window) are excluded and the exclusion is recorded in the summary
table. The random-window split is again deliberate: the probe
measures how much session information the features carry when the
model is allowed to see the same sessions, which is the quantity
that matters for the fingerprinting argument.

Leakage checklist (honest version):
- [x] Normalisation (StandardScaler) fit on training fold only (Pipeline, per fold)
- [x] Median imputation fit on training fold only
- [x] Class weights computed from training fold only (class_weight='balanced')
- [ ] NO session sharing between train and test: VIOLATED BY DESIGN (the probe's object is session information under session sharing)
- [x] No tuning on test data (C=1.0 locked default)

Environment: local VM (CPU), scikit-learn. Seed = split shuffle
seed {seed}.
""")
    out = pd.DataFrame(per_participant)
    out.to_csv(TAB / "probe_sessionid_daylevel_summary.csv", index=False)
    probed = out[out.probed]
    pt, lo, hi, n = group_boot_ci(probed.balanced_accuracy.values)
    print(f"  probe mean balanced accuracy={pt:.3f} [{lo:.3f}, {hi:.3f}] "
          f"over {n} participants (chance 0.333)")
    return {"mean_ba": pt, "lo": lo, "hi": hi, "n": n}


def main():
    arr, idx = load_retained()
    feats = extract_features(arr, idx)
    print(f"retained windows: {len(idx)}")
    lopo = run_lopo(feats, idx)
    leaky = run_leaky(feats, idx)
    probe = run_probe(feats, idx)
    print("\n=== day-level arm summary ===")
    print(f"LOPO: BA={lopo['balanced_accuracy']:.3f} "
          f"[{lopo['ba_ci95_low']:.3f}, {lopo['ba_ci95_high']:.3f}]")
    print(f"Leaky random split: BA mean={leaky.balanced_accuracy.mean():.3f} "
          f"(range {leaky.balanced_accuracy.min():.3f} to "
          f"{leaky.balanced_accuracy.max():.3f}), fatigued recall mean="
          f"{leaky.recall_fatigued.mean():.3f} (range "
          f"{leaky.recall_fatigued.min():.3f} to "
          f"{leaky.recall_fatigued.max():.3f})")
    print(f"Session-ID probe: mean balanced accuracy={probe['mean_ba']:.3f} "
          f"[{probe['lo']:.3f}, {probe['hi']:.3f}] (chance 0.333)")


if __name__ == "__main__":
    main()
