"""RQ4 2x2 preliminary-replication arm (POST HOC, pre-specified by the
panel, 21 Nov 2026).

Why this run exists: the day-level arm of RQ4 priced whole-session
labels, a leaky split and the session fingerprint separately, but it
could not say which of the preliminary experiment's remaining
ingredients carried its 99.18% fatigued-class recall, because the
preliminary differed from the day-level leaky run in three ways at
once: one participant, a chronological split inside the recordings,
and a 1D-CNN on raw EEG. This arm crosses the two priced factors on
the preliminary's own participant: model (band-power logistic
regression vs the frozen EEGNet) by split (chronological vs random
window), under day-level labels, on Subject 01 only. Subject 01 is
the preliminary's participant, chosen for that reason and never for
performance (panel specification, review 24).

Data (important, and stated wherever the numbers are quoted): the
windows are THIS pipeline's retained Amendment-1 windows for P01
(primary pool plus check set, M1/M3 blocks only, no cap), not the
preliminary's own window set. The preliminary epoched the whole
sessions with 4 s windows at a 2 s stride (50% overlap) and had
1,353 windows; the retained set here is 24 windows (Session 01: 22,
Session 03: 2), because most of P01's Session 03 did not survive
Amendment-1 retention. So this arm prices model x split on the
windows this pipeline kept; it does not recreate the preliminary's
data volume, and the fatigued test class is a single window in every
split, which makes fatigued recall a 0-or-1 quantity here. The
counts are printed in every run note and in the summary table so
the numbers cannot be read without them.

Splits (exact rules):
  chronological  within each session, windows are ordered by
                 start_sample; the first floor(0.7 * n) are train,
                 the rest are test. No purge gap: the windows are
                 non-overlapping, so the cut shares no samples; the
                 temporal adjacency across the cut is part of the
                 design being priced. (The preliminary's own audit
                 notebook used an 80/20 chronological cut per its
                 repository README; this arm uses the panel-specified
                 70/30 and says so.)
  random         stratified by class over the pooled windows: per
                 class, n_test = max(1, round(0.3 * n_class)); the
                 class's windows are shuffled with the split seed
                 and the first n_test go to test. Split seeds 0 to 4.
                 The same split is used for the LR and CNN cells, so
                 the model comparison is paired by split.
  chronological-reversed (LR sensitivity only): the last
                 floor(0.7 * n) windows of each session are train,
                 the first windows are test. Checks that the
                 chronological result is not an artefact of which
                 end of the session is tested.

CNN training: the frozen EEGNet configuration from src/models/cnn.py
(architecture, Adam lr 1e-3, batch 64, dropout 0.25, balanced class
weights from the training windows, per-channel normalisation from
the training windows, 6-channel mask-indicator input). One forced
adaptation, stated here and in every CNN run note: cnn.py early-stops
on a validation split carved from the training windows, but no
both-class carve exists in this arm, because the fatigued class has
exactly ONE training window in every split (train counts: 15 alert,
1 fatigued). Any carve either removes that window from training
(leaving a single-class training set) or leaves validation
single-class. The CNN cells therefore train on all training windows
for a fixed budget of 100 epochs (the frozen maximum) with no early
stopping. Model seeds 0, 1, 2 per cell.

Pre-declared reading (also written into docs/rq4_session_labels.md
before the results): if the chronological CNN cell approaches the
preliminary's ~0.99 fatigued recall while the chronological LR cell
does not, the gap prices as model + split; if no cell approaches
0.99, the preliminary number is not reproducible under this pipeline
and the write-up says so.

Leakage stance: every cell shares both sessions (and the
participant) between train and test BY DESIGN; this arm prices a
leaky design family and none of its numbers is a performance
estimate. Preprocessing statistics (feature scaling for LR,
normalisation and class weights for the CNN) are still fit on the
training windows only, so the split is the only leak, as in the
Week 2 leakage-gap demo.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.metrics import binary_metrics
from src.evaluation.rq4_day_level import load_retained
from src.features.bandpower import extract_features
from src.models.baseline import make_model
from src.models.cnn import BATCH, LR, MAX_EPOCHS, EEGNet, build_inputs

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "experiments" / "runs"
TAB = ROOT / "results" / "tables"
DATE = "2026-11-21"
PARTICIPANT = "P01"
SPLIT_SEEDS = [0, 1, 2, 3, 4]
MODEL_SEEDS = [0, 1, 2]
TRAIN_FRAC = 0.7


def git_commit():
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()


def p01_frame(idx):
    """P01's day-level frame: retained windows of S01 (alert) and
    S03 (fatigued), each session sorted by start_sample (recording
    order). Returns row positions into the retained array + frame."""
    keep = (idx.participant == PARTICIPANT) \
        & idx.session.isin(["S01", "S03"])
    pos = np.where(keep.values)[0]
    df = idx.iloc[pos].reset_index(drop=True).copy()
    df["label"] = (df.session == "S03").astype(int)
    order = np.lexsort((df.start_sample.values,
                        (df.session != "S01").values))
    return pos[order], df.iloc[order].reset_index(drop=True)


def chronological_masks(df, reversed_=False):
    """Per session: first floor(0.7*n) windows in time order are
    train (or the last ones, for the reversed sensitivity)."""
    train = np.zeros(len(df), dtype=bool)
    for _, g in df.groupby("session"):
        n_train = int(TRAIN_FRAC * len(g))
        rows = g.index.to_numpy()
        train_rows = rows[-n_train:] if reversed_ else rows[:n_train]
        train[np.isin(np.arange(len(df)), train_rows)] = True
    return train, ~train


def random_masks(df, seed):
    """Stratified random split, same for every model at this seed:
    per class, shuffle with the seed, first max(1, round(0.3*n))
    windows go to test."""
    rng = np.random.default_rng(seed)
    test = np.zeros(len(df), dtype=bool)
    for _, g in df.groupby("label"):
        rows = g.index.to_numpy().copy()
        rng.shuffle(rows)
        n_test = max(1, int(round((1 - TRAIN_FRAC) * len(rows))))
        test[np.isin(np.arange(len(df)), rows[:n_test])] = True
    return ~test, test


def fit_metrics(y_true, y_pred, y_proba):
    m = binary_metrics(y_true, y_pred, y_proba)
    m["accuracy"] = float(accuracy_score(y_true, y_pred))
    return m


def run_lr(feats, df, train_mask, test_mask):
    m = make_model()
    m.fit(feats.values[train_mask], df.label.values[train_mask])
    pred = m.predict(feats.values[test_mask])
    proba = m.predict_proba(feats.values[test_mask])[:, 1]
    return pred, proba


def train_cnn_fixed(X, y, seed):
    """EEGNet with the frozen cnn.py configuration, trained on ALL
    training windows for MAX_EPOCHS with no early stopping (see the
    module docstring for why no validation carve exists here).
    Normalisation and class weights come from the training windows
    only, exactly as in cnn.py's train_fold."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    mean = X[:, 0, :4].mean(axis=(0, 2))
    std = X[:, 0, :4].std(axis=(0, 2)) + 1e-8
    Xn = X.copy()
    Xn[:, 0, :4] = (Xn[:, 0, :4] - mean[None, :, None]) / std[None, :, None]
    Xtr = torch.tensor(Xn)
    ytr = torch.tensor(y)
    cls = np.bincount(y, minlength=2)
    w = torch.tensor(len(y) / (2 * cls), dtype=torch.float32)
    model = EEGNet()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = torch.nn.CrossEntropyLoss(weight=w)
    for _ in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            loss = lossf(model(Xtr[b]), ytr[b])
            loss.backward()
            opt.step()
    return model, mean, std


def run_cnn(X, df, train_mask, test_mask, seed):
    model, mean, std = train_cnn_fixed(X[train_mask],
                                       df.label.values[train_mask], seed)
    Xn = X[test_mask].copy()
    Xn[:, 0, :4] = (Xn[:, 0, :4] - mean[None, :, None]) / std[None, :, None]
    model.eval()
    with torch.no_grad():
        proba = torch.softmax(model(torch.tensor(Xn)), dim=1).numpy()
    return proba.argmax(1), proba[:, 1]


SPLIT_TEXT = {
    "chronological": "within each session, first floor(0.7*n) windows "
                     "in recording order = train, remainder = test; "
                     "no purge gap (windows are non-overlapping)",
    "chronological-reversed": "within each session, LAST floor(0.7*n) "
                              "windows = train, first windows = test "
                              "(direction sensitivity)",
    "random": "stratified random window split, per class n_test = "
              "max(1, round(0.3*n_class)), split seed as in the folder "
              "name; the same split is used for the LR and CNN cells",
}

LEAKY_CHECKLIST = """Leakage checklist (honest version):
- [x] Preprocessing statistics fit on training windows only (LR scaling/imputation; CNN normalisation and class weights)
- [ ] NO session sharing between train and test: VIOLATED BY DESIGN (both sessions, and the participant, appear on both sides; the violation is the object of the run)
- [x] No tuning on test data (all hyperparameters are the frozen pipeline's)
- [x] Label definition and split rule fixed in the driver docstring before the run (post hoc arm, panel-specified)
"""


def write_run(name, model_desc, split, seed_note, metrics_rows, preds,
              extra_notes=""):
    model_key = model_desc
    display = {"lr": "band-power logistic regression",
               "cnn": "EEGNet CNN (frozen cnn.py configuration)"}
    model_desc = display[model_key]
    d = RUNS / name
    d.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(d / "metrics.csv", index=False)
    preds.to_csv(d / "predictions.csv", index=False)
    cfg = {
        "model": model_desc,
        "features/input": ("bandpower_43_amendment1" if model_key == "lr"
                           else "raw windows, 6 channels (4 EEG + mask "
                                "indicator planes), EEGNet-8,2"),
        "protocol": "RQ4 2x2 preliminary-replication arm (POST HOC, "
                    "panel-specified): Subject 01 only, day-level "
                    "labels (S01 = alert, S03 = fatigued)",
        "split": SPLIT_TEXT[split],
        "seed": seed_note,
        "data": "retained Amendment-1 windows for P01 (uncapped pool + "
                "check set), 24 windows: S01 22 alert, S03 2 fatigued; "
                "no cap",
        "git_commit": git_commit(),
    }
    if model_key == "cnn":
        cfg["torch"] = str(torch.__version__)
    (d / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (d / "notes.md").write_text(f"""# RQ4 2x2 replication, {model_desc}, {split} split, {seed_note}

Post hoc arm, panel-specified (review 24). Subject 01 only (the
preliminary experiment's participant), day-level labels: whole
Session 01 = alert, whole Session 03 = fatigued. Retained
Amendment-1 windows: 24 (22 alert, 2 fatigued), so every split
trains on 16 windows (15 alert, 1 fatigued) and tests on 8
(7 alert, 1 fatigued); fatigued recall is therefore a 0-or-1
quantity in this arm and is reported with its counts.

Split definition: {SPLIT_TEXT[split]}.
{extra_notes}
{LEAKY_CHECKLIST}
Environment: pinned current stack (docs/environment_pin.md).
""")


def pred_frame(df, test_mask, pred, proba, **extra):
    rec = df.iloc[np.where(test_mask)[0]][["participant",
                                           "session"]].copy()
    rec["y_true"] = df.label.values[test_mask]
    rec["y_pred"] = pred
    rec["y_proba"] = proba
    for k, v in extra.items():
        rec[k] = v
    return rec


def main():
    arr, idx = load_retained()
    pos, df = p01_frame(idx)
    arr_f = arr[pos]
    feats = extract_features(arr_f, df)
    X = build_inputs(arr_f, df)
    y = df.label.values
    print(f"P01 day-level frame: {len(df)} windows, "
          f"classes {df.label.value_counts().to_dict()}")
    fits = []

    def record(cell, split, split_seed, model_seed, tr, te, pred, proba):
        m = fit_metrics(y[te], pred, proba)
        row = {"cell": cell, "split": split, "split_seed": split_seed,
               "model_seed": model_seed,
               "n_train": int(tr.sum()), "n_test": int(te.sum()),
               "n_test_alert": int((y[te] == 0).sum()),
               "n_test_fatigued": int((y[te] == 1).sum()), **m}
        fits.append(row)
        print(f"  {cell} {split} ss={split_seed} ms={model_seed}: "
              f"BA={m['balanced_accuracy']:.3f} acc={m['accuracy']:.3f} "
              f"recall_fat={m['recall_fatigued']:.3f} "
              f"recall_alert={m['recall_alert']:.3f} "
              f"AUC={m['roc_auc']:.3f}")
        return row

    # ---- LR, chronological (+ reversed sensitivity)
    for split, rev in [("chronological", False),
                       ("chronological-reversed", True)]:
        tr, te = chronological_masks(df, reversed_=rev)
        pred, proba = run_lr(feats, df, tr, te)
        row = record("lr", split, 0, None, tr, te, pred, proba)
        write_run(f"{DATE}_replication2x2_lr-{split}_s0", "lr", split,
                  "deterministic split (s0 by convention)",
                  [row], pred_frame(df, te, pred, proba))

    # ---- LR, random splits
    for ss in SPLIT_SEEDS:
        tr, te = random_masks(df, ss)
        pred, proba = run_lr(feats, df, tr, te)
        row = record("lr", "random", ss, None, tr, te, pred, proba)
        write_run(f"{DATE}_replication2x2_lr-random_s{ss}", "lr",
                  "random", f"split seed {ss}", [row],
                  pred_frame(df, te, pred, proba))

    cnn_note = ("CNN adaptation (forced, stated in the driver): no "
                "both-class validation carve exists at these counts "
                "(one fatigued training window), so the frozen "
                "early stopping is replaced by a fixed 100-epoch "
                "budget on all training windows; architecture, "
                "optimiser, class weights and normalisation are the "
                "frozen cnn.py configuration.\n")

    # ---- CNN, chronological
    tr, te = chronological_masks(df)
    for ms in MODEL_SEEDS:
        pred, proba = run_cnn(X, df, tr, te, ms)
        row = record("cnn", "chronological", 0, ms, tr, te, pred, proba)
        write_run(f"{DATE}_replication2x2_cnn-chronological_s{ms}",
                  "cnn", "chronological", f"model seed {ms}", [row],
                  pred_frame(df, te, pred, proba), cnn_note)

    # ---- CNN, random splits (one folder per split seed, model seeds
    # as rows, so the folder count stays readable)
    for ss in SPLIT_SEEDS:
        tr, te = random_masks(df, ss)
        rows, preds_all = [], []
        for ms in MODEL_SEEDS:
            pred, proba = run_cnn(X, df, tr, te, ms)
            rows.append(record("cnn", "random", ss, ms, tr, te,
                               pred, proba))
            preds_all.append(pred_frame(df, te, pred, proba,
                                        model_seed=ms))
        write_run(f"{DATE}_replication2x2_cnn-random_s{ss}", "cnn",
                  "random", f"split seed {ss}, model seeds "
                  f"{MODEL_SEEDS}", rows, pd.concat(preds_all), cnn_note)

    fits_df = pd.DataFrame(fits)
    fits_df.to_csv(TAB / "rq4_replication_2x2_fits.csv", index=False)
    agg = fits_df.groupby(["cell", "split"]).agg(
        n_fits=("balanced_accuracy", "size"),
        balanced_accuracy=("balanced_accuracy", "mean"),
        balanced_accuracy_sd=("balanced_accuracy", "std"),
        accuracy=("accuracy", "mean"),
        recall_fatigued=("recall_fatigued", "mean"),
        recall_alert=("recall_alert", "mean"),
        roc_auc=("roc_auc", "mean")).reset_index()
    agg.to_csv(TAB / "rq4_replication_2x2_summary.csv", index=False)
    print("\n=== 2x2 cell means ===")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
