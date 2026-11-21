"""RQ4 prototype-faithful replication (POST HOC, 21 Nov 2026).

Why this run exists: the 2x2 arm (src/evaluation/rq4_replication_2x2.py)
crossed model by split on this pipeline's retained Amendment-1 windows,
but that arm cannot answer the question it was built for, because two
things had changed at once between the preliminary and that arm: the
windowing (the preliminary epoched whole sessions at 4 s
windows with a 2 s stride; this pipeline cuts non-overlapping windows
inside blocks) and the retention (the preliminary applied no
Amendment-1 rejection; Subject 01 keeps only 24 of its 1,353 windows
under Amendment 1). This driver re-runs the preliminary's own
configuration as closely as this project's frozen models allow, so the
two changes are priced separately. The full specification, including
the outcome branches declared before the run, is written in
docs/rq4_session_labels.md ("Prototype-faithful replication").

What the preliminary's record documents (its repository README):
Subject 01 only; forehead_eeg_raw.csv of Sessions 01 (alert) and 03
(fatigued); MNE at 256 Hz; bandpass 1 to 50 Hz; sliding windows of 4 s
(1,024 samples) at a 2 s stride giving 1,353 labelled samples; an audit
notebook with a chronological split (first 80% of each session train,
last 20% test) and 99.26% chronological test accuracy; the headline
99.18% is a fatigued-class recall (121 of 122 fatigued test samples).

Windowing decision, settled by a count: the preliminary's MNE
construction loads a session's samples as one continuous array and has
no concept of the recording's timestamp gaps. Epoching the loaded
stream that way gives exactly 1,353 windows (720 + 633). Epoching
inside continuous runs only, this pipeline's discipline, gives 389.
This driver therefore epochs the whole stream, and the doc says what
that means: many windows join samples that are not contiguous in time
(Session 01 has 898 timestamp gaps over 100 ms).

Departures from the record, named in the doc and in every run note:
both frozen models take 512-sample windows at 128 Hz, so the stream is
filtered with this pipeline's chain (1 to 40 Hz zero-phase Butterworth,
then 256 to 128 Hz resampling) instead of the record's 1 to 50 Hz at
256 Hz, and the models are this project's (band-power logistic
regression; EEGNet) rather than the prototype's Keras 1D-CNN. The
record's tech-stack table mentions artifact removal but names no
method and its step list contains no such step, so none is applied.
What is replicated exactly: participant, whole-session labels,
windowing geometry and count, and the chronological 80/20 split.

Artefact handling: none in the primary run. Features are computed with
every channel treated as valid (the three mask-indicator features are
constant) and the CNN's mask-indicator planes are constant 1. The
Amendment-1 mask is still computed for every window, with the criteria
exactly as in src/preprocessing/build_windows_a1.py, and is used only
to slice the test predictions into mask-retained and mask-rejected
subsets afterwards. That slice is the run's key cut: if the high score
lives in the rejected windows, the preliminary number rode on windows
the quality bar removes.

Splits:
  chronological  the prototype's audit configuration: within each
                 session, first floor(0.8 * n) windows in stream order
                 are train, the rest test, no purge at the cut.
  blocked        the comparator: within each session, consecutive
                 blocks of 30 windows (60 s of window starts); every
                 fifth block is test, the rest train; training windows
                 that share samples with a test window (the windows
                 immediately before and after each test block) are
                 purged from training and counted. Reported as
                 "overlap removed, recording still shared". A
                 session-held-out control does not exist for this
                 participant: with two sessions and session-level
                 labels, holding out a session removes a whole class.

CNN training goes through the frozen cnn.py train_fold (architecture,
Adam, balanced class weights, subtraining normalisation, early
stopping patience 15 with best weights restored), model seeds 0, 1, 2.
One adaptation: cnn.py's carve is grouped by session, which is
impossible here because the session is the label. The carve unit is
the contiguous tail segment of the training span: validation is the
last quintile of each session's training windows, subtraining the
first four quintiles. No purge at the carve boundary (the neighbouring
windows overlap by 2 s); that can only make early stopping marginally
optimistic and does not touch the test set.

Intervals: contiguous-block bootstrap over the test windows (30-window
blocks within each session resampled with replacement, 10,000
resamples, rng 42); for the CNN the statistic in each resample is the
mean over the 3 model seeds. The effective sample size is far below
the window count and the doc says so.

Outputs: the window set (data/processed/windows_proto_p01.npy + index,
with the true Amendment-1 pass flags stored), one run folder per model
per split (per seed for the CNN) under experiments/runs/, and
results/tables/rq4_replication_proto_summary.csv.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.signal import butter, resample_poly, sosfiltfilt
from sklearn.metrics import accuracy_score

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.load_session import CHANNELS, load_eeg, load_fit
from src.evaluation.metrics import binary_metrics
from src.features.bandpower import extract_features
from src.models.baseline import make_model
from src.models.cnn import build_inputs, train_fold
from src.preprocessing.build_windows import fit_per_sample

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
RUNS = ROOT / "experiments" / "runs"
TAB = ROOT / "results" / "tables"
DATE = "2026-11-21"
PARTICIPANT, P_NUM = "P01", "01"
SESSIONS = [("01", "S01", 0), ("03", "S03", 1)]  # (folder, name, label)
FS_IN, FS_OUT = 256, 128
WIN = 512                    # 4 s at 128 Hz
STRIDE = 256                 # 2 s at 128 Hz (50% overlap)
WIN_IN, STRIDE_IN = 1024, 512  # the same geometry at 256 Hz
P2P_LIMIT, FLAT_SD = 150.0, 0.5  # Amendment-1 criteria (build_windows_a1)
FRONTAL = [CHANNELS.index("AF7"), CHANNELS.index("AF8")]
TRAIN_FRAC = 0.8
BLOCK = 30                   # windows per block (60 s of window starts)
MODEL_SEEDS = [0, 1, 2]
N_BOOT, BOOT_SEED = 10_000, 42
SOS = butter(4, [1.0, 40.0], btype="bandpass", fs=FS_IN, output="sos")


def git_commit():
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()


def build_windows():
    """Epoch P01's Sessions 01 and 03 the prototype's way: the loaded
    sample stream treated as one continuous signal, filtered and
    downsampled whole, then 4 s windows at a 2 s stride. Also computes
    the Amendment-1 mask per window (used only as an evaluation
    slicer). Returns (windows (n,4,512) float32, index DataFrame)."""
    wins, rows = [], []
    for folder, sname, label in SESSIONS:
        ts, x = load_eeg(P_NUM, folder)
        fit_ts, fit = load_fit(P_NUM, folder)
        fit_s = fit_per_sample(ts, fit_ts, fit)
        assert not np.isnan(x).any(), \
            f"{sname}: NaN samples present; stream epoching assumes none"
        filt = sosfiltfilt(SOS, x, axis=0)
        down = resample_poly(filt, up=1, down=2, axis=0)  # (n_down, 4)
        n_win = (len(down) - WIN) // STRIDE + 1
        for k in range(n_win):
            win = down[k * STRIDE:k * STRIDE + WIN]        # (512, 4)
            # Amendment-1 criteria, exactly as build_windows_a1 computes
            # them: per channel, largest 1 s sub-window peak-to-peak,
            # window SD, and Bad-fit fraction over the raw-rate span.
            sub = win.reshape(4, 128, 4)
            p2p_ch = (sub.max(axis=1) - sub.min(axis=1)).max(axis=0)
            sd_ch = win.std(axis=0)
            fseg = fit_s[k * STRIDE_IN:k * STRIDE_IN + WIN_IN]
            fit_bad_frac = np.nanmean(fseg == 4, axis=0)
            pass_ch = ~((p2p_ch > P2P_LIMIT) | (sd_ch < FLAT_SD)
                        | (fit_bad_frac > 0))
            wins.append(np.ascontiguousarray(win.T, dtype=np.float32))
            rows.append({
                "participant": PARTICIPANT, "session": sname,
                "stream_window": k, "start_sample": int(k * STRIDE_IN),
                "label": label,
                **{f"valid_{c}": bool(pass_ch[i])
                   for i, c in enumerate(CHANNELS)},
                "a1_retained": bool(pass_ch[FRONTAL].all()),
            })
        print(f"{sname}: {n_win} stream windows "
              f"({sum(r['a1_retained'] for r in rows if r['session'] == sname)}"
              f" mask-retained)")
    idx = pd.DataFrame(rows)
    return np.stack(wins), idx


def chronological_masks(idx):
    """Per session: first floor(0.8 * n) windows in stream order are
    train, the rest test (the prototype's audit cut, no purge)."""
    train = np.zeros(len(idx), dtype=bool)
    for _, g in idx.groupby("session"):
        rows = g.index.to_numpy()
        train[rows[:int(TRAIN_FRAC * len(rows))]] = True
    return train, ~train


def blocked_masks(idx):
    """Per session: consecutive blocks of BLOCK windows; every fifth
    block is test. Training windows that share samples with a test
    window (immediately before/after a test block in stream order)
    are purged from training. Returns (train, test, n_purged)."""
    test = np.zeros(len(idx), dtype=bool)
    for _, g in idx.groupby("session"):
        rows = g.index.to_numpy()
        for j, r in enumerate(rows):
            if (j // BLOCK + 1) % 5 == 0:
                test[r] = True
    train = ~test
    purged = np.zeros(len(idx), dtype=bool)
    for _, g in idx.groupby("session"):
        rows = g.index.to_numpy()
        pos_of = {r: j for j, r in enumerate(rows)}
        for r in rows:
            if not train[r]:
                continue
            j = pos_of[r]
            neighbours = [rows[j - 1]] if j > 0 else []
            neighbours += [rows[j + 1]] if j < len(rows) - 1 else []
            if any(test[q] for q in neighbours):
                purged[r] = True
    train &= ~purged
    return train, test, int(purged.sum())


def carve_groups(idx_sub):
    """Validation-carve groups for cnn.train_fold: the within-session
    quintile (0 to 4) of each training window, in stream order.
    split_grouped takes the last group as validation, which is then
    the last quintile of EACH session's training span (both classes),
    because the quintiles are computed per session."""
    groups = np.zeros(len(idx_sub), dtype=int)
    for _, g in idx_sub.groupby("session"):
        rows = g.index.to_numpy()  # positional: idx_sub was reset
        groups[rows] = np.minimum(4, (5 * np.arange(len(rows)))
                                  // len(rows))
    return groups


def subset_metrics(y_true, y_pred, y_proba):
    """binary_metrics plus plain accuracy, tolerant of a single-class
    slice (balanced accuracy then reduces to that class's recall and
    AUC is undefined; both are reported as computed, with n beside
    them in every table)."""
    m = binary_metrics(y_true, y_pred, y_proba)
    m["accuracy"] = float(accuracy_score(y_true, y_pred))
    return m


def run_lr(feats, y, tr, te):
    m = make_model()
    m.fit(feats.values[tr], y[tr])
    return m.predict(feats.values[te]), \
        m.predict_proba(feats.values[te])[:, 1]


def run_cnn(X, y, idx, tr, te, seed):
    """One EEGNet fit through the frozen cnn.train_fold, with the
    tail-quintile carve groups. Returns (pred, proba, carve note)."""
    groups = carve_groups(idx.iloc[np.where(tr)[0]].reset_index(drop=True))
    model, mean, std, note = train_fold(X[tr], y[tr], groups, seed)
    if model is None:
        raise RuntimeError(f"CNN carve failed: {note}")
    Xn = X[te].copy()
    Xn[:, 0, :4] = (Xn[:, 0, :4] - mean[None, :, None]) / std[None, :, None]
    model.eval()
    with torch.no_grad():
        proba = torch.softmax(model(torch.tensor(Xn)), dim=1).numpy()
    return proba.argmax(1), proba[:, 1], note


def block_bootstrap(idx_te, y, retained, preds_by_seed):
    """Contiguous-block bootstrap of balanced accuracy and fatigued
    recall on the all / retained / rejected test slices. Blocks are
    consecutive chunks of BLOCK windows within each session, in
    stream order, resampled with replacement per session. For the
    CNN, preds_by_seed holds one (pred, proba) pair per model seed
    and the statistic is the seed mean. Returns {(slice, metric):
    (low, high)}."""
    blocks = []
    for _, g in idx_te.groupby("session"):
        rows = g.index.to_numpy()
        for j in range(0, len(rows), BLOCK):
            blocks.append((g.session.iloc[0], rows[j:j + BLOCK]))
    by_session = {}
    for sname, b in blocks:
        by_session.setdefault(sname, []).append(b)
    rng = np.random.default_rng(BOOT_SEED)
    stats = {(s, m): [] for s in ("all", "retained", "rejected")
             for m in ("balanced_accuracy", "recall_fatigued")}
    for _ in range(N_BOOT):
        take = np.concatenate([
            np.concatenate([bl[rng.integers(len(bl))]
                            for _ in range(len(bl))])
            for bl in by_session.values()])
        yt, rt = y[take], retained[take]
        for sname, mask in (("all", np.ones(len(take), bool)),
                            ("retained", rt), ("rejected", ~rt)):
            if mask.sum() == 0 or len(np.unique(yt[mask])) < 2:
                for m in stats:
                    if m[0] == sname:
                        stats[m].append(np.nan)
                continue
            bas, rfs = [], []
            for pred, _proba in preds_by_seed:
                yp = pred[take][mask]
                bas.append(0.5 * ((yp[yt[mask] == 0] == 0).mean()
                                  + (yp[yt[mask] == 1] == 1).mean()))
                rfs.append((yp[yt[mask] == 1] == 1).mean())
            stats[(sname, "balanced_accuracy")].append(np.mean(bas))
            stats[(sname, "recall_fatigued")].append(np.mean(rfs))
    return {k: (float(np.nanpercentile(v, 2.5)),
                float(np.nanpercentile(v, 97.5)))
            for k, v in stats.items()}


SPLIT_TEXT = {
    "chronological": "chronological 80/20 within each session (first "
                     "floor(0.8*n) windows in stream order = train, "
                     "remainder = test, no purge at the cut): the "
                     "prototype audit notebook's configuration",
    "blocked": "blocked split: consecutive 30-window blocks within "
               "each session, every fifth block = test, the rest "
               "train, with training windows that share samples "
               "with a test window purged (overlap removed, "
               "recording still shared)",
}

CHECKLISTS = {
    "chronological": """Leakage checklist (honest version):
- [x] Preprocessing statistics fit on training windows only (LR imputation/scaling; CNN normalisation and class weights from the subtraining windows)
- [ ] NO session sharing between train and test: VIOLATED BY DESIGN (the prototype's configuration; the violation is the object of the run)
- [ ] NO overlapping windows across the split: VIOLATED BY DESIGN (50% overlap is the prototype's windowing; the test window after the cut overlaps the last training window by 2 s)
- [x] No artefact rejection or masking anywhere (the Amendment-1 mask is computed only to slice test predictions afterwards)
- [x] No tuning on test data (both models are the frozen pipeline's)
- [x] Specification and outcome branches written into docs/rq4_session_labels.md before the run (post hoc arm, panel-owed)
""",
    "blocked": """Leakage checklist (honest version):
- [x] Preprocessing statistics fit on training windows only (LR imputation/scaling; CNN normalisation and class weights from the subtraining windows)
- [ ] NO session sharing between train and test: VIOLATED BY DESIGN (train and test come from the same two recordings; this split removes window overlap only and is reported as "overlap removed, recording still shared", never as an honest fatigue evaluation)
- [x] No overlapping windows across the split (blocks assigned wholly; boundary training windows purged, count in config)
- [x] No artefact rejection or masking anywhere (the Amendment-1 mask is computed only to slice test predictions afterwards)
- [x] No tuning on test data (both models are the frozen pipeline's)
- [x] Specification written into docs/rq4_session_labels.md before the run (post hoc arm, panel-owed)
""",
}


def write_run(name, model_key, split, seed_note, idx_te, y_te, pred,
              proba, cis, extra_cfg, extra_notes):
    display = {"lr": "band-power logistic regression (frozen baseline)",
               "cnn": "EEGNet CNN (frozen cnn.py train_fold)"}
    d = RUNS / name
    d.mkdir(parents=True, exist_ok=True)
    ret = idx_te.a1_retained.values
    rows = []
    for sname, mask in (("all", np.ones(len(idx_te), bool)),
                        ("a1_retained", ret), ("a1_rejected", ~ret)):
        m = subset_metrics(y_te[mask], pred[mask], proba[mask])
        rows.append({"subset": sname, **m,
                     "ba_ci95_low": cis[(sname.replace("a1_", ""),
                                         "balanced_accuracy")][0],
                     "ba_ci95_high": cis[(sname.replace("a1_", ""),
                                          "balanced_accuracy")][1],
                     "recall_fatigued_ci95_low":
                         cis[(sname.replace("a1_", ""),
                              "recall_fatigued")][0],
                     "recall_fatigued_ci95_high":
                         cis[(sname.replace("a1_", ""),
                              "recall_fatigued")][1]})
    pd.DataFrame(rows).to_csv(d / "metrics.csv", index=False)
    preds = idx_te[["participant", "session", "stream_window",
                    "start_sample", "a1_retained"]].copy()
    preds["y_true"] = y_te
    preds["y_pred"] = pred
    preds["y_proba"] = proba
    preds.to_csv(d / "predictions.csv", index=False)
    cfg = {
        "model": display[model_key],
        "features/input": ("bandpower_43, all channels treated as valid "
                            "(no masking)" if model_key == "lr" else
                           "raw stream windows, 6 channels (4 EEG + "
                           "constant-1 mask indicator planes), EEGNet-8,2"),
        "protocol": "RQ4 prototype-faithful replication (POST HOC, "
                    "panel-owed): Subject 01 only, day-level labels "
                    "(whole Session 01 = alert, whole Session 03 = "
                    "fatigued), prototype windowing (4 s windows, 2 s "
                    "stride, whole-stream epoching, 1,353 windows)",
        "split": SPLIT_TEXT[split],
        "seed": seed_note,
        "data": "windows_proto_p01 (this run's window build; see the "
                "driver docstring for the count check and the named "
                "preprocessing departures)",
        "git_commit": git_commit(), **extra_cfg,
    }
    if model_key == "cnn":
        cfg["torch"] = str(torch.__version__)
    (d / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (d / "notes.md").write_text(f"""# RQ4 prototype-faithful replication, {display[model_key]}, {split} split, {seed_note}

Post hoc arm, panel-owed (reviews 25 and 26 in docs/methods_notes.md);
specification and outcome branches were written into
docs/rq4_session_labels.md before the run. Subject 01 only, the
preliminary experiment's own participant and configuration as far as
this project's frozen models allow: whole sessions epoched as one
continuous stream (720 + 633 = 1,353 windows, the preliminary's own
count), 4 s windows at a 2 s stride, whole-session labels, no
artefact rejection or masking. Named departures from the
preliminary's documented chain: this pipeline's 1 to 40 Hz filter
and 128 Hz models in place of its 1 to 50 Hz at 256 Hz, and this
project's two models in place of its Keras 1D-CNN. The Amendment-1
mask is computed for every window and used only to slice the test
predictions into retained and rejected subsets (metrics.csv rows
a1_retained / a1_rejected).

Split definition: {SPLIT_TEXT[split]}.
{extra_notes}
{CHECKLISTS[split]}
Environment: pinned current stack (docs/environment_pin.md).
""")


def main():
    arr, idx = build_windows()
    np.save(PROC / "windows_proto_p01.npy", arr)
    idx.to_csv(PROC / "windows_proto_p01_index.csv", index=False)
    total = len(idx)
    per_sess = idx.groupby("session").size().to_dict()
    print(f"window set: {total} windows {per_sess}; mask-retained "
          f"{int(idx.a1_retained.sum())}, mask-rejected "
          f"{int((~idx.a1_retained).sum())}")

    # Modelling copy of the index: every channel valid, so features and
    # CNN inputs carry no masking (the primary run applies none). The
    # saved index keeps the true Amendment-1 flags.
    idx_m = idx.copy()
    for c in CHANNELS:
        idx_m[f"valid_{c}"] = True
    feats = extract_features(arr, idx_m)
    X = build_inputs(arr, idx_m)
    y = idx.label.values
    retained_all = idx.a1_retained.values

    summary_rows = []
    for split in ("chronological", "blocked"):
        if split == "chronological":
            tr, te = chronological_masks(idx)
            purge_note, extra_cfg = "", {}
        else:
            tr, te, n_purged = blocked_masks(idx)
            purge_note = (f" Blocked-split purge: {n_purged} training "
                          f"windows removed for sharing samples with a "
                          f"test window.")
            extra_cfg = {"blocked_purge_n": n_purged}
        idx_te = idx.iloc[np.where(te)[0]].reset_index(drop=True)
        y_te = y[te]
        ret_te = retained_all[te]
        print(f"\n{split}: train {int(tr.sum())}, test {int(te.sum())} "
              f"(alert {int((y_te == 0).sum())}, fatigued "
              f"{int((y_te == 1).sum())}); test mask-retained "
              f"{int(ret_te.sum())}, rejected {int((~ret_te).sum())}")

        # ---- LR (deterministic, one fit)
        pred, proba = run_lr(feats, y, tr, te)
        cis = block_bootstrap(idx_te, y_te, ret_te, [(pred, proba)])
        write_run(f"{DATE}_replication-proto_lr-{split}_s0", "lr", split,
                  "deterministic (s0 by convention)", idx_te, y_te, pred,
                  proba, cis, extra_cfg,
                  "LR is deterministic given the data." + purge_note)
        for sname, mask in (("all", np.ones(len(idx_te), bool)),
                            ("a1_retained", ret_te),
                            ("a1_rejected", ~ret_te)):
            m = subset_metrics(y_te[mask], pred[mask], proba[mask])
            key = sname.replace("a1_", "")
            summary_rows.append({
                "model": "lr", "split": split, "subset": sname, **m,
                "ba_ci95_low": cis[(key, "balanced_accuracy")][0],
                "ba_ci95_high": cis[(key, "balanced_accuracy")][1],
                "recall_fatigued_ci95_low": cis[(key, "recall_fatigued")][0],
                "recall_fatigued_ci95_high": cis[(key, "recall_fatigued")][1],
                "seed_note": "deterministic"})
            print(f"  LR {sname}: n={m['n']} acc={m['accuracy']:.3f} "
                  f"BA={m['balanced_accuracy']:.3f} "
                  f"recall_fat={m['recall_fatigued']:.3f} "
                  f"AUC={m['roc_auc']:.3f}")

        # ---- CNN (3 model seeds through the frozen train_fold)
        cnn_preds = []
        carve_notes = []
        for seed in MODEL_SEEDS:
            pred_s, proba_s, note = run_cnn(X, y, idx, tr, te, seed)
            cnn_preds.append((pred_s, proba_s))
            carve_notes.append(note)
            print(f"  CNN seed {seed}: {note}")
        cis_cnn = block_bootstrap(idx_te, y_te, ret_te, cnn_preds)
        # Per-seed run folders carry that seed's own point metrics
        # only; the bootstrap intervals belong to the seed-mean
        # statistic and live in the summary table, so the interval
        # columns here are NaN by design and the note says so.
        nan_cis = {k: (np.nan, np.nan) for k in cis_cnn}
        for j, (seed, (pred_s, proba_s)) in enumerate(
                zip(MODEL_SEEDS, cnn_preds)):
            write_run(f"{DATE}_replication-proto_cnn-{split}_s{seed}",
                      "cnn", split, f"model seed {seed}", idx_te, y_te,
                      pred_s, proba_s, nan_cis, extra_cfg,
                      f"CNN carve (frozen train_fold, tail-quintile "
                      f"groups): {carve_notes[j]}. Interval columns "
                      f"are empty in this per-seed folder: intervals "
                      f"are computed for the seed-mean statistic and "
                      f"reported in results/tables/"
                      f"rq4_replication_proto_summary.csv." + purge_note)
        for sname, mask in (("all", np.ones(len(idx_te), bool)),
                            ("a1_retained", ret_te),
                            ("a1_rejected", ~ret_te)):
            per_seed = [subset_metrics(y_te[mask], p[mask], pr[mask])
                        for p, pr in cnn_preds]
            agg = {c: float(np.mean([m[c] for m in per_seed]))
                   for c in per_seed[0]}
            agg["n"] = per_seed[0]["n"]
            agg["n_alert"] = per_seed[0]["n_alert"]
            agg["n_fatigued"] = per_seed[0]["n_fatigued"]
            key = sname.replace("a1_", "")
            summary_rows.append({
                "model": "cnn", "split": split, "subset": sname, **agg,
                "ba_ci95_low": cis_cnn[(key, "balanced_accuracy")][0],
                "ba_ci95_high": cis_cnn[(key, "balanced_accuracy")][1],
                "recall_fatigued_ci95_low":
                    cis_cnn[(key, "recall_fatigued")][0],
                "recall_fatigued_ci95_high":
                    cis_cnn[(key, "recall_fatigued")][1],
                "seed_note": "mean over model seeds 0,1,2"})
            print(f"  CNN {sname} (seed mean): n={agg['n']} "
                  f"acc={agg['accuracy']:.3f} "
                  f"BA={agg['balanced_accuracy']:.3f} "
                  f"recall_fat={agg['recall_fatigued']:.3f} "
                  f"AUC={agg['roc_auc']:.3f}")

    out = pd.DataFrame(summary_rows)
    out.to_csv(TAB / "rq4_replication_proto_summary.csv", index=False)
    print("\nsummary written to results/tables/"
          "rq4_replication_proto_summary.csv")


if __name__ == "__main__":
    main()
