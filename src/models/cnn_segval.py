"""Phase C (revised) - 1D-CNN (EEGNet-8,2), within-subject protocol with
SEGMENT-GROUPED validation (decided 20 Nov 2026; replaces the
session-grouped validation of src/models/cnn.py, which was
structurally infeasible on this dataset - see the STOPPED run
folders). I changed the carve because whole sessions are too coarse
a unit here: several participants cannot spare a full session for
validation and still leave both classes in training, so the
session-grouped version silently left them uncovered.

Protocol: per participant, hold out one session as test (rotate over
sessions containing both classes); training = the participant's other
sessions. Validation for early stopping is carved from the TRAINING
sessions only, as whole contiguous run-segments (run_id groups from the
window index; every run is single-class - verified: 0/287 runs span both
classes), with a 2-window purge at segment boundaries (training windows
within 2 positions of a validation window in the same session's time
order are dropped from subtraining).

IMPORTANT (must be stated wherever these numbers are cited): this
validation measures SEGMENT generalisation within known sessions, NOT
unseen-session transfer. The test session is still fully held out.

Amendment 1 mask handling: as in cnn.py, 6 input channels (4 EEG,
zero-filled masked temporals + 2 mask-indicator planes). ADDITION
(mask-as-fingerprint mitigation): CHANNEL DROPOUT on the temporal
channels during training. I added this because the mask pattern
itself could act as a fingerprint of the session: if the network can
always see which channels were masked, it can learn a session's
masking signature instead of anything about fatigue, and the
within-subject numbers would flatter it for the wrong reason. For
each training sample in each batch, TP9 and TP10 are independently
dropped with p=0.2 (EEG channel zeroed AND its indicator plane set
to 0), so no fixed masking pattern is reliable enough to lean on.
Validation/test are never dropped.

Normalisation: per-EEG-channel mean/SD from the subtraining fold only.
Class weights from the subtraining fold only. EEGNet-8,2 as in cnn.py;
Adam lr 1e-3, batch 64, <=100 epochs, early stopping patience 15
(best-val-loss weights restored).

Feasibility criteria (fixed before running): a fold runs only if the
test session has both classes, the carved validation has both classes
and >=6 windows, and the post-purge subtraining set has both classes
and >=20 windows. Folds/participants failing this are logged as
UNCOVERED with the reason - never forced.

Data: canonical cap-seed-0 set (windows_a1). Model seeds {0,1} (reduced
set, as before; full LOSO x 5 seeds = Weeks 3-4 on Colab).
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.metrics import binary_metrics
from src.models.cnn import BATCH, LR, MAX_EPOCHS, PATIENCE, EEGNet, build_inputs

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
RUNS = ROOT / "experiments" / "runs"
TAB = ROOT / "results" / "tables"
DATE = "2026-11-20"
DROP_P = 0.2
MIN_VAL, MIN_SUBTRAIN, PURGE = 6, 20, 2


def carve_validation(train_df):
    """train_df: rows of the window index for the training sessions of one
    fold (positions are row numbers in the fold-local arrays).
    Returns (subtrain_mask, val_mask, description) or (None, None, reason)."""
    df = train_df.reset_index(drop=True)
    runs = (df.groupby(["session", "run_id"])
            .agg(label=("label", "first"), n=("label", "size"),
                 start=("start_sample", "min"))
            .reset_index().sort_values(["session", "start"])
            .reset_index(drop=True))
    if runs.label.nunique() < 2:
        return None, None, "training sessions contain a single class only"
    alert_runs = runs[runs.label == 0]
    fat_runs = runs[runs.label == 1]

    def val_mask_for(val_keys):
        m = df.set_index(["session", "run_id"]).index.isin(val_keys)
        return np.asarray(m)

    def apply_purge(sub_mask, va_mask):
        sub = sub_mask.copy()
        for sess, g in df.groupby("session"):
            pos = np.where((df.session == sess).to_numpy())[0]
            order = pos[np.argsort(df.start_sample.to_numpy()[pos])]
            vpos = set(np.where(va_mask[pos])[0].tolist())
            for i, p in enumerate(order):
                if va_mask[p]:
                    for q in range(max(0, i - PURGE),
                                   min(len(order), i + PURGE + 1)):
                        if not va_mask[order[q]]:
                            sub[order[q]] = False
        return sub

    candidates = []
    for k in (1, 2, 3, 4, 5):
        candidates.append((f"last{k}", list(alert_runs.tail(k).itertuples())
                           + list(fat_runs.tail(k).itertuples())))
        candidates.append((f"first{k}", list(alert_runs.head(k).itertuples())
                           + list(fat_runs.head(k).itertuples())))
    for name, chosen in candidates:
        keys = [(r.session, r.run_id) for r in chosen]
        va = val_mask_for(keys)
        if len(np.unique(df.label.values[va])) < 2 or va.sum() < MIN_VAL:
            continue
        sub = apply_purge(~va, va)
        if len(np.unique(df.label.values[sub])) < 2 or sub.sum() < MIN_SUBTRAIN:
            continue
        desc = (f"strategy={name}, val_runs={keys}, val_n={int(va.sum())}, "
                f"subtrain_n={int(sub.sum())} (post-purge)")
        return sub, va, desc
    return None, None, ("no run-segment carve gives both classes in "
                        "validation (>=6 windows) and subtrain (>=20 windows "
                        "after purge)")


def channel_dropout(xb):
    """xb: (B,1,6,512) training batch. Independently drop TP9 (EEG ch 0,
    indicator plane 4) and TP10 (EEG ch 3, indicator plane 5) with
    p=DROP_P per sample."""
    xb = xb.clone()
    b = xb.shape[0]
    for eeg_ch, plane in ((0, 4), (3, 5)):
        drop = torch.rand(b) < DROP_P
        if drop.any():
            xb[drop, 0, eeg_ch] = 0.0
            xb[drop, 0, plane] = 0.0
    return xb


def train_fold_segval(X, y, train_df, seed):
    sub_mask, va_mask, desc = carve_validation(train_df)
    if sub_mask is None:
        return None, None, None, desc
    torch.manual_seed(seed); np.random.seed(seed)
    mean = X[sub_mask][:, 0, :4].mean(axis=(0, 2))
    std = X[sub_mask][:, 0, :4].std(axis=(0, 2)) + 1e-8

    def norm(A):
        A = A.copy()
        A[:, 0, :4] = (A[:, 0, :4] - mean[None, :, None]) / std[None, :, None]
        return A

    Xtr = torch.tensor(norm(X[sub_mask])); ytr = torch.tensor(y[sub_mask])
    Xva = torch.tensor(norm(X[va_mask])); yva = torch.tensor(y[va_mask])
    cls = np.bincount(y[sub_mask], minlength=2)
    w = torch.tensor(len(y[sub_mask]) / (2 * cls), dtype=torch.float32)
    model = EEGNet()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss(weight=w)
    best, best_state, bad, ep = np.inf, None, 0, 0
    for ep in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), BATCH):
            b = perm[i:i + BATCH]
            xb = channel_dropout(Xtr[b])
            opt.zero_grad()
            loss = lossf(model(xb), ytr[b])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = lossf(model(Xva), yva).item()
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    note = (f"{desc}, epochs_run={ep + 1}, best_val_loss={best:.4f}")
    return model, mean, std, note


def main():
    arr = np.load(PROC / "windows_a1.npy")
    idx = pd.read_csv(PROC / "windows_a1_index.csv").reset_index(drop=True)
    X = build_inputs(arr, idx)
    y = idx.label.to_numpy()
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()

    # fold plan + coverage accounting
    fold_defs, uncovered = [], []
    for part, g in idx.groupby("participant"):
        testable = [s for s, gs in g.groupby("session")
                    if gs.label.nunique() == 2]
        if not testable:
            uncovered.append(f"{part}: no session with both classes in the "
                             f"Amendment-1 set (cannot form a test fold)")
            continue
        for sess in testable:
            te = ((idx.participant == part) & (idx.session == sess)).to_numpy()
            tr = ((idx.participant == part) & (idx.session != sess)).to_numpy()
            if len(np.unique(y[tr])) < 2:
                uncovered.append(f"{part}/{sess}: remaining training "
                                 f"sessions are single-class")
                continue
            fold_defs.append((f"{part}/{sess}", tr, te))
    all_parts = set(idx.participant.unique())

    all_per = []
    for seed in (0, 1):
        recs, skipped, fold_notes = [], [], list(uncovered)
        covered_parts = set()
        for fid, tr, te in fold_defs:
            tr_pos = np.where(tr)[0]
            model, mean, std, note = train_fold_segval(
                X[tr], y[tr], idx.iloc[tr_pos], seed)
            if model is None:
                skipped.append(f"{fid} (UNCOVERED: {note})")
                continue
            fold_notes.append(f"{fid}: {note}")
            covered_parts.add(fid.split("/")[0])
            Xn = X[te].copy()
            Xn[:, 0, :4] = (Xn[:, 0, :4] - mean[None, :, None]) / std[None, :, None]
            model.eval()
            with torch.no_grad():
                proba = torch.softmax(model(torch.tensor(Xn)), dim=1).numpy()
            rec = idx.iloc[np.where(te)[0]][["participant", "session"]].copy()
            rec["fold"] = fid; rec["y_true"] = y[te]
            rec["y_pred"] = proba.argmax(1); rec["y_proba"] = proba[:, 1]
            recs.append(rec)
        preds_df = pd.concat(recs)
        per = pd.DataFrame([
            {"participant": gid, **binary_metrics(g.y_true, g.y_pred, g.y_proba)}
            for gid, g in preds_df.groupby("participant")])
        name = f"{DATE}_cnn_within_subjective_s{seed}"
        d = RUNS / name; d.mkdir(parents=True, exist_ok=True)
        agg = {c: per[c].mean() for c in per.columns if c != "participant"}
        pd.concat([per, pd.DataFrame([{"participant": "AGGREGATE_mean_over_groups",
                                       **agg}])],
                  ignore_index=True).to_csv(d / "metrics.csv", index=False)
        preds_df.to_csv(d / "predictions.csv", index=False)
        not_covered = sorted(all_parts - covered_parts)
        cfg = {"model": "EEGNet-8,2 (kernel 64 @128 Hz)",
               "input": "6 channels: 4 EEG + mask_TP9 + mask_TP10 indicator planes",
               "validation": "SEGMENT-GROUPED: whole run-segments carved from "
                             "training sessions, 2-window purge at boundaries",
               "channel_dropout": f"temporal channels dropped p={DROP_P} per "
                                  f"sample per batch during training (EEG zeroed "
                                  f"+ indicator plane set to 0)",
               "training": f"Adam lr={LR}, batch={BATCH}, max_epochs={MAX_EPOCHS}, "
                           f"early stopping patience={PATIENCE} on segment-grouped "
                           f"validation loss",
               "class_weights": "subtraining-fold balanced",
               "normalisation": "per-EEG-channel mean/SD from subtraining fold only",
               "protocol": "within (segment-grouped validation)",
               "labels": "subjective (Amendment 1 windows)",
               "cap_seed": 0, "model_seed": seed,
               "feasibility": f"validation both classes and >={MIN_VAL} windows; "
                              f"subtrain both classes and >={MIN_SUBTRAIN} windows "
                              f"after purge",
               "seed_note": "REDUCED seed set {0,1}; full LOSO x 5 seeds = Weeks 3-4 on Colab",
               "git_commit": commit, "torch": str(torch.__version__)}
        (d / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        (d / "notes.md").write_text(f"""# CNN (EEGNet) within - subjective labels, model seed {seed} - SEGMENT-GROUPED validation

Split definition: per participant, rotate held-out session over sessions
containing both classes (train = the participant's other sessions).
Early stopping: validation = whole contiguous RUN-SEGMENTS carved from
the training sessions only (run_id groups; every run is single-class),
with a 2-window purge of subtraining windows adjacent to validation
segments in the same session's time order; best-val-loss weights
restored. Feasibility criteria (fixed before running): validation both
classes and >={MIN_VAL} windows; post-purge subtrain both classes and
>={MIN_SUBTRAIN} windows.

**Interpretation boundary (state wherever cited): this validation
measures segment generalisation within known sessions, NOT unseen-session
transfer. The test session itself is fully held out.**

Fold details: {'; '.join(fold_notes) if fold_notes else 'none'}
Skipped / uncovered folds: {skipped if skipped else 'none'}
Participants with no covered fold in this seed: {not_covered if not_covered else 'none'}

Mitigations: temporal CHANNEL DROPOUT during training (p={DROP_P} per
sample, EEG zeroed + indicator plane zeroed - Reviewer 1 review 5
mask-as-fingerprint mitigation); masked temporals are zero-filled AND
flagged via indicator planes (Amendment 1 - never silent zeros).

Leakage checklist:
- [x] Normalisation statistics from the subtraining fold only
- [x] Class weights from the subtraining fold only
- [x] Early stopping never touches the test fold (validation carved from training sessions only, segment-grouped with purge)
- [x] No overlapping windows across splits (tumbling windows; session-level test split; purge at validation boundaries)
- [x] Label rules locked (docs/label_rules.md incl. Amendment 1) before training

Environment: local VM, CPU, torch {torch.__version__}. Reduced seed set
{{0,1}}; full LOSO x 5 seeds runs Weeks 3-4 on Colab. This run supersedes
the session-grouped-validation attempt preserved in the *_STOPPED_sessionval
folders (P08-only partial run, kept for audit).
""")
        per["seed"] = seed
        all_per.append(per)
        a = per[["balanced_accuracy", "macro_f1", "roc_auc", "mcc"]].mean()
        print(f"seed {seed}: participants={len(per)} not_covered={not_covered} "
              f"bal_acc={a.balanced_accuracy:.3f} macroF1={a.macro_f1:.3f} "
              f"AUC={a.roc_auc:.3f} MCC={a.mcc:.3f}", flush=True)
        print(f"  skipped: {skipped}", flush=True)
    allp = pd.concat(all_per)
    summ = allp.groupby("participant")[["balanced_accuracy", "macro_f1", "roc_auc",
                                        "mcc", "recall_alert", "recall_fatigued",
                                        "precision_alert", "precision_fatigued",
                                        "n"]].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.reset_index().to_csv(TAB / "cnn_within_subjective_a1_summary.csv",
                              index=False)
    print("summary written", flush=True)


if __name__ == "__main__":
    main()
