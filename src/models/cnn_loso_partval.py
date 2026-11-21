"""Phase C sensitivity - 1D-CNN (EEGNet-8,2), leave-one-participant-out
with PARTICIPANT-CARVED validation (21 Nov 2026, POST HOC).

Why this run exists: in the RQ2 headline run
(src/models/cnn_loso_segval.py), early stopping validated on whole
run-segments carved from the training participants, and the frozen
strategy search's first feasible carve placed those segments inside a
single training participant's session, sharing sessions with the
subtraining data. That sharing can bias the CNN downward: validation
that shares a session with training rewards session fingerprinting
and can stop training late, which is exactly the direction that
would manufacture a "no CNN advantage" result. The headline cannot
stand next to that worry unanswered, so this run is the labelled
post hoc sensitivity that answers it: same everything, with the
validation carve moved somewhere the sharing is impossible.

What changes, and only this: the early-stopping validation set is one
WHOLE training participant, held out of subtraining entirely, so no
validation window shares a participant (let alone a session) with any
subtraining window. The validation participant is chosen by a fixed
rule that uses no scores of any kind: the participant immediately
after the held-out test participant in the sorted participant list,
wrapping around to the first. Everything else is identical to the
headline run: same canonical cap-seed-0 window set (806 windows, 11
participants), same folds, same model seeds 0-4, same frozen network
and training constants, same channel dropout, same patience and
improvement threshold, normalisation and class weights from the
subtraining fold only. No purge is needed at the validation boundary:
validation and subtraining windows belong to different participants,
and the windows themselves are non-overlapping.

The frozen files are not modified. The network, input builder and
channel dropout are imported from src/models/cnn.py and
src/models/cnn_segval.py unchanged; the training loop below mirrors
the frozen train_fold_segval loop step for step, with the segment
carve replaced by the participant masks. Because the frozen carve
function cannot express a participant-level carve, the loop is
re-implemented here rather than imported; the docstring of the
headline driver records why the frozen carve needed a driver around
it there too.

Checkpointing mirrors the headline run: one run folder per fold,
written as soon as the fold finishes, skipped on re-run if its
metrics.csv exists. Run seed by seed (--seeds 0, then --seeds 1, ...)
so each completed seed can be committed before the next starts.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.metrics import binary_metrics
from src.models.cnn import BATCH, LR, MAX_EPOCHS, PATIENCE, EEGNet, build_inputs
from src.models.cnn_segval import channel_dropout

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
RUNS = ROOT / "experiments" / "runs"
DATE = "2026-11-21"


def fold_folder(part, seed):
    return RUNS / f"{DATE}_cnn_loso_partval_subjective_{part}_s{seed}"


def train_fold_partval(X, y, sub_mask, va_mask, seed):
    """The frozen train_fold_segval training loop with explicit
    subtrain/validation masks instead of the segment carve. Returns
    (model, mean, std, note) in the same shape as the frozen function."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    mean = X[sub_mask][:, 0, :4].mean(axis=(0, 2))
    std = X[sub_mask][:, 0, :4].std(axis=(0, 2)) + 1e-8

    def norm(A):
        A = A.copy()
        A[:, 0, :4] = (A[:, 0, :4] - mean[None, :, None]) / std[None, :, None]
        return A

    Xtr = torch.tensor(norm(X[sub_mask]))
    ytr = torch.tensor(y[sub_mask])
    Xva = torch.tensor(norm(X[va_mask]))
    yva = torch.tensor(y[va_mask])
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
            loss.backward()
            opt.step()
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
    note = (f"val_participant_windows={int(va_mask.sum())}, "
            f"subtrain_n={int(sub_mask.sum())}, epochs_run={ep + 1}, "
            f"best_val_loss={best:.4f}")
    return model, mean, std, note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--folds", default=None,
                    help="comma-separated participants to run "
                         "(default: all 11 with both classes)")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    arr = np.load(PROC / "windows_a1.npy")
    idx = pd.read_csv(PROC / "windows_a1_index.csv").reset_index(drop=True)
    X = build_inputs(arr, idx)
    y = idx.label.to_numpy()
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()

    participants = sorted(idx.participant.unique())
    if args.folds:
        participants = [p for p in args.folds.split(",") if p in participants]
    all_parts = sorted(idx.participant.unique())

    for seed in seeds:
        for part in participants:
            d = fold_folder(part, seed)
            if (d / "metrics.csv").exists():
                print(f"[skip] {part} s{seed}: metrics.csv exists", flush=True)
                continue
            # Validation participant: the next participant after the
            # held-out one in sorted order, wrapping (fixed rule,
            # stated in the module docstring; uses no scores).
            val_part = all_parts[(all_parts.index(part) + 1) % len(all_parts)]
            te = (idx.participant == part).to_numpy()
            va = (idx.participant == val_part).to_numpy()
            sub = ~te & ~va
            if (len(np.unique(y[sub])) < 2 or len(np.unique(y[va])) < 2
                    or len(np.unique(y[te])) < 2):
                print(f"[uncovered] {part} s{seed}: single-class split",
                      flush=True)
                continue
            t0 = time.time()
            model, mean, std, note = train_fold_partval(X, y, sub, va, seed)
            wall = time.time() - t0
            Xn = X[te].copy()
            Xn[:, 0, :4] = (Xn[:, 0, :4] - mean[None, :, None]) \
                / std[None, :, None]
            model.eval()
            with torch.no_grad():
                proba = torch.softmax(model(torch.tensor(Xn)), dim=1).numpy()
            rec = idx.iloc[np.where(te)[0]][["participant", "session"]].copy()
            rec["fold"] = part
            rec["y_true"] = y[te]
            rec["y_pred"] = proba.argmax(1)
            rec["y_proba"] = proba[:, 1]
            met = {"participant": part,
                   **binary_metrics(rec.y_true, rec.y_pred, rec.y_proba)}
            d.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([met]).to_csv(d / "metrics.csv", index=False)
            rec.to_csv(d / "predictions.csv", index=False)
            cfg = {"model": "EEGNet-8,2 (kernel 64 @128 Hz)",
                   "input": "6 channels: 4 EEG + mask_TP9 + mask_TP10 "
                            "indicator planes",
                   "validation": "PARTICIPANT-CARVED (POST HOC sensitivity): "
                                 "one whole training participant held out "
                                 "of subtraining as the early-stopping "
                                 "validation set; rule = next participant "
                                 "after the held-out one in sorted order, "
                                 "wrapping",
                   "validation_participant": val_part,
                   "channel_dropout": "temporal channels dropped p=0.2 per "
                                      "sample per batch during training",
                   "training": "Adam lr=0.001, batch=64, max_epochs=100, "
                               "early stopping patience=15 on "
                               "participant-carved validation loss",
                   "class_weights": "subtraining-fold balanced",
                   "normalisation": "per-EEG-channel mean/SD from "
                                    "subtraining fold only",
                   "protocol": "loso (leave-one-participant-out)",
                   "held_out_participant": part,
                   "labels": "subjective (Amendment 1 windows)",
                   "cap_seed": 0, "model_seed": seed,
                   "carve": note,
                   "wall_time_s": round(wall, 1),
                   "git_commit": commit, "torch": str(torch.__version__)}
            (d / "config.yaml").write_text(yaml.safe_dump(cfg,
                                                          sort_keys=False))
            (d / "notes.md").write_text(f"""# CNN (EEGNet) LOPO, PARTICIPANT-CARVED validation - POST HOC sensitivity, held-out {part}, model seed {seed}

Split definition: leave-one-participant-out. Test = all {int(te.sum())}
windows of {part}. Validation = ALL windows of {val_part} (the next
participant after {part} in sorted order, wrapping; a fixed rule that
uses no scores). Subtraining = the remaining participants' windows
from the canonical cap-seed-0 Amendment-1 set (806 windows, 11
participants; P05 has no Amendment-1 windows and has no fold).

**This run is a POST HOC sensitivity, labelled as such everywhere it
is cited.** It changes exactly one thing from the headline CNN LOPO
run (src/models/cnn_loso_segval.py): the early-stopping validation is
carved by whole training participant instead of by run-segments
inside the training participants. The change answers a review point:
segment-carved validation shares sessions with subtraining, which can
reward session fingerprinting and stop training late, biasing the
CNN downward. Everything else (window set, folds, seeds, network,
training constants, channel dropout, patience, improvement
threshold) is identical to the headline run. The training loop
mirrors the frozen train_fold_segval loop; the frozen files are not
modified. No purge is applied at the validation boundary because
validation and subtraining windows belong to different participants
and the windows are non-overlapping.
Carve detail: {note}
Wall time for this fold (training + prediction): {wall:.1f} s on the
local VM (CPU), sharing the machine with other project work.

**Interpretation boundary (state wherever cited): the validation
measures generalisation to one known-but-unseen-in-training
participant, not to the test participant; the held-out test
participant is never touched during fitting.**

Mitigations: temporal CHANNEL DROPOUT during training (p = 0.2 per
sample, EEG zeroed + indicator plane zeroed); masked temporals are
zero-filled AND flagged via indicator planes (Amendment 1 - never
silent zeros).

Leakage checklist:
- [x] Held-out participant contributes no window to subtrain, validation, normalisation statistics or class weights
- [x] Validation participant is a training participant, chosen by a fixed score-free rule, and is excluded from subtraining
- [x] Normalisation statistics from the subtraining fold only
- [x] Class weights from the subtraining fold only
- [x] Early stopping never touches the test fold (validation is a whole training participant)
- [x] No overlapping windows across splits (tumbling windows; participant-level splits throughout)
- [x] Label rules locked (docs/label_rules.md incl. Amendment 1) before training

Environment: local VM, CPU, torch {torch.__version__} (pinned stack,
docs/environment_pin.md). Model seeds 0-4 on the fixed cap-seed-0
window set; one run folder per fold so an interrupted run resumes.
""")
            print(f"[done] {part} s{seed}: BA={met['balanced_accuracy']:.3f} "
                  f"AUC={met['roc_auc']:.3f} MCC={met['mcc']:.3f} "
                  f"n={met['n']} wall={wall:.1f}s (val={val_part}; {note})",
                  flush=True)


if __name__ == "__main__":
    main()
