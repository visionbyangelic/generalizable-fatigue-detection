"""Phase C - 1D-CNN (EEGNet-8,2), leave-one-participant-out (LOPO)
protocol with SEGMENT-GROUPED validation (21 Nov 2026).

This is the RQ2 headline run that the Week 2 design always pointed at:
the frozen plan in src/models/cnn.py says the full LOSO x 5-seed run
belongs to Weeks 3-4, and src/models/cnn_segval.py fixed the validation
design after the session-grouped attempt proved structurally infeasible
within subjects. This driver applies that same segment-grouped design
across participants. It does not modify any frozen file: the network,
the input builder, the validation carve and the channel dropout are all
imported from cnn.py / cnn_segval.py unchanged.

Protocol: hold out one participant entirely as the test fold; train on
the other 10 participants' windows from the canonical cap-seed-0 set
(data/processed/windows_a1*, 806 windows, 11 participants; P05 has no
Amendment-1 windows at all and so has no fold - stated, not hidden).
Early stopping uses a validation set carved from the TRAINING
participants only: whole contiguous run-segments (run_id groups),
chosen per class by the frozen strategy search in
cnn_segval.carve_validation, with a 2-window purge of subtraining
windows adjacent to validation segments in the same recording's time
order. The held-out participant contributes nothing to fitting: not to
the subtrain, not to the validation carve, not to the normalisation
statistics and not to the class weights (all computed on the
subtraining fold only, inside the frozen train_fold_segval).

One mechanical adaptation, documented here because the frozen carve was
written for one participant's training sessions: the carve groups runs
by (session, run_id) and purges within a session. Across participants
the session labels repeat (everyone has an S01), so this driver passes
the carve a composite session key, participant + "/" + session. Runs
then group per participant-session recording and the purge stays inside
one recording, which is what the frozen code intends. Nothing else
about the carve changes.

Seeds: model seeds 0-4 on the fixed cap-seed-0 window set, exactly the
seed role the frozen cnn.py design specifies (its Week 2 within run
used model seeds {0,1} on this same set; the baseline's seeds instead
vary the cap subsample - the per-participant seed-mean comparison is
the established RQ2 contrast and the asymmetry is disclosed wherever
the contrast is reported).

Checkpointing: every fold lands in its own run folder
(experiments/runs/2026-11-21_cnn_loso_subjective_<PART>_s<seed>/) with
config.yaml, metrics.csv, predictions.csv and notes.md, written as
soon as the fold finishes. A fold whose metrics.csv already exists is
skipped, so re-running this driver after an interruption resumes where
it stopped. Run it seed by seed (--seeds 0, then --seeds 1, ...) so
each completed seed can be committed before the next starts.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.metrics import binary_metrics
from src.models.cnn import build_inputs
from src.models.cnn_segval import train_fold_segval

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
RUNS = ROOT / "experiments" / "runs"
DATE = "2026-11-21"


def fold_folder(part, seed):
    return RUNS / f"{DATE}_cnn_loso_subjective_{part}_s{seed}"


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

    for seed in seeds:
        for part in participants:
            d = fold_folder(part, seed)
            if (d / "metrics.csv").exists():
                print(f"[skip] {part} s{seed}: metrics.csv exists", flush=True)
                continue
            te = (idx.participant == part).to_numpy()
            tr = ~te
            if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
                print(f"[uncovered] {part} s{seed}: single-class fold",
                      flush=True)
                continue
            # Fold-local training index with the composite session key
            # the frozen carve needs (see module docstring).
            tr_pos = np.where(tr)[0]
            train_df = idx.iloc[tr_pos].copy()
            train_df["session"] = (train_df.participant + "/"
                                   + train_df.session)
            t0 = time.time()
            model, mean, std, note = train_fold_segval(
                X[tr], y[tr], train_df, seed)
            wall = time.time() - t0
            if model is None:
                print(f"[uncovered] {part} s{seed}: {note}", flush=True)
                continue
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
                   "validation": "SEGMENT-GROUPED: whole run-segments "
                                 "carved from training participants only, "
                                 "2-window purge at boundaries",
                   "channel_dropout": "temporal channels dropped p=0.2 per "
                                      "sample per batch during training",
                   "training": "Adam lr=0.001, batch=64, max_epochs=100, "
                               "early stopping patience=15 on "
                               "segment-grouped validation loss",
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
            (d / "notes.md").write_text(f"""# CNN (EEGNet) LOPO - subjective labels, held-out {part}, model seed {seed}

Split definition: leave-one-participant-out. Test = all {int(te.sum())}
windows of {part}; training = the other 10 participants' windows from
the canonical cap-seed-0 Amendment-1 set (806 windows, 11 participants;
P05 has no Amendment-1 windows and has no fold).
Early stopping: validation = whole contiguous RUN-SEGMENTS carved from
the training participants only (run_id groups; every run is
single-class), with a 2-window purge of subtraining windows adjacent to
validation segments in the same recording's time order; best-val-loss
weights restored. The carve received a composite participant/session
key so runs group per recording (driver docstring explains why).
Carve detail: {note}
Wall time for this fold (training + prediction): {wall:.1f} s on the
local VM (CPU), sharing the machine with other project work.

**Interpretation boundary (state wherever cited): the validation
measures segment generalisation inside known training participants,
NOT unseen-participant transfer. The held-out participant is the test
fold and is never touched during fitting.**

Mitigations: temporal CHANNEL DROPOUT during training (p = 0.2 per
sample, EEG zeroed + indicator plane zeroed - the mask-as-fingerprint
mitigation carried over from the Week 2 within run); masked temporals
are zero-filled AND flagged via indicator planes (Amendment 1 - never
silent zeros).

Leakage checklist:
- [x] Held-out participant contributes no window to subtrain, validation, normalisation statistics or class weights
- [x] Normalisation statistics from the subtraining fold only
- [x] Class weights from the subtraining fold only
- [x] Early stopping never touches the test fold (segment-grouped validation from training participants only, with purge)
- [x] No overlapping windows across splits (tumbling windows; participant-level split; purge at validation boundaries)
- [x] Label rules locked (docs/label_rules.md incl. Amendment 1) before training

Environment: local VM, CPU, torch {torch.__version__} (pinned stack,
docs/environment_pin.md). Model seeds 0-4 on the fixed cap-seed-0
window set; one run folder per fold so an interrupted run resumes.
""")
            print(f"[done] {part} s{seed}: BA={met['balanced_accuracy']:.3f} "
                  f"AUC={met['roc_auc']:.3f} MCC={met['mcc']:.3f} "
                  f"n={met['n']} wall={wall:.1f}s ({note})", flush=True)


if __name__ == "__main__":
    main()
