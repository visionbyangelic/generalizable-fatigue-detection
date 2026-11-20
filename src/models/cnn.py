"""Phase C — 1D-CNN (EEGNet-8,2) under Amendment 1.

Architecture (Lawhern et al. 2018, per how-others-did-it.md section 6):
EEGNet F1=8, D=2, F2=16, temporal kernel 64 @128 Hz, pools (1,4)/(1,8),
dropout 0.25, Adam (lr 1e-3), cross-entropy with training-fold class
weights, batch 64, max 100 epochs, early stopping (patience 15, restore
best) on a GROUPED validation split drawn from training data only:
  - within protocol: validation = one whole training SESSION of the same
    participant (session-grouped); the model is trained on the remaining
    training session(s) only.
  - loso protocol: validation = one whole training PARTICIPANT
    (subject-grouped).

Amendment 1 mask handling (documented choice): the network input has 6
channels — the 4 EEG channels (masked temporals zero-filled, exactly as
stored in windows_a1*) PLUS 2 extra constant mask-indicator channels
(mask_TP9, mask_TP10: the window's validity value broadcast over time).
Missingness is therefore explicit to the model, never silent zeros alone.

Normalisation: per-EEG-channel mean/SD computed on the (sub)training fold
only; mask channels are left as 0/1.

Fold validity: train and test must each contain both classes (same rule as
the baseline). Local Week-2 run: within protocol, cap seed 0 (the canonical
windows_a1 set), model seeds {0,1} — a REDUCED seed set; the full
LOSO x 5-seed run is Weeks 3-4 on Colab (notebooks/04_cnn_colab.ipynb).
"""
import argparse
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

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
RUNS = ROOT / "experiments" / "runs"
TAB = ROOT / "results" / "tables"
EEG_CH = ["TP9", "AF7", "AF8", "TP10"]
MAX_EPOCHS, PATIENCE, BATCH, LR, DROPOUT = 100, 15, 64, 1e-3, 0.25


class EEGNet(nn.Module):
    def __init__(self, n_ch=6, n_time=512, f1=8, d=2, f2=16, dropout=DROPOUT):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f1 * d, (n_ch, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * d), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(dropout))
        self.block2 = nn.Sequential(
            nn.Conv2d(f1 * d, f1 * d, (1, 16), padding=(0, 8),
                      groups=f1 * d, bias=False),
            nn.Conv2d(f1 * d, f2, (1, 1), bias=False),
            nn.BatchNorm2d(f2), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(dropout))
        self.fc = nn.Linear(f2 * (n_time // 32), 2)

    def forward(self, x):
        x = self.block2(self.block1(x))
        return self.fc(x.flatten(1))


def build_inputs(arr, idx):
    """(n,4,512) EEG + validity -> (n,1,6,512) tensor-ready array."""
    mask_tp9 = idx["valid_TP9"].to_numpy(dtype=np.float32)
    mask_tp10 = idx["valid_TP10"].to_numpy(dtype=np.float32)
    n = len(idx)
    extra = np.zeros((n, 2, 512), dtype=np.float32)
    extra[:, 0, :] = mask_tp9[:, None]
    extra[:, 1, :] = mask_tp10[:, None]
    return np.concatenate([arr.astype(np.float32), extra], axis=1)[:, None]


def train_fold(X, y, groups, seed):
    """Session/subject-grouped validation: last sorted group = validation."""
    torch.manual_seed(seed); np.random.seed(seed)
    uniq = sorted(np.unique(groups))
    val_group = uniq[-1]
    va = groups == val_group
    tr = ~va
    if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
        tr, va = ~va, va  # fall back to the other group as validation
    if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
        return None, "no grouped validation split with both classes"
    mean = X[tr][:, 0, :4].mean(axis=(0, 2))
    std = X[tr][:, 0, :4].std(axis=(0, 2)) + 1e-8
    Xn = X.copy()
    Xn[:, 0, :4] = (X[:, 0, :4] - mean[None, :, None]) / std[None, :, None]
    cls = np.bincount(y[tr], minlength=2)
    w = torch.tensor(len(y[tr]) / (2 * cls), dtype=torch.float32)
    model = EEGNet()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss(weight=w)
    Xtr = torch.tensor(Xn[tr]); ytr = torch.tensor(y[tr])
    Xva = torch.tensor(Xn[va]); yva = torch.tensor(y[va])
    best, best_state, bad = np.inf, None, 0
    for ep in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            loss = lossf(model(Xtr[b]), ytr[b])
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
    return (model, Xn), f"val_group={val_group}, epochs_run={ep + 1}, best_val_loss={best:.4f}"


def predict(model, Xn, mask):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(Xn[mask]))
    proba = torch.softmax(logits, dim=1).numpy()
    return proba.argmax(1), proba[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["within", "loso"], default="within")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--cap-seed", type=int, default=0)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    if args.cap_seed == 0:
        arr = np.load(PROC / "windows_a1.npy")
        idx = pd.read_csv(PROC / "windows_a1_index.csv").reset_index(drop=True)
    else:
        sys.path.insert(0, str(ROOT))
        from src.models.baseline import cap_pool
        arr_u = np.load(PROC / "windows_a1_uncapped.npy")
        idx_u = pd.read_csv(PROC / "windows_a1_uncapped_index.csv").reset_index(drop=True)
        capped = cap_pool(idx_u, args.cap_seed)
        arr = arr_u[capped.index.to_numpy()]
        idx = capped.reset_index(drop=True)
    X = build_inputs(arr, idx)
    y = idx.label.to_numpy()
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()

    if args.protocol == "within":
        fold_defs = []
        for part, g in idx.groupby("participant"):
            for sess in g.session.unique():
                te = ((idx.participant == part) & (idx.session == sess)).to_numpy()
                tr = ((idx.participant == part) & (idx.session != sess)).to_numpy()
                fold_defs.append((f"{part}/{sess}", tr, te,
                                  idx.session.to_numpy()))
        group_col = "participant"
    else:
        fold_defs = []
        for part in idx.participant.unique():
            te = (idx.participant == part).to_numpy()
            fold_defs.append((part, ~te, te, idx.participant.to_numpy()))
        group_col = "participant"

    all_per = []
    for seed in seeds:
        preds, skipped, fold_notes = [], [], []
        for fid, tr, te, grp in fold_defs:
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                skipped.append(fid); continue
            out, note = train_fold(X[tr], y[tr], grp[tr], seed)
            if out is None:
                skipped.append(f"{fid} ({note})"); continue
            model, Xn_tr = out
            # Recompute normalisation-matched test inputs: train_fold returns
            # normalised TRAIN array only; recompute stats identically here.
            mean = X[tr][:, 0, :4][
                np.isin(np.arange(tr.sum()),
                        np.where(grp[tr] != sorted(np.unique(grp[tr]))[-1])[0])
            ].mean(axis=(0, 2)) if False else None
            fold_notes.append(f"{fid}: {note}")
            # NOTE: predict on test with the SAME normalisation: recompute by
            # re-running the normalisation from train_fold via stored stats.
            # (train_fold normalises internally; for the test set we rebuild
            # Xn for the full fold below.)
            preds.append((fid, tr, te, model))
        # Second pass: normalise test with training-fold stats stored per fold
        # is not retained, so instead train_fold is re-entered via a wrapper:
        # — simpler: we stored the model; normalisation stats are recomputed
        # deterministically from the training fold (same rule as train_fold).
        recs = []
        for fid, tr, te, model in preds:
            groups_tr = (idx.session if args.protocol == "within"
                         else idx.participant).to_numpy()[tr]
            uniq = sorted(np.unique(groups_tr))
            va = groups_tr == uniq[-1]
            if len(np.unique(y[tr][~va])) < 2 or len(np.unique(y[tr][va])) < 2:
                va = ~va
            sub = np.where(tr)[0][~va]
            mean = X[sub][:, 0, :4].mean(axis=(0, 2))
            std = X[sub][:, 0, :4].std(axis=(0, 2)) + 1e-8
            Xn_te = X[te].copy()
            Xn_te[:, 0, :4] = (Xn_te[:, 0, :4] - mean[None, :, None]) / std[None, :, None]
            model.eval()
            with torch.no_grad():
                logits = model(torch.tensor(Xn_te))
            proba = torch.softmax(logits, dim=1).numpy()
            rec = idx.iloc[np.where(te)[0]][["participant", "session"]].copy()
            rec["fold"] = fid; rec["y_true"] = y[te]
            rec["y_pred"] = proba.argmax(1); rec["y_proba"] = proba[:, 1]
            recs.append(rec)
        preds_df = pd.concat(recs)
        per = []
        for gid, g in preds_df.groupby(group_col):
            per.append({group_col: gid, **binary_metrics(g.y_true, g.y_pred, g.y_proba)})
        per = pd.DataFrame(per)
        name = f"2026-11-20_cnn_{args.protocol}_subjective_s{seed}"
        d = RUNS / name; d.mkdir(parents=True, exist_ok=True)
        agg = {c: per[c].mean() for c in per.columns if c != group_col}
        metrics = pd.concat([per, pd.DataFrame([{group_col: "AGGREGATE_mean_over_groups", **agg}])],
                            ignore_index=True)
        metrics.to_csv(d / "metrics.csv", index=False)
        preds_df.to_csv(d / "predictions.csv", index=False)
        cfg = {"model": "EEGNet-8,2 (kernel 64 @128 Hz)", "input": "6 channels: 4 EEG + mask_TP9 + mask_TP10 indicator planes",
               "training": f"Adam lr={LR}, batch={BATCH}, max_epochs={MAX_EPOCHS}, early stopping patience={PATIENCE} on grouped validation (see notes)",
               "dropout": DROPOUT, "class_weights": "subtraining-fold balanced",
               "normalisation": "per-EEG-channel mean/SD from subtraining fold only",
               "protocol": args.protocol, "labels": "subjective (Amendment 1 windows)",
               "cap_seed": args.cap_seed, "model_seed": seed,
               "seed_note": "REDUCED seed set {0,1} for Week 2 within run; full LOSO x 5 seeds = Weeks 3-4 on Colab",
               "git_commit": commit, "torch": torch.__version__}
        (d / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        (d / "notes.md").write_text(f"""# CNN (EEGNet) {args.protocol} — subjective labels, model seed {seed}

Split definition: {'per participant, rotate held-out session' if args.protocol == 'within' else 'leave-one-subject-out'}. Folds require both classes in train and test; skipped: {skipped if skipped else 'none'}.
Early stopping: grouped validation inside the training data only ({'one whole training session held out as validation; model trained on the remaining session(s)' if args.protocol == 'within' else 'one whole training participant held out as validation'}); best-val-loss weights restored. Fold details: {'; '.join(fold_notes)}.

Leakage checklist:
- [x] Normalisation statistics from the (sub)training fold only
- [x] Class weights from the (sub)training fold only
- [x] Early stopping never touches the test fold (grouped validation from training data)
- [x] No overlapping windows across splits (tumbling windows; session-level splits)
- [x] Masked temporal channels are zero-filled AND flagged via explicit mask-indicator input channels (Amendment 1 — never silent zeros)
- [x] Label rules locked (docs/label_rules.md incl. Amendment 1) before training

Environment: local VM, CPU, torch {torch.__version__}. Reduced seed set: Week 2
within-protocol seeds {{0,1}} only; full LOSO x 5 seeds runs Weeks 3-4 on Colab.
""")
        per["seed"] = seed
        all_per.append(per)
        a = per[["balanced_accuracy", "macro_f1", "roc_auc", "mcc"]].mean()
        print(f"seed {seed} {args.protocol}: participants={len(per)} skipped={skipped} "
              f"bal_acc={a.balanced_accuracy:.3f} macroF1={a.macro_f1:.3f} "
              f"AUC={a.roc_auc:.3f} MCC={a.mcc:.3f}")
    allp = pd.concat(all_per)
    summ = allp.groupby(group_col)[["balanced_accuracy", "macro_f1", "roc_auc", "mcc",
                                    "recall_alert", "recall_fatigued",
                                    "precision_alert", "precision_fatigued", "n"]].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.reset_index().to_csv(TAB / f"cnn_{args.protocol}_subjective_a1_summary.csv", index=False)


if __name__ == "__main__":
    main()
