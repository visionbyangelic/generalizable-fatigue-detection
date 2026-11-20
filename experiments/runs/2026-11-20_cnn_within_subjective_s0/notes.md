# CNN (EEGNet) within — subjective labels, model seed 0

Split definition: per participant, rotate held-out session (train = other sessions of the same participant). Folds require both classes in subtrain, validation and test; skipped: ['P01/S01 (no grouped validation split with both classes)', 'P01/S02 (no grouped validation split with both classes)', 'P02/S02 (no grouped validation split with both classes)', 'P02/S03 (no grouped validation split with both classes)', 'P03/S01 (no grouped validation split with both classes)', 'P03/S02 (no grouped validation split with both classes)', 'P04/S01 (no grouped validation split with both classes)', 'P04/S02', 'P04/S03 (no grouped validation split with both classes)', 'P06/S01', 'P06/S02 (no grouped validation split with both classes)', 'P06/S03 (no grouped validation split with both classes)', 'P07/S03', 'P09/S01 (no grouped validation split with both classes)', 'P09/S03 (no grouped validation split with both classes)', 'P10/S01', 'P10/S02', 'P11/S01 (no grouped validation split with both classes)', 'P11/S02 (no grouped validation split with both classes)', 'P12/S01', 'P12/S02'].
Early stopping: grouped validation inside the training data only (one whole training session held out as validation; model trained on the remaining session(s)); best-val-loss weights restored.
Fold details: P08/S01: val_group=S03, subtrain_n=54, val_n=42, epochs_run=16, best_val_loss=0.6944; P08/S02: val_group=S03, subtrain_n=31, val_n=42, epochs_run=18, best_val_loss=0.6778; P08/S03: val_group=S02, subtrain_n=31, val_n=54, epochs_run=17, best_val_loss=1108416.3750

Leakage checklist:
- [x] Normalisation statistics from the subtraining fold only
- [x] Class weights from the subtraining fold only
- [x] Early stopping never touches the test fold (grouped validation from training data only)
- [x] No overlapping windows across splits (tumbling windows; session-level splits)
- [x] Masked temporal channels are zero-filled AND flagged via explicit mask-indicator input channels (Amendment 1 — never silent zeros)
- [x] Label rules locked (docs/label_rules.md incl. Amendment 1) before training

Environment: local VM, CPU, torch 2.15.0+cpu. Reduced seed set: Week 2
within-protocol model seeds {0,1} only; full LOSO x 5 seeds runs Weeks 3-4 on Colab.

## COVERAGE LIMITATION (added 20 Nov 2026 — read before citing)
This run covers ONLY P08. Reason: after the per-session M1 cap, most
participants have exactly 2 usable sessions, so a within-subject training
fold contains a single session once the test session is held out — a
session-grouped validation split then leaves no training data. Where 2
training sessions exist, one is frequently single-class. A both-classes
session-grouped validation split therefore exists only for P08's folds.
The CNN within-subject protocol as specified (session-grouped early
stopping) is structurally infeasible for 7 of the 8 baseline-evaluable
participants on this dataset; it was STOPPED here rather than silently
deviating (see project lead report 20 Nov 2026). Do not cite these numbers
as the within-subject CNN result — they are a single-participant partial
run, kept for audit.
