# CNN (EEGNet) within - subjective labels, model seed 1 - SEGMENT-GROUPED validation

Split definition: per participant, rotate held-out session over sessions
containing both classes (train = the participant's other sessions).
Early stopping: validation = whole contiguous RUN-SEGMENTS carved from
the training sessions only (run_id groups; every run is single-class),
with a 2-window purge of subtraining windows adjacent to validation
segments in the same session's time order; best-val-loss weights
restored. Feasibility criteria (fixed before running): validation both
classes and >=6 windows; post-purge subtrain both classes and
>=20 windows.

**Interpretation boundary (state wherever cited): this validation
measures segment generalisation within known sessions, NOT unseen-session
transfer. The test session itself is fully held out.**

Fold details: P07/S03: remaining training sessions are single-class; P10/S02: remaining training sessions are single-class; P12/S02: remaining training sessions are single-class; P02/S02: strategy=last1, val_runs=[('S03', 11), ('S03', 46)], val_n=17, subtrain_n=35 (post-purge), epochs_run=19, best_val_loss=0.6804; P03/S02: strategy=first2, val_runs=[('S01', 43), ('S01', 48), ('S01', 240), ('S01', 241)], val_n=6, subtrain_n=26 (post-purge), epochs_run=16, best_val_loss=0.6994; P04/S01: strategy=last2, val_runs=[('S03', 35), ('S03', 37), ('S03', 365), ('S03', 382)], val_n=6, subtrain_n=23 (post-purge), epochs_run=100, best_val_loss=0.0090; P06/S03: strategy=first1, val_runs=[('S02', 33), ('S01', 1931)], val_n=18, subtrain_n=25 (post-purge), epochs_run=29, best_val_loss=0.6860; P08/S01: strategy=last1, val_runs=[('S03', 40), ('S03', 88)], val_n=19, subtrain_n=71 (post-purge), epochs_run=16, best_val_loss=0.6952; P08/S02: strategy=last1, val_runs=[('S03', 40), ('S03', 88)], val_n=19, subtrain_n=48 (post-purge), epochs_run=54, best_val_loss=0.5036; P08/S03: strategy=last1, val_runs=[('S02', 30), ('S02', 88)], val_n=16, subtrain_n=63 (post-purge), epochs_run=88, best_val_loss=0.2900; P09/S01: strategy=first1, val_runs=[('S03', 49), ('S03', 114)], val_n=6, subtrain_n=40 (post-purge), epochs_run=33, best_val_loss=0.6052; P09/S03: strategy=last1, val_runs=[('S01', 112), ('S01', 289)], val_n=6, subtrain_n=48 (post-purge), epochs_run=34, best_val_loss=0.3199; P11/S01: strategy=last1, val_runs=[('S02', 11), ('S02', 20)], val_n=15, subtrain_n=47 (post-purge), epochs_run=100, best_val_loss=0.1590; P11/S02: strategy=first1, val_runs=[('S01', 65), ('S01', 176)], val_n=6, subtrain_n=36 (post-purge), epochs_run=16, best_val_loss=0.5793
Skipped / uncovered folds: ['P01/S01 (UNCOVERED: no run-segment carve gives both classes in validation (>=6 windows) and subtrain (>=20 windows after purge))', 'P01/S02 (UNCOVERED: no run-segment carve gives both classes in validation (>=6 windows) and subtrain (>=20 windows after purge))', 'P02/S03 (UNCOVERED: no run-segment carve gives both classes in validation (>=6 windows) and subtrain (>=20 windows after purge))', 'P03/S01 (UNCOVERED: no run-segment carve gives both classes in validation (>=6 windows) and subtrain (>=20 windows after purge))', 'P04/S03 (UNCOVERED: no run-segment carve gives both classes in validation (>=6 windows) and subtrain (>=20 windows after purge))', 'P06/S02 (UNCOVERED: no run-segment carve gives both classes in validation (>=6 windows) and subtrain (>=20 windows after purge))']
Participants with no covered fold in this seed: ['P01', 'P07', 'P10', 'P12']

Mitigations: temporal CHANNEL DROPOUT during training (p=0.2 per
sample, EEG zeroed + indicator plane zeroed - Reviewer 1 review 5
mask-as-fingerprint mitigation); masked temporals are zero-filled AND
flagged via indicator planes (Amendment 1 - never silent zeros).

Leakage checklist:
- [x] Normalisation statistics from the subtraining fold only
- [x] Class weights from the subtraining fold only
- [x] Early stopping never touches the test fold (validation carved from training sessions only, segment-grouped with purge)
- [x] No overlapping windows across splits (tumbling windows; session-level test split; purge at validation boundaries)
- [x] Label rules locked (docs/label_rules.md incl. Amendment 1) before training

Environment: local VM, CPU, torch 2.15.0+cpu. Reduced seed set
{0,1}; full LOSO x 5 seeds runs Weeks 3-4 on Colab. This run supersedes
the session-grouped-validation attempt preserved in the *_STOPPED_sessionval
folders (P08-only partial run, kept for audit).
