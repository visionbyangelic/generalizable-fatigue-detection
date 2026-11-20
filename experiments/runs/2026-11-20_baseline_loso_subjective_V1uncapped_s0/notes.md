# 2026-11-20_baseline_loso_subjective_V1uncapped_s0

PRE-DECLARED SENSITIVITY - estimates and intervals only. No verdict
under the decision-table lock is declared in this run; reporting
status awaits the lead reviewer's ruling.

Split definition: leave-one-PARTICIPANT-out: hold out ALL windows of one participant; train on the other 11 participants (uncapped pool). P05 cannot be a test fold (its retained windows are single-class: 5 alert / 0 fatigued) but its windows are in the training pools of the other folds - unlike the capped primary, where the cap removed them entirely. A fold counts only if train and test
both contain both classes; skipped folds: ['P05'].

Variant V1 - UNCAPPED + CLASS-WEIGHTED: the full Amendment-1
retained pool (1,053 windows: 564 alert / 489 fatigued) with NO
per-session M1 cap. Class imbalance is handled by the training-fold
balanced class weights instead of by discarding alert windows (the
cap discarded ~44% of retained alert windows in the primary). Design
difference vs the capped primary, stated openly: single-class
sessions, which the cap reduced to zero windows, contribute their
windows to TRAINING pools here; they still cannot be test folds.

Leakage checklist:
- [x] Normalisation (StandardScaler) fit on training fold only (Pipeline, per fold)
- [x] Median imputation fit on training fold only
- [x] Class weights from training fold only (class_weight='balanced')
- [x] No overlapping windows across splits (tumbling windows; splits at session/participant level)
- [x] No tuning on test data (C=1.0 locked default)
- [x] Label rules locked (docs/label_rules.md incl. Amendment 1) before any training
- [x] Variant declared in advance (track.md, 20 Nov 2026) - not chosen from results

Environment: local VM (CPU), scikit-learn 1.9.1.
