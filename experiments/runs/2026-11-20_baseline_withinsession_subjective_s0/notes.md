# Baseline within-session ceiling check - subjective labels, cap seed 0

Diagnostic requested by Reviewer 2 review 6 (docs/methods_notes.md):
does ANY early-vs-late signal exist inside a single session? Sessions
eligible (both classes, >=20 windows/class after cap): ['P02/S03', 'P08/S02', 'P08/S03', 'P09/S01', 'P09/S03', 'P10/S02', 'P11/S01', 'P11/S02', 'P12/S02'].

Split: within each session, each class's windows (time-ordered) are cut
into 3 contiguous chunks; fold k tests on chunk k of BOTH classes and
trains on the remaining chunks of the same session, dropping a 2-window
purge on each side of every test chunk. Metrics per session are computed
on the concatenated out-of-fold predictions.

Aggregate (mean over sessions): balanced accuracy 0.744,
ROC-AUC 0.778, MCC 0.498.

Leakage checklist:
- [x] Imputation/scaling fit on the training segments only (Pipeline, per fold)
- [x] Class weights from training segments only
- [x] No window in both train and test; purge buffer at segment boundaries
- [x] No tuning on test data (locked C=1.0)
- [x] Label rules locked before any training

Environment: local VM (CPU), scikit-learn. NOTE: this is a diagnostic,
not one of the PLAN section 6 protocols; within a session the label is
perfectly confounded with block/time by construction (label_rules.md
section 1), so a high score here shows session-specific separability,
not fatigue physiology.
