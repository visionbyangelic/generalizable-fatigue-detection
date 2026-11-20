# 2026-11-20_baseline_session-out_subjective_V2sessnorm_s0

PRE-DECLARED SENSITIVITY - estimates and intervals only. No verdict
under the decision-table lock is declared in this run; reporting
status awaits the lead reviewer's ruling.

Split definition: pooled: hold out one (participant, session); train on all remaining windows of all participants (uncapped pool, so single-class sessions contribute training windows but cannot be test folds). A fold counts only if train and test
both contain both classes; skipped folds: ['P01/S03', 'P02/S01', 'P03/S03', 'P04/S02', 'P05/S01', 'P06/S01', 'P09/S02', 'P10/S01', 'P11/S03', 'P12/S01'].

Variant V2 - WITHIN-SESSION-NORMALISED FEATURES: each
continuous feature was centred/scaled per session using that
session's own windows (nanmean / nanstd ddof=0 over the session's
retained windows; labels never used). TRANSDUCTIVITY, stated as
required: this normalisation uses each session's own UNLABELLED
windows - including, for a test session, that test session's own
windows - so it is transductive at session level by design (Reviewer 2
review 10 declared sensitivity). It is NOT a substitute for the
primary's train-fold-only normalisation; the Pipeline's imputation
and StandardScaler still fit on training folds only, on top of the
session-normalised features. Indicator features are unchanged.

Leakage checklist:
- [x] Normalisation (StandardScaler) fit on training fold only (Pipeline, per fold)
- [x] Median imputation fit on training fold only
- [x] Class weights from training fold only (class_weight='balanced')
- [x] No overlapping windows across splits (tumbling windows; splits at session/participant level)
- [x] No tuning on test data (C=1.0 locked default)
- [x] Label rules locked (docs/label_rules.md incl. Amendment 1) before any training
- [x] Variant declared in advance (track.md, 20 Nov 2026) - not chosen from results

Environment: local VM (CPU), scikit-learn 1.9.1.
