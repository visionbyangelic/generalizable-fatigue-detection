# 2026-11-20_baseline_loso_subjective_V2sessnorm_s0

PRE-DECLARED SENSITIVITY - estimates and intervals only. No verdict
under the decision-table lock is declared in this run; reporting
status awaits the lead reviewer's ruling.

Split definition: leave-one-PARTICIPANT-out: hold out ALL windows of one participant; train on the other 11 participants (uncapped pool). P05 cannot be a test fold (its retained windows are single-class: 5 alert / 0 fatigued) but its windows are in the training pools of the other folds - unlike the capped primary, where the cap removed them entirely. A fold counts only if train and test
both contain both classes; skipped folds: ['P05'].

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
