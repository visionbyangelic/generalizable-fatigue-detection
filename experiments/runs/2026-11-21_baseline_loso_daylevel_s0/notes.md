# Baseline LOPO, DAY-LEVEL session labels (RQ4, POST HOC), s0

RQ4 diagnostic run, post hoc arm. Labels are whole-session: earliest
recorded session (S01) = alert, latest recorded session (S03) =
fatigued, middle session excluded, every retained window in the
session carrying the session's label. P05 is excluded entirely: its
latest session retained no windows, so it has no day-level contrast.
Session labels are a diagnostic for label-definition artefacts
(docs/label_rules.md section 3) and are never presented as fatigue
ground truth.

Split definition: hold out ALL day-labelled windows of one
participant; train on the other 10 participants. A fold counts only
if train and test both contain both classes; skipped folds:
none.

Seed convention: this run has no stochastic element. The primary
baseline's seeds were cap-subsample seeds; there is no cap under
day-level labels, and the logistic regression is deterministic given
the data. The s0 suffix is the project's naming convention only.

Leakage checklist:
- [x] Normalisation (StandardScaler) fit on training fold only (Pipeline, per fold)
- [x] Median imputation for masked-channel NaNs fit on training fold only
- [x] Class weights computed from training fold only (class_weight='balanced')
- [x] No overlapping windows across splits (tumbling windows; split at participant level)
- [x] No tuning on test data (C=1.0 locked default from how-others-did-it.md section 6)
- [x] Label definition fixed in the driver docstring before the run (post hoc arm, labelled as such)

Environment: local VM (CPU), scikit-learn.
