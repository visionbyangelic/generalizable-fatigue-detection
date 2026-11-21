# Baseline within — SESSION-PHASE labels (RQ4), arm all36, cap seed 4

RQ4 diagnostic run. Labels come from block position only (M1 = alert,
M3 = fatigued); no subjective rating is consulted. Arm: all 36 sessions (primary pool + the 4 non-increasing drift/sensitivity check sessions).
Session labels are a diagnostic for label-definition artefacts
(docs/label_rules.md section 3) and are never presented as fatigue
ground truth.

Split definition: per participant, rotate held-out session (train = other sessions of the same participant). A fold counts only if train
and test both contain both classes; skipped folds: ['P04/S02', 'P06/S01', 'P10/S01', 'P10/S02', 'P10/S03', 'P12/S01', 'P12/S02', 'P12/S03'].

Leakage checklist:
- [x] Normalisation (StandardScaler) fit on training fold only (Pipeline, per fold)
- [x] Median imputation for masked-channel NaNs fit on training fold only
- [x] Class weights computed from training fold only (class_weight='balanced')
- [x] No overlapping windows across splits (tumbling windows; splits at session/participant level)
- [x] No tuning on test data (C=1.0 locked default from how-others-did-it.md section 6)
- [x] Label rules locked (docs/label_rules.md section 3, RQ4 diagnostic) before any training

Environment: local VM (CPU), scikit-learn. Seed = M1-cap subsample seed 4;
logistic regression itself is deterministic given the data.
