# Baseline session-out — SESSION-PHASE labels (RQ4), arm restricted32, cap seed 1

RQ4 diagnostic run. Labels come from block position only (M1 = alert,
M3 = fatigued); no subjective rating is consulted. Arm: 32 delta-mental > 0 sessions only (identical windows to the Week 2 primary pool).
Session labels are a diagnostic for label-definition artefacts
(docs/label_rules.md section 3) and are never presented as fatigue
ground truth.

Split definition: pooled: hold out one (participant, session); train on all remaining sessions of all participants. A fold counts only if train
and test both contain both classes; skipped folds: ['P04/S02', 'P06/S01', 'P10/S01', 'P12/S01'].

Leakage checklist:
- [x] Normalisation (StandardScaler) fit on training fold only (Pipeline, per fold)
- [x] Median imputation for masked-channel NaNs fit on training fold only
- [x] Class weights computed from training fold only (class_weight='balanced')
- [x] No overlapping windows across splits (tumbling windows; splits at session/participant level)
- [x] No tuning on test data (C=1.0 locked default from how-others-did-it.md section 6)
- [x] Label rules locked (docs/label_rules.md section 3, RQ4 diagnostic) before any training

Environment: local VM (CPU), scikit-learn. Seed = M1-cap subsample seed 1;
logistic regression itself is deterministic given the data.
