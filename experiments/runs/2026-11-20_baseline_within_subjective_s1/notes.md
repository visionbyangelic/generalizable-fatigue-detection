# Baseline within — subjective labels, cap seed 1

Split definition: per participant, rotate held-out session (train = other sessions of the same participant). A fold counts only if train and test both contain both classes; skipped folds: ['P04/S02', 'P06/S01', 'P07/S03', 'P10/S01', 'P10/S02', 'P12/S01', 'P12/S02'].

Leakage checklist:
- [x] Normalisation (StandardScaler) fit on training fold only (Pipeline, per fold)
- [x] Median imputation for masked-channel NaNs fit on training fold only
- [x] Class weights computed from training fold only (class_weight='balanced')
- [x] No overlapping windows across splits (tumbling windows; splits at session level)
- [x] No tuning on test data (C=1.0 locked default from how-others-did-it.md section 6)
- [x] Label rules locked (docs/label_rules.md incl. Amendment 1) before any training

Environment: local VM (CPU), scikit-learn. Seed = M1-cap subsample seed 1;
logistic regression itself is deterministic given the data.
