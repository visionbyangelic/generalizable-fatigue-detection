# RQ3 keepall / peripheral / sessionout — cap seed 1

Design: docs/rq3_preregistration.md incl. Amendment 1. Pool: the full 1,053-window primary pool, M1 capped per session per cap seed, exactly as the Week 2 baseline.

Split definition: pooled: hold out one (participant, session); train on all remaining sessions of all participants; metrics aggregated per participant. A fold counts only if train and test
each contain both classes; skipped folds: ['P04/S02', 'P06/S01', 'P10/S01', 'P12/S01'].
Metrics in metrics.csv are per participant (out-of-fold predictions
concatenated per participant), the aggregation the amendment fixes.

Leakage checklist:
- [x] Normalisation (StandardScaler) fit on training fold only (Pipeline, per fold)
- [x] Median imputation fit on training fold only
- [x] Class weights computed from training fold only (class_weight='balanced')
- [x] No overlapping windows across splits (tumbling windows; splits at participant/session level)
- [x] No tuning on test data (C=1.0 locked, identical to the Week 2 baseline)
- [x] Label rules locked (docs/label_rules.md incl. Amendment 1) before any training
- [x] Peripheral features computed by the declared extractor only (SHA-256 ced46c059143...), no feature added or tuned

Environment: local VM (CPU), scikit-learn, pinned stack
(docs/environment_pin.md). Seed = M1-cap subsample seed 1;
logistic regression itself is deterministic given the data.
