# RQ3 restricted / indicators / lopo — cap seed 4

Design: docs/rq3_preregistration.md incl. Amendment 1. Pool: the 993 windows with hr_valid = eda_valid = 1, capped inside the restricted pool. This pool is a co-gate with a conditional estimand (performance when the band worked); it is not a cleaner pool: the restriction removes fatigued windows at about 12% and alert windows at about 0.2%.

Split definition: hold out ALL sessions of one participant; train on the other participants. A fold counts only if train and test
each contain both classes; skipped folds: none.
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
(docs/environment_pin.md). Seed = M1-cap subsample seed 4;
logistic regression itself is deterministic given the data.
