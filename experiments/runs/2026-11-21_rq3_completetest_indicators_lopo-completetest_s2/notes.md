# RQ3 completetest / indicators / lopo-completetest — cap seed 2

Design: docs/rq3_preregistration.md incl. Amendment 1. Training on the full keep-all capped pool; the held-out participant is scored on their complete-data windows only (P08 and P10 flagged: their stopped bands removed 27 and 32 fatigued windows respectively). On complete test windows the validity indicators are constant, so the missingness pattern carries no information there.

Split definition: hold out one participant; train on the other participants' full capped pool; score only the held-out participant's complete-data windows. A fold counts only if train and test
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
(docs/environment_pin.md). Seed = M1-cap subsample seed 2;
logistic regression itself is deterministic given the data.
