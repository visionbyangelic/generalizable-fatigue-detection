# RQ3 ablation (post hoc) / hr / lopo — cap seed 2

POST HOC AND DESCRIPTIVE. This arm was not in docs/rq3_preregistration.md
or Amendment 1. It was added after the RQ3 results were reviewed, to
record what this signal family scores on its own under the identical
locked design. No feature-set selection is made from it, and it is
reported beside the peripheral arm as description only.

Pool: the full 1,053-window primary pool, M1 capped per session per
cap seed, exactly as the Week 2 baseline and the RQ3 grid.

Split definition: hold out ALL sessions of one participant; train on
the other participants. A fold counts only if train and test each
contain both classes; skipped folds: none.
Metrics in metrics.csv are per participant (out-of-fold predictions
concatenated per participant), the aggregation the project fixes.

Leakage checklist:
- [x] Normalisation (StandardScaler) fit on training fold only (Pipeline, per fold)
- [x] Median imputation fit on training fold only
- [x] Class weights computed from training fold only (class_weight='balanced')
- [x] No overlapping windows across splits (tumbling windows; splits at participant level)
- [x] No tuning on test data (C=1.0 locked, identical to the Week 2 baseline)
- [x] Label rules locked (docs/label_rules.md incl. Amendment 1) before any training
- [x] Peripheral features computed by the declared extractor only (SHA-256 ced46c059143...), no feature added or tuned

Environment: local VM (CPU), scikit-learn, pinned stack
(docs/environment_pin.md). Seed = M1-cap subsample seed 2;
logistic regression itself is deterministic given the data.
