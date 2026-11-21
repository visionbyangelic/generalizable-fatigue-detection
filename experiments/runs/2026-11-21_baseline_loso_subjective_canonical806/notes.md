# Baseline LOPO - subjective labels, canonical cap-seed-0 set (806 windows)

Split definition: leave-one-participant-out on the canonical
cap-seed-0 Amendment-1 set (data/processed/windows_a1*): the exact
806 windows, and the exact 11 folds, of the CNN LOPO run. One
deterministic pass: the cap seed is embodied in the set and logistic
regression has no model seed. Skipped folds: none.

WHY (panel-owed sensitivity): the RQ2 contrast as first reported
mixed model class with window set (CNN on these 806 windows,
baseline headline over five cap seeds on the 1,053-window pool).
This run holds the window set fixed so the recomputed contrast is a
model comparison only. It is a sensitivity; the primary baseline
headline remains the five-cap-seed estimate.

Leakage checklist:
- [x] Held-out participant contributes no window to training
- [x] Normalisation (StandardScaler) fit on training fold only (Pipeline, per fold)
- [x] Median imputation for masked-channel NaNs fit on training fold only
- [x] Class weights computed from training fold only (class_weight='balanced')
- [x] No overlapping windows across splits (tumbling windows; participant-level split)
- [x] No tuning on test data (C=1.0 locked default)
- [x] Label rules locked (docs/label_rules.md incl. Amendment 1) before any training

Environment: local VM (CPU), scikit-learn (pinned stack,
docs/environment_pin.md).
