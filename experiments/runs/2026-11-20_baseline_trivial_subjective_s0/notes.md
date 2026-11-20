# Trivial baselines - session-out, cap seeds 0-4

Same folds and pipeline as the main baseline. Variants: (a) a single
feature = the window's normalised position within its session (0..1 by
start_sample among the session's capped windows) - a pure time-on-task
probe; (b) delta-band log-power only (4 channels, valid channels, NaNs
median-imputed on the training fold); (c) the full 43-feature model for
reference. Aggregate per seed = mean over valid held-out sessions.

Leakage checklist:
- [x] Scaling/imputation fit on training folds only
- [x] Window position uses only protocol metadata (window order), no labels
- [x] No tuning on test data (locked C=1.0)

Environment: local VM (CPU), scikit-learn.
