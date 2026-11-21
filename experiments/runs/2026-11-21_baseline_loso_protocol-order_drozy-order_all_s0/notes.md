# 2026-11-21_baseline_loso_protocol-order_drozy-order_all_s0

Split definition: leave-one-participant-out, as in the DROZY label
arms. Labels come from session order only: each participant's
earliest session with data is labelled alert, their latest session
fatigued, any middle session excluded (results/tables/
drozy_labels_order.csv). Participant 12 stays excluded (one session,
no contrast). 13 folds; skipped folds: none.

Why this run exists: it is an exploratory check written after the
DROZY headline results were seen. The KSS arm's fatigued session is
the latest session for most participants, so a model that only
learned 'later session' could in principle reproduce the KSS score
without reading sleepiness at all. This arm measures exactly that.

Result: balanced accuracy 0.536, 95% CI [0.493, 0.572] by
participant bootstrap (10,000 resamples over per-participant values,
rng 42). Permutation null (200 within-participant session-label
swaps, computed for this arm): mean 0.505, SD
0.039, p = 0.229.

Leakage checklist:
- [x] Normalisation (StandardScaler) fit on training fold only (Pipeline, per fold)
- [x] Median imputation for masked-channel NaNs fit on training fold only
- [x] Class weights computed from training fold only (class_weight='balanced')
- [x] No overlapping windows across splits (tumbling windows; splits at participant level)
- [x] Held-out participant contributes nothing to fitting, imputation, scaling or tuning
- [x] No tuning on test data (C=1.0 locked default)
- [x] Label construction fixed in this file's docstring and the checks document before the run

Environment: local VM (CPU), scikit-learn 1.9.1, MNE 1.13.2.
