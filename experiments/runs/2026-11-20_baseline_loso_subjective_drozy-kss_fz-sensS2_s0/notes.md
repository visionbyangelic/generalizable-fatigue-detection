# 2026-11-20_baseline_loso_subjective_drozy-kss_fz-sensS2_s0

Split definition: leave-one-participant-out. Test = all windows of one
participant (their alert + fatigued sessions for the KSS label
arm); train = the other participants. 12 folds. A fold counts only if
train and test both contain both classes; skipped folds: none.
Participants dropped for this sensitivity run: ['P07']. Single deterministic pass: the DROZY design has no subsampling step
(no per-session cap; each participant contributes exactly one alert and
one fatigued session), so there is no seed for the data to vary with,
and logistic regression is deterministic given the data. The folder
suffix s0 marks that single pass, the same convention as the Week 2
precision variants.

Result: balanced accuracy 0.528, 95% CI [0.484, 0.575] by
participant bootstrap (10,000 resamples over per-participant values,
rng 42). No permutation null for sensitivity runs.

Leakage checklist:
- [x] Normalisation (StandardScaler) fit on training fold only (Pipeline, per fold)
- [x] Median imputation for masked-channel NaNs fit on training fold only
- [x] Class weights computed from training fold only (class_weight='balanced')
- [x] No overlapping windows across splits (tumbling windows; splits at participant level)
- [x] Held-out participant contributes nothing to fitting, imputation, scaling or tuning
- [x] No tuning on test data (C=1.0 locked default)
- [x] Label rules fixed in docs/dataset2_preregistration.md incl. Amendment 1 before any training

Environment: local VM (CPU), scikit-learn 1.9.1, MNE 1.13.2.
