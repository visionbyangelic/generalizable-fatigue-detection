# 2026-11-21_poscontrol_dose0-identity_drozy-kss_all_s0

Amendment 2 dose-0 identity check, KSS arm, all-channel
features. Dose 0 (theta power x1.0, alpha power x1.0) was pushed
through the injection code path on the true labels and the frozen
LOPO run repeated. The point is adapter validation, not a result:
if this number is not the reported uninjected score, the injection
path (or the window alignment behind it) is not faithful and the
control does not proceed.

Result: balanced accuracy 0.6022 against the reported 0.6022
(abs diff 0.00e+00); identity OK.
Skipped folds: none.

Leakage checklist:
- [x] Normalisation (StandardScaler) fit on training fold only (Pipeline, per fold)
- [x] Median imputation for masked-channel NaNs fit on training fold only
- [x] Class weights computed from training fold only (class_weight='balanced')
- [x] No overlapping windows across splits (tumbling windows; splits at participant level)
- [x] Held-out participant contributes nothing to fitting, imputation, scaling or tuning
- [x] No tuning on test data (C=1.0 locked default)
- [x] Injection is the declared Amendment 2 manipulation: it uses the run's own labels by design (a synthetic effect painted on fatigued-labelled windows), applied to the whole arm before splitting, exactly as in the Week 2 FatigueSet control; it moves no information from any test fold into fitting beyond that declared manipulation

Environment: local VM (CPU), scikit-learn 1.9.1, MNE 1.13.2.
