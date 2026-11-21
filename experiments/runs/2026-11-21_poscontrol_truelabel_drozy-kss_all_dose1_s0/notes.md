# 2026-11-21_poscontrol_truelabel_drozy-kss_all_dose1_s0

Amendment 2 companion (true-label) positive control, KSS
arm, dose 1 (theta power x2.000, alpha power
x1.500). The injection was applied to the truly fatigued
windows and the frozen LOPO ran under the true labels. Reported as
the increment over the arm's uninjected score, because on true
labels the natural signal and the injected effect are mixed.

Result: balanced accuracy 0.851 against the uninjected
0.602, increment +0.249. Swap null on the increment
(injected and uninjected statistics computed under the same swap
pattern): 100 of 100 planned swaps
run, 0 reached the observed increment,
p = 0.010.
Manipulation check (mean log10 shift in injected windows): theta
0.291 (target 0.301), alpha
0.181 (target 0.176).
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
