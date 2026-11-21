# 2026-11-21_poscontrol_permuted_drozy-rt_all_dose0.25_r0_s0

Amendment 2 primary (permuted-label) positive control, RT
arm, dose 0.25 (theta power x1.250, alpha power
x1.125), repeat 0. The repeat's labels are a within-
participant session-label swap of the true labels (rng
70000), so any natural label-signal link is null by
construction; the injection was applied to the permuted-fatigued
windows and the frozen LOPO ran under the permuted labels. Recovery
here can only be the injected effect.

Result: recovery balanced accuracy 0.550 [0.506, 0.591]
(participant bootstrap). Swap null on this repeat's injected
features: 41 of 100 planned swaps
run, 5 reached the observed value,
p >= 0.059 (early stop; detection verdict decided).
Manipulation check (mean log10 shift in injected windows): theta
0.093 (target 0.097), alpha
0.053 (target 0.051).
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
