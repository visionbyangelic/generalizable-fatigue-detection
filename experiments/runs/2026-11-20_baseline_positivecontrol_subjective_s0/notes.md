# Positive control - injected theta/alpha effect, session-out

Panel addition (Reviewer 1 review 7): can this pipeline detect a
fatigue-like spectral effect when one is DEFINITELY present? Method:
for every fatigued-class window (label 1) in windows_a1_uncapped, each
VALID channel's own FFT was scaled in the theta (4-8 Hz) and alpha
(8-12 Hz) bins by sqrt(power gain) - phase preserved, alert windows
untouched, masked channels untouched (they carry no features). Two
graded strengths: full (theta power x2.0 = +100%, alpha x1.5 = +50%)
and half (theta x1.5, alpha x1.25). Features were then re-extracted
with the standard extractor and the standard session-out LR pipeline
run unchanged. Manipulation check (observed vs target mean log10
frontal band-power shift in fatigued windows) is in
results/tables/diagnostics_positive_control_check.csv.

Recovered (cross-seed means):
                                      balanced_accuracy   roc_auc       mcc
variant                                                                    
full_strength_theta_x2.0_alpha_x1.5            0.833903  0.901671  0.638751
half_strength_theta_x1.5_alpha_x1.25           0.695391  0.768101  0.359706

Uninjected reference: BA 0.515, AUC 0.549, MCC 0.045.

Leakage checklist:
- [x] Injection is label-dependent BY DESIGN (it is a sensitivity
      control, not an experiment); it is applied window-wise before
      splitting and identical for all folds
- [x] Scaling/imputation fit on training folds only
- [x] No tuning on test data (locked C=1.0)

Environment: local VM (CPU), scikit-learn.
