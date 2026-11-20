# Leakage-gap demonstration - cap seed 0

Panel addition (Reviewer 1 review 7): identical features, identical
model, identical data - the ONLY changed variable is the split.
Random window-level split (windows pooled across all sessions and
participants, stratified 5-fold): balanced accuracy
0.571, ROC-AUC 0.609, MCC
0.141. Honest session-out on the same seed: 0.499 / 0.493 /
0.009. The gap between these numbers is what the split discipline is
worth; the leaky number is the kind of figure the project's RQ1/RQ4
argue is uninterpretable as fatigue detection.

Leakage checklist (for the honest pipeline this demo contrasts with):
- [x] Scaling/imputation fit on training folds only, even in the leaky
      variant - the split is the sole changed variable
- [x] No tuning on test data (locked C=1.0)

Environment: local VM (CPU), scikit-learn.
