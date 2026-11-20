# Mask-only baseline - session-out, cap seeds 0-4

Panel addition (Reviewer 1 review 7): how much label information do
the Amendment-1 validity indicators and coverage counts alone carry?
This is the shortcut ceiling for any model given the indicators.

Cross-seed aggregate: balanced accuracy 0.531,
ROC-AUC 0.521, MCC 0.034 (full EEG model:
0.515 / 0.549 / 0.045). Frontal validity is constant (1) in retained
windows by Amendment-1 gating, so all information here is temporal
masking pattern + coverage.

Leakage checklist:
- [x] Scaling fit on training folds only (Pipeline, per fold)
- [x] Features are protocol/quality metadata only - no EEG, no labels
- [x] No tuning on test data (locked C=1.0)

Environment: local VM (CPU), scikit-learn.
