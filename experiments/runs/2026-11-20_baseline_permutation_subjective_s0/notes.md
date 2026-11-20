# Baseline label-permutation test - session-out, cap seed 0

Labels permuted WITHIN each session (per-session class counts preserved),
200 permutations, identical pipeline and folds to the real session-out run
(20 valid folds). Real seed-0 aggregate: balanced accuracy
0.499, MCC 0.009, ROC-AUC 0.493. (The headline
0.515 / 0.045 figures are the cross-seed means over cap seeds 0-4; their
percentiles in this seed-0 null are also reported.)

Null (mean +/- sd): BA 0.498 +/- 0.028,
MCC -0.005 +/- 0.054. Permutation p-values
((1 + #null >= real) / 201): BA 0.493,
MCC 0.423.

Leakage checklist:
- [x] Identical folds/imputation/scaling discipline as the real run
- [x] Permutation restricted to within-session (class counts preserved)
- [x] No tuning on test data (locked C=1.0)

Environment: local VM (CPU), scikit-learn.
