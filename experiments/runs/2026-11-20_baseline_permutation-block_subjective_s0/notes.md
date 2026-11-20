# Block-permutation test - session-out, cap seed 0

Correction required by Reviewer 1 review 7: the original permutation
test shuffled labels iid within session, which is anti-conservative
under autocorrelation. This version CIRCULARLY SHIFTS each session's
time-ordered label sequence by a random offset (block structure and
per-session class counts preserved exactly), 200 permutations,
identical folds/pipeline. Real seed-0 aggregate: BA 0.499, MCC 0.009,
AUC 0.493.

Block null: BA 0.508 +/- 0.032 (95% range
0.449-0.574); iid null was
0.498 +/- 0.028. Full comparison in
results/tables/diagnostics_permutation_combined_summary.csv. The block
null is the one to cite.

Leakage checklist:
- [x] Identical folds/imputation/scaling discipline as the real run
- [x] Permutation restricted to within-session circular shifts
- [x] No tuning on test data (locked C=1.0)

Environment: local VM (CPU), scikit-learn.
