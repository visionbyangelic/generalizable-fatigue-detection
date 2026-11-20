# 2026-11-20_baseline_loso_subjective_frontal_s3

Split definition: hold out ALL sessions of one participant; train on the other 11 participants. NOTE on P05: P05 has NO fold in this protocol, and the mechanism is upstream of the fold rule - P05's only retained session (S01) is single-class (5 alert / 0 fatigued windows after Amendment 1), so the per-session M1 cap (cap_pool, cap M1 to the retained M3 count = 0) reduces P05 to zero windows in every capped dataset. P05 is therefore absent from all capped analyses (baseline, diagnostics, LOPO), not merely skipped here. Verified directly against windows_a1_uncapped_index.csv and cap_pool output during the closure pack; the earlier draft wording of this note (fold-rule skip) was wrong and has been corrected. A fold counts only if train and test
both contain both classes; skipped folds: none.

Leakage checklist:
- [x] Normalisation (StandardScaler) fit on training fold only (Pipeline, per fold)
- [x] Median imputation fit on training fold only
- [x] Class weights from training fold only (class_weight='balanced')
- [x] No overlapping windows across splits (tumbling windows; splits at session/participant level)
- [x] No tuning on test data (C=1.0 locked default)
- [x] Label rules locked (docs/label_rules.md incl. Amendment 1) before any training

Environment: local VM (CPU), scikit-learn. Seed = M1-cap subsample seed 3;
logistic regression itself is deterministic given the data.
