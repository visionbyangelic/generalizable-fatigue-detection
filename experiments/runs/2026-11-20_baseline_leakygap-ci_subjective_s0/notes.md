# Leakage-gap confidence interval - cap seed 0

Reviewer 1 review 11: report the leakage gap WITH a confidence
interval and drop the literature-contrast framing. The gap is paired
by session: for each held-out session, leaky BA = balanced accuracy of
that session's windows under the random window-level split
(out-of-fold predictions, sessions with both classes only); honest BA
= the same session as a session-out test fold. Bootstrap over the
20 paired sessions (10,000 resamples):

- headline aggregate gap (pooled leaky OOF BA minus honest
  mean-of-folds BA): +0.072
- paired mean gap: +0.052, 95% CI [-0.019, +0.121]

Per-session pairs in metrics.csv and
results/tables/diagnostics_leakage_gap_paired_sessions.csv.

Leakage checklist:
- [x] Leaky variant differs from the honest pipeline ONLY in the split
- [x] Scaling/imputation fit on training folds only in both variants
- [x] No tuning on test data (locked C=1.0)

Environment: local VM (CPU), scikit-learn.
