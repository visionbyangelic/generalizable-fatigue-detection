# Positive control - minimum detectable effect (MDE) dose curve

Reviewer 1 review 11: the homogeneous positive control was the best
case; extend the dose curve downward to the minimum detectable effect.
Four homogeneous doses (theta/alpha power gains), session-out, cap
seeds 0-4, per-fold outputs saved. Block-permutation null band upper
edge (q975, circular shifts): 0.574.

                    dose  theta_power_gain  alpha_power_gain  n_sessions  balanced_accuracy  ba_ci95_low  ba_ci95_high  block_null_q975  ci_clears_null_band
   theta_x2.0_alpha_x1.5              2.00             1.500          20           0.833903     0.773842      0.885028         0.573719                 True
  theta_x1.5_alpha_x1.25              1.50             1.250          20           0.695391     0.630919      0.756366         0.573719                 True
theta_x1.25_alpha_x1.125              1.25             1.125          20           0.561706     0.508990      0.613797         0.573719                False
  theta_x1.1_alpha_x1.05              1.10             1.050          20           0.495240     0.451688      0.533512         0.573719                False

MDE (smallest tested dose whose recovery CI clears the null band):
**theta_x1.5_alpha_x1.25**. This bounds what the pipeline could have detected in
the real labels: a homogeneous cross-session theta/alpha effect below
the MDE could exist undetected; at/above it, it would have cleared the
null band.

Leakage checklist:
- [x] Injection is label-dependent BY DESIGN (sensitivity control);
      applied window-wise before splitting, identical for all folds
- [x] Scaling/imputation fit on training folds only
- [x] No tuning on test data (locked C=1.0)

Environment: local VM (CPU), scikit-learn.
