# Equivalence assessment - headline balanced accuracy

Non-significance is not equivalence (Reviewer 1 review 7). Criterion
(pre-specified): the claim "equivalent to chance within the smallest
effect of interest" is supported for a protocol iff the UPPER bound of
the 95% bootstrap CI for balanced accuracy is below the SESOI of 0.58.

   protocol resample_unit  n_units  balanced_accuracy  ba_ci95_low  ba_ci95_high  sesoi_ba  upper_bound_below_sesoi       mcc  mcc_ci95_low  mcc_ci95_high
session-out       session       20           0.515133     0.467918      0.554966      0.58                     True  0.044639     -0.060931       0.139194
     within   participant        8           0.483556     0.389817      0.562247      0.58                     True -0.032221     -0.233716       0.144431

Resampling units: held-out sessions (n=20) for session-out, participants
(n=8) for within - the CI therefore reflects across-unit variability of
the cross-seed per-unit means, not window-level noise.

Environment: local VM (CPU).
