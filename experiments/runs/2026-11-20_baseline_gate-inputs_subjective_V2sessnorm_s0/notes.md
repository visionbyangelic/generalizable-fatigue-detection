# Gate inputs - V2sessnorm

Feasibility-gate INPUTS for the V2 variant (lock doc
section 2 supplies the formula; the bound itself was confirmed by
I at Delta = 0.05 and does not move). This run declares NO
verdict: it reports the label-free permutation null SDs, the implied
Delta_min at ~90% power, and the observed estimates/intervals next to
the 0.55 line, as inputs for the lead reviewer's ruling.

Permutation scheme: circular shifts of labels within session
(preserves per-session class counts and the within-session label
block structure), 200 permutations, rng 60000+p - one
common set reused across all variant feature sets and both protocols,
so the null SDs are paired across variants. Per-permutation fits use
the identical model; the median imputer and StandardScaler are
label-free and are fit once per fold on the real features (only the
LR, with its training-fold balanced class weights, is refit per
permutation - mathematically identical to refitting the Pipeline).

Gate inputs:
   variant feature_set_protocol        statistic  n_permutations                                                    permutation  null_sd  delta_min_90pct_power  observed_balanced_accuracy  observed_ba_ci95_low  observed_ba_ci95_high  sesoi_upper_line                                                                                       note
V2sessnorm          session-out     session_mean             200 circular shifts within session, common set of 200, rng 60000+p 0.039633               0.115983                    0.583008              0.498085               0.665246              0.55 gate INPUTS only - no verdict declared; reporting status awaits the lead reviewer's ruling
V2sessnorm          session-out participant_mean             200 circular shifts within session, common set of 200, rng 60000+p 0.040399               0.118223                    0.583008              0.498085               0.665246              0.55 gate INPUTS only - no verdict declared; reporting status awaits the lead reviewer's ruling
V2sessnorm                 loso participant_mean             200 circular shifts within session, common set of 200, rng 60000+p 0.034755               0.101707                    0.555928              0.479680               0.612673              0.55 gate INPUTS only - no verdict declared; reporting status awaits the lead reviewer's ruling

Environment: local VM (CPU), scikit-learn 1.9.1.
