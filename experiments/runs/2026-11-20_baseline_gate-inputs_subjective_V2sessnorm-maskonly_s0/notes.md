# Gate inputs - V2sessnorm-maskonly

Feasibility-gate INPUTS for the V2MASK variant (lock doc
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
V2sessnorm-maskonly          session-out     session_mean             200 circular shifts within session, common set of 200, rng 60000+p 0.023203               0.067901                    0.531636              0.476593               0.594513              0.55 gate INPUTS only - no verdict declared; reporting status awaits the lead reviewer's ruling
V2sessnorm-maskonly          session-out participant_mean             200 circular shifts within session, common set of 200, rng 60000+p 0.020444               0.059826                    0.531636              0.476593               0.594513              0.55 gate INPUTS only - no verdict declared; reporting status awaits the lead reviewer's ruling
V2sessnorm-maskonly                 loso participant_mean             200 circular shifts within session, common set of 200, rng 60000+p 0.032354               0.094680                    0.544406              0.499008               0.592880              0.55 gate INPUTS only - no verdict declared; reporting status awaits the lead reviewer's ruling

Environment: local VM (CPU), scikit-learn 1.9.1.
