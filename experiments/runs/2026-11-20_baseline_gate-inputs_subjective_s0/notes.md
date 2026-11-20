# Feasibility-gate INPUTS - NO VERDICT DECLARED

docs/decision_table_lock.md section 2: the equivalence bound Delta is
mine to confirm (candidate value 0.05). This run supplies ONLY
the gate's inputs - standard errors of the headline statistics from
label-free sources - and the conditional table: IF a given Delta were
locked, WOULD equivalence be testable at ~90% power (N as stated)?
No equivalence verdict is declared here, for any Delta.

Gate inputs (SE basis -> minimum Delta for ~90% power):
                                        basis       se  delta_min_90pct_power                                                                                                                     definition
    perm_null_circular_session_mean_statistic 0.031861               0.093238                                         SD of the session-mean BA statistic under 200 circular-shift permutations (cap seed 0)
      perm_null_block8_session_mean_statistic 0.032722               0.095759                                         SD of the session-mean BA statistic under 100 block-length-8 permutations (cap seed 0)
perm_null_circular_participant_mean_statistic 0.032255               0.094392 SD of the participant-mean BA statistic (mean over participants of per-participant fold means) under the same 200 permutations
  perm_null_block8_participant_mean_statistic 0.034248               0.100224                                              SD of the participant-mean BA statistic under the 100 block-length-8 permutations
                       analytical_session_N24 0.021269               0.062242                                                   per-session BA SD 0.1042 (20 held-out sessions, cross-seed means) / sqrt(24)
              analytical_session_N20_observed 0.023299               0.068183                                                                             per-session BA SD 0.1042 / sqrt(20 observed folds)
                   analytical_participant_N12 0.018361               0.053731                                                        per-participant LOPO BA SD 0.0636 (11 participants observed) / sqrt(12)

Conditional table (rows: candidate Delta; columns: SE basis):
 candidate_delta perm_null_circular_session_mean_statistic perm_null_block8_session_mean_statistic perm_null_circular_participant_mean_statistic perm_null_block8_participant_mean_statistic analytical_session_N24 analytical_session_N20_observed analytical_participant_N12
            0.03                              unattainable                            unattainable                                  unattainable                                unattainable           unattainable                    unattainable               unattainable
            0.05                              unattainable                            unattainable                                  unattainable                                unattainable           unattainable                    unattainable               unattainable
            0.08                              unattainable                            unattainable                                  unattainable                                unattainable               testable                        testable                   testable
            0.10                                  testable                                testable                                      testable                                unattainable               testable                        testable                   testable
            0.12                                  testable                                testable                                      testable                                    testable               testable                        testable                   testable
            0.15                                  testable                                testable                                      testable                                    testable               testable                        testable                   testable

Robustness: the permutation null was computed under two block
structures (whole-sequence circular shifts, n=200; block-length-8
shuffles, n=100). Per-permutation statistics in
results/tables/diagnostics_gate_permutation_nulls.csv.

Environment: local VM (CPU), scikit-learn.
