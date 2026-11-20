# DROZY feasibility gate, recomputed at the realised N = 13

Computed 20 Nov 2026, before any model was fit to the real DROZY
labels. The pre-registration (Amendment 1) requires the gate to be
recomputed with the realised participant count substituted into the
identical locked computation, and the numbers logged before modelling.
This document is that log.

## How the gate works here

The rule is the one locked in docs/decision_table_lock.md section 2:
the smallest equivalence margin Delta that would give about 90% power
to show equivalence at a true effect of zero is
Delta_min = (z_0.95 + z_0.90) x SE = 2.9265 x SE, with the standard
error taken from a label-free permutation null. If Delta_min exceeds
the confirmed Delta of 0.05, equivalence is unattainable at this
sample size and the results are reported as estimates with intervals,
never as an equivalence conclusion. The margin itself does not move.

One adaptation was forced by the data, and it is a change of
granularity, not of method. For FatigueSet the labels vary inside a
session (early block vs late block), so the null permutes labels by
circular shifts within the session. In DROZY a label belongs to a
whole session: every window of a session carries the same label, and
a within-session shift would permute identical values and return the
observed statistic unchanged. The permutation therefore acts at the
level where DROZY labels actually live: within each participant, the
two session labels are swapped or kept, coin flip per participant.
Window order inside a session is untouched, each participant keeps
exactly one alert and one fatigued session, and the statistic is the
same one the headline uses, the mean over participants of
per-participant balanced accuracy under leave-one-participant-out.
Two independently seeded swap sets (200 and 100 permutations) mirror
the two block structures used for FatigueSet, and an analytical basis
(the average per-participant spread under the null, over sqrt(13)) is
reported alongside, as it was in Week 2.

## Numbers

Primary combination (KSS arm, all-channel features):

| Basis | SE | Delta_min |
|---|---|---|
| Permutation, swap set A (200) | 0.0374 | 0.109 |
| Permutation, swap set B (100) | 0.0411 | 0.120 |
| Analytical, per-participant / sqrt(13) | 0.0279 | 0.082 |

Other combinations, permutation set A (analytical in brackets):

| Combination | SE | Delta_min |
|---|---|---|
| KSS arm, Fz-only | 0.0417 | 0.122 (0.086) |
| RT arm, all-channel | 0.0293 | 0.086 (0.067) |
| RT arm, Fz-only | 0.0377 | 0.110 (0.079) |

Full table: results/tables/drozy_gate_inputs.csv. Per-permutation
statistics: results/tables/drozy_gate_permutation_nulls.csv.

## What the locked rule concludes

Every basis for every combination gives Delta_min above 0.05 (the
smallest is 0.067, the RT arm's analytical basis; the primary
combination's permutation bases give 0.109 and 0.120). At the
confirmed Delta of 0.05, equivalence is therefore unattainable at
N = 13, and the DROZY leave-one-participant-out results are reported
estimation-only: point estimates with participant-bootstrap
intervals, no equivalence or null claim, whatever the intervals
happen to do. This matches the standing that FatigueSet ended in,
for the same reason: the uncertainty lives between participants,
and 13 participants cannot pin a mean balanced accuracy tightly
enough to prove a 0.55 bound.
