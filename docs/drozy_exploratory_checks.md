# DROZY exploratory checks (post-results, 21 Nov 2026)

These five checks were run on 21 Nov 2026, after the DROZY LOPO
results were already reported in docs/drozy_results.md. They are
exploratory and post-results by construction: the results review
asked for them because the headline pattern (sleepiness labels
decodable across participants, reaction-time labels not, on identical
recordings) has soft spots that should be measured rather than argued
about. Nothing here is a new primary analysis, and nothing here
changes a headline number. Code: src/evaluation/drozy_exploratory.py,
reusing the frozen fold machinery and the DROZY baseline wiring.

## Check 1: audit of the bootstrap intervals and the swap nulls

What I checked. The reported intervals come from participant_boot_ci
in src/evaluation/closure_pack.py, called with the per-participant
balanced accuracies (one number per participant, from the summarise
step grouped by participant). The function resamples that array of
13 values with replacement, 10,000 times, rng 42, and takes the 2.5th
and 97.5th percentiles of the resampled means. So by construction the
unit being resampled is the participant, not the window. To prove the
stored numbers really came out of that call, I recomputed all eight
reported intervals (the four primary combinations and the four
sensitivity runs) from the stored per-participant summary tables.

Verdict: the intervals are computed correctly. Every recomputed point
estimate and interval bound matches the reported value exactly (the
largest absolute difference across all eight rows is 0.0 at full
stored precision). The swap nulls are centred where they should be:
the stored null means are 0.504 (KSS all-channel), 0.498 (KSS
Fz-only), 0.500 (RT all-channel) and 0.497 (RT Fz-only), all within
0.004 of 0.5. Recounting the p-values from the stored null
distributions reproduces the reported values exactly (0.005, 0.134,
0.726, 0.811). Full table: results/tables/drozy_ci_audit.csv.

One correction, disclosed rather than smoothed over: the headline
table in docs/drozy_results.md originally printed the null means as
0.499, 0.501, 0.500 and 0.499. The stored distributions give 0.504,
0.498, 0.500 and 0.497. The null SDs printed there were right, and
the p-values, which are counted from the same stored distributions,
are unaffected. I have corrected the means in the results document;
the balanced accuracies, intervals and p-values do not change.

The Fz-only discrepancy, explained. The KSS Fz-only interval
[0.505, 0.582] excludes 0.5 while its swap-null p is 0.134. Neither
quantity is miscomputed; they measure different things, and the
numbers show it concretely:

- The bootstrap interval is built from the spread of the 13 observed
  fold scores. Those scores have SD 0.074, so the standard error of
  their mean is 0.021, and an interval of roughly plus or minus 0.039
  around the observed mean of 0.543 lands at [0.505, 0.582], just
  clear of 0.5.
- The swap null asks a different question: how far does this same
  statistic wander when the labels are exchangeable? Its SD is
  0.042, twice the bootstrap standard error. Against that spread, the
  observed mean sits only about 1.07 null SDs above the null mean of
  0.498, which is what p = 0.134 says.
- The null is wider for two measurable reasons. Under a label swap,
  each participant's fold score scatters more (SD about 0.106 across
  permutations, against 0.074 for the observed scores), because the
  refitted model can chase real between-session differences in either
  direction. And the 13 fold scores inside one permutation are not
  independent: every fold's model is trained on the other twelve
  participants' swapped labels, so one swap pattern moves all folds
  together. The implied average correlation between fold scores is
  about 0.08, which roughly doubles the variance of their mean
  relative to independence (independence would give a null SD of
  0.029, which is exactly the gate's analytical basis). The bootstrap
  inherits the independence assumption: it resamples the 13 observed
  scores as if they were independent draws and never refits anything,
  so its standard error understates the statistic's null spread.

Conclusion: the bootstrap interval is an honest interval for the mean
of the observed fold scores, but it is not a test against 0.5 in this
design, and reading it as one is what produced the apparent
contradiction. The swap-null p is the inferential quantity. The
Fz-only KSS estimate therefore stays what the results document
already called it: an estimate only, with no claim attached. The same
caution applies to every interval in the DROZY tables, including the
KSS all-channel one; that arm's claim rests on its permutation test
(p = 0.005), not on its interval clearing 0.5.

## Check 2: disagreement accounting between the label arms

Counted from the locked label tables (results/tables/
drozy_labels_kss.csv and drozy_labels_rt.csv), over the 13 included
participants. Full per-participant table: results/tables/
drozy_label_agreement.csv.

- KSS arm: the fatigued session is the participant's latest session
  with data for 11 of 13 participants. The exceptions are P02
  (fatigued = test 2, latest = test 3) and P13 (fatigued = test 1,
  latest = test 2).
- RT arm: the fatigued session is the latest session for 8 of 13.
  The exceptions are P01, P02, P04, P05 and P13.
- The two arms pick a different fatigued session for 4 of 13
  participants: P01, P02, P04 and P05. They pick a different alert
  session for 3 of 13 (P02, P03, P11), and 6 of 13 participants have
  at least one of their two sessions differing between the arms.

So the KSS-versus-RT contrast rests, in part, on a disagreeing subset
of four participants. That is now a counted fact rather than a
suspicion, and it is one reason the claim wording stays narrow.

## Check 3: protocol-order arm

Labels from session order alone: each participant's earliest session
with data is labelled alert, their latest fatigued, any middle
session excluded, Participant 12 excluded as before. Same features,
same model code, same LOPO protocol, and a fresh swap null computed
for this arm (200 swaps, same convention as the gate). Run folders:
experiments/runs/2026-11-21_baseline_loso_protocol-order_drozy-*.
Label table: results/tables/drozy_labels_order.csv.

| Features | BA [95% CI] | Null mean (SD) | p |
|---|---|---|---|
| Order, all-channel | 0.536 [0.493, 0.572] | 0.505 (0.039) | 0.229 |
| Order, Fz-only | 0.545 [0.502, 0.587] | 0.500 (0.046) | 0.154 |

Reading. The order arm does not reproduce the KSS result. On the
all-channel features, where the KSS estimate is 0.602 with p = 0.005,
order labels score 0.536 and do not separate from their own null
(p = 0.229). This is a sharper test than it might look: the order
labels are identical to the KSS labels for 11 of the 13 participants
(everyone except P02 and P13), yet swapping in order labels for those
two drops the estimate by 0.066 and loses the null separation. P13 is
the clearest single case: their KSS contrast runs against time order
(their sleepiest session was their first), and they score 0.584 under
KSS labels against 0.367 under order labels. One caution on reading
the per-participant differences: a fold score also moves when other
participants' training labels change, so the drop is a property of
the whole label assignment, not of two participants alone.

On Fz-only features the order arm (0.545) and the KSS arm (0.543) are
the same number to two decimals. That does not rescue the order
explanation, because the KSS Fz-only estimate never separated from
its own null either (p = 0.134, see Check 1): at Fz there is no
established effect for order to explain.

Conclusion, following the rule I stated before running it: because
the order arm sits near chance rather than near 0.60, the KSS result
cannot be dismissed as session order wearing a label's clothes; the
pure session-order explanation is weakened. It is not eliminated as a contributor. Order labels
still score 0.536 as a point estimate with a wide interval, KSS
labels align with order for 11 of 13 participants by construction,
and time awake remains entangled with sleepiness in this protocol.
The paired-contrast and disagreeing-subset caveats from Checks 2, 4
and 5 stand in full.

## Check 4: paired contrast between the arms

Participant-level bootstrap on the per-participant difference
BA(KSS) minus BA(RT), all-channel features, paired by participant
(10,000 resamples, rng 42): mean difference 0.120, 95% interval
[0.062, 0.179], n = 13. Every one of the 10,000 resampled means was
above zero. Table: results/tables/drozy_paired_contrast.csv.

This is the within-DROZY contrast between the two label constructions
on identical recordings. It is not the cross-dataset gap: the
DROZY-minus-FatigueSet difference of 0.094 stays a descriptive number
under the pre-registered 0.10 line, exactly as reported before. And
like everything in this document, the paired contrast is exploratory
and post-results; its interval inherits the independence caveat from
Check 1, since it is built from the same fold scores.

## Check 5: accuracy split by whether the arms agree

Mean per-participant balanced accuracy within each group, from the
stored fold scores. The groups are small and the disagreeing group is
very small, so the counts matter more than the decimals:
results/tables/drozy_agree_disagree_split.csv.

| Features | Agree (n = 9) | Disagree (n = 4: P01, P02, P04, P05) |
|---|---|---|
| KSS, all-channel | 0.590 | 0.630 |
| RT, all-channel | 0.524 | 0.388 |
| KSS, Fz-only | 0.521 | 0.592 |
| RT, Fz-only | 0.488 | 0.422 |

Reading. The KSS arm scores about the same in both groups (0.590
agreeing, 0.630 disagreeing). The gap between the arms is
concentrated in the four disagreeing participants, where the RT
labels point at different sessions than the KSS labels and score
below chance (0.388). Part of that is mechanical: where the two
constructions choose different sessions, at most one of them can line
up with whatever session-level structure the recordings carry, and
the other is scored against it. With n = 4 in the disagreeing group
this is a description of where the contrast lives, not a finding in
its own right, and it is exactly why the headline wording says the
contrast rests on a small disagreeing subset.

## What these checks change

No headline number changes. The claim language is now the locked
sentence: KSS-extreme labels were decodable across participants;
RT-extreme labels from the same recordings were not; this contrast
is not isolated from session order and rests on a small disagreeing
subset. The checks weaken the strongest alternative explanation
(session order alone does not reproduce the KSS score), verify that
the reported intervals and p-values were computed as described, put
exact counts on the disagreeing subset (4 of 13), and measure the
paired arm difference at 0.120 [0.062, 0.179]. Everything remains
estimation-only under the feasibility gate, and the proposed
positive control (Amendment 2, drafted in
docs/dataset2_preregistration.md) is still the missing piece for any
statement about the RT arm's sensitivity or about cross-dataset
ceilings.
