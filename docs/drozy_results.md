# DROZY baseline results: leave-one-participant-out, both label arms

Runs executed 20 Nov 2026, after the feasibility gate was recomputed
and logged (docs/drozy_gate.md). The gate's conclusion stands over
everything below: at the confirmed margin of 0.05, equivalence is
unattainable at N = 13 on every basis, so all DROZY numbers are
reported as estimates with intervals. No equivalence or proven-null
language is used anywhere in this document.

Method in one paragraph: the locked band-power + logistic-regression
pipeline, unchanged. DROZY recordings (Massoz et al., WACV 2016) were
turned into windows and features by the adapter
(src/data/drozy_adapter.py): 1-40 Hz bandpass, downsampled to 128 Hz,
4 s windows, a window kept when Fz passes the artefact checks, other
channels masked per window when they fail. Each participant
contributes one alert and one fatigued session per label arm, chosen
by the rules in docs/dataset2_preregistration.md (Amendment 1
included). Evaluation is leave-one-participant-out over 13
participants. The mask-only control is not applicable on clinical
recordings and was not run, as declared.

## Data kept

Of 5,400 candidate windows (36 sessions x 150), 4,970 (92.0%) passed
the Fz gate. Masking of non-frontal channels was rare (a few windows
per session at most). The KSS arm uses 3,704 windows (1,865 alert,
1,839 fatigued); the RT arm uses 3,663 (1,871 alert, 1,792 fatigued).
Lowest retention: Participant 4's tests 2 and 3 (55.3% and 69.3%) and
Participant 12's single session (23.3%, excluded from both arms in
any case). Per-session table: results/tables/drozy_rejection_rates.csv.

## Headline estimates

Balanced accuracy with 95% participant-bootstrap intervals (10,000
resamples, rng 42). Permutation nulls: 200 within-participant
session-label swaps, p = (1 + count(null >= real)) / 201.

| Arm / features | BA [95% CI] | AUC | MCC | Null mean (SD) | p |
|---|---|---|---|---|---|
| KSS, all-channel | 0.602 [0.563, 0.644] | 0.681 | 0.228 | 0.504 (0.037) | 0.005 |
| KSS, Fz-only | 0.543 [0.505, 0.582] | 0.597 | 0.091 | 0.498 (0.042) | 0.134 |
| RT, all-channel | 0.482 [0.421, 0.549] | 0.477 | -0.045 | 0.500 (0.029) | 0.726 |
| RT, Fz-only | 0.467 [0.414, 0.527] | 0.462 | -0.081 | 0.497 (0.038) | 0.811 |

(The null means in this table were corrected on 21 Nov 2026 after
the audit in docs/drozy_exploratory_checks.md found the first
version printed values that did not match the stored null
distributions. The null SDs and p-values were and are correct.)

Per-participant values are in the run folders and in
results/tables/baseline_loso_drozy_*_summary.csv. For the KSS
all-channel arm they run from 0.486 (Participant 3) to 0.756
(Participant 2), with 12 of 13 participants above 0.52.

Reading, in the wording the results review locked on 21 Nov 2026:

KSS-extreme labels were decodable across participants; RT-extreme
labels from the same recordings were not; this contrast is not
isolated from session order and rests on a small disagreeing subset.

What sits behind that sentence:

- The KSS all-channel estimate is 0.602 [0.563, 0.644], and its
  separation from chance rests on the permutation test (p = 0.005).
  The RT all-channel estimate, built from the same recordings with
  only the label construction changed, is 0.482 [0.421, 0.549],
  p = 0.726. A paired participant-level contrast between the two
  arms (exploratory, post-results) puts the difference at
  0.120 [0.062, 0.179].
- Against FatigueSet's leave-one-participant-out estimate for the
  same pipeline (0.508 [0.471, 0.542]), the KSS point estimate
  differs by 0.094. That gap sits just under the 0.10 size the
  pre-registration treats as clearly interpretable for a cross-dataset
  comparison, so it is reported as a descriptive contrast, not as a
  test and not as a settled cross-dataset verdict.
- The Fz-only estimates (KSS 0.543, RT 0.467) are reported alongside
  the all-channel ones with no spatial interpretation: their
  intervals overlap the all-channel intervals, and significant-versus-
  not is not a test of the difference between feature sets. The KSS
  Fz-only interval [0.505, 0.582] excludes 0.50 while its permutation
  test does not separate it from the null (p = 0.134); the audit in
  docs/drozy_exploratory_checks.md shows both numbers are computed
  correctly and that the interval is a location interval for the mean
  of the fold scores, not a test against 0.5. The permutation p is
  the inferential quantity, so the Fz-only estimate carries no claim.

## Sensitivities (declared in Amendment 1)

| Sensitivity | KSS all-channel BA [CI] | KSS Fz-only BA [CI] |
|---|---|---|
| S1: Participant 11 removed (N = 12) | 0.602 [0.557, 0.652] | 0.551 [0.516, 0.587] |
| S2: Participant 7 removed (N = 12) | 0.599 [0.557, 0.644] | 0.528 [0.484, 0.575] |

Neither declared sensitivity moves the KSS all-channel estimate by
more than 0.003. The headline does not hinge on the tie-break
participant or on the two-session participant.

## Exploratory checks (post-results, 21 Nov 2026)

Five checks requested by the results review were run after the
results above were reported. They are written up in full in
docs/drozy_exploratory_checks.md; the short version:

- Interval audit: every reported interval recomputes exactly from
  the stored per-participant values, the bootstrap resamples
  participants (13 clusters), the swap nulls are centred on 0.5, and
  the p-values recount exactly. The Fz-only interval-versus-p
  discrepancy is explained there (the interval is not a test against
  0.5; the fold scores are correlated through shared training data,
  so the bootstrap standard error understates the null spread).
- Disagreement accounting: the KSS arm's fatigued session is the
  latest session for 11 of 13 participants, the RT arm's for 8 of
  13, and the two arms choose a different fatigued session for 4 of
  13 participants (P01, P02, P04, P05). That counted subset is the
  one the locked claim wording refers to.
- Protocol-order arm: labels from session order alone (earliest
  session versus latest) score 0.536 [0.493, 0.572], p = 0.229
  all-channel and 0.545 [0.502, 0.587], p = 0.154 Fz-only. The order
  arm does not reproduce the KSS score, so the pure session-order
  explanation of the KSS result is weakened. It is not removed as a
  contributor: order labels agree with KSS labels for 11 of 13
  participants by construction, and time awake stays entangled with
  sleepiness in this protocol. The contrast is therefore still not
  isolated from session order, as the locked wording says.
- Paired contrast: BA(KSS) minus BA(RT) is 0.120 [0.062, 0.179]
  paired by participant (all-channel).
- Agree/disagree split: the KSS arm scores about the same whether
  the arms agree on the fatigued session (0.590, n = 9) or disagree
  (0.630, n = 4); the RT arm drops to 0.388 in the disagreeing four.
  With n = 4 that split is a description of where the contrast
  lives, not a finding.

## Against the pre-declared outcome table

The pre-registration fixed four rows in advance. None matches the
observed pattern exactly, and I am not going to pretend otherwise:

- Row 1 (clearly above FatigueSet, interval excludes 0.50, positive
  control recovered): the first two conditions hold for the KSS arm,
  but no positive control was run on DROZY in this stage, so the row
  is not claimed. The pipeline's positive control exists on FatigueSet
  with the identical frozen feature and model code; whether it
  recovers on DROZY data is untested here.
- Row 2 (above chance with RT labels, at chance with KSS labels, a
  label-construction effect): the direction that occurred is the
  reverse. KSS labels were decodable across participants, RT labels
  were not, on the same EEG.
  The phenomenon the row names, label construction deciding the
  answer, is what the data show, with the arms swapped relative to
  the row's wording.
- Row 3 (at chance, positive control recovered): does not describe
  the KSS arm, which is not at chance. It describes the RT arm's
  point pattern, without the control.
- Row 4 (high only under random splits): no random-split run was made
  on DROZY, so this row is not in play.

The defensible summary is the locked one: KSS-extreme labels were
decodable across participants; RT-extreme labels from the same
recordings were not; this contrast is not isolated from session
order and rests on a small disagreeing subset. This is an
exploratory dissociation obtained with a pre-registered pipeline,
in an estimation-only regime, and its direction reverses the
pre-registered label prediction; all three qualifiers travel with
it wherever it is reported. Anything stronger, in particular any
statement about the RT arm's sensitivity or about absolute ceilings
on either dataset, waits on the DROZY positive control, which is
drafted as Amendment 2 in docs/dataset2_preregistration.md and has
not been run.

## Problems encountered

- The VM restarted mid-stage and took the locally installed Python
  packages with it (the gate outputs were already written and
  committed, so nothing was lost). Packages were reinstalled at the
  same versions (scikit-learn 1.9.1, MNE 1.13.2, numpy 2.5.5,
  scipy 1.18.1, pandas 3.1.1) into a persistent folder, and before
  any DROZY run the Week 2 baseline seed 0 was reproduced from the
  stored FatigueSet windows exactly (balanced accuracy 0.4991,
  MCC 0.0090), the same faithfulness check used in Week 2.
- No modelling problems: no fold was skipped in any run (every
  participant's test fold contains both classes by construction, and
  every training fold does too).
