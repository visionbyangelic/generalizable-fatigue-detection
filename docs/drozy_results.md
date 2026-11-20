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
| KSS, all-channel | 0.602 [0.563, 0.644] | 0.681 | 0.228 | 0.499 (0.037) | 0.005 |
| KSS, Fz-only | 0.543 [0.505, 0.582] | 0.597 | 0.091 | 0.501 (0.042) | 0.134 |
| RT, all-channel | 0.482 [0.421, 0.549] | 0.477 | -0.045 | 0.500 (0.029) | 0.726 |
| RT, Fz-only | 0.467 [0.414, 0.527] | 0.462 | -0.081 | 0.499 (0.038) | 0.811 |

Per-participant values are in the run folders and in
results/tables/baseline_loso_drozy_*_summary.csv. For the KSS
all-channel arm they run from 0.486 (Participant 3) to 0.756
(Participant 2), with 12 of 13 participants above 0.52.

Reading, kept to what the numbers support:

- The KSS arm transfers to unseen participants at an estimated 0.602,
  with the interval excluding 0.50 and the permutation test agreeing
  (p = 0.005). Against FatigueSet's leave-one-participant-out estimate
  for the same pipeline (0.508 [0.471, 0.542]), the point estimates
  differ by 0.094. That gap sits just under the 0.10 size the
  pre-registration treats as clearly interpretable for a cross-dataset
  comparison, so I report it as a substantial point difference with
  that caution attached, not as a settled cross-dataset verdict.
- The RT arm, built from the same recordings with only the label
  construction changed, sits at chance (0.482, interval spanning
  0.50, p = 0.726). The two arms disagree on identical EEG, which is
  the label-construction phenomenon this dataset was chosen to test.
- The Fz-only KSS estimate (0.543) has an interval that just excludes
  0.50 while its permutation test does not separate it from the null
  (p = 0.134). The two measures disagree at the edge, so I report it
  as an estimate only and make no localisation claim from it.

## Sensitivities (declared in Amendment 1)

| Sensitivity | KSS all-channel BA [CI] | KSS Fz-only BA [CI] |
|---|---|---|
| S1: Participant 11 removed (N = 12) | 0.602 [0.557, 0.652] | 0.551 [0.516, 0.587] |
| S2: Participant 7 removed (N = 12) | 0.599 [0.557, 0.644] | 0.528 [0.484, 0.575] |

Neither declared sensitivity moves the KSS all-channel estimate by
more than 0.003. The headline does not hinge on the tie-break
participant or on the two-session participant.

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
  reverse. KSS labels transfer, RT labels do not, on the same EEG.
  The phenomenon the row names, label construction deciding the
  answer, is what the data show, with the arms swapped relative to
  the row's wording.
- Row 3 (at chance, positive control recovered): does not describe
  the KSS arm, which is not at chance. It describes the RT arm's
  point pattern, without the control.
- Row 4 (high only under random splits): no random-split run was made
  on DROZY, so this row is not in play.

The defensible summary is the narrow one: under this pipeline,
subjective sleepiness labels built as a within-person contrast
transfer across participants on DROZY at an estimated 0.60, and
reaction-time labels built the same way on the same recordings do
not. Anything stronger waits on the DROZY positive control and on
the cross-dataset comparison being read at the size it is.

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
