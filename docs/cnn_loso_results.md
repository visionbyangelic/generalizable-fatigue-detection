# CNN leave-one-participant-out results (RQ2 headline, RQ1 gap) — 21 Nov 2026

The field's open problem is not whether a fatigue model can score well
on people it has already seen; it is whether anything survives the move
to a person it has never seen. Week 2 answered that for the band-power
baseline on FatigueSet (LOPO balanced accuracy 0.508, participant
bootstrap CI [0.471, 0.542]). This phase answers the same question for
the 1D-CNN on identical splits, which is RQ2, and puts the two models'
within-subject and cross-participant numbers side by side, which is
RQ1. Everything here is reported as estimates with intervals: the
locked feasibility-gate ruling (docs/decision_table_lock.md) makes this
phase estimation-only, so no equivalence or null verdict is claimed
for either model.

## Protocol (as run)

- Data: the canonical cap-seed-0 Amendment-1 window set
  (data/processed/windows_a1*): 806 windows over 11 participants, drawn
  from the 1,053-window primary pool (564 alert / 489 fatigued before
  the per-session M1 cap). P05 has no Amendment-1 windows at all, so
  there are 11 folds, not 12; that is a property of the locked
  preprocessing, stated here rather than absorbed silently.
- Split: leave-one-participant-out. The held-out participant
  contributes no window to training, validation, normalisation
  statistics or class weights.
- Model: EEGNet-8,2 exactly as frozen in src/models/cnn.py (temporal
  kernel 64 at 128 Hz, 6 input channels: 4 EEG plus the two temporal
  mask-indicator planes), Adam lr 1e-3, batch 64, at most 100 epochs,
  early stopping patience 15, training-fold class weights, temporal
  channel dropout p = 0.2 during training.
- Validation for early stopping: whole contiguous run-segments carved
  from the TRAINING participants only, with a 2-window purge at segment
  boundaries, using the frozen strategy search from
  src/models/cnn_segval.py (the design the Week 2 within-subject run
  settled on after session-grouped validation proved infeasible). The
  driver (src/models/cnn_loso_segval.py) imports the frozen network,
  input builder, carve and training loop unchanged; the one mechanical
  adaptation is that the carve receives a composite participant/session
  key so runs group per recording instead of merging different
  participants' same-named sessions.
- Seeds: model seeds 0 to 4 on the fixed cap-seed-0 window set, the
  seed role the frozen CNN design specifies. The baseline's five seeds
  instead vary the cap subsample. The RQ2 contrast below therefore
  compares per-participant seed means, the same convention Week 2 used
  for the within-subject comparison; the asymmetry is disclosed here
  because it is part of what the contrast means.
- Execution: a timing probe (one fold, 24.9 s of training) projected
  the 55-fold grid at well under an hour locally, so the run stayed on
  the local VM instead of moving to Colab. Every fold was checkpointed
  into its own run folder as it finished (fold wall times 13 to 44 s;
  about 21 minutes of fold time in total), and seeds were committed in
  order, so a restart could have cost at most one fold. In the event,
  all 55 folds completed with no failures and no skipped folds.

## Headline result

CNN LOPO, mean over the 11 participants of per-participant seed means
(intervals: participant bootstrap, 10,000 resamples over the
per-participant values):

| Metric | Estimate | 95% CI |
|---|---|---|
| Balanced accuracy | 0.493 | [0.452, 0.529] |
| ROC-AUC | 0.517 | [0.466, 0.564] |
| MCC | -0.000 | [-0.081, 0.076] |

Per participant (balanced accuracy, mean ± SD over the 5 model seeds;
n = held-out windows):

| Participant | BA | AUC | n |
|---|---|---|---|
| P01 | 0.577 ± 0.041 | 0.651 | 40 |
| P02 | 0.525 ± 0.046 | 0.546 | 64 |
| P03 | 0.570 ± 0.067 | 0.625 | 40 |
| P04 | 0.394 ± 0.061 | 0.359 | 43 |
| P06 | 0.472 ± 0.047 | 0.553 | 69 |
| P07 | 0.356 ± 0.104 | 0.412 | 31 |
| P08 | 0.516 ± 0.052 | 0.530 | 127 |
| P09 | 0.523 ± 0.103 | 0.550 | 112 |
| P10 | 0.526 ± 0.041 | 0.534 | 98 |
| P11 | 0.491 ± 0.059 | 0.467 | 116 |
| P12 | 0.474 ± 0.020 | 0.456 | 66 |

Full per-seed values are in the 55 run folders
(experiments/runs/2026-11-21_cnn_loso_subjective_*) and the summary
table results/tables/cnn_loso_subjective_a1_summary.csv. The spread is
the story as much as the mean: two participants sit clearly below
chance (P07 0.356, P04 0.394), the best (P01 0.577) is modest, and the
seed SDs (up to 0.104) say the network's answer for a given unseen
person is not stable either.

## RQ2 — CNN vs band-power baseline on identical splits

Paired per-participant contrast, CNN minus baseline balanced accuracy
(each model averaged over its own seeds first; participant bootstrap
CI): **-0.014 [-0.075, +0.042]** over the 11 participants
(results/tables/cnn_loso_rq2_contrast.csv). Per participant the
difference runs from -0.226 (P07, where the baseline is comparatively
good and the CNN is not) to +0.144 (P10, the reverse). Read as an
estimate, the CNN's cross-participant performance is centred a little
below the baseline's, with an interval that includes small differences
in either direction. Nothing here supports a CNN advantage on unseen
participants for this dataset and label design, which matches the
Week 2 within-subject finding (CNN at chance, no advantage) on the
protocol where the comparison is hardest to fake.

## RQ1 — the generalisation gap

Gap = within-subject balanced accuracy minus LOPO balanced accuracy,
using the stored Week 2 within-subject numbers as the comparator
(session-held-out within-subject protocol; baseline 0.484 over 8
participants, CNN 0.493 over the 7 participants its feasibility
criteria covered, model seeds {0,1}):

| Model | Within | LOPO | Gap (aggregate) | Gap (paired, same participants) |
|---|---|---|---|---|
| Band-power baseline | 0.484 | 0.508 | -0.024 | -0.030 [-0.112, +0.053], n = 8 |
| 1D-CNN (EEGNet) | 0.493 | 0.493 | -0.000 | -0.006 [-0.104, +0.085], n = 7 |

(Tables: results/tables/rq1_generalisation_gap.csv and
rq1_generalisation_gap_paired.csv.)

The honest reading is that there is no positive gap to report under
these protocols, for either model, because the within-subject estimates
themselves sit at chance: moving from "new sessions of known people" to
"a wholly new person" does not measurably cost anything when the
starting point already carries no transferable signal. That is a
different statement from the within-session estimate of 0.744 in the
Week 2 diagnostics, and the two must not be conflated: the 0.744 comes
from windows inside the same session, where window position is a
perfect proxy by construction, and it measures same-session
separability, not a within-subject generalisation level. The collapse
this project set out to measure is between that same-session number
and every honest split, not between the two honest splits.

Two caveats travel with the CNN gap. Its within-subject side rests on
7 of 12 participants and two seeds, and one covered participant's
within value (P03, two test windows) is flagged in the Week 2
diagnostics as uninterpretable; the paired CNN gap inherits that
fragility. And the gap rows mix participant sets at the aggregate
level, which is why the paired version is reported alongside.

## Interpretation boundaries

- The early-stopping validation measures segment generalisation inside
  known training participants, not unseen-participant transfer; the
  held-out participant is only ever a test fold. One behaviour of the
  frozen carve is worth stating plainly: its first feasible strategy
  placed the validation segments in a single training participant's
  session (P12/S02 for ten folds, P11/S02 for P12's own fold), so
  early stopping was tuned against one known person's segments in each
  fold. That is what the locked strategy search produces; changing it
  would mean changing the design after seeing the folds, so it is
  reported instead.
- These are estimates of what this pipeline, this label design and
  this dataset support. The interval for the CNN's balanced accuracy
  sits below the 0.55 usefulness line adopted for the baseline gate;
  that is reported as an observation about the interval, not as an
  equivalence verdict, which the gate ruling reserves to the analysis
  it was computed for.
- Claims are scoped to this feature family, this evaluation and the
  Muse S headset data of FatigueSet, as everywhere in this project.

## Problems encountered

None in the runs themselves: 55 of 55 folds completed, each
checkpointed and committed per seed. The environment restarted twice
elsewhere in the project tonight, but not during this grid; the
per-fold checkpointing was in place in case it had.
