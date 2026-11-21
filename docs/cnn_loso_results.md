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
  channel dropout p = 0.2 during training. "Frozen" throughout this
  document means this locked configuration, initialised and trained
  fresh inside every fold; no pre-trained weights are loaded at any
  point.
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

Neither the frozen-configuration EEGNet CNN nor the band-power
baseline exceeds chance in cross-participant fatigue classification
on FatigueSet (N = 11 analysed of 12 enrolled), with no advantage
for the CNN.

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
table results/tables/cnn_loso_subjective_a1_summary.csv. The
per-participant values run from 0.356 (P07) to 0.577 (P01), and the
seed SDs run up to 0.104. With 11 participants and intervals of this
width, that spread is reported as values only; it is not read as
evidence about individual participants, and no per-participant
pattern is claimed.

## How much did the network actually learn? Prediction degeneracy across the 55 fits

A balanced accuracy near 0.5 can hide fits that output almost a
single class, so I extracted the stored predictions from all 55 run
folders (no retraining; src/evaluation/cnn_loso_degeneracy.py, table
results/tables/cnn_loso_degeneracy.csv) and counted, per fit, the
fraction of test windows predicted fatigued and the epoch at which
training stopped. A fit is called degenerate here if one class
accounts for at least 95% of its test predictions.

- 16 of the 55 fits are degenerate by that rule. Three of them
  (P04 seed 3, P07 seed 3, P08 seed 3) predict fatigued for every
  single test window, so their balanced accuracy is exactly 0.500 by
  construction. No fit falls in the 90 to 95% band: the fits are
  either clearly two-class or past the 95% line, with nothing in
  between.
- The degeneracy sits in the seeds, not the participants. All 16
  degenerate fits belong to seed 0 (5 fits, each predicting alert for
  about 95 to 97% of windows) or seed 3 (11 fits, one for every
  participant, each predicting fatigued for about 97 to 100% of
  windows). Seeds 1, 2 and 4 have none.
- Training usually stopped as early as it could. 43 of the 55 fits
  stopped at epoch 16, the earliest stop the patience of 15 permits,
  which means the best validation loss was the first epoch's and
  never improved afterwards; the median run is 16 epochs and the
  longest is 36. Every degenerate fit stopped at epoch 16.

Plainly: in a large share of these folds the network barely learned
anything beyond a near-constant output, and the headline estimate
averages over those fits. The RQ2 comparison below is therefore a
comparison between the baseline and a network that, in many folds,
produced little more than a constant prediction. That is part of the
result, not a footnote to it.

## RQ2 — CNN vs band-power baseline on identical splits

Paired per-participant contrast, CNN minus baseline balanced accuracy
(each model averaged over its own seeds first; participant bootstrap
CI): **-0.014 [-0.075, +0.042]** over the 11 participants
(results/tables/cnn_loso_rq2_contrast.csv). Per participant the
difference runs from -0.226 (P07) to +0.144 (P10); those values are
reported for completeness and are not read as evidence about
individual participants. The interval includes zero and small
differences in either direction: the data are consistent with no
difference between the two models, and they do not support an
advantage for the CNN. They are not used to exclude an advantage of
0.05 or more either: the interval's upper limit (+0.042) falls 0.008
short of the 0.05 smallest effect size of interest, and under the
estimation-only ruling no exclusion claim is made in either
direction. Two qualifications from the follow-up checks belong with
this contrast. First, the CNN side includes the 16 near-constant
fits counted above, so the contrast is against a network that often
barely learned. Second, the baseline's numbers in this contrast come
from the primary Week 2 run, whose cap seeds vary the window
subsample while the CNN's seeds are model seeds on the fixed
cap-seed-0 set; the sensitivity directly below removes that
asymmetry.

**Sensitivity: baseline rerun on the CNN's exact window set.** The
band-power baseline was rerun LOPO on the identical canonical
cap-seed-0 set of 806 windows, same folds
(experiments/runs/2026-11-21_baseline_loso_subjective_canonical806;
src/models/baseline_loso_canonical806.py). On those windows the
baseline averages 0.488 balanced accuracy over the 11 participants
(against 0.508 for the primary headline over the larger pool), and
the paired CNN-minus-baseline contrast recomputed on the identical
window set is **+0.005 [-0.068, +0.071]**
(results/tables/cnn_loso_rq2_contrast_canonical806.csv). The small
negative tilt of the primary contrast was therefore a property of
the differing window sets, not of the models: on identical windows
the two estimates are centred on the same value, with an interval
that again includes zero.

## Sensitivity (post hoc): early-stopping validation carved by whole participant

This check was run after the headline results, in response to the
review of them, and is labelled post hoc everywhere it is reported.
The concern it answers: in the primary run the early-stopping
validation was carved from run-segments of the training participants,
and the frozen strategy search placed those segments inside a single
training participant's session in every fold, so early stopping was
tuned against one known person's segments. The sensitivity repeats
the CNN LOPO run identically except that the validation set is one
whole training participant: the participant immediately after the
held-out one in sorted order, wrapping around (a fixed rule that
looks at no scores), whose windows are removed from subtraining and
used only for early stopping. A new driver
(src/models/cnn_loso_partval.py) imports the frozen network, input
builder and training pieces and changes only this carve; because
validation and subtraining are then different people, and the
windows are non-overlapping, no purge is needed. Same 806 windows,
same folds, same seeds 0 to 4, same everything else (run folders
experiments/runs/2026-11-21_cnn_loso_partval_subjective_*;
aggregate src/evaluation/cnn_loso_partval_aggregate.py).

Result: CNN LOPO balanced accuracy **0.509 [0.482, 0.538]** (ROC-AUC
0.539 [0.493, 0.583], MCC 0.022 [-0.033, 0.084]), against
0.493 [0.452, 0.529] under the segment carve. The paired
per-participant movement is **+0.016 [-0.012, +0.047]**
(results/tables/cnn_loso_partval_vs_segval.csv): the estimate moved
up slightly, by an amount whose interval includes no movement, and
it remains centred on chance. That movement is reported as an
observation; it is not a verdict that the carve choice is irrelevant
in general, only that under this dataset and design the two carves
gave estimates this close.

One related point the review asked to have verified from the code
rather than assumed: whether the original carve-strategy search ever
used the held-out participant's scores. It did not, and it could not
have. The search (carve_validation in src/models/cnn_segval.py) is
handed only the training participants' rows: the LOSO driver passes
idx.iloc[tr_pos], from which the test participant has already been
removed. It walks a fixed list of candidate strategies in a fixed
order and returns the first one that is structurally feasible:
validation windows of both classes, at least 6 of them, and a
post-purge subtraining set of both classes with at least 20 windows.
It computes no performance score of any kind and optimises nothing;
the first feasible carve is simply the one used.

## RQ1 — the generalisation gap, in three tiers

RQ1 is clearest as three tiers of estimate, from the least honest
split to the most honest:

1. **Same-session estimate: 0.744.** Windows from inside the same
   session, split within it (Week 2 diagnostics). This number
   carries a quantified confound and is never used as a level the
   honest tiers are measured against: window position alone ranks
   the labels perfectly (position-only AUC 1.000, exact in all 20
   sessions, because the block label design puts every alert window
   before every fatigued window), and the block permutation test
   puts the session-out score inside its own no-signal null (null
   balanced accuracy 0.508 ± 0.032 against a real 0.499, p = 0.612;
   docs/week2_diagnostics.md).
2. **Honest within-subject (new sessions of known people): baseline
   0.484, CNN 0.493.** Both at chance.
3. **Cross-participant (LOPO, a wholly new person): baseline 0.508,
   CNN 0.493.** Both at chance.

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

The paired subsets are named so the numbers can be checked: the
baseline paired gap covers P01, P02, P03, P04, P06, P08, P09 and P11;
the CNN paired gap covers P02, P03, P04, P06, P08, P09 and P11. The
other LOPO participants drop out of the within-subject side for
structural reasons, not by choice: P07, P10 and P12 have no valid
session rotation, because in each case no rotation leaves both
classes in the training sessions and both classes in the held-out
session (P07 has retained windows in only one session; P10 and P12
each have a session containing only fatigued windows, and rotating
to their other session leaves a single-class training side). P01
drops out of the CNN set in addition, because no run-segment carve
met the frozen feasibility criteria for any of its within-subject
session folds, so no CNN within-subject value exists for P01.

**The gap is uninformative in this dataset, and it is reported as
such rather than as a finding.** Both ends of the subtraction sit at
chance (a floor effect): a difference between two chance-level
estimates cannot measure what moving to a new person costs, and the
paired samples (n = 8 and n = 7) are too small to rescue it. The
collapse this project set out to measure is between the first tier
and the two honest tiers, and the first tier is itself confounded
by construction, as quantified above. Nothing stronger than that is
claimed from RQ1 here.

Two caveats travel with the CNN gap. Its within-subject side rests on
7 participants and two seeds, and one covered participant's
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
  reported instead. The post hoc sensitivity above re-ran the grid
  with the validation carved by whole participant; its paired
  movement against this run was +0.016 [-0.012, +0.047].
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
per-fold checkpointing was in place in case it had. The follow-up
runs reported above (the participant-carved sensitivity and the
baseline on the canonical window set) also completed without
failures: 55 of 55 sensitivity folds, committed per seed as they
finished.
