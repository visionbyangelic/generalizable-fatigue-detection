# RQ4 — Session-Label Diagnostic (21 Nov 2026)

**Question (PLAN.md RQ4, docs/label_rules.md section 3):** how much of
the apparent performance in this literature is an artefact of labelling
whole sessions as fatigued? My preliminary experiment reached 99.18%
fatigued-class recall with session labels. This diagnostic reruns the
locked Week 2 baseline with labels that come from block position alone,
under the same honest protocols, and prices what the label definition
itself contributes. A second arm, added post hoc after panel review
and labelled as such throughout, runs the day-level form of session
labels (whole earliest session vs whole latest session), which is the
preliminary experiment's own construction; it is reported in the final
section.

**Operationalisation.** The main arm uses the block-phase form of
session labels: every retained window in a session's M1 block is
labelled alert (0) and every retained window in its M3 block is
labelled fatigued (1), with no rating consulted. I use this form for
the main comparison because it is the one that can be compared against
the subjective labels on identical windows, features and splits, so
the label definition is the only moving part. The day-level form
changes the window set and the splits at the same time as the labels,
which confounds exactly the comparison the main arm exists to make;
it therefore runs separately, post hoc, answering a different question
(what the preliminary's design family does to the numbers). Session
labels in either form are a diagnostic only, never ground truth, and
every claim in this document is scoped to this dataset and this
pipeline.

## The identity at the centre of this diagnostic

The subjective construction (label_rules.md section 1) inherits the
block label: M1 windows are alert, M3 windows are fatigued, and the
only further act is the delta filter, which keeps a session in the
primary pool only if the participant's own mental-fatigue rating rose
(delta-mental > 0, 32 of 36 sessions). The two label sets are
therefore **identical by construction on every window where both are
defined; the rating acts only as a session-inclusion filter.** A
Cohen's kappa computed on the shared windows comes out at 1.000, but
that is a tautology, not a finding and not agreement between two
methods: both labellings are functions of the same block variable, so
they cannot disagree. I do not report it as a result anywhere in this
document (the counts behind it are in
`results/tables/rq4_label_agreement.csv` for completeness). The
identity also holds only where both sets are defined; the definitions
still differ in two ways that matter:

- **Coverage**: block-phase labels also label the 68 retained windows
  (6.1% of 1,121) in the 4 non-increasing sessions, where subjective
  labels are undefined because the rating did not rise.
- **What they claim to measure**: the subjective set claims a
  rating-backed fatigue state; the block-phase set claims only block
  position. Identical numbers, different meaning, which is precisely
  why the comparison has to be run rather than asserted.

That identity shapes everything below. "Session vs subjective" inside
this dataset is not two labellings of the same windows; it is two
session sets. The comparison therefore has two arms:

- **restricted32**: the Week 2 primary pool (1,053 windows, 564 alert /
  489 fatigued). Block-phase labels on these windows are numerically
  the subjective labels, so this arm must reproduce Week 2 exactly, and
  it does (see Methods). It is the apples-to-apples anchor.
- **all36**: restricted32 plus the drift/sensitivity check set
  (windows_a1_control, 68 windows), 1,121 windows total (604 / 517).
  34 of the 36 sessions contribute retained windows; P05/S02 and
  P05/S03 retained no M1/M3 windows under Amendment 1 in either arm
  (frontal-contact failure, the known P05 problem). The check sessions
  are heavily one-sided: P07/S01 21 alert / 0 fatigued, P07/S02 19 / 2,
  P10/S03 0 / 1, P12/S03 0 / 25.

The 4 non-increasing sessions in full, because "non-increasing" covers
two different rating behaviours and the difference matters for how the
all36 arm should be read:

| Session | Intensity | Mental rating M1 → M3 | Δ | Retained windows (alert / fatigued) | What the ratings did |
|---|---|---|---|---|---|
| P07/S01 | low | 0.1 → 0.0 | −0.1 | 21 / 0 | Flat at zero. P07 is a near-zero reporter: all 9 of their mental ratings sit between 0.0 and 0.42, so their sessions fail the filter because no fatigue is ever reported, not because it fell. |
| P07/S02 | medium | 0.0 → 0.0 | 0.0 | 19 / 2 | Flat at zero, same story. |
| P10/S03 | low | 34.6 → 19.2 | −15.4 | 0 / 1 | A genuine decrease: the session started moderately fatigued and the rating fell substantially by M3. |
| P12/S03 | high | 37.6 → 32.5 | −5.1 | 0 / 25 | Started moderately fatigued and eased slightly. |

So two of the four are one participant whose ratings never move, and
two are sessions that began fatigued and went down, the pattern the
label rules' review amendments already flagged (a non-increasing
rating can mean an already-fatigued M1 rather than a clean alert
control). None of the four is a straightforward "stayed alert" session.

## Methods

Identical machinery to the Week 2 primary baseline (P configuration):
Amendment-1 windows, 43 band-power features, logistic regression
(C = 1.0, class_weight balanced), median imputation and scaling fit on
training folds only, per-session M1 cap with cap seeds 0 to 4.
Protocols: within-subject, pooled leave-one-session-out, and
leave-one-participant-out (LOPO). The windows themselves are
non-overlapping by design (4 s tumbling windows cut within
continuous runs), so no two windows share samples; every leakage
question in this project is about shared sessions or shared
participants, never about shared samples. Aggregates follow the Week 2
convention (mean over groups of per-group cross-seed means) with
group-level bootstrap intervals (10,000 resamples, rng 42; participants
for within/LOPO, sessions for session-out). Driver:
`src/evaluation/rq4_session_labels.py`, a thin file that reuses the
frozen fold, model and metric code; no frozen file was edited.

**Reproduction check.** The restricted32 arm's point estimates match
the Week 2 subjective headlines closely on all three protocols
(within 0.484, session-out 0.515, LOPO 0.508), and the LOPO interval
reproduces exactly ([0.471, 0.542]). The label identity above is
therefore not just a construction argument; it is visible in the
numbers. The panel then required the stronger statement, exact
reproduction, and the small gap that remains is audited in full below.

**Reproduction audit (panel-owed, 21 Nov 2026).** With identical
windows, features, model and seeds, this arm should in principle
reproduce the stored Week 2 primary exactly. It does not quite, and
the difference has a mechanical cause that I traced suspect by
suspect:

- *Cap sampling: identical.* Both paths call the same frozen
  `cap_pool` on the same window index. The stored predictions of the
  Week 2 runs and of this arm line up row for row (participant,
  session, fold, true label) in every run compared, which only
  happens if the capped pools are the same sets.
- *Feature extraction: identical.* `extract_features` works window by
  window, so extracting on the full uncapped array and subsetting
  afterwards (this driver, and the Week 2 closure pack for LOPO)
  gives an identical feature matrix to extracting on the capped
  subset only (the Week 2 baseline script): I verified the two
  matrices are exactly equal (maximum absolute difference 0.0, NaN
  pattern equal), and that the two paths produce identical
  predictions when run under a common stack.
- *Seed handling, normalisation, feature ordering: identical.* Same
  frozen fold builders, same training-fold imputation and scaling,
  same metric code; the logistic regression carries no seed of its
  own.
- *The `np.trapz` alias: not the cause.* `np.trapezoid` is the
  renamed `np.trapz`, the same computation, and the feature matrices
  above are identical with the alias in place.
- *What actually differs: the environment the stored numbers were
  computed in.* The Week 2 runs were made on 20 Nov, before the VM
  restart; this arm ran on 21 Nov, after the numeric stack was
  reinstalled (numpy 2.5.5, scipy 1.18.1, scikit-learn 1.9.1,
  pandas 3.1.1). One hard marker of the change: the 20 Nov runs
  execute `np.trapz` natively, and NumPy 2.5 removed that name, so
  the 20 Nov runs used an earlier NumPy. Re-running the identical
  code on the current stack reproduces this arm's stored predictions
  exactly. Against the stored Week 2 predictions, the features, the
  sampling and the seeds are identical, and the predictions are
  identical under a common stack; the stored Week 2 probabilities
  differ by at most 0.0025, which moves exactly one window across
  the 0.5 threshold (next point) and the LOPO headline by 0.0003.
  The attribution of that difference to solver-level differences
  between the two software environments is by elimination: every
  code-level suspect above was tested and cleared, but the old
  stack itself was never reproduced, and its package versions were
  not recorded, so the attribution cannot be demonstrated directly.
  What the difference looks like mechanically is solver-iterate
  movement, the logistic-regression solver stopping at a slightly
  different iterate of the same optimum, not any difference in
  data, features or code path; the tolerance sensitivity below
  supports that reading. (Consistent with this, the earlier
  reproduction checks that passed exactly, such as the Week 2
  seed-0 session-out aggregate of 0.4991, are all label-based
  metrics, which a probability move of this size does not touch.)
  All figures in this document are reported from the pinned current
  stack (docs/environment_pin.md); the stored Week 2 values remain
  canonical for the primary results, and the two stacks' headlines
  round to the same three decimals on every protocol (within 0.484,
  session-out 0.515, LOPO 0.508), so no conclusion depends on which
  stack's figures are used.
- *The one flipped window.* Leave-one-participant-out, cap seed 4,
  P04's fold: a fatigued window of P04/S01 whose stored probability
  is 0.4994 in the Week 2 run (predicted alert) and 0.5006 here
  (predicted fatigued). It sits within 0.0013 of the decision
  boundary in both environments. It moves P04's seed-4 balanced
  accuracy by 0.0147 (one fatigued window of 34), P04's cross-seed
  mean by 0.0029, and the LOPO headline by +0.000267. Of the 195
  per-group balanced accuracies across the three protocols and five
  seeds, 194 are identical; this window is the only exception. The
  within-subject and session-out headline aggregates are identical
  to the Week 2 values to six decimal places (0.483556 and 0.515133
  in both). The probability movement does move some fold-level AUCs
  (largest 0.048, in a 10-window fold), because AUC reads the
  probabilities while balanced accuracy reads only the labels.
- *Solver record and tolerance sensitivity (added 21 Nov 2026,
  panel-owed).* The Week 2 run folders did not store the solver's
  iteration counts, so I re-ran the primary LOPO folds on the
  current stack with the record captured (driver:
  `src/evaluation/rq4_tidyup.py`). Under the locked default
  tolerance (1e-4), all 55 folds (11 participants × 5 cap seeds)
  converged with no ConvergenceWarning, in at most 74 iterations
  (median 67), so the default run is not an unconverged run.
  Tightening the tolerance to 1e-8 (iterations rise to a median of
  176, maximum 194, still no warnings) moves probabilities by at
  most 0.026 (mean 0.0006), flips 5 of the 4,030 window predictions,
  and moves the LOPO headline from 0.5078 to 0.5066. The solver's
  own tolerance slack on a single stack is therefore about ten
  times the cross-environment difference being audited (at most
  0.0024 on the LOPO folds, one flip), which is what "solver-iterate
  differences" means concretely here: both environments' solvers
  stop at slightly different iterates of the same optimum, well
  inside the slack the default tolerance permits. Runs:
  `experiments/runs/2026-11-21_baseline_loso_subjective-tolcheck-*`;
  table: `results/tables/rq4_tolerance_sensitivity.csv`.

So the restricted arm does not reproduce the stored Week 2 numbers
exactly, for an understood and mechanical reason: same data, same
features, same code, a different build of the numeric stack, and one
borderline window. Neither path contains a bug, and no Week 2
conclusion is affected; its headlines move by at most 0.0003.

**Environment note (recorded, not hidden).** The frozen feature code
calls `np.trapz`, which NumPy 2.5 removed after renaming it
`np.trapezoid`. The driver aliases the old name to the renamed
function (same computation) instead of editing the frozen file; the
reproduction check above confirms the features are unchanged.

## Results

Balanced accuracy (group bootstrap 95% CI in brackets):

| Protocol | Subjective labels (Week 2 primary) | Block-phase, restricted32 | Block-phase, all36 |
|---|---|---|---|
| Within-subject | 0.484 [0.390, 0.562], 8 participants | 0.484 [0.390, 0.564], 8 | 0.461 [0.371, 0.545], 9 |
| Leave-one-session-out | 0.515 [0.468, 0.555], 20 sessions | 0.515 [0.468, 0.555], 20 | 0.515 [0.470, 0.556], 21 |
| LOPO | 0.508 [0.471, 0.542], 11 participants | 0.508 [0.471, 0.542], 11 | 0.516 [0.482, 0.548], 11 |

Supporting metrics (all36 vs restricted32): LOPO MCC 0.037 vs 0.019,
ROC-AUC 0.547 vs 0.540. Session-out MCC 0.043 vs 0.045, AUC 0.569 vs
0.549. Within MCC −0.076 vs −0.032, AUC 0.460 vs 0.487. Full aggregates:
`results/tables/rq4_aggregates.csv`; per-group tables:
`results/tables/baseline_*_sessionlabels-*_summary.csv`; run folders:
`experiments/runs/2026-11-21_baseline_*_sessionlabels-*_s0..s4`.

**Within-subject detail.** The 8 participants scored in both arms have
identical per-participant values in both arms (within folds never touch
another participant's data, and their own sessions are unchanged). The
aggregate drop from 0.484 to 0.461 is pure composition: all36 adds
P07, whose within-subject score is 0.283. P07 is the near-zero reporter
whose ratings never rose; under block-phase labels its M3 blocks are
called fatigued anyway, and the model separates them worse than chance
inside P07's own sessions.

**Session-out detail.** The aggregate does not move at all (0.515 in
both arms). Only one check session can serve as a held-out fold:
P07/S02, which scores exactly 0.500. The other three check sessions
are single-class after the cap (P07/S01 has no fatigued windows,
P10/S03 and P12/S03 have no alert windows), so they can enter training
pools but can never be test folds.

**LOPO detail (the headline comparison).** Per-participant balanced
accuracy, cross-seed means:

| Participant | Subjective (= restricted32) | Block-phase, all36 |
|---|---|---|
| P01 | 0.496 | 0.508 |
| P02 | 0.522 | 0.550 |
| P03 | 0.490 | 0.490 |
| P04 | 0.451 | 0.498 |
| P06 | 0.566 | 0.538 |
| P07 | 0.582 | 0.580 |
| P08 | 0.447 | 0.451 |
| P09 | 0.579 | 0.605 |
| P10 | 0.382 | 0.397 |
| P11 | 0.559 | 0.553 |
| P12 | 0.513 | 0.505 |
| **Mean (11)** | **0.508** | **0.516** |

(P05 has no LOPO fold in either arm: its only retained session is
single-class and the cap reduces it to zero windows, the documented
Week 2 mechanism.) P07's test fold grows from one session (S03) to all
three in all36 and its score does not move (0.582 to 0.580). The
largest gains are P04 (+0.047) and P09 (+0.027), the largest loss P06
(−0.028). The aggregate difference is +0.008, and the two intervals
overlap almost completely.

## Interpretation

**1. Dropping the rating filter buys nothing.** RQ4's main question
was whether the subjective-rise inclusion filter matters. In this
dataset, under honest splits, it does not. The one label-level
difference between the subjective rules and pure block-phase labels
is the delta filter, and removing it changes the headline numbers by
at most 0.008 balanced accuracy, in inconsistent directions across
protocols (LOPO +0.008, session-out 0.000, within −0.023 driven by one
added participant). A researcher who skipped the ratings entirely and
labelled by block would, in this dataset and with this pipeline, get
the same honest-split results I did. The inflation RQ4 hunts is not
hiding in the delta filter.

**2. The block-phase definition does not transfer either.** The
natural hope for block-phase session labels is that a cruder, more
regular target might be easier to generalise. It is not, here:
leave-one-participant-out (LOPO) with block-phase labels on all 36
sessions gives 0.516 [0.482, 0.548]. That interval overlaps the
subjective arm's ([0.471, 0.542]) almost completely, so at this
precision the two arms cannot be separated, and no equivalence is
claimed: both sit below the 55% line and the feasibility gate's
estimation-only ruling governs both, exactly as it governs the
primary (a clearing interval in an underpowered design would be an
observation, not a ruling change). And per Week 2, even this level
cannot be credited to fatigue physiology: a model given only the
recording-quality mask pattern matched the full EEG model
(session-out 0.530 vs 0.515), so whatever weak cross-participant
structure exists in this dataset is carried as much by artefacts as
by spectra. Block-phase labels inherit that limit unchanged.

**3. Where the block-phase artefact actually lives.** Inside a
session, the M1-vs-M3 contrast is genuinely decodable (the Week 2
within-session estimate is 0.744), and window position alone
separates the blocks with AUC 1.0 by construction, because every
alert window precedes every fatigued window. The reason position and
drift can separate blocks inside one session yet carry nothing
across people is that the trajectory is session-specific: contact
quality, drift and time-on-task evolve smoothly within a recording,
so early and late windows of that recording differ systematically,
but each session drifts in its own direction and amount, so there is
no participant-invariant direction for a model to carry to a new
person. That is how a within-session estimate of 0.744 coexists with
LOPO at 0.51. Block-phase labels are perfectly confounded with
time-on-task, block position and electrode drift, so any evaluation
that lets training and test windows come from the same recordings
(random window splits, or a chronological split inside one
participant's recordings)
can harvest position and drift and report it as fatigue. My
preliminary experiment belonged to that design family (one
participant, whole-session labels, a split inside the same
recordings, 99.18% recall). Under splits that hold out the session or
the participant, the same features and the same model with
block-phase labels give 0.46 to 0.52 in this dataset. So the
artefact is real, but it lives in the pairing of session-confounded
labels with recording-sharing evaluation, not in the choice between
rating-filtered and unfiltered block labels. Two bounds keep this
honest: the within-session estimate prices position and drift at
about 0.74, and the Week 2 random-split leakage demo reached 0.571,
so the block-phase arm alone does not account for a number like
99.18%. The day-level arm below tests the remaining candidate
mechanism, session fingerprinting under whole-session labels,
directly.

**Caveats.** (i) The claims in this section are scoped to block-phase
labels in this dataset with this pipeline; the day-level form is
priced separately in the post hoc section below, and nothing here is
a claim about session labels in the wider literature. (ii) The 4
check sessions contribute few windows (68) and three of them can
never be test folds, so the all36 comparison is powered to detect
only a large filter effect; a small one (under ~0.01 to 0.02) would
be invisible here, which is why I report the difference as "at most
about 0.01 at headline level" rather than "exactly zero".
(iii) P05 is absent from LOPO in both arms, as in Week 2.

## Day-level labels (post hoc)

This arm was not part of the original RQ4 design. It is owed to the
panel review of the block-phase arm, which pointed out that the
preliminary experiment did not label blocks at all: it labelled whole
sessions. Leaving the day-level form untested would have left the
write-up implying an explanation of the 99.18% that the block-phase
numbers cannot carry. Everything in this section is post hoc and is
labelled as such in every run note.

**Definition (exact).** For each participant, every retained
Amendment-1 window in the whole earliest recorded session (S01) is
labelled alert (0) and every retained window in the whole latest
recorded session (S03) is labelled fatigued (1); the middle session
(S02) is excluded entirely. Session numbering follows recording
order. This mirrors the preliminary prototype's construction
(Session 01 as alert, Session 03 as fatigued). The retained windows
are the Amendment-1 files (primary pool plus check set, 1,121
windows before the session restriction): they cover the M1 and M3
blocks, because M2 windows were set aside upstream by the primary
construction, before any label question arose. No M1 cap is applied:
the cap balances M1 against M3 inside a session, and under day-level
labels there is no block distinction left to balance. The arm covers
11 participants and 671 windows (377 alert, 294 fatigued). P05 is
excluded entirely, from training as well as from test folds, because
its latest session retained no windows, so it has no day-level
contrast; this also keeps the participant set identical to the
primary LOPO. Same 43 features, same logistic-regression baseline,
same training-fold-only imputation and scaling. Driver:
`src/evaluation/rq4_day_level.py`.

Seed conventions, stated because they differ from the primary: the
LOPO run has no stochastic element (no cap, and the model is
deterministic given the data), so its folder carries s0 by naming
convention only. The other two runs are priced at split seeds 0 to 4,
because there the randomness under study is the split itself.

**Run 1: LOPO with day-level labels (the honest estimate).**
Balanced accuracy **0.484 [0.353, 0.620]** over 11 participants
(participant bootstrap interval), MCC −0.084, ROC-AUC 0.492. This
arm is post hoc and estimation-only: the interval crosses the 55%
line, so no null claim rests on the interval alone. The exact
session-level swap null over the 11 fold scores (all 2^11 = 2,048
sign flips of the per-participant deviations from 0.5, enumerated
exactly rather than sampled) has mean 0.500 and SD 0.069, and the
observed 0.484 sits well inside it (two-sided p = 0.822;
`results/tables/rq4_daylevel_swapnull.csv`). Whole
sessions as labels do not transfer to unseen participants either.
The per-participant values are spread across almost the whole scale,
which is itself informative: day identity helps some participants
and inverts others, as expected if what separates two recording days
is mostly day-specific rather than a shared fatigue direction.

| Participant | Alert windows (S01) | Fatigued windows (S03) | Balanced accuracy |
|---|---|---|---|
| P01 | 22 | 2 | 0.682 |
| P02 | 4 | 66 | 0.500 |
| P03 | 57 | 7 | 0.337 |
| P04 | 10 | 33 | 0.864 |
| P06 | 33 | 22 | 0.174 |
| P07 | 21 | 31 | 0.601 |
| P08 | 31 | 50 | 0.194 |
| P09 | 79 | 54 | 0.261 |
| P10 | 32 | 1 | 0.438 |
| P11 | 84 | 3 | 0.821 |
| P12 | 4 | 25 | 0.455 |

Read the small folds coarsely: P10's fatigued class is a single
window, P01's is 2 and P11's is 3, so those participants' scores move
in steps of a third or a half and the wide interval is the honest
summary.

**Run 2: random-window split with recordings shared between train
and test (the leaky design, deliberately).** The same windows,
labels, features and model, with windows pooled across participants
and split at random into 5 stratified folds, so every session
appears in both train and test; because the windows are pooled,
every participant appears on both sides too, so session sharing and
participant sharing are confounded in this arm and I describe it as
sharing recordings between train and test. Preprocessing stays
fold-fitted, so the split is the single leak, matching the Week 2
leakage-gap demo's discipline. Balanced accuracy by split seed:
0.740, 0.747, 0.735, 0.751, 0.730 (mean **0.741**); plain accuracy
mean **0.753** (0.742 to 0.765 across seeds); alert-class recall
mean **0.842**; fatigued-class recall: 0.636, 0.656, 0.629, 0.643,
0.633 (mean **0.639**); ROC-AUC mean 0.784 (per-seed values in
`results/tables/rq4_leaky_accuracy.csv`). The fatigued recall is
the metric the preliminary's headline used (its 99.18% was a
fatigued-class recall), so it is the number to compare against:
sharing recordings between train and test inflates the day-level
numbers from 0.484 to 0.741 balanced accuracy on identical
machinery, and the fatigued recall reaches 0.639. That is a large,
real inflation, and it is nowhere near 99.18% recall.

**Run 3: session-identification probe.** Within each participant,
classify which session (S01, S02 or S03) a window came from, from the
same 43 features, under a random-window split (folds per participant
set to the smallest session count, capped at 5, so stratification
stays valid; split seeds 0 to 4). The probe's classes are the
participant's own three sessions (S01, S02, S03) and its split is a
within-participant random-window split; against the 0.333 chance
level for three classes it reaches a mean balanced accuracy of
**0.821 [0.785, 0.853]** over the 10 probed participants;
per-participant values run from 0.704 (P01) to 0.894
(P07), and raw accuracy from 0.78 to 0.93. P05 (one session with
retained windows) and P10 (a session with a single window) cannot be
probed and are excluded, recorded in
`results/tables/probe_sessionid_daylevel_summary.csv`. Sessions are
therefore strongly identifiable from these features within a person:
the fingerprint that run 2's split rewards is demonstrably there for
a model to use.

**What can now honestly be said about the preliminary 99.18%.**
Three things are established. First, the preliminary's label
construction transfers no better than any other in this dataset:
whole-session labels under an honest participant-held-out evaluation
sit at 0.484 (post hoc arm, estimation-only), the same chance-level
story as every other arm.
Second, the mechanism family the preliminary's design belongs to is
real and large: sessions carry a strong feature fingerprint (the
session-identification probe, whose classes are the participant's
three sessions under a within-participant random-window split,
scores 0.821 against a 0.333 chance level), and letting train and
test share recordings
inflates the same labels, features and model from 0.484 to 0.741
balanced accuracy. Third, and this is the boundary of the claim:
that inflation reaches a fatigued recall of about 0.64, not 0.99, so
this arm does not account for the preliminary number, and I do not
claim that it does. The preliminary differed from run 2 in at least
three further ways that this arm does not price separately: it used
a single participant, so the model only had to separate one person's
two days; its split was chronological inside the same recordings, so
test windows sat next to training windows in time, a stronger sharing
than a random split; and it used a 1D-CNN on raw EEG with its own
preprocessing rather than band-power features and logistic
regression. Any or all of those may carry the remaining distance to
99.18%. What the day-level arm settles is narrower and sufficient
for the thesis: the 99.18% is not evidence about fatigue detection,
because the same labels collapse to chance the moment the evaluation
stops sharing recordings, and a large part of what a
recording-sharing evaluation measures is which session a window
came from.

## The 2x2 replication arm (post hoc, panel-specified)

The day-level arm leaves three of the preliminary's ingredients
unpriced as a combination: it was one participant, its split was
chronological inside the same recordings, and its model was a 1D-CNN
on raw EEG. This arm crosses the two factors that can be crossed on
this pipeline's windows, model (band-power logistic regression vs
the EEGNet CNN) by split (chronological vs random window), under
day-level labels, on Subject 01 only. Subject 01 is the
preliminary's own participant, chosen for that reason and never for
performance. The design and the reading below were specified by the
panel (review 24) and are written here before the results.

**Data, stated before the numbers.** The windows are this
pipeline's retained Amendment-1 windows for P01 (primary pool plus
check set, M1/M3 blocks only, no cap): 24 windows, Session 01
carrying 22 (alert) and Session 03 carrying 2 (fatigued). That is
not the preliminary's window set. The preliminary epoched the whole
sessions at a 2 s stride (50% overlap) and worked with 1,353
windows; most of P01's Session 03 did not survive Amendment-1
retention here. So this arm prices model x split on the windows
this pipeline kept, at a far smaller n than the preliminary had,
and every split below trains on 16 windows (15 alert, 1 fatigued)
and tests on 8 (7 alert, 1 fatigued). The fatigued test class is a
single window in every split, so fatigued recall here is a 0-or-1
quantity and is reported with its counts wherever it appears.

**Splits (exact rules).** Chronological: within each session,
windows are ordered by recording position; the first 70% (floor)
are train, the last 30% are test. No purge gap: the windows are
non-overlapping, so the cut shares no samples, and the temporal
adjacency across the cut is part of the design being priced. (The
preliminary repository's README describes its own audit notebook as
using an 80/20 chronological cut; this arm uses the panel-specified
70/30 and says so.) Random: stratified by class over the pooled 24
windows (per class, the test count is max(1, round(0.3 x n))), at
split seeds 0 to 4; the same split is used for both models, so the
model comparison is paired by split. A reversed-chronological LR
run (last 70% train, first 30% test) is included as a direction
sensitivity.

**Models.** LR is the locked baseline (43 band-power features,
training-fold imputation and scaling). The CNN is the frozen EEGNet
configuration from `src/models/cnn.py` (6-channel input with mask
indicator planes, Adam, balanced class weights and per-channel
normalisation from the training windows only), at model seeds 0, 1
and 2 per cell. One forced adaptation: cnn.py early-stops on a
validation split carved from the training windows, but no
both-class carve exists here, because the fatigued class has
exactly one training window in every split; any carve either
removes it from training or leaves the validation set single-class.
The CNN cells therefore train on all 16 training windows for a
fixed budget of 100 epochs (the frozen maximum) with no early
stopping, and every CNN run note says so.

**Pre-declared reading.** If the chronological CNN cell approaches
the preliminary's ~0.99 fatigued recall while the chronological LR
cell does not, the preliminary's gap prices as model + split. If no
cell approaches 0.99, the preliminary number is not reproducible
under this pipeline, and the write-up says so. Either way, no cell
in this arm is a performance estimate: every cell shares both
recordings between train and test by design.

**Results.** Every fit tests on the same 8 windows (7 alert, 1
fatigued), so each fatigued recall below is a count out of 1 per
fit. Driver: `src/evaluation/rq4_replication_2x2.py`; per-fit table:
`results/tables/rq4_replication_2x2_fits.csv`; cell summary:
`results/tables/rq4_replication_2x2_summary.csv`.

| Cell | Fits | Balanced accuracy | Accuracy | Fatigued recall | Alert recall | ROC-AUC |
|---|---|---|---|---|---|---|
| LR, chronological | 1 | 1.000 | 1.000 | 1.000 (1/1) | 1.000 | 1.000 |
| LR, chronological reversed | 1 | 1.000 | 1.000 | 1.000 (1/1) | 1.000 | 1.000 |
| LR, random | 5 | 1.000 (all seeds) | 1.000 | 1.000 (5/5 fits) | 1.000 | 1.000 |
| CNN, chronological | 3 | 1.000 (all seeds) | 1.000 | 1.000 (3/3 fits) | 1.000 | 1.000 |
| CNN, random | 15 | 0.900 (SD 0.207) | 0.975 | 0.800 (12/15 fits) | 1.000 | 1.000 |

**Which branch of the pre-declared reading applies: neither, and
the write-up says so plainly.** The panel's two branches were
"CNN-chronological approaches 0.99 while LR-chronological does
not" (prices as model + split) and "no cell approaches 0.99" (not
reproducible under this pipeline). What happened is a third
pattern the pre-declaration did not list: the chronological LR
cell also scores a perfect 8 of 8, as does the reversed split and
every random LR split. On this pipeline's retained windows, the
recording-sharing split inside one participant is by itself
sufficient for a perfect score with either model, so the model
factor prices at nothing here; if anything the CNN is the less
reliable of the two, missing the single fatigued test window in 3
of its 15 random fits (in each miss its AUC is still 1.000: it
ranks the fatigued window first but below the 0.5 threshold, a
calibration artefact of training on one fatigued example). The
honest conclusion is the stronger form of the day-level arm's:
P01's two retained sessions are trivially separable once both
appear in training, which is the session-fingerprint mechanism
operating at its purest, and a headline of the preliminary's
magnitude needs no CNN and no fatigue signal to arise under this
design family.

The bounds travel with the result. The fatigued class is 2 windows
in total and 1 per test set, so each recall is a single coin flip
landing heads; the retained 24 windows are a heavily filtered
subset of the preliminary's 1,353, and nothing here shows the
preliminary's own window set behaves the same way; and the
chronological test fatigued window sits immediately after the one
fatigued training window in time. What the arm settles is the
factor pricing it was built for: split design alone, on one
participant with whole-session labels, can produce the
preliminary's headline number with the plain baseline model. The
remaining distance to the preliminary (its exact windows, its own
preprocessing) is not priced, and I do not claim it is.
