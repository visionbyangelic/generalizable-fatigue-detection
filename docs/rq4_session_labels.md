# RQ4 — Session-Label Diagnostic (21 Nov 2026)

**Question (PLAN.md RQ4, docs/label_rules.md section 3):** how much of the
apparent performance in this literature is an artefact of labelling whole
sessions as fatigued? My preliminary experiment reached 99.18%
fatigued-class recall with session labels. This diagnostic reruns the
locked Week 2 baseline with labels that come from block position alone,
under the same honest protocols, and prices what the label definition
itself contributes.

**Operationalisation.** Session label here means the block-phase form:
every retained window in a session's M1 block is labelled alert (0) and
every retained window in its M3 block is labelled fatigued (1), with no
rating consulted. I use this form because it is the one that can be
compared against the subjective labels on identical windows, features
and splits, so the label definition is the only moving part. The locked
rules (label_rules.md section 3) also describe the day-level form
(recording session / treadmill-intensity condition, the preliminary
experiment's design); a day-level run would change the window set and
the splits at the same time as the labels, which confounds exactly the
comparison RQ4 exists to make. Session labels in either form are a
diagnostic only, never ground truth.

## The identity at the centre of this diagnostic

The subjective construction (label_rules.md section 1) inherits the
block label: M1 windows are alert, M3 windows are fatigued, and the
only further act is the delta filter, which keeps a session in the
primary pool only if the participant's own mental-fatigue rating rose
(delta-mental > 0, 32 of 36 sessions). So on any window where both
label sets are defined, they are the same label by construction:

- **Cohen's kappa (session-phase vs subjective, window level): 1.000**
  (observed agreement 1.000, n = 1,053 windows, the full uncapped
  restricted pool; `results/tables/rq4_label_agreement.csv`).
- The definitions differ only in **coverage**: session-phase labels
  also label the 68 retained windows (6.1% of 1,121) in the 4
  non-increasing sessions, where subjective labels are undefined
  because the rating did not rise.

That identity shapes everything below. "Session vs subjective" inside
this dataset is not two labellings of the same windows; it is two
session sets. The comparison therefore has two arms:

- **restricted32**: the Week 2 primary pool (1,053 windows, 564 alert /
  489 fatigued). Session-phase labels on these windows are numerically
  the subjective labels, so this arm must reproduce Week 2 exactly, and
  it does (see Methods). It is the apples-to-apples anchor.
- **all36**: restricted32 plus the drift/sensitivity check set
  (windows_a1_control, 68 windows), 1,121 windows total (604 / 517).
  34 of the 36 sessions contribute retained windows; P05/S02 and
  P05/S03 retained no M1/M3 windows under Amendment 1 in either arm
  (frontal-contact failure, the known P05 problem). The check sessions
  are heavily one-sided: P07/S01 21 alert / 0 fatigued, P07/S02 19 / 2,
  P10/S03 0 / 1, P12/S03 0 / 25.

## Methods

Identical machinery to the Week 2 primary baseline (P configuration):
Amendment-1 windows, 43 band-power features, logistic regression
(C = 1.0, class_weight balanced), median imputation and scaling fit on
training folds only, per-session M1 cap with cap seeds 0 to 4.
Protocols: within-subject, pooled leave-one-session-out, and
leave-one-participant-out (LOPO). Aggregates follow the Week 2
convention (mean over groups of per-group cross-seed means) with
group-level bootstrap intervals (10,000 resamples, rng 42; participants
for within/LOPO, sessions for session-out). Driver:
`src/evaluation/rq4_session_labels.py`, a thin file that reuses the
frozen fold, model and metric code; no frozen file was edited.

**Reproduction check.** The restricted32 arm's point estimates match
the Week 2 subjective headlines to within 0.0004 on all three protocols
(within 0.484, session-out 0.515, LOPO 0.508), and the LOPO interval
reproduces exactly ([0.471, 0.542]). The label identity above is
therefore not just a construction argument; it is visible in the
numbers.

**Environment note (recorded, not hidden).** The frozen feature code
calls `np.trapz`, which NumPy 2.5 removed after renaming it
`np.trapezoid`. The driver aliases the old name to the renamed
function (same computation) instead of editing the frozen file; the
reproduction check above confirms the features are unchanged.

## Results

Balanced accuracy (group bootstrap 95% CI in brackets):

| Protocol | Subjective labels (Week 2 primary) | Session-phase, restricted32 | Session-phase, all36 |
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
whose ratings never rose; under session-phase labels its M3 blocks are
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

| Participant | Subjective (= restricted32) | Session-phase, all36 |
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

**1. Dropping the rating filter buys nothing.** The one label-level
difference between the subjective rules and pure session-phase labels
is the delta filter, and removing it changes the headline numbers by
at most 0.008 balanced accuracy, in inconsistent directions across
protocols (LOPO +0.008, session-out 0.000, within −0.023 driven by one
added participant). If a researcher skipped the ratings entirely and
labelled by block, their honest-split results would be the same as
mine. The inflation RQ4 hunts is not hiding in the delta filter.

**2. The session definition does not transfer either.** The natural
hope for session labels is that a cruder, more regular target might be
easier to generalise. It is not: LOPO with session-phase labels on all
36 sessions is 0.516 [0.482, 0.548], statistically indistinguishable
from the subjective 0.508, and both intervals sit below the 55% line
(the estimation-only ruling from the feasibility gate still governs;
these are estimates, not equivalence claims). And per Week 2, even
this level cannot be credited to fatigue physiology: a model given
only the recording-quality mask pattern matched the full EEG model
(session-out 0.530 vs 0.515), so whatever weak cross-participant
structure exists is carried as much by artefacts as by spectra.
Session labels inherit that ceiling unchanged.

**3. Where the session-label artefact actually lives.** Inside a
session, the M1-vs-M3 contrast is genuinely decodable (Week 2
within-session ceiling 0.744), and window position alone separates the
blocks with AUC 1.0 by construction, because every alert window
precedes every fatigued window. Session-phase labels are perfectly
confounded with time-on-task, block position and electrode drift, so
any evaluation that lets training and test windows share a session
(random window splits, or a chronological split inside one
participant's recordings) can harvest position and drift and report
it as fatigue. That is the design family my preliminary experiment
belonged to (one participant, session labels, a split inside the same
recordings, 99.18% recall). Under splits that hold out the session or
the participant, the same labels, the same features and the same model
give 0.46 to 0.52. RQ4's answer is therefore: the artefact is real,
but it lives in the pairing of session-confounded labels with
session-sharing evaluation, not in the choice between rating-filtered
and unfiltered block labels. Fix the evaluation and the session-label
advantage evaporates; keep the evaluation and even honest labels
would look better than they are.

**Caveats.** (i) This prices the block-phase form of session labels;
the day-level form (intensity condition) remains untested as a
standalone arm, by the design argument in the operationalisation note.
(ii) The 4 check sessions contribute few windows (68) and three of
them can never be test folds, so the all36 comparison is powered to
detect only a large filter effect; a small one (under ~0.01 to 0.02)
would be invisible here, which is why I report the difference as "at
most about 0.01 at headline level" rather than "exactly zero".
(iii) P05 is absent from LOPO in both arms, as in Week 2.
