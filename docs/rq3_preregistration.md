# RQ3 pre-registration: does adding heart rate and EDA to EEG help cross-participant fatigue detection?

**Status: DECLARED 21 Nov 2026, before any RQ3 modelling.** RQ3 is the optional multimodal extension in PLAN.md (Weeks 5–6), approved for running on 21 Nov 2026 ("just run it since we have time"). This note fixes the design before a single peripheral feature is computed for modelling. What has been seen so far is the inventory only (`docs/RQ3_INVENTORY.md`): coverage, timing and value-sanity statistics for the wrist streams. No model has been fit on peripheral data and no arm's outcome is known. Any change to what is declared here needs a dated amendment written before results are seen, never discretion after the fact.

## Why RQ3 at all, and why it is not a rescue

On FatigueSet, EEG alone does not transfer across participants under honest splits: the band-power baseline's leave-one-participant-out balanced accuracy is 0.508 [0.471, 0.542] and the CNN's is 0.493 [0.452, 0.529] (docs/week2_diagnostics.md, docs/cnn_loso_results.md). The peripheral signals are the one signal family in this dataset that the primary pipeline has never touched, and the proposal always carried the question as an optional extension. RQ3 asks it under the same discipline as everything else in this project: same labels, same windows, same splits, same model family, same estimation-only reporting. A null result is a complete answer here, not a failure: it would say the cross-participant failure on this dataset is not specific to EEG.

## The one primary contrast

Leave-one-participant-out balanced accuracy of the **combined** arm minus the **EEG-only** arm, paired per participant over cap-seed means, with a participant bootstrap interval. The headline is that contrast together with the two arms' levels. Within-subject and leave-one-session-out results are secondary tiers, reported the same way, and cannot upgrade the headline.

## Arms

- **(a) EEG-only (reference).** The Week 2 primary baseline exactly as run: the 43 Amendment-1 band-power features (`src/features/bandpower.py`) with the locked logistic regression. Its stored per-participant values are the comparator for the primary contrast. It is not rerun for the primary analysis. (For the restriction sensitivity below, it is additionally evaluated on the restricted window subset, from its stored features, so that sensitivity's contrast stays paired.)
- **(b) Peripheral-only.** The six peripheral columns defined below, same model, same splits.
- **(c) Combined.** The 43 EEG features plus the six peripheral columns (49 in total), same model, same splits.

**Declared diagnostic (not an arm): indicators-only.** A logistic regression on the two validity indicators alone (`hr_valid`, `eda_valid`), same protocols. The inventory shows a window's missing peripheral data is itself label-informative in this dataset (Section: missing data), so any peripheral-arm score has to be apportionable between physiology and the missingness pattern. This project has run the equivalent control at every step (mask-only baselines in Week 2 and RQ4); RQ3 is no exception. The diagnostic is reported alongside the arms and never competes with them for the headline.

## Labels and windows

- **Labels:** the primary subjective labels only (M1 windows alert, M3 windows fatigued, in the 32 sessions whose mental-fatigue rating rose). Performance labels are not run for RQ3. That is a scope decision, stated plainly: RQ3 is a question about signals on the primary label set; the label-construction question belongs to RQ4 and to the DROZY arm, and both are already answered.
- **Windows:** the Amendment-1 primary pool, unchanged: the 1,053 retained M1/M3 windows (`data/processed/windows_a1_uncapped_index.csv`), with cap seeds 0 to 4 applied exactly as the Week 2 baseline applies them (per session, M1 windows capped to the retained M3 count, seeded per cap seed; `src/models/baseline.py`). Within a cap seed, every arm sees the identical window set. The four non-increasing sessions stay excluded, as in every primary analysis. P05 contributes no fold in any protocol (its five pool windows are all alert, a single class), exactly as in Week 2.
- **Peripheral assignment:** a window's span is the closed epoch-ms interval from its first to its last EEG timestamp (1,024 raw samples; the stored `centre_ms` is that span's exact midpoint, verified for all 1,053 windows in the inventory). A wrist sample belongs to the window if and only if its timestamp falls in that span. Streams are read through the project's existing loader, so `*2.csv` continuations are concatenated in timestamp order (this affects only P02/S01's wrist data).

## The peripheral feature set (complete, and closed)

Six columns per window, computed from the Empatica E4 streams. Nothing else is computed, and nothing here is tuned:

1. **hr_mean**: mean of `wrist_hr` samples in the window span with a positive value. (`wrist_hr` is the E4's own 1 Hz product, an average over the preceding 10 s.)
2. **eda_mean**: mean of `wrist_eda` samples in the span, in µS.
3. **eda_slope**: least-squares slope of `wrist_eda` against time over the in-span samples, in µS per second.
4. **eda_scr_count**: number of skin conductance responses in the span, by this fixed rule: scanning the raw in-span 4 Hz series in time order, a response is a rise of at least 0.01 µS from a local minimum to the next local maximum; each such rise counts once. No smoothing and no filtering are applied before counting.
5. **hr_valid**: 1 if the window holds at least 2 usable HR samples, else 0.
6. **eda_valid**: 1 if the window holds at least 8 EDA samples and their maximum exceeds 0.002 µS, else 0.

`hr_mean` is undefined (NaN) when `hr_valid` is 0; the three EDA features are undefined when `eda_valid` is 0. NaNs are handled by the locked pipeline's median imputation, fit on the training fold only, and the indicators are always present as features: a missing stream is represented explicitly, never silently filled.

**Considered and excluded, with the reasons fixed here:**

- **IBI-derived variability (for example RMSSD): excluded, fallback taken.** The inventory's coverage counts decide this, by the rule stated when RQ3 was designed (variability features only if IBI coverage supports them). It does not: only 30.7% of pool windows hold at least 3 usable beats, per participant from 7.2% to 67.7%, and the shortfall is worse in the fatigued class (24.7%) than the alert class (35.8%). A feature missing for about 69% of windows, that unevenly, would function as a second missingness indicator with noise attached. Heart rate therefore enters through the complete 1 Hz `wrist_hr` product only. (Inventory, Section 4.)
- **Skin temperature: excluded.** It moves over minutes, is dominated by ambient conditions and by the treadmill activity the protocol imposes between the blocks, and within a 4 s window it is effectively constant per session: it would hand the model a session-identity feature, the exact failure mode this project exists to catch.
- **BVP waveform features: excluded.** The E4's sanctioned derivatives of the PPG are its HR and IBI products; raw-BVP morphology would be a second, unvalidated feature family bolted on mid-project.
- **Wrist accelerometer: excluded.** Motion in this protocol is a designed confound (the activity block sits between the labelled blocks by construction), not a fatigue signal.
- **Zephyr chest HR/HRV and eSense ear PPG: excluded.** Second devices, with their own clocks, missingness patterns and reliability flags. EDA exists only on the E4, so the E4 is the peripheral device; the inventory records that the others exist and are usable in principle, so this choice is visible rather than silent.

## Missing data: the rule and the reason

**Primary rule: keep every window.** Peripheral features are computed for all pool windows; where a stream is absent the features are NaN, the validity indicators say so, and imputation is fit on training folds only, exactly as the locked pipeline already treats masked EEG channels under Amendment 1. Three reasons, in order of weight:

1. The primary contrast stays a pure feature-set contrast: identical windows in every arm, paired per participant, so any movement is attributable to the added signals and nothing else.
2. It mirrors the project's established treatment of missing data (Amendment 1's mask-and-indicate for EEG channels), rather than inventing a second philosophy for a second device.
3. The inventory shows restriction would not be a neutral filter. The missingness is whole-session and class-asymmetric: two participants' E4 streams end before the M3 block, so fatigued windows lose their peripheral data at about 12% against about 0.2% for alert windows, and 59 of the 60 affected windows are the fatigued windows of P08/S02 and P10/S01. Dropping incomplete windows would delete most of two participants' fatigued class and quietly change what the LOPO estimate means.

**Sensitivity: complete-peripheral restriction.** All three arms evaluated on the windows with `hr_valid` = 1 and `eda_valid` = 1 (993 of the 1,053 pool windows before capping), with the restriction counts reported per participant and per class. If the primary and sensitivity readings disagree, both are reported and the disagreement is stated; neither is picked for being nicer.

**Standing reporting obligation.** The validity rates by class and by participant (inventory, Section 3) are reported beside the RQ3 results, together with the indicators-only diagnostic. In this dataset a validity indicator is not a boring technicality: because the truncated sessions are M3-heavy, "the wristband had stopped" is itself evidence about the label, and a reader is entitled to see how much of any peripheral-arm performance that fact alone can buy.

## Model and protocols

The locked baseline configuration, unchanged: logistic regression, L2, C = 1.0, class weights balanced, with median imputation and standardisation inside a pipeline fit on each training fold only (`src/models/baseline.py`). Cap seeds 0 to 4, the Week 2 convention. Protocols: leave-one-participant-out (headline, 11 participants form folds), leave-one-session-out pooled, and within-subject session rotation, under the same fold-validity rules as the Week 2 baseline (a fold counts only if train and test each contain both classes). Metrics: balanced accuracy primary; macro-F1, ROC-AUC and MCC alongside; per-participant values always reported next to aggregates.

**No CNN for RQ3.** The question is whether these signals add transferable information to the locked pipeline, and PLAN.md's multimodal sketch specifies feature-level fusion into the classical baseline, which is what this is. The CNN's FatigueSet cross-participant result is already a null whose evidence the panel ruled weak, because its training instability cannot be separated from absence of signal (docs/cnn_loso_results.md); adding a second model class here would confound the attribution of any change, in either direction.

## Reporting and the gate

Estimation-only, as everywhere in this project. The Δ = 0.05 equivalence gate was already unattainable for the FatigueSet LOPO design at this sample size (docs/decision_table_lock.md), and RQ3 changes neither the participant count nor the bound, so no equivalence or null verdict will be claimed and no new gate computation will be run. Intervals are participant bootstrap (10,000 resamples) over per-participant seed means, for arm levels and for the paired contrast alike.

## Pre-declared readings

| Outcome pattern | What the project may say |
|---|---|
| Combined above EEG-only, paired interval excludes 0 | Peripheral signals add transferable information that EEG alone does not carry on this dataset, under this label design. The levels are still reported plainly: adding information is not the same as solving cross-participant fatigue detection. |
| Combined and EEG-only both at chance, paired interval spans 0 | Peripheral signals do not rescue cross-participant transfer on this dataset under the locked pipeline. The FatigueSet null is not specific to EEG. |
| Peripheral-only above chance while combined is not | A surprising pattern, reported as a puzzle, not a headline. The indicators-only diagnostic and the validity-by-class table are examined first, because a missingness artefact is the standing alternative explanation; then the restriction sensitivity. No claim until those are on the page. |
| Any arm above chance only in the within-subject or session-out tiers | Read exactly as RQ1 reads the EEG tiers: separability among known people or sessions is not transfer. The LOPO tier governs every claim. |

One caveat travels with every row of that table. M1 and M3 differ in protocol position by construction, and heart rate in particular answers to the treadmill activity and to time on task, not only to fatigue. Honest splits remove participant and session identity from the test fold; they cannot remove block position, because block position is part of how the labels are built, in the EEG arms just as much as here. RQ3 inherits that limitation. It does not create it, and no RQ3 result will be worded as if it solves it.

## Alignment risk

The inventory's verdict (its Section 5) is that wrist-to-EEG alignment is phase-level and solid, but not sample-locked: the streams share the published epoch-ms axis, were trimmed to the marker span by the dataset creators, agree with the markers to within about a second at the span edges, and a motion cross-check against the forehead accelerometer bounds any residual offset to a few seconds in every session. There is no shared hardware clock, so no sub-second synchrony claim exists.

What that means for this design, stated in advance:

- **Window features are insensitive to the residual uncertainty.** Heart rate is a 10-second trailing average: a shift of even 2 s changes which of its 1 Hz samples fall inside a 4 s window, not their values in any material way. EDA level, slope and response counts over 4 s are likewise slow quantities. Block membership is not touched at all: the blocks run for 2 to 7.5 minutes and the offsets are seconds.
- **What phase-level alignment forbids is any analysis needing sub-second synchrony** between EEG and periphery: event-locked averaging, latency comparisons, phase coupling. RQ3 contains no such analysis, and if a future amendment proposed one, the alignment evidence on record would not support it. That amendment would have to argue its own alignment case first.

## Code

Phase 2 adds one new file, the peripheral feature extractor, implementing only the rules declared here and joining the frozen EEG features by window identity (participant, session, block, run_id, start_sample). No frozen file changes. The extractor's hash is recorded in the Phase 2 run notes before any result is examined. Anything else that has to change is a logged deviation with its reason, written before results are seen.
