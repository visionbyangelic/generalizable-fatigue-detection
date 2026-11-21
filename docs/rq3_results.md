# RQ3 results: heart rate and EDA on top of EEG, cross-participant

**Status: run 21 Nov 2026 under `docs/rq3_preregistration.md` as amended by Amendment 1 (committed before any modelling).** Estimation-only throughout, as everywhere in this project. Code: `src/features/peripheral.py` (the declared extractor, SHA-256 ced46c05914303145c7e88d755999288de759a01613f2d99d39c90821d6a9f12) and `src/evaluation/rq3_arms.py`. Run folders: `experiments/runs/2026-11-21_rq3_*` (90 folders, one per pool/arm/protocol/cap seed, each with its leakage checklist). Tables: `results/tables/rq3_arm_summary.csv`, `rq3_per_participant_seedmeans.csv`, `rq3_contrasts.csv`, `rq3_indicator_coefficients.csv`, plus the pre-modelling checks in `rq3_feature_sanity.csv`, `rq3_feature_distributions.csv` and `rq3_indicator_correlation.csv`. Post hoc additions after the results review: the HR-only and EDA-only ablation (`src/evaluation/rq3_ablation.py`; 10 further run folders; tables `rq3_ablation_per_participant.csv`, `rq3_ablation_summary.csv`, `rq3_ablation_indicator_coefficients.csv`), the descriptive heart-rate-by-intensity table (`src/evaluation/rq3_hr_by_intensity.py`; `rq3_hr_by_intensity.csv`), and the tier-parity verification (`src/evaluation/rq3_tier_parity.py`; `rq3_within_per_participant.csv`).

## The question, and why a positive answer had to survive three defences

RQ3 asks whether the Empatica E4 signals (mean heart rate from the E4's own 1 Hz product, and three EDA summaries: mean, slope, rise count) carry the protocol-defined M1/M3 block contrast across participants on FatigueSet, on their own or added to the EEG features, under the primary subjective labels, the locked logistic regression, and the same windows and splits as the Week 2 baseline. EEG alone does not transfer on this dataset (LOPO balanced accuracy 0.508). The wrist data bring one specific danger, measured in the Phase 1 inventory before any modelling: the missingness is class-asymmetric. Session-level data loss in two participants (P08/S02 and P10/S01; the cause is not established) removed the wrist streams before the M3 block, so 59 of the 60 windows with unusable peripheral data are fatigued windows. A model given the validity indicators could in principle score above chance by learning "the band had stopped" instead of anything physiological. The amended design therefore built three defences in advance, and this result is reported only through them:

1. **The headline quantities are increments, not levels**: peripheral-only minus indicators-only, and combined minus EEG-only, paired per participant.
2. **The restricted pool is a co-gate**: all arms re-evaluated on the 993 windows where both streams are usable. That pool estimates performance when the band worked; it is a conditional estimand, not a cleaner pool, because the restriction itself removes fatigued windows at about 12% against about 0.2% of alert windows.
3. **Complete-test-window scoring**: models trained on the full pool, then scored only on each held-out participant's complete-data windows, where the indicators are constant and the missingness pattern carries no information at all.

## Headline (leave-one-participant-out, keep-all pool)

The claim this phase is licensed to make, stated before the numbers: **peripheral features recover the protocol-defined M1/M3 contrast across participants, and the design cannot separate fatigue from exercise-related physiology.** In its strongest licensed form, on the leave-one-participant-out tier: wrist HR and EDA features separate the pre-treadmill from the post-treadmill window class in unseen participants (balanced accuracy 0.741, against 0.539 for the missingness indicators alone on the same tier; the paired increment is +0.202 [0.085, 0.314]); EEG features from the same sessions show no such separation (0.508). The tier is named with every number in this document, and the peripheral level is never reported without the indicators level beside it.

Per-participant balanced accuracy, cap-seed means first; every interval in this document is a participant-level bootstrap over the per-participant seed means (10,000 resamples), the project convention:

| Arm | BA | 95% CI | AUC | MCC |
|---|---|---|---|---|
| Indicators-only (shortcut ceiling) | 0.539 | [0.500, 0.595] | 0.539 | 0.085 |
| Peripheral-only | 0.741 | [0.633, 0.851] | 0.836 | 0.491 |
| Combined (EEG + peripheral) | 0.737 | [0.636, 0.838] | 0.802 | 0.471 |
| EEG-only (stored Week 2) | 0.508 | [0.471, 0.542] | 0.540 | 0.019 |

The headline increments:

| Contrast | Difference | 95% CI |
|---|---|---|
| Peripheral-only minus indicators-only | +0.202 | [0.085, 0.314] |
| Combined minus EEG-only (stored Week 2) | +0.230 | [0.128, 0.332] |

The peripheral features carry a cross-participant M1/M3 contrast that the validity indicators alone do not explain, and adding them to the EEG features moves the combined arm well above the EEG-only level. The combined arm's level sits at the peripheral-only level (0.737 against 0.741); the two levels are reported side by side and no reading is placed on their similarity beyond that.

## The shortcut, quantified

The indicators-only arm scores 0.539 overall, and its per-participant values show the exact shape the design review predicted: 0.500 for nine of the eleven participants, above chance only in the two folds where a participant with session-level data loss is held out (P08: 0.682; P10: 0.746). The shortcut is real, worth about +0.04 of pooled balanced accuracy, and concentrated entirely in those two folds. It is reported as a ceiling, not as a share: no percentage of any other arm's score is attributed to missingness, because scores are not additive and the increments above are the honest form of the comparison.

The fitted coefficients on the two indicators (per fold, peripheral and combined arms, keep-all LOPO) stay in a sane range: the largest absolute value is 1.17 (combined arm; fold means about 1.0), and 0.83 in the peripheral arm. Nothing shows the coefficient blow-up that near-separation inside a fold would produce. The two indicators are near-duplicates of each other (pool correlation 0.991: the same stopped bands take out both streams at once), and they correlate only weakly with the EEG mask indicators (Pearson r between -0.21 and +0.20 across the pool), so the two devices' missing-data patterns are largely separate.

## The co-gate: restricted pool (993 complete windows)

| Arm | BA | 95% CI | AUC | MCC |
|---|---|---|---|---|
| Indicators-only | 0.500 | [0.500, 0.500] | 0.500 | 0.000 |
| Peripheral-only | 0.724 | [0.633, 0.815] | 0.848 | 0.467 |
| Combined | 0.725 | [0.647, 0.799] | 0.809 | 0.454 |
| EEG-only (same windows) | 0.530 | [0.497, 0.568] | 0.561 | 0.067 |

On windows where both streams are usable, the indicators are constant, so the indicators-only arm is at exactly 0.500 by construction; it is in the table as that anchor, not as a competitor. The increments on this pool: peripheral-only minus indicators-only **+0.224 [0.133, 0.315]**; combined minus EEG-only **+0.194 [0.117, 0.274]**. The pools do not diverge: the keep-all reading survives the co-gate.

## Complete-test-window scoring

Trained on the full capped pool, scored only on each held-out participant's complete-data windows (P08 and P10 flagged, as declared: their complete test windows are what remains after the session-level data loss):

| Arm | BA | 95% CI | AUC | MCC |
|---|---|---|---|---|
| Indicators-only | 0.500 | [0.500, 0.500] | 0.500 | 0.000 |
| Peripheral-only | 0.725 | [0.598, 0.848] | 0.819 | 0.456 |
| Combined | 0.726 | [0.624, 0.828] | 0.784 | 0.450 |
| EEG-only (same scoring) | 0.524 | [0.491, 0.554] | 0.555 | 0.055 |

The peripheral and combined arms hold their level on test windows where the missingness pattern carries no information. Whatever these arms use, it is present in the measured signals on complete windows, not in the pattern of what is missing.

## Secondary tiers

The LOPO tier governs every claim, as declared. The tiers differ only in fold structure. LOPO trains on ten participants and tests on the held-out participant's sessions. Session-out pools all participants and holds out one (participant, session) at a time, with metrics aggregated per participant. Within-subject rotates the held-out session inside one participant, training on that participant's other sessions. Before the tiers are set side by side, their parity was verified from the run configs and the code path (`src/evaluation/rq3_tier_parity.py`): the target definition (primary subjective labels on the Amendment-1 windows), the capped window pool at each cap seed, the class-balance handling (the per-session M1 cap, plus class weights balanced inside the pipeline), and the normalisation route (training-fold median imputation and standardisation, fitted per fold) are identical across the tiers, and every run config matches except the protocol field. At each cap seed the LOPO folds cover all 806 capped windows exactly once; the within-subject folds cover the 557 capped windows that sit in feasible rotations. P07, P10 and P12 have no feasible rotation (no rotation leaves both classes on both sides) and form no within-subject fold, so the within tier covers eight participants.

| Tier | Peripheral-only BA [CI] | Combined BA [CI] |
|---|---|---|
| Session-out (pooled, per-participant aggregation) | 0.709 [0.599, 0.828] | 0.705 [0.607, 0.807] |
| Within-subject (session rotation, n = 8) | 0.549 [0.448, 0.655] | 0.506 [0.367, 0.637] |

The session-out tier pools across days and participants, and it agrees with the LOPO tier. The within-subject tier is at chance, imprecisely: its interval is too wide to establish absence either. Per participant (seed means; full table in `results/tables/rq3_within_per_participant.csv`):

| Participant | Peripheral | Combined |
|---|---|---|
| P01 | 0.359 | 0.374 |
| P02 | 0.512 | 0.606 |
| P03 | 0.680 | 0.480 |
| P04 | 0.654 | 0.729 |
| P06 | 0.480 | 0.138 |
| P08 | 0.377 | 0.365 |
| P09 | 0.830 | 0.768 |
| P11 | 0.500 | 0.588 |

The spread across the eight participants is wide in both arms (peripheral: SD 0.161, range 0.359 to 0.830; combined: SD 0.210, range 0.138 to 0.768), and the values are reported as values, with no individual-difference claim made from them.

Set against the EEG pattern (at chance on every tier), the peripheral pattern (strong across participants, at chance within) is an inversion. The more parsimonious reading of that inversion: a physiological response that is similar across people but varies in size from day to day within a person, of the kind the treadmill exercise between the blocks produces, predicts exactly this ordering. That is a reading, not a finding. One candidate mechanism behind it is the intensity structure of the protocol: the three sessions are counterbalanced treadmill-intensity days, a within-subject rotation trains on two days and tests on a third, and if the size of the block-to-block shift differs by day, the within tier is the one that suffers. This explanation is a hypothesis the design cannot resolve; it is recorded as a hypothesis, and it is not used to account for the weak tier. The descriptive table in the next section is the agreed non-modelled companion to it.

## Descriptive table: mean heart rate by block and treadmill-intensity day

No model is fitted to this table, and no interval or test is attached to it (`src/evaluation/rq3_hr_by_intensity.py`; `results/tables/rq3_hr_by_intensity.csv`). It records the mean of the heart-rate feature over the primary pool's HR-usable windows (993 of 1,053), grouped by the treadmill intensity of the session the windows come from (the dataset's own metadata mapping, as in `docs/DATA_INVENTORY.md`):

| Treadmill intensity | M1 windows (n) | Mean HR, M1 (bpm) | M3 windows (n) | Mean HR, M3 (bpm) | M3 minus M1 (bpm) |
|---|---|---|---|---|---|
| Low | 119 | 71.6 | 88 | 70.3 | -1.3 |
| Medium | 234 | 72.4 | 200 | 85.7 | +13.3 |
| High | 210 | 77.3 | 142 | 95.4 | +18.2 |

In the table, the M3-minus-M1 difference is -1.3 bpm on low-intensity days, +13.3 bpm on medium-intensity days and +18.2 bpm on high-intensity days.

## Within-participant-normalised sensitivity

The declared V2-style sensitivity (continuous features centred and scaled within participant, using that participant's own unlabelled windows; indicators unchanged; labelled sensitivity, transductive at participant level in the same declared way the Week 2 V2 was at session level):

| Arm | BA [CI] | Movement vs unnormalised |
|---|---|---|
| Peripheral-only | 0.753 [0.623, 0.866] | +0.012 [-0.062, +0.090] |
| Combined | 0.721 [0.624, 0.809] | -0.016 [-0.080, +0.052] |

Normalising away each participant's own level changes very little: the increments survive it (normalised peripheral-only minus indicators-only: +0.214 [0.085, 0.331]). The cross-participant contrast in these features is therefore not carried by stable between-person level differences alone.

## Post hoc ablation: HR-only and EDA-only arms (descriptive)

Post hoc and descriptive: these arms were not in the pre-registration. They were added after the results review to record what each signal family scores on its own under the identical design (keep-all pool, LOPO tier, cap seeds 0-4, the locked logistic regression; `src/evaluation/rq3_ablation.py`; tables `rq3_ablation_per_participant.csv`, `rq3_ablation_summary.csv`, `rq3_ablation_indicator_coefficients.csv`). The HR-only arm uses hr_mean and hr_valid; the EDA-only arm uses eda_mean, eda_slope, eda_rise_count and eda_valid.

| Arm | BA | 95% CI | AUC | MCC |
|---|---|---|---|---|
| HR-only | 0.647 | [0.556, 0.748] | 0.821 | 0.285 |
| EDA-only | 0.640 | [0.558, 0.735] | 0.747 | 0.321 |

Beside them, on the same tier and pool: peripheral-only 0.741, indicators-only 0.539. Per participant (seed means):

| Participant | HR-only | EDA-only |
|---|---|---|
| P01 | 0.500 | 0.500 |
| P02 | 0.872 | 0.531 |
| P03 | 0.575 | 0.500 |
| P04 | 0.779 | 0.500 |
| P06 | 0.680 | 0.557 |
| P07 | 0.500 | 0.979 |
| P08 | 0.501 | 0.688 |
| P09 | 0.964 | 0.675 |
| P10 | 0.746 | 0.791 |
| P11 | 0.500 | 0.793 |
| P12 | 0.500 | 0.529 |

The fitted coefficients on each arm's validity indicator stay in a sane range across folds (largest absolute value 1.45), with no sign of the blow-up that near-separation inside a fold would produce. Both arms are reported as description: no feature set is selected on the basis of these numbers, and no conclusion about which channel matters more is drawn from them beyond the two estimates themselves.

## Per-participant values (LOPO, keep-all pool, seed means)

| Participant | Indicators | Peripheral | Combined | EEG-only (stored) |
|---|---|---|---|---|
| P01 | 0.500 | 0.500 | 0.473 | 0.496 |
| P02 | 0.500 | 0.778 | 0.653 | 0.522 |
| P03 | 0.500 | 0.675 | 0.925 | 0.490 |
| P04 | 0.500 | 0.779 | 0.750 | 0.448 |
| P06 | 0.500 | 0.638 | 0.625 | 0.566 |
| P07 | 0.500 | 1.000 | 1.000 | 0.582 |
| P08 | 0.682 | 0.497 | 0.592 | 0.447 |
| P09 | 0.500 | 0.977 | 0.912 | 0.579 |
| P10 | 0.746 | 0.997 | 0.876 | 0.382 |
| P11 | 0.500 | 0.793 | 0.788 | 0.559 |
| P12 | 0.500 | 0.514 | 0.516 | 0.513 |

(EEG-only per-participant values are the stored Week 2 seed means; the other columns are this phase's runs. P05 forms no fold in any protocol, as in Week 2.) The peripheral effect is broad but not uniform: eight of eleven participants sit above 0.6 in the peripheral arm, P01 and P12 sit at chance, and P08 sits at chance in the keep-all peripheral arm despite carrying half the dataset's missingness. In the complete-test scoring, P08's peripheral value is 0.326 on its complete windows; that value is reported here as part of the flagged-participant record, without a story attached to it. Per-participant values are reported, as everywhere in this project, as values: with 11 participants and intervals of this width, no individual-difference claim is made from them.

## Standing reporting obligations

Validity rates (Phase 1 inventory, reproduced exactly by the Phase 2 extractor before any modelling):

| Scope | HR usable | EDA usable |
|---|---|---|
| Pool (1,053 windows) | 94.3% | 94.4% |
| Alert windows (564) | 99.8% | 100% |
| Fatigued windows (489) | 87.9% | 87.9% |

Both streams usable: 993 of 1,053 windows. Per participant, the complete rate is 80.9% for P08 and 71.2% for P10, and at least 99% for everyone else. The 60 incomplete windows: 27 fatigued from P08/S02, 32 fatigued from P10/S01, 1 alert from P02/S01 (the session whose recording restarted mid-session).

What the features look like on usable windows, by class (means; medians in the sanity table): mean heart rate 74.1 bpm on alert windows against 85.8 on fatigued windows; EDA mean 0.79 µS against 1.74 µS; EDA slope near zero in both classes; rise count 1.18 against 0.94 per window. These are descriptions of the pool the model sees, reported because the amendment requires the class-conditional picture to be on the page; the caveat in the next section applies to them directly.

## Verification

Before any RQ3 number was trusted, the harness re-ran the EEG-only arm on the identical capped sets: the per-participant seed means reproduce the stored Week 2 values to within 0.003 (largest absolute difference over the 11 participants), the size of the cross-environment solver-tolerance difference already documented for this project (docs/environment_pin.md). The headline combined-minus-EEG contrast computed against the stored values (+0.230) and against the rerun (+0.2295) agree to 0.0003. The declared comparator is the stored Week 2 table, as the pre-registration fixes it.

## Which outcome row this matches, and what it means

**Row 2 of the amended outcome table**: peripheral-only above indicators-only in both pools, both increments' intervals excluding zero. The pre-declared reading of that row, applied as written: the peripheral signals carry transferable information beyond the missingness pattern, scoped to this pool, this feature set, this device, these labels and this model. An increment is not a working detector, and the levels say so themselves: the LOPO levels, 0.72 to 0.74 across the two pools, are far from the usefulness line this project uses, and nowhere near the within-session numbers the literature reports.

The caveat that was written into the pre-registration before running travels with this result, unchanged and unsoftened. **M1 and M3 differ in protocol position by construction, and heart rate in particular answers to the treadmill activity the protocol places between the blocks, and to time on task, not only to fatigue.** Honest splits remove participant and session identity from the test fold; they cannot remove block position, because block position is part of how the labels are built, in the EEG arms just as much as here. What has been shown is that the M1-to-M3 peripheral contrast (higher mean heart rate, higher EDA level on the later block) is consistent enough across people for a model trained on other people to use it on a new person. Whether that contrast is fatigue, exercise recovery, warming skin over a long session, or some mixture, this design cannot separate, and no sentence in this document claims that it does. RQ3 inherits that limitation from the label design; it does not solve it, and it is the first thing a reader should hold against the headline.

Two scope statements, for the record. The HR arm is mean heart rate only, taken from the E4's smoothed trailing-average product; nothing is claimed beyond that feature. Skin temperature was excluded by design before modelling; nothing here is evidence about whether it carries information, in either direction.

## Problems encountered

One, minor and caught before use: the first version of the sanity-check script tried to read `both_temporals_masked` from the window index, where it does not exist (it is a derived band-power feature, both temporal channels masked). The script was fixed to derive it from the index's per-channel validity flags, and the correlation table was produced by the fixed version. The extractor itself ran without errors, and its validity counts matched the inventory on the first pass.

One more, in the post hoc ablation script: its first run completed the HR-only arm and then stopped on a variable-shadowing error (the fold summary reused the name of the peripheral frame, so the EDA features could not be found). The script was fixed and re-run in full; the HR-only numbers were identical on the re-run, as expected for a deterministic model on identical capped sets.
