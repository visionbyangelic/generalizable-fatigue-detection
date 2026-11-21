# Methods facts

*Support material for my own writing: the frozen pipeline facts I need for the methodology chapter, each with the document it comes from so I can verify before I cite it. Values are as frozen before the runs they describe; nothing here was tuned on results. This is not a draft of the paper.*

## Data

- FatigueSet: 12 participants, 3 sessions each (36 sessions), Muse S EEG (AF7, AF8, TP9, TP10) plus Empatica E4 peripherals. Sessions are treadmill days of low, medium and high intensity. Source: docs/DATA_INVENTORY.md.
- DROZY: 14 participants recorded, 13 analysed (Participant 12 excluded under Amendment 1), 36 EDF recordings at 512 Hz, clinical PSG montage; labels from Karolinska Sleepiness Scale ratings and psychomotor vigilance task reaction times. Source: docs/DROZY_INVENTORY.md, docs/drozy_results.md.

## Preprocessing (frozen, src/preprocessing.py)

- Band-pass filter 1 to 40 Hz, zero-phase, applied per continuous run; no notch filter in the primary pipeline.
- Downsample 256 Hz to 128 Hz.
- Windows: 4 seconds, non-overlapping, cut within continuous runs only; a run is split wherever the sampling gap exceeds 100 ms, and windows are anchored per block from the block marker so a window never spans blocks.
- No ICA or other spatial cleaning.
- DROZY adapter: the same pipeline, with 512 Hz downsampled to 128 Hz and Fz as the designated frontal channel. Source: docs/methods_notes.md (implementation review), docs/drozy_results.md.

## Artefact rule (Amendment 1, docs/week2_diagnostics.md §12)

- A window enters the pool only if BOTH frontal channels pass all checks for that window: peak-to-peak amplitude within 150 microvolts, no flatline (SD below 0.5 microvolts fails), device fit checks pass.
- A failing temporal channel does not reject the window; its features are masked for that window and a per-channel validity indicator is carried as a feature instead.
- Retention: 1,121 of 1,568 candidate windows (71.5%); primary pool 1,053 windows (564 alert, 489 fatigued); 43 features per window including 3 indicators.

## Baseline model (frozen, docs/label_rules.md §6, src/features.py)

- Features: per-channel log band power and relative band power in delta, theta, alpha and beta from a 4-second Hann-windowed spectrum (0.25 Hz resolution), plus frontal theta/alpha and (theta+alpha)/beta ratios and the validity indicators: 43 features in total.
- Model: logistic regression, L2 penalty, C = 1.0, balanced class weights.
- Standardisation and median imputation are fitted on the training fold only, never on the full data.

## Labels (docs/label_rules.md)

- Primary labels are subjective: windows in block M1 are alert, windows in block M3 are fatigued, and a session enters only if the subjective mental-fatigue rating rose from M1 to M3 (32 of 36 sessions pass).
- Balance rule: within each session, the M1 windows kept are capped at the retained M3 count by a seeded random subsample, so class balance is a property of the design.
- Performance labels (reaction-time based) are secondary on FatigueSet, where the group fatigability slope was about zero; they are a primary arm on DROZY (KSS extremes and RT extremes, symmetric dose tie-breaks, Participant 12 excluded).
- Session labels (whole day = one class) are a diagnostic for RQ4 only, never a ground truth.

## Splits and seeds

- Three evaluation tiers: within-subject (session rotation inside a participant), leave-one-session-out pooled across participants, and leave-one-participant-out (LOPO).
- Baseline: 5 seeds (seeds 0 to 4) varying the M1 cap subsample. CNN: 5 model seeds on the canonical cap-seed-0 window set (806 windows), so the RQ2 contrast is paired on identical windows.
- No window, and no participant in LOPO, ever appears on both sides of a split; normalisation, imputation and tuning use training folds only.

## CNN (frozen configuration, docs/cnn_loso_results.md)

- EEGNet-8,2: temporal kernel length 64 at 128 Hz (half a second), depth multiplier 2, separable convolutions; input is 6 planes (4 EEG channels plus 2 binary validity-mask planes).
- Training: Adam, learning rate 1e-3, batch 64, at most 100 epochs, early stopping with patience 15 on validation loss, channel dropout p = 0.2, class weights from the training fold.
- Validation for early stopping is carved from the training participants only (a run-segment carve with a 2-window purge at the boundary); a whole-participant carve was run afterwards as a disclosed sensitivity.
- The DROZY cross-dataset runs used the baseline pipeline only; no DROZY CNN run exists.

## Metrics and inference

- Primary metric: balanced accuracy. Also reported: macro-F1, ROC-AUC, MCC.
- Intervals: participant-level bootstrap, 10,000 resamples, percentile 95% intervals.
- Nulls: block permutation null (Week 2), within-participant session-label swap null (DROZY and the RQ4 day-level arm), exact enumeration where the count allows (2^11 sign patterns).
- Usefulness gate: SESOI delta = 0.05, fixed before the LOPO runs; the feasibility gate (smallest detectable margin about 0.093 to 0.100) ruled equivalence unattainable, so all phases are reported estimation-only. Source: docs/decision_table_lock.md.

## RQ3 additions (docs/rq3_preregistration.md, docs/RQ3_INVENTORY.md)

- Peripheral features per window: mean heart rate, mean EDA, EDA slope, EDA rise count (a rise is a sample-to-sample increase of at least 0.01 microsiemens), plus heart-rate and EDA validity indicators.
- Validity: a window's heart rate is valid with at least 2 in-span positive samples of the E4 heart-rate product (a 10-second trailing average sampled at 1 Hz); EDA is valid with at least 8 in-span samples and a maximum above 0.002 microsiemens.
- Inter-beat intervals were audited and rejected as too sparse for HRV features (about 31% of windows hold 3 or more usable beats); the heart-rate arm is mean HR only.
- Alignment between the EEG and peripheral clocks is phase-level, verified to about plus or minus 2 seconds.
- Models: logistic regression only (the frozen baseline specification); the headline quantities are the paired per-participant increments of the peripheral arms over the indicators-only and EEG-only arms, with the restricted complete-window pool (993 windows) as co-gate.

## Environment (docs/environment_pin.md)

- Python 3.12.3; numpy 2.5.5; scipy 1.18.1; scikit-learn 1.9.1; pandas 3.1.1; MNE 1.13.2; PyTorch 2.15.0+cpu; SciPy-linked OpenBLAS 0.3.34.
- Week 2 headline numbers were computed before the stack was recorded; the stored result files are canonical, and pinned-stack re-runs reproduce them within solver tolerance.
- Code, frozen label rules and every run's outputs are in the project repository; the daily record is track.md.
