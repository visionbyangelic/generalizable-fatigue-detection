# Generalizable Fatigue Detection from Wearable Physiological Signals

**Angelic Charles**

EEG fatigue detection has a generalisation problem. Across the published literature, models that report very high accuracy on the people they were trained on collapse toward chance when they are tested on people they have never seen. This project measures that collapse under leakage-controlled evaluation, on two datasets, with the decision rules written down before the results were interpreted. It asks a simple question: does a fatigue model still work on people it has never seen?

## Project goal

Consumer EEG headbands are cheap enough to put fatigue monitoring in trucks, factories and clinics, but only if the model works on **new people**, not just on the people it was trained on. Many published fatigue-detection results are evaluated in ways that let information about the test person leak into training, which inflates the scores. The goal of this project is to measure how well fatigue detection from wearable signals **actually generalises**: across sessions, and across participants the model has never seen.

## Research questions

- **RQ1, Generalisation.** How does performance change as the evaluation gets more honest: within a session, within a participant across sessions, and on completely unseen participants (leave-one-participant-out)?
- **RQ2, Model comparison.** On identical, honest splits, does an EEGNet 1D-CNN trained on raw EEG beat a classical band-power + logistic-regression baseline?
- **RQ3, Multimodal extension.** Do heart-rate and electrodermal signals (recorded alongside the EEG) carry anything across participants where EEG alone does not?
- **RQ4, Label diagnostics.** How much of the apparent performance in this literature is an artefact of how labels are assigned (for example, labelling whole sessions as "fatigued") rather than physiology?

## Hypotheses

- **H1:** Models that score highly under random or within-session splits will fall to near chance under participant-held-out evaluation, because many headline numbers in this literature are evaluation artefacts, not physiology.
- **H2:** A substantial part of what models learn under loose evaluation is recording artefacts (session identity, sensor-contact quality), not fatigue.
- **H3:** Where any genuine signal exists, it will be small at this scale, so the study is designed to bound effects honestly (estimation with confidence intervals) rather than overclaim.

## Design

- **FatigueSet** (primary): 12 participants, Muse S EEG, three sessions each on separate days (low, medium and high treadmill intensity), with subjective fatigue ratings and cognitive-task performance.
- **DROZY** (second dataset, pre-registered): clinical polysomnography EEG under sleep deprivation, 13 participants analysed. Its decisive feature is that it carries **two label types on identical recordings**: subjective sleepiness (KSS) and reaction-time performance (PVT lapses). The same locked pipeline runs on both datasets, so the comparison tests whether the findings are about the hardware, the labels, or the evaluation.
- **Evaluation:** within-subject, leave-one-session-out, and leave-one-participant-out (LOPO), with normalisation, feature selection and tuning fitted on training folds only, and no overlapping windows across splits.
- **Labels:** subjective ratings are primary, task performance is secondary, and the label rules were locked before any training.
- **Reporting:** estimation-only. The equivalence margin set in advance (delta = 0.05 in balanced accuracy, the smallest effect considered useful) turned out to be unattainable at this sample size, so results are reported as estimates with intervals, not as equivalence verdicts.

## Datasets and data notes

- **FatigueSet**: 12 participants × 3 sessions, Muse S EEG (TP9, AF7, AF8, TP10 @ 256 Hz) plus Empatica E4, eSense and Zephyr wearables, with subjective fatigue ratings and cognitive-task performance. © Nokia Bell Labs 2021; no licence is stated for the data, so it is **not redistributed** here. Obtain it from the official source and cite the FatigueSet paper if you use it.
- **DROZY**: 14 participants recorded, 13 analysed (one participant has data for a single session only and cannot form a within-person contrast), clinical polysomnography EEG (Fz, Cz, Pz, C3, C4 @ 512 Hz), three vigilance sessions under increasing sleep deprivation, with Karolinska Sleepiness Scale scores and PVT reaction times. Research use only, **not redistributed** here; cite Massoz et al. (WACV 2016) and obtain it from the official ULg record. The analysis plan for DROZY was locked, and the pipeline code frozen and hashed, before the data was downloaded: `docs/dataset2_preregistration.md`.

## Method in brief

- 4-second non-overlapping windows; band-power features (delta/theta/alpha/beta, log + relative power, frontal ratios) with a logistic-regression baseline; EEGNet 1D-CNN on raw EEG, compared on identical splits.
- A pre-declared artefact rule (Amendment 1) keeps a window only if both frontal channels pass amplitude and flatline checks; channels that fail are masked per window and carried as validity indicators, never silently zeroed.
- Every headline number ships with diagnostics: permutation nulls, a positive control (the pipeline must recover an injected effect of known size), a mask-only control (can recording quality alone predict the labels?), and a leakage-gap measurement (random split vs honest split).
- Decision rules, including what may be claimed from each possible outcome, were written down and locked **before** results were interpreted: `docs/decision_table_lock.md`. The full build log, with the reason for every step, is in `track.md`.

## Results

### RQ1: generalisation is three tiers, not one gap

- **Same-session estimate: 0.744** balanced accuracy. This number is position-confounded: window position inside the block alone ranks the labels perfectly, and a block permutation null gives 0.508 (p = 0.612). It is an upper bound on separability, not evidence of decoding.
- **Honest within-subject: at chance.** Baseline 0.484, CNN 0.493.
- **Leave-one-participant-out: at chance.** Baseline 0.508, CNN 0.493.

Because both honest tiers sit at the floor, the pre-registered within-minus-LOPO gap is uninformative, and RQ1 is reported as the three tiers above. The pipeline itself is not blind: on FatigueSet it recovers an injected effect down to a known minimum size (theta ×1.5 / alpha ×1.25), so these nulls are informative rather than a broken pipeline.

### RQ2: the CNN does not beat the baseline on unseen participants

- CNN LOPO balanced accuracy **0.493** [0.452, 0.529]; baseline LOPO **0.508** on the primary window pool.
- On the identical canonical window set, the corrected paired contrast is **+0.005** [-0.068, +0.071].

The locked conclusion: **no evidence that either model exceeds chance on unseen participants on FatigueSet, and no evidence of a CNN advantage**, scoped to the frozen EEGNet configuration and early-stopping scheme used here. The CNN null is weak evidence: 16 of 55 fits were near-constant (predicting one class for almost every test window, which scores exactly 0.5), so the result is a null for this configuration and does not bound what a differently trained network could do.

### DROZY: the labels, not the recordings, decide what transfers

This is an exploratory dissociation from a pre-registered pipeline, reported in that framing.

- Subjective (KSS-extreme) labels: LOPO **0.602** [0.563, 0.644], p = 0.005.
- Reaction-time (RT-extreme) labels, on the same recordings: LOPO **0.482** [0.421, 0.549], p = 0.726.
- Paired contrast between the two label types: **+0.120** [0.062, 0.179].

The locked headline: **KSS-extreme labels were decodable across participants; RT-extreme labels from the same recordings were not; this contrast is not isolated from session order and rests on a small disagreeing subset.**

The positive control passed its gate: on label-permuted data, the pipeline recovered an injected effect at **0.798** at full dose and **0.672** at half dose, against pre-set bars of 0.75 and 0.60. The control detected RT-aligned injected shifts at dose 0.5 and above on a 0.25-step dose grid, so smaller RT-related band-power changes are **not** ruled out, and the dose scale, calibrated on label-permuted data, is not mapped onto the observed KSS score. The finding is pipeline-specific and sensitivity-bounded. One pre-registered outcome row is deliberately not claimed: the KSS-minus-FatigueSet gap is +0.094, which falls under the pre-registered 0.10 line for calling the cross-dataset difference clearly interpretable.

### RQ4: session labels, and what my preliminary score was measuring

- Swapping the primary labels for plain block-phase labels (early block = alert, late block = fatigued, no subjective filter) changes almost nothing under honest splits: LOPO **0.516** vs **0.508**. The subjective-rise inclusion filter does not matter; the labels were never the source of the FatigueSet null.
- I rebuilt my preliminary single-participant experiment exactly (Subject 01, its own windowing, chronological split). The headline **reproduces**: logistic regression 1.000 and CNN 0.993 balanced accuracy. But in that setup the fatigued class coincides almost perfectly with a session-wide amplitude regime that fails the pre-declared quality criterion: the artefact mask retains **97.1%** of the alert session's windows and only **2.5%** of the fatigued session's. The score therefore identifies the session and cannot be read as evidence of fatigue decoding. The available data do not separate placement or contact differences from state-correlated physiology or behaviour, and the write-up says so rather than choosing one.

### RQ3: wrist signals recover the protocol contrast, with the confound inside the claim

- Peripheral-only features (mean heart rate, electrodermal level and slope) LOPO: **0.741** [0.633, 0.851]. Indicators-only (which windows have valid wrist data, nothing else): **0.539** [0.500, 0.595]. These two numbers are always reported together.
- Increments: **+0.202** [0.085, 0.314] over the indicators-only arm, and **+0.230** [0.128, 0.332] over EEG-only. The result holds on the complete-data window pool as well.

The locked claim: **peripheral features recover the protocol-defined M1/M3 contrast across participants, and the design cannot separate fatigue from exercise-related physiology.** The early block sits before the treadmill and the late block after it, so this is not fatigue detection and is never described as such. A descriptive check points the same way: the late-minus-early mean heart-rate gap scales with the day's treadmill intensity (minus 1.3 bpm on low days, +13.3 on medium, +18.2 on high).

## Preliminary work

This project follows up my earlier prototype, [Neuro-Fatigue-Prediction](https://github.com/visionbyangelic/Neuro-Fatigue-Prediction/tree/v0-preliminary), a 1D-CNN on one participant's Muse EEG that reported 99.18% recall distinguishing an "alert" session from a "fatigued" session. That prototype is the motivation for RQ4 above: this project rebuilt its setup exactly, reproduced its score, and established what the score was actually measuring.

## Repository structure

```
├── PLAN.md                  project plan and week-by-week checklist
├── track.md                 complete build log, every step, its reason, every error and result
├── docs/                    label rules, decision lock, dataset notes, diagnostics and results write-ups
├── src/                     data loading, preprocessing, features, models, evaluation
├── notebooks/               inspection and training notebooks (incl. the Colab CNN notebook)
├── experiments/             one folder per run: config, metrics, notes
├── results/                 summary tables and figures
└── data/                    local only, never committed
```

## How to run

`notebooks/01_data_inspection.ipynb` walks through the data, and `notebooks/04_cnn_colab.ipynb` runs the CNN training on Colab. The pipeline itself lives in `src/`; every number reported above traces to a run folder in `experiments/` and a summary table in `results/tables/`, with the full reasoning in `track.md`.

## Status

Complete. All four research questions are answered, both datasets are analysed, and the positive controls, diagnostics and sensitivity checks are done and written up in `docs/`. What remains is the writing.
