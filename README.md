# Generalizable Fatigue Detection from Wearable Physiological Signals

**Angelic Charles**

Can a fatigue detector trained on some people's brain signals be trusted on a person it has never seen? This project builds a complete, leakage-controlled machine-learning pipeline to answer that question honestly — and reports what it finds, including when the answer is "not yet".

## Project goal

Consumer EEG headbands are cheap enough to put fatigue monitoring in trucks, factories and clinics — but only if the model works on **new people**, not just on the people it was trained on. Most published fatigue-detection results are evaluated in ways that let information about the test person leak into training, which inflates the scores. The goal of this project is to measure how well fatigue detection from wearable signals **actually generalises**: across sessions, and across participants the model has never seen.

## Research questions

- **RQ1 — The generalisation gap.** How large is the drop in performance between testing on the same participants (within-subject) and testing on completely unseen participants (leave-one-participant-out)?
- **RQ2 — Model comparison.** On identical, honest splits, does a 1D-CNN trained on raw EEG beat a classical band-power + logistic-regression baseline?
- **RQ3 — Multimodal extension.** Do heart-rate and electrodermal signals (recorded alongside the EEG) improve generalisation where EEG alone does not?
- **RQ4 — Label diagnostics.** How much of the apparent performance in this literature is an artefact of how labels are assigned (for example, labelling whole sessions as "fatigued") rather than physiology?

## Hypotheses

- **H1:** Models that score highly under random or within-session splits will fall to near chance under participant-held-out evaluation — the headline numbers in much of this literature are evaluation artefacts, not physiology.
- **H2:** A substantial part of what models learn under loose evaluation is recording artefacts (session identity, sensor-contact quality), not fatigue.
- **H3:** Where any genuine signal exists, it will be small at this scale (12 participants) — so the study is designed to bound effects honestly (estimation with confidence intervals) rather than overclaim.

## Datasets

- **FatigueSet** (primary) — 12 participants × 3 sessions, Muse S EEG (TP9, AF7, AF8, TP10 @ 256 Hz) plus Empatica E4, eSense and Zephyr wearables, with subjective fatigue ratings and cognitive-task performance. Labels are built from the subjective ratings (early-task vs late-task blocks), with the rules locked before any training.
- **DROZY** (comparison, pre-registered) — 14 participants, clinical polysomnography EEG, three vigilance sessions under increasing sleep deprivation, with Karolinska Sleepiness Scale scores and reaction times. The same pipeline runs on both datasets so the comparison tests whether the FatigueSet findings are about the hardware, the labels, or the evaluation. The analysis plan for DROZY was locked (and the pipeline code frozen) before the data was downloaded: `docs/dataset2_preregistration.md`.

Raw datasets are **not** redistributed in this repository (no licence to do so) — download them from their official sources.

## Method in brief

- 4-second non-overlapping windows; band-power features (δ/θ/α/β, log + relative power, frontal ratios) with a logistic-regression baseline; EEGNet-style 1D-CNN on raw EEG.
- Evaluation at three levels: within-subject, leave-one-session-out, and leave-one-participant-out — with normalisation, feature selection and tuning fitted on training folds only, and no overlapping windows across splits.
- Every headline number ships with diagnostics: permutation nulls, a positive control (the pipeline must recover an injected effect of known size), a mask-only control (can recording quality alone predict the labels?), and a leakage-gap measurement (random split vs honest split).
- Decision rules — including the smallest effect considered useful (balanced accuracy 0.55) and what may be claimed from each possible outcome — were written down and locked **before** results were interpreted: `docs/decision_table_lock.md`. The full build log, with the reason for every step, is in `track.md`.

## Key findings so far

- Inside a single session, alert vs fatigued blocks are clearly separable (balanced accuracy ≈ **0.74**) — the signal exists locally.
- Across sessions and across unseen participants, the band-power baseline sits **at chance** (leave-one-participant-out balanced accuracy **0.508**, 95% CI [0.471, 0.542]).
- The pipeline is not blind: it recovers injected effects down to a known minimum size (theta ×1.5 / alpha ×1.25), so the null result is informative, not a broken pipeline.
- The one systematic cross-session pattern found was a **recording-quality shortcut** (sensor-contact patterns), exposed by a mask-only control and a causal ablation — exactly the kind of artefact loose evaluation would have mistaken for fatigue signal.
- The honest summary at this scale: **no detectable transfer, with effects up to ≈0.56 not excludable** at N = 12. Full write-up: `docs/week2_diagnostics.md`.

## Preliminary work

This project follows up my earlier prototype, [Neuro-Fatigue-Prediction](https://github.com/visionbyangelic/Neuro-Fatigue-Prediction) — a 1D-CNN on one participant's Muse EEG that reported **99.18% recall** distinguishing an "alert" session from a "fatigued" session. That result is the motivation for everything here: this project asks whether a score like that survives contact with unseen sessions and unseen people. (So far: it does not — and finding out *why*, rigorously, is the contribution.)

## Repository structure

```
├── PLAN.md                  project plan and week-by-week checklist
├── track.md                 complete build log — every step, its reason, every error and result
├── docs/                    label rules, decision lock, dataset notes, diagnostics write-ups
├── src/                     data loading, preprocessing, features, models, evaluation
├── notebooks/               inspection and training notebooks (incl. the Colab CNN notebook)
├── experiments/             one folder per run: config, metrics, notes
├── results/                 summary tables and figures
└── data/                    local only — never committed
```

## Status

Weeks 1–2 complete (data inspection, locked label rules, preprocessing, baseline + diagnostics). In progress: the pre-registered DROZY comparison and the CNN's leave-one-participant-out runs. Progress is tracked daily in `track.md`.
