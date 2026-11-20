# Thesis Project Plan — Generalizable Fatigue Detection from Wearable Physiological Signals

**Author:** Angelic Charles — Final-Year Computer Science
**Question:** Does a fatigue model still work on people it has never seen?
**Proposal source:** `~/workspace/user/files/Angelic_Charles_Revised_Thesis_Proposal.pdf` (read 20 Nov 2026)

## 1. Core Idea

Many fatigue papers report 95%+ accuracy — but train and test on the *same* people. The model may be learning a person's EEG fingerprint or a recording session's noise, not fatigue.

This thesis tests that directly:
- Preliminary result to interrogate: single-participant 1D-CNN on Muse EEG with **session labels** = 99.18% fatigued-class recall. Was that fatigue, or session artefacts?
- Answer it with proper labels + participant-independent evaluation.

## 2. Research Questions

- **RQ1 — Generalisation gap:** Within-subject performance vs Leave-One-Subject-Out (LOSO). Gap = within − LOSO.
- **RQ2 — Model comparison:** 1D-CNN (raw EEG windows) vs classical baseline (band-power features + classical classifier) on *identical* splits.
- **RQ3 — Multimodal extension (Weeks 5–6, if time):** Does adding HR / EDA (Empatica E4, Zephyr, eSense) improve LOSO over EEG alone?
- **RQ4 — Label diagnostic:** Session labels vs subjective-rating labels vs performance-based labels. Session labels are diagnostic ONLY, never primary ground truth.

## 3. Dataset — FatigueSet

- Source: https://esense.io/datasets/fatigueset — Kalanadhabhatta et al. 2022
- 12 participants, multimodal
- **Core:** Muse S EEG (4 channels)
- **Extension:** Empatica E4 (EDA, PPG/HR), Nokia eSense, Zephyr BioHarness (HR, respiration, ACC)
- Plus: subjective fatigue ratings + cognitive task performance (RT, errors)
- No new data collection. No hardware. Computational pipeline only.

> **OUTCOME — 20 Nov 2026: DOWNLOADED (official source, no form/gate).** Full research: `docs/fatigueset-research.md`; verified inventory: `docs/DATA_INVENTORY.md`.
> - Downloaded `fatigueset.zip` (641,686,295 bytes; SHA-256 `2db4c9d785307a309f66a404d90a5dbdbe1a45d0a954f980204dd39e2a7fd8ca`) from the official page's public Google Drive link into `data/raw/`. CRC check passed on all 1,473 files; 3.38 GB uncompressed; all 12 participants × 3 sessions × 39 core files present.
> - **Key corrections to assumptions** (details in research doc §8): (1) subjective ratings are only **3 per session (108 total)**, VAS 0–100 — window labels require a documented interpolation/block rule; (2) the 3 sessions are **low/medium/high treadmill-intensity days**, not fatigued-vs-rested — RQ4 wording must reflect what session labels encode; (3) the original paper found **no significant correlation** between subjective ratings and task RTs — expect label disagreement; (4) **no explicit licence** — cite-and-use, do NOT redistribute raw data; (5) loaders must handle `*2.csv` split continuations (65 files), epoch-ms vs relative-seconds clocks, and at least one missing marker timestamp.

## 4. Labels — Locked Before Training

Two primary label sets, built and evaluated **separately**. Rules documented in `docs/label_rules.md` BEFORE any training, and never tuned on test results.

1. **Subjective:** from each participant's fatigue ratings — rule based on scale/distribution in the dataset.
2. **Performance:** from change in RT / errors vs a baseline period.
3. **Session (RQ4 only):** recording session as label — to reproduce/diagnose the 99.18% result.

Also: examine agreement/disagreement between (1) and (2); report alignment limits where ratings/performance are not time-aligned to windows.

## 5. Models

- **1D-CNN:** learns temporal patterns from segmented raw EEG windows. Architecture + training settings documented; selection on training data only.
- **Classical baseline:** band-power features per window (delta/theta/alpha/beta + ratios, incl. frontal theta) → classical classifier (e.g. Logistic Regression / Random Forest / SVM — pick via training-only validation, then freeze across protocols).
- **Multimodal (extension):** simple feature-level fusion — EEG features + HR/EDA summary features. No complex fusion architecture.

## 6. Evaluation Protocols

All three run for both models, both primary label sets:

1. **Within-subject:** train/test same participant, respecting recording boundaries.
2. **Leave-one-session-out:** hold out one session, train on other eligible sessions.
3. **LOSO:** hold out 1 participant entirely, train on other 11, repeat ×12.

**Metrics:** recall, precision, F1, balanced accuracy. Reported per-participant (all 12) + aggregate. Mean ± variation across **5+ seeds** reported separately from variation across participants.

## 7. Leakage Discipline — The Heart of the Thesis

Non-negotiable, applies to every fold:
- Normalisation parameters fit on **training fold only**
- Feature selection / hyperparameter tuning on training fold only (nested validation inside train)
- No overlapping windows across train/test splits — windows grouped by recording
- In LOSO: zero data from held-out participant used for fitting, tuning, or normalisation
- No participant-specific baseline adjustment unless estimable without held-out labels/future info — otherwise excluded from LOSO
- Label rules fixed before training; test results never used to choose preprocessing/features/configs

Every experiment log must state how these were satisfied (see `experiments/README.md`).

## 8. Platform Architecture — Colab (my decision, 20 Nov 2026)

I directed: **use Google Colab**, with the Google account I already has access to (visonbyangelic@gmail.com, already signed in, free tier — no premium, which is fine for this project's model sizes). This supersedes the earlier Kaggle-first simplification.

| Layer | Role | Access needed |
|---|---|---|
| **GitHub** (private repo `fatigue-detection-thesis`) | Home — all code, notebooks, docs, results logs, thesis notes. The reproducible deliverable. | ✅ DONE 20 Nov 2026 — repo created, SSH-key push working |
| **Google Colab** | **The GPU training engine.** Notebooks live in GitHub and open/run in Colab from the browser, using the already-signed-in visonbyangelic@gmail.com account. Dataset reaches Colab from its official Drive link / my Drive — never a public re-upload (no licence to redistribute). | Already signed in — nothing needed from me |
| **Local VM** (small CPU) | Development, data inspection, preprocessing, classical baseline. | Already available |
| ~~Kaggle~~ | Not used. Named fallback only if Colab free-tier limits ever block a run. No token needed. | — |

Flow: develop locally → push to GitHub → run heavy CNN in Colab (notebook pulled from GitHub, data pulled from the official source) → results (CSV + logs + plots) committed back to GitHub.

**Status (20 Nov 2026):** dataset downloaded + verified locally; repo live with scaffold pushed; Colab verification in progress.

## 9. Repo Structure

```
fatigue-detection-thesis/
├── README.md                  # How to reproduce — entry point for examiners
├── PLAN.md                    # This plan (also kept in local workspace)
├── .gitignore                 # data/, checkpoints, secrets excluded
├── requirements.txt
├── configs/                   # One YAML per experiment config (frozen per run)
├── data/
│   ├── raw/                   # FatigueSet as downloaded — NEVER in git
│   ├── interim/               # Cleaned / aligned intermediates — not in git
│   └── processed/             # Windowed arrays + label files — not in git (or git-LFS if small)
├── src/
│   ├── data/                  # download/inspect/load FatigueSet
│   ├── preprocessing/         # filtering, windowing, normalisation (train-only fit)
│   ├── labels/                # subjective / performance / session label builders
│   ├── features/              # band-power extraction
│   ├── models/                # cnn.py, baseline.py
│   ├── evaluation/            # splits (within / session-out / LOSO), metrics
│   └── utils/                 # seeds, logging, plotting
├── notebooks/
│   ├── 01_data_inspection.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline.ipynb
│   ├── 04_cnn.ipynb
│   └── 05_loso_analysis.ipynb
├── experiments/
│   ├── README.md              # Results-logging convention
│   └── runs/                  # One folder per run: config + metrics + notes
├── results/
│   ├── tables/                # Per-participant + aggregate CSVs (in git)
│   └── figures/               # Plots for thesis (in git)
└── docs/
    ├── label_rules.md         # LOCKED before training — versioned
    ├── dataset_notes.md       # FatigueSet inspection findings (Week 1)
    └── methods_notes.md       # Decisions + rationale as we go
```

## 10. Week-by-Week Checklist

### Week 1 — Data + preprocessing + label rules
- [x] Obtain FatigueSet, verify files/checksums, inventory per participant/session/modality — **done 20 Nov 2026** (official Drive download, CRC-verified; see `docs/DATA_INVENTORY.md`)
- [ ] Inspect Muse EEG: sampling rate, channels, quality, missing segments
- [ ] Inspect ratings + task performance: scales, timing, alignment to EEG
- [ ] Implement preprocessing: filtering, artefact handling, fixed-length windows (no cross-split overlap)
- [ ] Write + lock `docs/label_rules.md` (subjective + performance rules, baseline period, thresholds)
- [ ] Examine label agreement/disagreement

### Week 2 — Baseline + CNN + within / session-out
- [ ] Band-power feature extraction
- [ ] Classical baseline with training-only validation
- [ ] 1D-CNN implementation + training loop (seeded)
- [ ] Within-subject evaluation (both models, both label sets)
- [ ] Leave-one-session-out evaluation

### Weeks 3–4 — LOSO + seeds + RQ4 (core results)
- [ ] LOSO × 12 participants, both models, both label sets
- [ ] Repeat with 5+ seeds; separate seed variance from participant variance
- [ ] RQ4: session-label run under identical protocols; compare
- [ ] Generalisation gap analysis; per-participant tables + figures
- [ ] Start thesis draft (Methods + Results skeleton)

### Weeks 5–6 — Multimodal extension (optional, RQ3)
- [ ] Align E4/Zephyr/eSense signals to EEG windows + labels (only where reliable)
- [ ] Compare: EEG only / HR only / EDA only / EEG+HR / EEG+HR+EDA under LOSO
- [ ] Decide keep/cut based on LOSO gain

### Weeks 7–8 — Writing / slides / repo cleanup
- [ ] Full analysis + discussion + limitations
- [ ] Thesis report finalised
- [ ] Presentation slides
- [ ] Repo: README reproduction steps verified from a clean clone, results tables/figures final

**Priority rule:** If time squeezes, cut Weeks 5–6 first. Core (RQ1, RQ2, RQ4) must be complete and clean.

## 11. Access Checklist — COMPLETE (20 Nov 2026)

- [x] **GitHub:** repo `fatigue-detection-thesis` created under Visionbyangelic (private); login saved via Secure Vault; pushes via SSH key "Muse — thesis VM"
- [x] **Google Colab:** uses the already-signed-in visonbyangelic@gmail.com account (free tier) — nothing needed from me
- Kaggle token: **no longer needed** (my Colab decision, 20 Nov 2026)

**First milestone once access lands:** obtain FatigueSet + complete Week 1 data inspection (`notebooks/01_data_inspection.ipynb` + `docs/dataset_notes.md`).

## 12. Deliverables (from proposal §2.9, authorship clarified 20 Nov 2026)

1. Documented, reproducible preprocessing + label-construction pipeline
2. Implemented 1D-CNN + classical baseline
3. Evaluation scripts + participant-level results
4. Code repository with implementation + experiment documentation
5. **track.md — the complete build log I write my paper from** (every step, parameter, error, result)
6. Thesis report — **written by me** in the school format (see docs/school_format.md: 40+ pages, TNR 12, double-spaced, APA 6th; AI content ≤30% per institutional rules). Any drafts produced along the way are only rewrite material.
7. Presentation slides for the physical defense (Oct–Nov window)
