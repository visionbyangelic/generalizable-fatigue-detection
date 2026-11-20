# FatigueSet — Full Research Report

**Date:** 20 Nov 2026 · **Status:** Dataset **DOWNLOADED AND VERIFIED** (see §2 and `docs/DATA_INVENTORY.md`)
**Purpose:** Establish, from primary sources, how FatigueSet is distributed, what it actually contains, how others have used it, and where reality differs from the thesis proposal's assumptions.

---

## 1. Official source

- **Official page:** https://www.esense.io/datasets/fatigueset
- **Paper (open-access PDF):** https://alessandro-montanari.github.io/papers/Kalanadhabhatta-PervasiveHealth2021-Camera.pdf
- **Published version:** Kalanadhabhatta, M., Min, C., Montanari, A., & Kawsar, F. (2022). *FatigueSet: A Multi-modal Dataset for Modeling Mental Fatigue and Fatigability.* In Pervasive Computing Technologies for Healthcare (PervasiveHealth 2021), Springer, pp. 204–217. DOI: https://doi.org/10.1007/978-3-030-99194-4_14
  - Affiliations: UMass Amherst (Kalanadhabhatta) and Nokia Bell Labs, Cambridge UK (Min, Montanari, Kawsar).
  - Conference held 6–8 Dec 2021 (virtual); Springer proceedings volume dated 2022 — both years appear in citations; this is normal, not an error.

## 2. Distribution, licence, size, formats

### How it is distributed — DIRECT DOWNLOAD, no gate
The official page carries a single **"Download"** link pointing to a public Google Drive file:

- `https://drive.google.com/file/d/1i-LPuF6j91T77tAjmZ7TqZfYYPBlTE5_/view?usp=sharing`

Verified 20 Nov 2026 by fetching the page HTML directly. **There is no form, no data-use agreement, no email request, and no sign-in.** The file downloads anonymously (we used `gdown`; a browser click works identically).

> ⚠️ The download link is **invisible to text-only page extraction** (it sits in the page header markup) — a plain read of the page looks like it has no download at all. Anyone reporting "no download available" from a text scrape is wrong; check the HTML.

### What we downloaded (evidence)
| Item | Value |
|---|---|
| File | `fatigueset.zip` |
| Size | 641,686,295 bytes (642 MB / 612 MiB) |
| MD5 | `a358db31c27a8c7e0a1ef255b7c6ba28` |
| SHA-256 | `2db4c9d785307a309f66a404d90a5dbdbe1a45d0a954f980204dd39e2a7fd8ca` |
| Archive integrity | `zipfile.testzip()` → **no bad files** (full CRC check passed) |
| Entries | 1,522 (1,473 files + directories) |
| Uncompressed size | **3.38 GB** (zip uses *store*, i.e. essentially uncompressed) |
| Location | `~/workspace/thesis-fatigue/data/raw/fatigueset.zip` (gitignored, per plan) |

No checksum is published by the authors, so the hashes above are ours — recorded so any future copy can be verified against this one.

### Licence / terms of use — **NO EXPLICIT LICENCE FOUND** ⚠️
- The dataset page and the in-archive `README.md` state **no licence** (no CC-BY, no "research use only" clause, nothing).
- The page footer reads "© Nokia Bell Labs 2021. All Rights Reserved" and links a "Terms of Use" page whose link is dead (`href="#"`).
- The only stated condition is a **citation request**: "If you use our dataset in your work, please cite the original publication."
- **Implication for the thesis:** Use is clearly *intended* for research (public anonymous download + citation request), and academic use with citation is the community norm — every paper in §6 does exactly this. But strictly, no licence grants redistribution rights. **Do not re-upload the raw dataset** (e.g. do not commit it to GitHub or post it to Kaggle as a public dataset). Code, derived features and results are fine. This is worth one honest sentence in the thesis.

### Formats
Everything is plain text: **CSV** for all sensor/experiment files, **XLSX** for the two survey workbooks, **Markdown** for the README. No proprietary/binary formats. This is excellent news for reproducibility.

## 3. Contents inventory (from the paper, the official page, and the downloaded archive's own README)

### Participants and sessions
- **12 participants** (9 male, 3 female), ages 21–40 (mean 30.75, SD 5.78). One participant with mild asthma, one with seasonal asthma, rest no reported conditions. £30 gift-card compensation. (Paper §3.1)
- **3 sessions per participant, on 3 different days** (gap up to 19 days), at roughly the same time of day to control circadian effects → **36 sessions total**.
- Sessions are **physical-activity intensity conditions**, counterbalanced by balanced Latin square: **low** = walking 5 km/h (3.2 METs), **medium** = jogging 7 km/h (5.3 METs), **high** = jogging 9 km/h (8.8 METs). 12 sessions of each.
- `metadata.csv` (root, verified in archive) gives each participant's session→intensity mapping, e.g. P01: low=01, medium=02, high=03.
- **~13 hours total**; mean session duration 21.24 min (SD 3.23). (Paper §4)

### Session protocol (paper §3, Fig. 1; confirmed against `exp_markers.csv`)
| Phase | What happens | Duration |
|---|---|---|
| Pre-task | Stanford Sleepiness Scale (1–7) + Global Vigour & Affect Scale; preliminary session only: demographics, BFI-10, MCTQ chronotype, Fatigue Severity Scale, International Fitness Scale | — |
| S1 | Seated rest, baseline physiology | 3 min |
| **M1** | Fatigue VAS + CRT + 2-back (baseline measurement) | — |
| S2 | Treadmill activity at the session's intensity (stopped early if Borg RPE ≥ 10/14/16 for low/med/high) | 3 min |
| **M2** | Fatigue VAS + CRT + 2-back | — |
| S3 | Dual letter/number **task-switching** paradigm (PsychoPy, 200 trials) to induce mental fatigue | ~15 min (paper text) |
| **M3** | Fatigue VAS + CRT + 2-back | — |

`exp_markers.csv` per session timestamps (UTC epoch ms): `start_experiment, start_baseline, end_baseline, submit_survey, start_crt, start_nback, start_activity, end_activity, submit_survey, start_crt, start_nback, start_fatigue, end_fatigue, submit_survey, start_crt, start_nback, end_session`. (Verified for P01; one marker in P01/S01 — the third `start_nback` — has an **empty timestamp**. Data are not perfectly clean; see §8.)

### Devices and modalities (paper Table 1 = archive README, identical)
**14 sensors across 4 devices:**

| Device | Files (per session) | Signals & rates |
|---|---|---|
| **Muse S headband** (forehead) | `forehead_eeg_raw.csv` | **Raw EEG, µV, 256 Hz** — channels **TP9, AF7, AF8, TP10** |
| | `forehead_eeg_{alpha,beta,delta,gamma,theta}_abs.csv` | Absolute band power (Bels), same 4 channels, 10 Hz |
| | `forehead_acc.csv`, `forehead_gyro.csv` | ACC ±2 g, gyro ±245 °/s, both 52 Hz |
| | `muse_blinks.csv`, `muse_jaw_clenches.csv` | Muse-detected blink / jaw-clench events |
| | `muse_device_fit.csv` | Per-channel fit: 1=Good, 2=Medium, 4=Bad, 10 Hz |
| | `muse_device_battery.csv`, `muse_device_touch.csv` | Battery/voltage/temp; forehead-touch flag |
| **Empatica E4** (wrist) | `wrist_bvp.csv` 64 Hz · `wrist_eda.csv` 4 Hz (µS) · `wrist_skin_temperature.csv` 4 Hz · `wrist_acc.csv` 32 Hz · `wrist_hr.csv` 1 Hz (10-s average) · `wrist_ibi.csv` per beat | PPG/BVP, EDA, temperature, ACC, HR, IBI |
| **Nokia eSense earable prototype** (both ears) | `ear_acc_{left,right}.csv`, `ear_gyro_{left,right}.csv`, `ear_ppg_{left,right}.csv` (green/IR/red) | ACC ±2 g, gyro ±500 °/s, PPG — all **100 Hz** |
| **Zephyr BioHarness 3.0** (chest) | `chest_raw_ecg.csv` 250 Hz (12-bit) · `chest_raw_breathing.csv` 25 Hz · `chest_raw_acc.csv` 100 Hz · `chest_rr_interval.csv`, `chest_bb_interval.csv` per event · `chest_physiology_summary.csv` 1 Hz (HR, breathing rate, posture, HRV + reliability flags) · `chest_sensor_summary.csv`, `zephyr_activity_summary.csv`, `zephyr_device_status.csv` 1 Hz | ECG, respiration, ACC, HR/HRV |

**Per session: 39 core files** (36 sessions × 39 = 1,404) + split-file continuations (see inventory doc). All 36 sessions have every core file — verified from the archive listing; no session is missing its EEG.

### Subjective ratings — what they actually look like ⚠️ (major, see §8)
`exp_fatigue.csv`, one per session, **exactly 3 rows per session** (M1, M2, M3 — verified across all 36 files; **108 rows total in the entire dataset**):

```
measurementNumber, physicalFatigueAnswerTime, mentalFatigueAnswerTime,
fatigueSurveySubmissionTime, physicalFatigueScore, mentalFatigueScore
0, 226.38, 234.14, 238.97, 14.44, 14.17      ← P01/S01, M1
1, 686.57, 690.08, 694.57, 24.03, 11.25      ← M2
2, 1462.91, 1464.46, 1467.55, 20.00, 45.83   ← M3
```

- Scores are **visual analogue scale positions converted to 0–100** (paper §3.4: "Not at all fatigued" → "Maximally fatigued"). Observed mental-fatigue range in the data: **0.0–77.2**.
- Answer times are in **seconds** (session-relative), while sensor files and markers use **epoch milliseconds** — alignment requires combining markers + relative times carefully.
- Pre-task surveys (`pre_task_survey.xlsx`): SSS sleepiness + GVAS vigour/affect per session. Preliminary questionnaire (`preliminary_questionnaire.xlsx`, once per participant): demographics, BFI-10, MCTQ, FSS, IFIS.

### Task performance — what it actually looks like
Trial-level CSVs per session, each with a `measurementBlock` column (0/1/2 = M1/M2/M3):
- `exp_crt.csv` — Deary–Liewald Choice Reaction Time: **108 trials/session** (36 × 3 blocks; verified). Fields: `trialStartTime, stimulusStartTime, stimulusCoords, correctResponse, participantResponse, isCorrectResponse, responseTime, thisBlockTrialNumber, measurementBlock`. Times in seconds.
- `exp_nback.csv` — 2-back: 60 trials/session (20 × 3 blocks). Fields add `trialEndTime, stimulus`; `responseTime` empty on no-response trials (correct rejections).
- `exp_task_switch.csv` — the S3 fatigue-*induction* task (200 trials). **The paper explicitly did not analyse its performance** ("the objective was only to induce mental fatigue"); it is not a fatigability measure.
- Paper's operationalisation of fatigability: **change in CRT / 2-back RT and errors vs the M1 baseline** (§3.4) — exactly the construction the proposal plans.

## 4. How the archive is laid out (verified)

```
fatigueset/
├── README.md                       # full per-file data dictionary
├── metadata.csv                    # participant → session/intensity mapping
├── pre_task_survey.xlsx
├── preliminary_questionnaire.xlsx
└── 01/ … 12/                       # participant IDs (zero-padded)
    └── 01/ 02/ 03/                 # sessions (intensity via metadata.csv)
        └── 39 CSVs (see §3)
```
Some sessions contain extra `*2.csv` continuation files (e.g. `ear_ppg_left2.csv` in 8 sessions; one session has continuations for nearly all files) — evidently a recording restart/split. A loader must concatenate `X.csv` + `X2.csv`. Full per-participant breakdown: `docs/DATA_INVENTORY.md`.

## 5. Muse S EEG practicalities

Specs (paper Table 1 + archive README, corroborated by Muse technical documentation, e.g. https://docs.rs/crate/muse-rs/latest and Muse's technical specifications summary at https://manuals.plus/m/c9470a6feb2dc3896a6bd13b98ed1e05fb616d771c7de2511f03e4cf4a043bc3.pdf):
- **Channels: TP9, AF7, AF8, TP10** (dry electrodes; AF7/AF8 frontal, TP9/TP10 behind the ears), **reference FPz**. No central/parietal/occipital coverage — frontal theta/alpha claims are testable; anything requiring other sites is not.
- **Raw EEG at 256 Hz**, 12-bit, values in µV. Timestamps are epoch **milliseconds**, so consecutive samples share timestamps (256 Hz needs ~3.9 ms resolution) — sample *order*, not timestamp deltas, defines the time base. A loader must not deduplicate on timestamp.
- Muse also ships **pre-computed absolute band powers at 10 Hz** — usable as a sanity check on our own band-power pipeline, not a substitute for it.

Known artefact / quality issues, specific to this dataset and device:
1. **Motion is the dominant confound, by design.** Participants walk/jog on a treadmill for 3 minutes mid-session while wearing the headband. EEG (and every other modality) during S2 will be heavily contaminated; even S3 follows immediately after exercise. Any window labelling must know which phase a window is in (via `exp_markers.csv`).
2. **Jaw/eye artefacts at frontal/temporal sites.** AF7/AF8 sit over frontalis, TP9/TP10 over temporalis — jaw clenches and blinks produce signals 10–100× EEG amplitude. Helpfully, Muse's own `muse_blinks.csv` / `muse_jaw_clenches.csv` event files are included, so artefact-contaminated windows can be flagged/excluded rather than guessed.
3. **Dry-electrode contact loss.** `muse_device_fit.csv` gives per-channel fit (1/2/4) at 10 Hz — a ready-made quality gate. Use it; do not assume all EEG is good EEG.
4. **Bluetooth gaps.** Muse streams over BLE; Muse's own support documentation notes gaps occur on Bluetooth disconnects and on >20 s sensor-quality loss (https://choosemuse.my.site.com/s/article/Understanding-Gaps-in-my-Session-Data). Expect timestamp discontinuities; segment windows within continuous runs only.
5. **Consumer-grade validity is contested.** Krigolson et al. (2017, https://doi.org/10.3389/fnins.2017.00109 — cited in the proposal) validated Muse for ERP work. A 2024 comparison against a research-grade Brain Products system (https://www.mdpi.com/1424-8220/24/24/8108) found Muse S Gen 2 had the *poorest* spectral alignment of the devices tested and notes dry-electrode susceptibility to artefacts and test–retest variability. Honest framing for the thesis: the device is validated enough for band-power/ERP effects, not a medical-grade instrument — which is precisely why the generalisation question matters.

## 6. Mirrors, existing code, and later work using FatigueSet

### Mirrors / alternative hosts
| Source | What it is | Notes |
|---|---|---|
| **Official Google Drive** (via esense.io) | The raw dataset | Used here. No gate. |
| `https://sensix.tech/datasets/fatigueset/fatigueset.zip` | Direct zip URL referenced by rs2lab's preprocessing repo (Nokia Bell Labs' sensing group now presents as Sensix) | Plausibly the same archive behind a direct link; our HEAD request to it timed out, so it is **unverified** — unnecessary anyway, since the official Drive copy is verified. |
| Kaggle: `anaxmenobrito/fatigueset-pre-processed` (https://www.kaggle.com/datasets/anaxmenobrito/fatigueset-pre-processed) | **Pre-processed derivative** by the rs2lab team (5-s statistical windows), with an LSTM meta-learning classifier notebook | Derivative, not the raw data. Useful as a preprocessing cross-check; the thesis pipeline must run from raw. Kaggle download itself requires a Kaggle account. |
| Zenodo | **No FatigueSet record found** (searches return unrelated fatigue-materials datasets) | — |
| Hugging Face | **No copy of this FatigueSet.** `Jlods32/FatigueSense` and `accelerated-komputing/FatigueNet` are *different* datasets (driver-drowsiness pose data; steel-shaft FEA) sharing only names | Easy to confuse — avoid. |

### Existing code (GitHub search, 20 Nov 2026 — 7 repositories)
- **rs2lab/fatigueset-data-preprocessing** — https://github.com/rs2lab/fatigueset-data-preprocessing — preprocessing following Kodikara et al. (below): statistical features over sliding windows (their run: 5 s). The most directly reusable loader reference.
- **Shuchih-Negi/fatigueset_parser_v1** — https://github.com/Shuchih-Negi/fatigueset_parser_v1 — ECG/HRV feature extraction with **task-aligned windowing from `exp_markers.csv`**, labels joined from `exp_fatigue.csv` (physical fatigue score, with SSS fallback). Confirms the marker-based alignment approach and the real file names.
- **ARama-K/MSARNet_FL** — https://github.com/ARama-K/MSARNet_FL — full preprocessing notebook (`Dataset_PreProcess.ipynb`): adaptive resampling, **30-s windows / 15-s step**, mean/std features, **fatigue scores interpolated between M1/M2/M3** to label windows, producing 2,819 windows × 14 features. Ships the derived spreadsheet.
- **Satveer27/mental-fatigue-prediction** — MSci dissertation: mental fatigue from HRV + wrist skin temperature only.
- **MatthewKrinn/Cognitive-Fatigue-Research-2024** — exploratory data analysis on FatigueSet.
- **EmilioGullon/TFG_FatigueSet + tfg-fatigueset-web** — undergraduate thesis + web viewer.

### Later papers — how they built labels and splits ⚠️
1. **Kodikara, Wijekoon & Meegahapola (2025), "FatigueSense", ACM Trans. Computing for Healthcare 6(2), Art. 14** — the main ML follow-up. Two label families: **"psychological"** fatigue (from the subjective VAS ratings) and **"physiological"** fatigue. Device-specific models: AUC 0.63–0.69 (psychological), 0.74–0.80 (physiological); multi-device fusion: 0.69–0.74 / 0.81–0.88. **These modest AUCs — not 95%+ accuracies — are the honest published reference point for this dataset.**
2. **Ramakrishna et al. (2026), "MSARNet", Frontiers in AI** — https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1952954/full — regression of physical + mental fatigue scores from HR/IBI/ACC/EDA/temp (no EEG). 2,819 overlapping 30-s windows (15-s step) from 12 participants; **participants treated as federated clients — a participant-partitioned design**, the closest published analogue to LOSO thinking on this dataset. Best RMSE 0.2293 (normalised scores).
3. **Jeribi & Khubrani (2026), "FATE", Frontiers in Public Health** — https://www.frontiersin.org/journals/public-health/articles/10.3389/fpubh.2026.2011960/abstract — leakage/confound-aware evaluation framework applied to FatigueSet (12 participants, 10 models): models reaching ROC-AUC 0.96–1.00 under **random splits fell to 0.47–0.62 on unseen participants**; features predicted time-of-day and task phase at least as well as fatigue. **This is direct published support for the thesis's central premise** — and a warning that phase/time confounds must be handled explicitly.
4. The original paper itself (§4.1): **no significant correlation (p > 0.05) between subjective fatigue scores and CRT/2-back response times**, and **no significant RT fatigability effect when pooling conditions** ("significant mental fatigability was not observed" without accounting for activity condition).

**No published 1D-CNN-on-raw-EEG LOSO study of FatigueSet was found.** The EEG-specific, participant-independent evaluation the proposal plans appears to be genuinely unoccupied ground.

## 7. What this means for label construction (design consequences)

- **Subjective labels are sparse by nature:** 3 ratings/session. The standard solution in the literature (MSARNet) is to **interpolate scores between M1→M2→M3** and label windows by interpolated value; another (FatigueSense-style) is to label **phase blocks** (e.g. windows near M1 vs near M3). Either is defensible — but the rule, and its weakness (interpolation *assumes* monotone drift), must be documented in `docs/label_rules.md` before training, exactly as planned.
- **Performance labels are block-level:** CRT/2-back give one RT/error summary per measurement block (M1/M2/M3) → fatigability = Δ vs M1, per the paper. Windows can inherit their block's label only within the corresponding phase.
- **Phase is a confound, not a label:** baseline / activity / task-switch phases differ in movement, posture and time-on-task. FATE (§6.3) shows models exploit exactly this. Splits and labels must be phase-aware.

## 8. ⚠️ CONTRADICTIONS & CORRECTIONS vs the proposal's assumptions

The proposal is unusually accurate (12 participants ✓, devices ✓, tasks ✓, ~13 h ✓). The problems are of **emphasis and granularity**, but two are thesis-critical:

1. **CRITICAL — Subjective labels are NOT densely time-aligned.** The proposal (§2.3) hopes to derive labels from ratings and notes alignment limits "will be reported". Reality is stronger: there are only **108 mental-fatigue ratings in total (3 per session)**, each a single VAS snapshot. There is no per-window ground truth. Any window-level label is an *interpolation/block assumption*, and the thesis must present it as such. This does not break the design — the whole literature does this — but the label rule is the single most consequential methodological choice in the project.
2. **CRITICAL — "Session" ≠ fatigued-vs-rested.** The 3 sessions are **low/medium/high treadmill-intensity days**, counterbalanced. The preliminary experiment's "session labels" (99.18% recall) therefore separated *days that also differed in exercise intensity, date (up to 19 days apart), sensor refitting, and circadian details* — an even richer artefact bundle than "session noise" suggests. Good news: this strengthens RQ4's motivation. But RQ4 wording should say what session labels actually encode.
3. **Subjective and performance labels may barely agree — by the authors' own analysis.** The paper found **no significant correlation** between fatigue ratings and task RTs, and no pooled RT fatigability effect. The proposal plans to "examine agreement/disagreement" — expect disagreement; it is a finding, not a failure. It also means performance-based binary labels may be noisy/weak for some participants, which must be reported per-participant.
4. **The fatigue-induction task's own performance is not a label source.** `exp_task_switch.csv` exists, but the paper did not analyse it as fatigability; the proposal correctly names CRT-style measures — keep it to CRT + 2-back.
5. **eSense is a PPG/IMU earable, not an EEG/EDA device.** Occasional secondary summaries blur this. Per Table 1/README: earables contribute ACC + gyro + PPG (green/IR/red) only. The multimodal extension's EDA comes solely from the E4; HR from E4/Zephyr/ear-PPG.
6. **No licence.** See §2 — cite-and-use is the norm, but the dataset is not formally open-licensed; do not redistribute raw data.
7. **Minor, documentation-level:** the official page misspells the survey files as `.xslx` (they are `.xlsx`); the paper's §4.1 text refers to measurement points as M0/M1/M2 while §3.4 and the data use M1/M2/M3 (= `measurementBlock` 0/1/2, `measurementNumber` 0/1/2). The data files are authoritative.
8. **Data are not pristine:** at least one missing marker timestamp (P01/S01), split `*2.csv` continuations in 9+ sessions, duplicated EEG timestamps (ms resolution at 256 Hz), and Muse fit/battery channels that exist precisely because contact was sometimes lost. Week 1's inspection notebook must quantify these per participant rather than discovering them mid-training.

## 9. Sources

- Official dataset page — https://www.esense.io/datasets/fatigueset (read in full, incl. HTML source for the download link, 20 Nov 2026)
- Original paper (open PDF) — https://alessandro-montanari.github.io/papers/Kalanadhabhatta-PervasiveHealth2021-Camera.pdf ; DOI https://doi.org/10.1007/978-3-030-99194-4_14 ; Springer listing https://www.springerprofessional.de/en/fatigueset-a-multi-modal-dataset-for-modeling-mental-fatigue-and/20244516
- The downloaded archive itself — `fatigueset/README.md`, `metadata.csv`, `exp_*.csv` samples (P01 S01/S03 read directly), full archive listing (20 Nov 2026)
- Kodikara et al. 2025, FatigueSense, ACM TCH — abstract via https://ouci.dntb.gov.ua/en/works/7PGGrxx9/ ; citation via https://dl.acm.org/doi/10.1145/3797885
- Ramakrishna et al. 2026, MSARNet — https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1952954/full ; code https://github.com/ARama-K/MSARNet_FL
- Jeribi & Khubrani 2026, FATE — https://www.frontiersin.org/journals/public-health/articles/10.3389/fpubh.2026.2011960/abstract
- rs2lab preprocessing — https://github.com/rs2lab/fatigueset-data-preprocessing ; Kaggle derivative https://www.kaggle.com/datasets/anaxmenobrito/fatigueset-pre-processed
- Parser — https://github.com/Shuchih-Negi/fatigueset_parser_v1
- Muse validation/limits — Krigolson et al. 2017 https://doi.org/10.3389/fnins.2017.00109 ; MDPI Sensors 2024 comparison https://www.mdpi.com/1424-8220/24/24/8108 ; Muse gap documentation https://choosemuse.my.site.com/s/article/Understanding-Gaps-in-my-Session-Data ; Muse protocol specs https://docs.rs/crate/muse-rs/latest
