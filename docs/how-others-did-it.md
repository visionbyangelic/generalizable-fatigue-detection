# How Others Did It — A Practical Recipe Book for This Pipeline

**Compiled:** 20 Nov 2026 · **Companion to:** `fatigueset-research.md`, `literature.md`, `methods_notes.md`, `PLAN.md`
**Purpose:** The concrete, proven choices other researchers made when building pipelines like ours — preprocessing, windows, architectures, features, splits — so our implementation follows evidence, not guesses. Every claim below carries its source; items marked **[abstract-only]** were verified from abstracts, and **[NOT VERIFIED]** means no open source could be found and the claim must not be cited until the full text is obtained.

---

## 1. FatigueSet pipelines specifically

| Source | Preprocessing | Window / overlap | Features / model input | Label construction | Splits | Headline results |
|---|---|---|---|---|---|---|
| **Original paper** — Kalanadhabhatta et al. 2022 (open PDF read in full: alessandro-montanari.github.io/papers/Kalanadhabhatta-PervasiveHealth2021-Camera.pdf) | **None** — dataset + behavioural analysis only; no filtering, windowing, or EEG processing of any kind | N/A | N/A | VAS physical + mental fatigue 0–100 at M1/M2/M3 per session; fatigability = CRT (36 trials/block) and 2-back (20 trials/block) performance differenced vs M1 baseline | N/A | Ratings rose with induction; **no significant RT difference across M1/M2/M3** (CRT or n-back, pooled); **no significant correlation (p > 0.05) between fatigue scores and RT** on either task |
| **FatigueSense** — Kodikara et al. 2025, ACM Trans. Computing for Healthcare 6(2) Art. 14 (DOI **10.1145/3709363** — see note below) | **[NOT VERIFIED]** — full text paywalled, no author PDF found | **[NOT VERIFIED]** | All 4 devices, multimodal; per-modality detail **[NOT VERIFIED]** | Two binary tasks: "psychological" vs "physiological" fatigue (exact construction **[NOT VERIFIED]** in the paper itself) | Device-specific vs multi-device training; subject-wise? **[NOT VERIFIED]** | AUC psychological **0.63–0.69**, physiological **0.74–0.80** (single-device); **0.69–0.74 / 0.81–0.88** (multi-device) — abstract-verified |
| **rs2lab preprocessing code** — github.com/rs2lab/fatigueset-data-preprocessing (code read in full; states it follows FatigueSense with modifications) | Timestamp-align all sensors (misaligned bins → NaN); **drop first 30 s** of each session. No filtering, artefact rejection, or normalisation | **Tumbling (non-overlapping)** windows; example 5 s; Kaggle release ships 1/2/5/10/15/30/60 s versions | Per channel per window: **min, max, std, mean, median** (5 stats). 35 channels incl. Muse raw EEG + Muse's precomputed absolute band powers (5 bands × TP9/AF7/AF8/TP10) → ~175 features | `psycho`: window = 1 if the interval's mentalFatigueScore > **that participant's median** (median from sessions 01+03 only), forward-propagated between answer times. `physio`: **session 01 = 0, session 03 = 1**; session 02 dropped entirely | None (preprocessing only) | None verified (linked Kaggle LSTM meta-learning notebook's results not readable) |
| **Shuchih-Negi parser** — github.com/Shuchih-Negi/fatigueset_parser_v1 (README read in full) | ECG artefact flag only (empty/corrupt slice → `unknown`) | **30 s, non-overlapping, task-aligned from `exp_markers.csv`** block geometry; LF/HF on enclosing 120 s windows | ECG/HRV only: HR, RMSSD, SDNN per 30 s; LF/HF via Lomb-Scargle per 120 s. **No EEG** | **Block inheritance:** every window in a measurement block inherits that block's `physicalFatigueScore` (0–100); SSS fallback if a block lacks a rating (never actually needed — 677/677 windows labelled) | None (labelled table + EDA) | 677 windows total; label range 0.00–77.22; 23% of LF/HF windows low-confidence |
| **MSARNet** — Ramakrishna et al. 2026, Frontiers in AI; code github.com/ARama-K/MSARNet_FL (repo is the verified record; journal full text not retrieved) | **Wrist (E4) only — no EEG.** Adaptive resampling: all channels to the session's coarsest grid via resample-mean + interpolation; cross-boundary windows discarded | **30 s window, 15 s step (50% overlap)** → 2,819 windows | **Mean + std of 7 wrist channels = 14 features** (HR, IBI duration, ACC xyz, EDA, temp) | **Regression:** physical + mental scores linearly **interpolated between M1/M2/M3** so every window has a continuous target in [0,1] | **Chronological 80/20 within each participant** (NOT LOSO); StandardScaler fit on that participant's train portion only; federated = 12 clients | Best: MSARNet + FedAdam **RMSE 0.2293, MAE 0.2135**. **All centralised R² negative** (−0.15 to −3.6) — the authors' own evidence that pooled cross-subject signal is weak |
| **FATE** — Jeribi & Khubrani 2026, Frontiers in Public Health (from `literature.md` §E2) | Framework paper — re-implemented pipelines, confound probes (time-of-day, task phase) | — | Ten off-the-shelf models | Fatigue labels as in source pipelines | **Partition ladder: random splits → held-out participants** | Random-split AUC **0.96–1.00 → 0.47–0.62 on unseen participants** (LII up to 0.49) |

**Three incompatible published answers to the "3 ratings per session" problem** — this is the key methods-justification material for `label_rules.md`:
1. **Median-split + stage** (rs2lab, claiming FatigueSense lineage): binary label from own-median threshold; physiological label = early vs late session.
2. **Block inheritance** (Shuchih-Negi): windows inherit their measurement block's rating — no interpolation assumptions, but coarse (3 distinct values/session) and block = phase-confounded.
3. **Linear interpolation** (MSARNet): smooth continuous targets — but interpolation uses *future* ratings to label earlier windows (look-ahead; see §6).

> **DOI correction (verified 21 Nov 2026):** the FatigueSense DOI is **10.1145/3709363**, confirmed against the ACM Digital Library record for the paper; the 10.1145/3797885 recorded earlier belongs to a different article in the same journal. Full verification note in `docs/literature.md`, entry F5.
>
> **The gap, stated plainly:** no published FatigueSet pipeline found uses LOSO or any subject-wise split, and none models the EEG specifically. Our design is not a replication of any of these — it is the evaluation they all skipped.

---

## 2. Muse S EEG preprocessing norms

### Filtering — what published Muse studies actually used (full-text verified unless noted)

| Bandpass | Notch | Context | Source |
|---|---|---|---|
| 0.1–30 Hz, dual-pass Butterworth | 60 Hz | Krigolson Muse ERP validation (N200/P300), 256 Hz | frontiersin.org/articles/10.3389/fnins.2017.00109 |
| 0.1–15 Hz | 60 Hz | Krigolson Lab step-by-step pipeline (ERP only) | krigolsonlab.com/muse-analysis.html (lab protocol) |
| 1–40 Hz, 4th-order Butterworth | 50 Hz (pre-recording) | Muse S meditation / moral-learning studies, *Scientific Reports* | nature.com/articles/s41598-026-53600-6 |
| 0.5–40 Hz, zero-phase FIR | — | Muse sleep staging; Muse S Gen 2 vs research-grade comparison (deliberately minimal) | mdpi.com/1424-8220/24/24/8108 |
| 0.5–30 Hz, 4th-order Butterworth | 50 Hz IIR | Muse 2 stress classification | orclever.com EJ R&D 2026 |
| 1–50 Hz | 60 Hz | Muse S real-time game / motor imagery | HCII 2025 (faculty.cs.gwu.edu) |
| 1–40 Hz + downsample 256→128 Hz + per-window zero-meaning | — | Banville et al. Muse S sleep / age work | MPG Pure supplement (Banville 2024) |

**Pattern:** for spectral/band-power and ML work, **0.5–40 Hz or 1–40 Hz** are the defensible literature-backed choices; 0.1 Hz high-pass belongs to ERP work; 1–50 Hz retains the most EMG and needs explicit justification. **Notch follows mains frequency: 50 Hz for Nigeria/Europe-recorded data, 60 Hz for North America.** (FatigueSet was recorded in Cambridge, UK → 50 Hz mains, though as §6 notes, a 40 Hz low-pass makes a notch largely moot.)

### Resampling
Keeping 256 Hz is the most common choice. Downsampling to 128 Hz is the most common *reduction* (Banville lineage; also matches EEGNet's original design rate). It is optional and must be justified by model input size, not presented as required.

### Artefact handling — what is and isn't defensible at 4 channels
- **Amplitude/gradient rejection is the published norm.** ±100 µV segment rejection → ~10–12% of segments lost (Muse S, *Scientific Reports* pipeline). Gradient >10 µV/ms or range >100 µV (Krigolson). A stricter >60 µV rule discarded **35% of segments** (Krigolson et al. 2021) — thresholds cost real data on Muse; report the rejection rate.
- **Recording-level QC (Banville):** reject a recording if >5% of samples are missing on any channel (Bluetooth loss), or ≥25% of 1-s windows (2–26 Hz) exceed variance 100 µV² on any channel.
- **Muse's own blink/jaw event files and fit (HSI) values exist and are documented, but no peer-reviewed Muse study was found that excluded epochs solely because a blink/jaw marker fired.** Using them as an exclusion rule is a novel, documented choice — defensible, but flag marker false-positives (e.g. swallowing/device handling misclassified as jaw clenches in field reports). ⚠️ **HSI coding differs across sources (1=Good/2=Medium/4=Bad vs 0/1/2)** — the FatigueSet archive README's own coding (1/2/4, per `fatigueset-research.md` §3) is authoritative for our data.
- **ICA: do not use as primary cleaning.** Explicitly not applied in the Muse S *Scientific Reports* pipeline "due to the limited number of EEG channels"; methodological consensus is ICA cannot separate brain + blink + jaw + motion + line-noise sources from 4 components (needs ≳20 channels to isolate ocular sources). One Muse S paper did apply ICA anyway — the literature is inconsistent in practice, consistent in guidance. Same for **ASR**: standard guidance is ≥20 channels; 4-channel use exists only with semi-simulated parameter tuning or as an unvalidated existence proof.
- **Referencing:** hardware reference is FPz. The Krigolson offline norm is re-referencing to **average mastoid = (TP9+TP10)/2**. Common-average reference over these 4 frontal/temporal sites is *not* the norm (dominated by frontal EMG/EOG).

### Pitfalls specific to our four sites
- **EMG geography:** frontalis EMG peaks 20–30 Hz at frontal sites; temporalis 40–80 Hz (plus ~20 Hz peak) at temporal sites (Goncharova et al. 2003). AF7/AF8 sit over frontalis, TP9/TP10 over temporalis → **beta is EMG-vulnerable and gamma is EMG-dominated** at every Muse channel. The Muse S *Scientific Reports* paper itself treats gamma results as exploratory for exactly this reason.
- **Dry-electrode low-frequency excess** and the poorest spectral alignment vs research-grade systems in head-to-head testing (Sensors 2024, Muse S Gen 2) — a stated limitation, and part of why honest LOSO numbers matter.
- **Bluetooth gaps:** typically 4–8 s losses in field data (UNIVERSE dataset logs); handle by interpolating short gaps *and masking*, or by windowing only within continuous runs — never window across a gap silently.
- **FatigueSet-specific:** the treadmill phase (S2) is motion-saturated by design, and phases must be known per window via `exp_markers.csv` (see `fatigueset-research.md` §5).

---

## 3. 1D-CNN architectures for raw-EEG fatigue/drowsiness

### EEGNet (Lawhern et al. 2018) — the canonical starting point (full text: arxiv.org/abs/1611.08024)
- **Notation:** EEGNet-F1,D; F2 = F1 × D. EEGNet-8,2 → F1=8, D=2, F2=16.
- **Block 1:** temporal Conv2D, F1 filters, kernel (1, 64) at the paper's 128 Hz — **kernel = sampling_rate/2, chosen to capture ≥2 Hz**. Then depthwise spatial conv (C, 1), max-norm 1 → BatchNorm → ELU → AvgPool (1,4) → Dropout.
- **Block 2:** separable conv (1, 16) → BatchNorm → ELU → AvgPool (1,8) → Dropout → Flatten → Dense softmax (max-norm 0.25). Total downsampling ÷32.
- **Training:** Adam (defaults), cross-entropy, up to 500 epochs with validation-based weight selection. **Dropout 0.5 within-subject, 0.25 cross-subject.** Only ~1–2.3k parameters — orders of magnitude below DeepConvNet (~170k). Braindecode's defaults (F1=8, D=2, kernel 64, pools 4/8, dropout 0.25) are the canonical implementation.
- **⚠️ Sampling-rate trap for us:** at Muse S's 256 Hz, preserving Lawhern's ≥2 Hz rule means kernel = **128 samples**, not 64. Keeping 64 at 256 Hz silently changes the model to a ≥4 Hz detector.

### EEGNet on fatigue — Lan et al. 2019 (full text verified)
- Data: Cao et al. driving dataset, preprocessed release (1 Hz HP + 50 Hz LP FIR, artefact removal, downsampled 500→128 Hz).
- **Window: 3 s immediately before lane-departure events → 30 ch × 384 samples.** (A dataset convention, inherited by all follow-ups — not a tuned optimum.)
- Labels from reaction time: alert-RT = 5th percentile of local RTs; fatigue requires local *and* global RT > 2.5× alert-RT; 11 subjects, 837+837 balanced epochs.
- Training: Adam, binary cross-entropy, **100 epochs, batch 100, dropout 0.5** (note: *not* Lawhern's 0.25 cross-subject value). LOSO with the next 2 subjects (circular) as validation.
- **LOSO: EEGNet-4,2 = 59.84% ± 9.55; EEGNet-8,2 = 63.98% ± 11.10** — vs LR 63.62% and TCA+LR 72.70% (see §4).

### InterpretableCNN (Cui, Lan et al.) — strongest shallow-CNN comparator, same dataset (full text: arxiv.org/abs/2107.09507)
- Spatial-first (reverse of EEGNet): pointwise conv (30,1) → 16 maps; depthwise conv, kernel 64 (= fs/2 at 128 Hz), 32 maps; ReLU; BatchNorm; **global average pooling**; dense; softmax. Input 3 s × 30 ch @128 Hz, 1–50 Hz bandpass.
- **Critical training detail:** BatchNorm running statistics were **disabled** (`track_running_stats=False`) in all deep models — default BN caused accuracy collapse at this dataset size. Directly relevant to our small-N CNN.
- Adam lr 0.001, batch 50, ≤50 epochs, 10 repeats × 11 LOSO folds.
- **LOSO (balanced set): InterpretableCNN 78.35%; EEGNet-4,2 71.75%; EEGNet-8,2 71.88%; Conv-ShallowNet 75.19%.** ⚠️ Cui's EEGNet numbers are ~8 points above Lan's on nominally the same data/split — differences include balanced-set construction, the BN modification, and peak-epoch reporting. **Cite one protocol, never average the two.**

### Window lengths across the raw-CNN fatigue literature
| Window | Where |
|---|---|
| 1–2 s | Lawhern's original paradigms only (P300/ERN/MRCP/SMR) — **not fatigue** |
| **3 s** | The driving-fatigue convention (Lan, Cui — Cao dataset, @128 Hz) |
| 4 s | The one Muse S EEGNet project found (below); SFR-Net on 4-ch Muse |
| 8 s | SEED-VIG trial definition — but usually fed as DE/PSD **features**, not raw |
| 13 s | Trejo's classical segments — not a CNN |

No well-sourced 1 s raw-CNN *fatigue* LOSO result exists.

### CNNs on Muse S raw EEG — the cupboard is nearly bare
**No peer-reviewed Muse/Muse S raw-EEG *fatigue* CNN was found** (supports the project's novelty, subject to a formal database search before claiming "first"). Closest recipe: a DecNef neurofeedback project page (non-peer-reviewed) — Muse S, **TP9+TP10 only** (AF7/AF8 amplitudes too low in their data), **4 s windows = 2 ch × 1,024 samples @256 Hz**, per-channel normalisation with **training-set-only mean/SD**, EEGNet with kernel kept at 1×64 (violating the fs/2 rule — a decision to make explicitly, not inherit silently). Motor-vs-rest AUC 0.904. Krigolson et al. 2021 (n=1,000, Muse) found a linear combination of ERP+EEG features predicted perceived fatigue — classical, no CNN, no LOSO accuracy.

---

## 4. Band-power baseline recipes

### Bands — two conventions; pick one and state it
| Convention | δ | θ | α | β | γ | Used by |
|---|---|---|---|---|---|---|
| Lan/Cui driving | 1–4 | 4–8 | 8–12 | 12–30 | — | Lan 2019, Cui 2023 |
| SEED family | 1–4 | 4–8 | 8–14 | 14–31 | 31–50 | SEED/SEED-VIG |
Direction of effects (reviews + Tran et al. 2020 meta-analysis): **θ ↑ (g = 1.03), α ↑ (g = 0.85)** are the largest fatigue effects; β typically ↓; δ inconsistent.

### Lan et al. 2019 — the direct comparator recipe (full text)
- PSD by **plain FFT per 3-s epoch — not Welch.**
- Features: δ/θ/α/β power × 30 channels = **120 features. No ratios, no log, no normalisation reported.**
- Classifier: **logistic regression** (batch GD, 100 iterations).
- **LOSO: LR 63.62% ± 9.93; TCA+LR 72.70% ± 9.42** (TCA = domain adaptation using test-subject data — excluded by our strict LOSO; an upper reference only).

### Cui et al. 2023 — the most complete baseline suite (full text)
- PSD by **Welch (SciPy)** — ⚠️ **segment length, overlap, and window function are NOT reported.** This reporting gap is itself citable: any Welch recipe we choose is a design decision, not a replication.
- Three feature forms per channel over δ/θ/α/β: **RelativePower** (band ÷ total), **LogPower** (natural log of band power), **PowerRatio** (Jap's ratios, below).
- Classifiers: DT, RF, KNN, GNB, LR, LDA, QDA, SVM — all sklearn defaults.
- **LOSO results (balanced, 11 subjects):** LogPower best overall — **GNB 72.68%**, SVM 71.95%, LR 70.24%. RelativePower best: SVM 68.64%. PowerRatio best: SVM 64.24%. **Takeaway: log-power was the strongest single band-power form (~+4 points over relative power).**

### Ratios — Jap et al. 2009 canon
**(θ+α)/β, α/β, (θ+α)/(α+β), θ/β** — all increase with fatigue; **(θ+α)/β showed the largest increase** in monotonous driving and is the most-cited single index. Note θ/α *decreases* with fatigue in Lan's review — do not include ratios as if interchangeable.

### Trejo et al. — the within-subject ceiling (abstract + NASA report)
KPLS on 1–18 Hz PSD of single **13-s segments**, first vs last 15 min of up to 3 h arithmetic: **91–100% accuracy, within-subject only** (frontal θ +29%, alpha +44% across the task). The canonical demonstration that within-subject spectral fatigue classification is near-ceiling while LOSO is not — i.e., the gap our RQ1 measures.

### A fully-specified Welch example (rare — not from fatigue work)
10-s epochs; **Welch with 2-s segments, 50% overlap**; PSD integrated over δ 0.5–4 / θ 4–8 / α 8–13 / β 13–30 (+ α/θ and θ/β ratios). Included because almost no fatigue paper specifies Welch this completely (Frontiers in Systems Neuroscience 2026 worked example).

---

## 5. LOSO protocol practicalities

### Class imbalance
- **The defensible stack from the literature:** class weights **computed from the training fold's class distribution only** (well-documented in subject-level wearable work, e.g. Parkinson gait severity with 5 seeds — adjacent domain), **balanced accuracy + macro-F1 as primary metrics** (explicitly justified this way in nested-LOSO EEG work), accuracy secondary.
- **No EEG LOSO paper using SMOTE-inside-the-fold was found** — the train-fold-only resampling rule rests on general ML guidance, not EEG precedent. Prefer class weights over resampling.
- **Report per-fold collapse:** a named LOSO failure mode is the model becoming a near-one-class predictor on a held-out subject despite acceptable averages (EEG-Fusion, motor imagery). Report each held-out subject's **class prevalence and predicted-class fraction** alongside their score.
- ⚠️ **Held-out-subject normalisation is not innocent:** per-subject z-scoring that uses the held-out subject's statistics is "LOSO with unsupervised test-time normalisation", not a fully inductive protocol (EEGMMIDB LOSO study, 109 subjects) — if used, it must be named as such. Our PLAN §7 already excludes it unless estimable without held-out labels/future info.

### Seeds — separating the two variances (the part most papers get wrong)
- **Report three things:** (i) per-subject score = **mean over seeds**; (ii) headline = **mean ± SD across subjects** of those seed-averaged scores; (iii) supplement: mean within-subject seed SD. **Never pool 12 subjects × 5 seeds = 60 runs into one SD.**
- The distinction is worth ~5–10×: a foundation-model montage study reports seed SD ≈ 0.01–0.03 next to subject SD ≈ 0.11–0.14 in the same table.
- Common counts: **3, 5, or 10** seeds/repeats (Lan/Cui used 10 repeats, averaging training stochasticity before taking each subject's score; BNCI reports use 5 seeds, 42–46, identical splits across seeds). Our PLAN's 5+ seeds sits squarely in this range.
- **Selective seed reporting** is a named interpretation threat (leakage-audited LOSO benchmark) — pre-commit the seed set.

### Per-subject tables and aggregation
- Aggregation is **macro-average over subjects** (unweighted mean of one score per held-out subject ± SD across subjects) — never pooled-window accuracy.
- Table shapes to copy: per-subject rows with Mean/SD columns (BCI IV-2a LOSO tables: EEGNet per-subject 39.5–61.4% → 51.3 ± 6.9 in one hybrid-NN comparison; another reports 76.49 ± 14.74% with per-subject range 60.7–92.0%). The SD across subjects is itself a result — it discriminates stable from lucky methods.
- Cheap, useful addition: **paired win counts** (model A beats model B on k of 12 subjects) — used in the 109-subject EEGMMIDB LOSO paper; ideal for our CNN-vs-baseline RQ2.

### Generalisation-gap reporting (beyond Brookshire/FATE, one line each in `literature.md`)
- **FACED emotion benchmark (cleanest gap design found):** identical pipeline, 3 protocols, 123 participants — intra-subject 85.4% → pooled segment-level 56.4% → strict subject-independent **18.2%** (chance 11.1%). Gaps of 67.2 and 38.2 points attributable to partitioning alone.
- **EEGDash leakage documentation:** same data + model scores 0.95 on random window splits vs 0.55 subject-disjoint; "the accuracy gap is the leakage tax… typically 0.20–0.40" on real EEG problems (library documentation, non-peer-reviewed).
- **Wang et al. 2024 (what causes cross-subject degradation):** the gap depends on task formulation — multi-class-per-subject tasks (labels vary *within* each subject; **FatigueSet is this type**) showed small F1 drops (−0.6 to −1.9), while single-class-per-subject (disease) tasks collapsed (−30 to −50) via subject-identity shortcuts. Implication: our gap should be moderate *unless* a label construction makes session/subject a shortcut — which is precisely what RQ4 tests.
- Typical published drops elsewhere: ESNNet 83.2% within → 51.3% LOSO (−31.9); 1D-CNN MI 79.1% → 55.5% (−23.6, calibration-based LOSO, so true zero-shot gap is larger).

---

## 6. Recommended defaults for OUR pipeline

Each default is tied to the sources above and checked against PLAN.md §7 leakage rules. ⚠️ = read the flag before adopting.

### Preprocessing (Muse S EEG)
| Choice | Default | Basis |
|---|---|---|
| Filter | **Bandpass 1–40 Hz** (zero-phase FIR or 4th-order Butterworth, applied per continuous run) | §2: 1–40 Hz is a published Muse S norm (Sci Rep; Banville); keeps δ–β while cutting drift and the worst EMG band |
| Notch | **None in the primary pipeline** (40 Hz low-pass already excludes 50 Hz mains); run a 50 Hz-notch variant as a sensitivity check only | §2: notch follows mains (UK recording → 50 Hz); with LP at 40 Hz a notch is redundant |
| Reference | Keep FPz hardware reference for the CNN; for band-power features, optionally also compute the average-mastoid (TP9+TP10)/2 re-reference as a sensitivity variant | §2: Krigolson norm |
| Downsampling | **256 → 128 Hz** after anti-alias filtering, for both models | §2/§3: Banville norm; halves CNN input; matches EEGNet's design rate so kernel = 64 samples restores Lawhern's rule exactly |
| Artefact handling | **Amplitude rejection: drop windows containing any sample beyond ±100 µV (post-filter), plus drop windows overlapping Muse fit = Bad (4) on any channel; report the rejection % per participant.** No ICA, no ASR. Blink/jaw marker overlap reported as a statistic, not used as the sole exclusion rule in the primary pipeline | §2: ±100 µV → ~10–12% loss (published); ICA/ASR guidance ≥20 ch; marker-only exclusion has no peer-reviewed precedent |
| Gaps | Window only **within continuous runs**; never window across a Bluetooth gap or a `*2.csv` file join without a continuity check | §2; `fatigueset-research.md` §8 |

### Windowing
| Choice | Default | Basis |
|---|---|---|
| Length | **4 s** (512 samples @128 Hz) for the CNN; **the same 4 s windows** for band-power features, so RQ2 compares models on identical data | §3: 3 s is the driving convention but is event-locked (pre-event epochs), which our continuous session data is not; 4 s is the only Muse S CNN precedent and gives ≥1 full cycle at δ with stable Welch estimates. ⚠️ This is a judgement call between two precedents — state it as such in Methods |
| Overlap | **Non-overlapping (tumbling) windows in the primary pipeline** | PLAN §7 (no overlapping windows across splits) + rs2lab precedent. Overlap *within* a recording on one side of a split is legal under our rules, but tumbling windows make the guarantee trivially auditable and cost little at 4 s. A 50%-overlap variant may be run as a clearly-labelled sensitivity analysis, with splits made at recording level **before** windowing |

### 1D-CNN (RQ2, deep arm)
| Choice | Default | Basis |
|---|---|---|
| Architecture | **EEGNet-8,2**, kernel length **64 samples @128 Hz** (= fs/2, Lawhern's rule), pools (1,4)/(1,8), no dense layer before softmax | §3: canonical; Lan's better EEGNet variant; ~2k params suits 12-subject data |
| BatchNorm | Use standard BN, but **pre-register the Cui fallback**: if LOSO folds collapse, re-run with BN running-stats disabled and report both | §3: Cui's BN collapse warning |
| Training | Adam (defaults), cross-entropy, batch 64, **max 100 epochs with early stopping on a validation split drawn from training subjects only**, dropout **0.25** (Lawhern's cross-subject value), class weights from the training fold | §3 (Lan used 100 epochs/dropout 0.5 — we follow Lawhern's cross-subject dropout and note the difference); §5 imbalance stack |
| Input normalisation | Per-channel standardisation with **mean/SD from the training fold only** | §3/§4: Muse S EEGNet project + train-only rule |

### Band-power baseline (RQ2, classical arm)
| Choice | Default | Basis |
|---|---|---|
| Bands | **δ 1–4, θ 4–8, α 8–12, β 12–30 Hz** (Lan/Cui convention) — γ excluded (EMG-dominated at all four Muse sites, §2) | §4 |
| PSD | **Welch: 2-s Hamming segments, 50% overlap** within each 4-s window (nperseg = 256 @128 Hz) — stated in Methods as *our* specification, since the fatigue literature does not report one | §4: Cui used Welch unreported; fully-specified example uses 2-s/50% |
| Features | Per channel (×4): **log band power** for δ/θ/α/β (16) + **relative power** for δ/θ/α/β (16) + ratios **(θ+α)/β, α/β, (θ+α)/(α+β), θ/β** computed on frontal (AF7/AF8 mean) and temporal (TP9/TP10 mean) aggregates (8) = **40 features/window** | §4: LogPower was Cui's strongest form; RelativePower second; Jap ratios canonical |
| Classifier | **Logistic regression (L2, standardised features, scaler fit on training fold only)** as the primary baseline — it is Lan's comparator and the cleanest RQ2 contrast. SVM-RBF and random forest as pre-registered secondary baselines, selected/frozen via training-only validation | §4: LR 63.62% LOSO is the number to beat/compare; Cui shows SVM/GNB can edge LR on LogPower — hence secondaries, chosen without touching test data |
| ❌ Excluded | TCA/domain adaptation, per-test-subject normalisation — anything that touches held-out-subject data | PLAN §7; §5 LOSO-z warning |

### Labels (for `label_rules.md` — options ranked)
1. **Block-based binary (primary recommendation).** Within each session, windows in the M1 block = the session's M1 mental-fatigue rating; windows in the M3 block = M3 rating; binarise per session by **M3 > M1 → the session contributes "alert" (M1-block) vs "fatigued" (M3-block) windows** (sessions where M3 ≤ M1 are reported and handled by a pre-registered rule — e.g. excluded from the subjective-label experiment, counted in the report). No interpolation, no look-ahead, no future information; the block structure is the dataset's own design (Shuchih-Negi block inheritance, §1) and the phase confound is *identical* across models/splits so RQ2/RQ1 comparisons stay fair — while RQ4 + FATE's phase-probe tell us how much of the signal is phase. ⚠️ **Flag:** block labels are perfectly confounded with time-on-task/phase within a session — this is a known, declared limitation, and it is why the performance-label set (2) and the confound checks matter.
2. **Performance-based (co-primary, per PLAN).** Per block: mean CRT RT and error rate (and 2-back equivalents); fatigability label from **Δ vs the same session's M1 block** (paper's own operationalisation, §1). Windows inherit their block's label. ⚠️ **Flag vs PLAN §7:** a held-out subject's M1 baseline is computed from that subject's *own unlabelled-by-us* block structure — the Δ uses the held-out subject's M1 *performance data* (not labels) — this is participant-specific baseline adjustment and is only legal under §7 if pre-registered as estimable without held-out *labels*; the M1 block identity comes from `exp_markers.csv` (protocol, not labels), so it qualifies — but state this reasoning explicitly in `label_rules.md`, and also run the no-baseline variant (absolute RT thresholds from training subjects) as a sensitivity check.
3. **Median-split block labels (sensitivity analysis).** rs2lab/FatigueSense-style: binarise each block rating against the participant's median rating. ⚠️ **Flag vs PLAN §7:** a *held-out* participant's median uses their full label set — **leakage under strict LOSO**. Legal only if the median is computed from training subjects (pooled threshold) or from the held-out subject's M1 block alone. Do not copy rs2lab's per-participant median into the LOSO pipeline as-is.
4. **❌ Interpolation (MSARNet-style) — excluded from the primary pipeline.** Linear interpolation between M1/M2/M3 labels earlier windows using later ratings = look-ahead leakage under our rules (methods_notes.md, Reviewer 2 review, cross-checked here). Permissible only as a clearly-labelled comparison showing how much it inflates results — arguably a useful RQ4-adjacent demonstration, never the headline.

### Evaluation (all protocols)
- **Primary metrics: balanced accuracy + macro-F1**; also recall/precision per class; accuracy secondary (§5).
- **Per-subject table for every protocol** (12 rows: score = mean over **5 seeds**, class prevalence, predicted-class fraction/collapse check, window count after rejection) + macro mean ± SD across subjects + mean within-subject seed SD in supplement (§5).
- **Generalisation gap = within-subject − LOSO** on the *same metric, same label set, same model*, reported per label set (PLAN RQ1); expect a moderate gap for our within-subject-varying labels unless a shortcut exists (Wang et al. §5) — a large gap is itself the RQ4 signal.
- **Confound probes (from FATE, §1):** train the same baseline to predict (a) task phase and (b) session intensity from the same features; if phase is as predictable as fatigue, say so in Results, not just Discussion.
- **Motion-confound check (from methods_notes.md):** can ACC/fit/artefact features alone predict the labels? Report the number.

### ⚠️ Conflicts with PLAN.md assumptions — summary
1. **rs2lab's per-participant median label threshold leaks in LOSO** if applied to the held-out participant (PLAN §7). Use pooled or M1-only thresholds. (§6, Labels #3)
2. **MSARNet-style interpolation is look-ahead leakage** under PLAN §7 — excluded from the primary pipeline despite being published practice on this dataset. (Labels #4)
3. **Performance Δ-vs-M1 labels use held-out-subject data (their M1 block)** — defensible under §7's "estimable without held-out labels" clause because block identity is protocol metadata, but it must be argued explicitly in `label_rules.md`, not assumed. (Labels #2)
4. **No conflict, but a correction:** FatigueSense's DOI is **10.1145/3709363** (verified against the ACM record, 21 Nov 2026), not the 10.1145/3797885 recorded earlier. (§1)
5. **No conflict, but a scope note:** PLAN §5 says the baseline classifier is picked "via training-only validation, then frozen" — the literature supports pre-registering LR as primary *now* (Lan comparability) and treating SVM/RF as secondaries, which satisfies §5/§7 without any test-informed choice. (§6, Baseline)
6. **FatigueSense's split strategy is unverifiable** — do not claim in the project that FatigueSense did or did not use subject-wise splits; claim only that no subject-wise FatigueSet evaluation was *found in verifiable sources* (FATE excepted). (§1)

---

## Sources (key)
- FatigueSet paper (open PDF): https://alessandro-montanari.github.io/papers/Kalanadhabhatta-PervasiveHealth2021-Camera.pdf
- FatigueSense abstract: https://ouci.dntb.gov.ua/en/works/7PGGrxx9/ ; DOI listing: https://www.citedrive.com/en/discovery/fatiguesense-multi-device-and-multi-modal-wearable-sensing-for-detecting-mental-fatigue/
- rs2lab code: https://github.com/rs2lab/fatigueset-data-preprocessing (fatigueset_preprocess.py)
- Shuchih-Negi parser: https://github.com/Shuchih-Negi/fatigueset_parser_v1
- MSARNet code/results: https://github.com/ARama-K/MSARNet_FL
- EEGNet: https://arxiv.org/abs/1611.08024 ; Braindecode defaults: https://braindecode.org/stable/generated/braindecode.models.EEGNet.html
- Lan et al. 2019 (full text): https://www.fraunhofer.sg/content/dam/singapur/documents/Publications/EEG-based%20Cross-subject%20Mental%20Fatigue%20Recognition.pdf
- Cui/Lan InterpretableCNN (full text + code): https://arxiv.org/pdf/2107.09507v4
- Krigolson Muse pipelines: https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2017.00109 ; https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.634147
- Muse S device comparison: https://www.mdpi.com/1424-8220/24/24/8108
- Muse S spectral/artefact practice: https://www.nature.com/articles/s41598-026-53600-6
- Banville Muse S supplements: https://pure.mpg.de/pubman/item/item_3507943_4/component/file_3675115/Banville_2024_Suppl.pdf
- Goncharova et al. 2003 (EMG topography): https://www.neurotechcenter.org/sites/default/files/misc/EMG%20contamination%20of%20EEG%20spectral%20and%20topographical%20characteristics.pdf
- FACED gap benchmark: https://www.mdpi.com/2079-3197/14/10/230
- Wang et al. 2024, cross-subject degradation: https://arxiv.org/abs/2410.03057
- Plus the full source lists in the three research passes summarised here and in `literature.md` §§D–E.

---

## Addendum (20 Nov 2026) — Reviewer 2 peer review of the §6 recommended defaults

Per my peer-review rule, Reviewer 2 reviewed the defaults above before finalisation. Verdict: pipeline is solid on leakage; **label construction is the weakest part and the highest defence risk.** Points adopted into the working defaults (full text in docs/methods_notes.md):

1. **Subjective/block labels:** M1-vs-M3 confounds fatigue with time-on-task and dry-electrode impedance drift. ADOPTED: primary analysis only includes sessions where the participant actually reported a fatigue increase (Δrating > 0); add a low-fatigue-session negative control (if the model separates M1/M3 there too, it's learning drift); report M1-vs-M2 and M2-vs-M3 as sensitivity analyses.
2. **Performance labels under LOSO:** using the held-out subject's own M1 to define their labels is a form of test-subject information use. ADOPTED framing: distinguish **Zero-Calibration LOSO** (thresholds from training subjects only — the strict primary) from **Baseline-Calibrated LOSO** (subject's own M1 allowed — clearly labelled variant). Simple deltas are noisy/regression-to-the-mean prone; a residualised variant is noted.
3. **Artefact rejection:** ±100 µV is a wet-cap norm; ADOPTED ±75 µV (peak-to-peak, 1-s sub-windows) + flatline check (std < 0.5 µV); add an ocular sanity check correlating predictions with blink events (Muse provides them) since AF7/AF8 pick up blinks and blink rate changes with fatigue.
4. **PSD:** Welch 2-s sub-windows give only 0.5 Hz resolution. ADOPTED: single 4-s Hann window per epoch (Δf = 0.25 Hz) for the baseline.
5. **CNN validation:** random window-level validation inside training subjects is optimistic. ADOPTED: subject-grouped validation (hold out 2–3 training subjects) for early stopping/tuning. EEGNet-4,2 + dropout 0.5 added as a pre-registered capacity variant alongside 8,2.
6. **Metrics:** ADOPTED: add ROC-AUC and MCC; report per-block (M1 vs M3) window rejection rates so class-balance skew is auditable.

These amendments feed docs/label_rules.md and the Week 2 configs; they do not silently rewrite §6 — the original defaults and this review both stay on record.
