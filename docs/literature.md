# Literature Notes — Generalizable Fatigue Detection from Wearable Physiological Signals

**Project:** Angelic Charles — *Does a fatigue model still work on people it has never seen?*
**Compiled:** 20 Nov 2026
**Method note:** Every summary below is drawn from a source actually opened/read during this review (abstract, full text, or author PDF, as noted). No findings or numbers are reported that were not verified in the source. Where full text was paywalled, this is stated and the summary uses the abstract/author manuscript only. Preprints are flagged **PREPRINT**.

**Research questions (from PLAN.md):**
- **RQ1** — Generalisation gap: within-subject vs LOSO performance.
- **RQ2** — 1D-CNN (raw EEG) vs classical band-power baseline on identical splits.
- **RQ3** — Multimodal extension: does adding HR/EDA improve LOSO?
- **RQ4** — Label diagnostic: session labels vs subjective-rating vs performance-based labels.

---

## A. Seed references from the proposal

### A1. Williamson et al. (2011) — Fatigue and safety
**Citation:** Williamson, A., Lombardi, D. A., Folkard, S., Stutts, J., Courtney, T. K., & Connor, J. L. (2011). The link between fatigue and safety. *Accident Analysis & Prevention, 43*(2), 498–515. https://doi.org/10.1016/j.aap.2009.11.011
**Summary:** Review of evidence linking fatigue to safety outcomes in transport and occupational settings, defining fatigue as "a biological drive for recuperative rest." It found clear evidence that sleep-homeostasis factors produce impaired performance and accidents, and that sustained-attention/monotonous task effects produce significant performance decrements (their link to accidents was unresolved for lack of studies). It did **not** support a direct link between circadian influences alone and performance/safety outcomes, suggesting circadian effects act in combination with sleep-related factors.
**Relevance: RQ1–RQ4 (framing)** — Motivates why fatigue detection matters for safety, and why task-induced fatigue (the FatigueSet paradigm) is a legitimate target. Background only; no ML evaluation content.

### A2. Kluger, Krupp & Enoka (2013) — Fatigue vs fatigability taxonomy
**Citation:** Kluger, B. M., Krupp, L. B., & Enoka, R. M. (2013). Fatigue and fatigability in neurologic illnesses: Proposal for a unified taxonomy. *Neurology, 80*(4), 409–416. https://doi.org/10.1212/WNL.0b013e31827f07be (open access: PMC3589241)
**Summary:** Proposes a unified taxonomy distinguishing **fatigue** (subjective sensations/perceptions of tiredness) from **fatigability** (objective changes in performance), arguing progress is hampered by terminology and assessment problems. It reviews strengths/weaknesses of common measures and argues many research questions are better addressed using **multiple measures** rather than one. The FatigueSet paper (A4) explicitly adopts this taxonomy.
**Relevance: RQ4 (core)** — This is the theoretical basis for the project's two separate primary label sets: subjective ratings = fatigue; task-performance decline = fatigability. It justifies *not* merging them into one label, and examining their agreement/disagreement.

### A3. Krigolson et al. (2017) — Muse validation for ERP research
**Citation:** Krigolson, O. E., Williams, C. C., Norton, A., Hassall, C. D., & Colino, F. L. (2017). Choosing MUSE: Validation of a low-cost, portable EEG system for ERP research. *Frontiers in Neuroscience, 11*, 109. https://doi.org/10.3389/fnins.2017.00109
**Summary:** Demonstrated ERP research is possible with a portable Muse system and a single computer, without reliance on external event markers. Across a visual oddball task and a reward-learning task, the N200, P300 and reward positivity were observed and quantified, with single-sample t-tests (all p < 0.05), Bayesian credible intervals and 95% CIs all verifying component existence. The Muse version tested was the 2014 model (5 channels, 220 Hz or 500 Hz sampling, ~$249 in their comparison table) — **not** the Muse S used in FatigueSet, a distinction the project should state.
**Relevance: RQ2** — Supports the plausibility of extracting meaningful EEG signals from a low-cost Muse headband, while its ERP (time-domain) focus leaves spectral/band-power validity to other work (see C3, C4).

### A4. Kalanadhabhatta et al. (2022) — The FatigueSet paper
**Citation:** Kalanadhabhatta, M., Min, C., Montanari, A., & Kawsar, F. (2022). FatigueSet: A multi-modal dataset for modeling mental fatigue and fatigability. In *Pervasive Computing Technologies for Healthcare* (PervasiveHealth 2021), Lecture Notes of the Institute for Computer Sciences, Social Informatics and Telecommunications Engineering. Springer. https://doi.org/10.1007/978-3-030-99194-4_14
*Access note: Springer chapter is paywalled; summary is from the authors' open camera-ready PDF and the Springer abstract page.*
**Summary:** Introduces FatigueSet: multi-modal data from **12 participants (9 male, 3 female, ages 21–40, mean 30.75)** wearing four devices — a Nokia eSense-based earable prototype, Empatica E4 wristband, **Muse S EEG headband**, and Zephyr BioHarness 3.0 chest band — covering EEG, PPG, ECG, EDA, skin temperature, accelerometer and gyroscope. Participants performed physically and mentally demanding tasks (including choice reaction time and 2-back tasks) under varying physical-activity conditions, so the design studies how physical activity affects mental fatigue/fatigability. It adopts the Kluger et al. (A2) taxonomy (fatigue = subjective; fatigability = objective performance change). Preliminary analysis indicated the design induced physical and mental fatigue, but **significant mental fatigability was not observed without considering physical-activity conditions**, and fatigue/fatigability differed across activity conditions.
**Relevance: RQ1–RQ4 (core dataset)** — Defines the project data, its label ingredients (subjective ratings + CRT/n-back performance), and a key caution for RQ4: performance-based labels in FatigueSet are entangled with physical-activity condition, so label rules must account for/report this rather than treating performance decline as pure mental fatigue.

### A5. Trejo et al. (2005) — Measures and models for predicting cognitive fatigue
**Citation:** Trejo, L. J., Kochavi, R., Kubitz, K., Montgomery, L. D., Rosipal, R., & Matthews, B. (2005). Measures and models for predicting cognitive fatigue. In *Proceedings of SPIE 5797*, 105–115. (NASA Ames; see also the journal-length version, §B2.)
**Summary:** During a continuous mental-arithmetic task, response time rose (6.7 s → 7.9 s in the 16-subject analysis), ERP component amplitudes did not change, and **frontal theta power rose by 29% and alpha power by 44%** over the task after artefact control/rejection. A kernel partial least squares (KPLS) classifier on 30-channel EEG spectra classified 13-s segments from the first vs last 15 minutes at **91–100% test accuracy** — using random subsamples of each class **within single subjects** (i.e., a within-subject, segment-level split, not a cross-subject test).
**Relevance: RQ2 + RQ1** — The canonical evidence for frontal theta (and alpha) as fatigue markers, justifying the band-power baseline's features. Its 91–100% figure is also a textbook example of the evaluation style this project interrogates: high within-subject segment accuracy says nothing about unseen participants.

### A6. Saeb et al. (2017) — Evaluation must approximate the use-case
**Citation:** Saeb, S., Lonini, L., Jayaraman, A., Mohr, D. C., & Kording, K. P. (2017). The need to approximate the use-case in clinical machine learning. *GigaScience, 6*(5), gix019. https://doi.org/10.1093/gigascience/gix019
**Summary:** Compared record-wise vs subject-wise cross-validation for wearable/smartphone clinical prediction. Subject-wise CV mirrors the clinically relevant use-case (diagnosis in newly recruited subjects); record-wise CV has no such interpretation and **"often massively overestimates"** prediction accuracy, in both a public dataset and simulation. A systematic review found the record-wise method was used by **almost half** of retrieved studies using accelerometers, wearables or smartphones to predict clinical outcomes.
**Relevance: RQ1 (core justification)** — The methodological foundation of the whole project: if the intended use is a new wearer, evaluation must be subject-wise (LOSO). Directly supports the generalisation-gap design.

### A7. Brookshire et al. (2024) — Data leakage in deep-learning EEG
**Citation:** Brookshire, G., Kasper, J., Blauch, N. M., Wu, Y. C., Glatt, R., Merrill, D. A., Gerrol, S., Yoder, K. J., Quirk, C., & Lucero, C. (2024). Data leakage in deep learning studies of translational EEG. *Frontiers in Neuroscience, 18*, 1373515. https://doi.org/10.3389/fnins.2024.1373515
**Summary:** Re-ran published CNN architectures under segment-based vs subject-based holdout on two datasets. Alzheimer's classification: **>99% accuracy with segment-based holdout fell to 53.0% (95% CI 43.1–64.8%, including chance) with subject-based holdout**. Seizure-segment classification: **79.1% fell to 65.1%**. Their survey of 63 translational DNN-EEG papers found **only 17 (27.0%) unambiguously avoided** subject leakage. They note leakage is not DL-specific (also shown with random forests/SVMs) and applies to any method creating non-independence between train and test.
**Relevance: RQ1 + RQ2 (core)** — Provides the expected magnitude of the generalisation gap and the reason both the CNN *and* the classical baseline must be evaluated under identical subject-disjoint splits in this project — leakage inflates classical models too.

---

## B. EEG fatigue markers and fatigue detection from EEG/wearables

### B1. Tran et al. (2020) — Meta-analysis of EEG changes with mental fatigue
**Citation:** Tran, Y., Craig, A., Craig, R., Chai, R., & Nguyen, H. (2020). The influence of mental fatigue on brain activity: Evidence from a systematic review with meta-analyses. *Psychophysiology, 57*(5), e13554. https://doi.org/10.1111/psyp.13554
*Access note: summarised from the published abstract (full text paywalled).*
**Summary:** Systematic review/meta-analysis of 21 studies of EEG spectral change with mental fatigue in non-diseased adults. Overall EEG activity increased with fatigue (medium effect, Cohen's *g* = 0.68, 95% CI 0.24–1.13); network meta-analysis found **large increases in theta (*g* = 1.03) and alpha (*g* = 0.85)**, small-to-moderate changes in delta and beta, with central scalp regions showing the largest change (*g* = 0.80) and large theta increases at frontal, central and posterior sites. Authors conclude frontal/central/posterior theta change is a robust biomarker and alpha change a second-line biomarker accounting for individual variability.
**Relevance: RQ2** — Strongest available justification for the baseline feature set (theta first, alpha second, ratios involving them). Also flags individual variability in alpha — relevant to why LOSO is hard (RQ1). *Caveat for this project:* Muse S has frontal (AF7/AF8) and temporoparietal (TP9/TP10) sites only — no central electrodes — so the "central largest" finding cannot be directly exploited.

### B2. Trejo et al. (2015) — Journal version: EEG estimation/classification of mental fatigue
**Citation:** Trejo, L. J., Kubitz, K., Rosipal, R., Kochavi, R. L., & Montgomery, L. D. (2015). EEG-based estimation and classification of mental fatigue. *Psychology, 6*, 572–589. https://doi.org/10.4236/psych.2015.65055
**Summary:** Journal-length report of the A5 programme (n = 22 in this version): theta and alpha PSD rose 29% and 44% respectively as fatigue developed over up to 3 h of maths problems; a KPLS classifier on 1–18 Hz PSD of single 13-s segments distinguished alert vs fatigued periods at **91–100% accuracy (within-subject)**, and classifier outputs tracked fatigue development over time, with EEG signs present in all subjects after 15–30 min.
**Relevance: RQ2 + RQ4** — Reinforces band-power features; its labels are effectively *time-on-task* (first vs last period) — a close cousin of this project's session/time labels, and therefore a useful RQ4 discussion point: time-based labels confound fatigue with everything else that changes over a session.

### B3. Azam, Jacoby & Coady (2020) — Fatigue classification from consumer-grade (Muse 2) EEG
**Citation:** Azam, M., Jacoby, D., & Coady, Y. (2020). Classification of fatigue in consumer-grade EEG using entropies as features. In *Computer Science & Information Technology (CS & IT) 14* (ICAITA/CDKP/SAI proceedings), 13–26. https://doi.org/10.5121/csit.2020.100902
**Summary:** Used approximate, sample and Rényi entropy features from Muse 2 EEG with five classifiers (Naive Bayes, RBF network, SVM, k-NN, decision tree) for binary fatigue classification; best result was **77.5% accuracy with SVM**. The reviewed text does not describe a subject-independent split, so this number should not be read as cross-subject performance.
**Relevance: RQ2** — Shows (a) Muse-class hardware being used for fatigue classification with classical models, and (b) a more modest accuracy range than the 95%+ headlines once features/models are simple — a useful comparator for the project baseline, with an evaluation-protocol caveat.

---

## C. Muse / Muse S validation beyond Krigolson

### C1. Ratti et al. (2017) — Medical vs consumer wireless EEG for clinical trials
**Citation:** Ratti, E., Waninger, S., Berka, C., Ruffini, G., & Verma, A. (2017). Comparison of medical and consumer wireless EEG systems for use in clinical trials. *Frontiers in Human Neuroscience, 11*, 398. https://doi.org/10.3389/fnhum.2017.00398
**Summary:** Compared two medical-grade (B-Alert, Enobio) and two consumer (Muse, MindWave) systems in 5 subjects over two visits, focusing on Fp1. B-Alert, Enobio and MindWave power spectra were similar; **Muse showed a broadband increase in power spectra and the highest relative variation across test–retest acquisitions**, and consumer systems were more prone to eye-blink/muscle artefacts in the frontal region. Conclusion: data can be collected from all systems, but medical systems offer clear advantages in quality, reliability and depth of analysis.
**Relevance: RQ2 + RQ1** — Directly relevant caution: Muse frontal channels (exactly the AF7/AF8 channels this project relies on for frontal theta) are artefact-prone and less test–retest stable, which can both add noise and create session-specific signatures a model could exploit (links to RQ4).

### C2. Cannard, Wahbeh & Delorme (2021) — Validating Muse for spectral analysis
**Citation:** Cannard, C., Wahbeh, H., & Delorme, A. (2021). Validating the wearable MUSE headset for EEG spectral analysis and frontal alpha asymmetry. In *2021 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)*, 3603–3610. https://doi.org/10.1109/BIBM52615.2021.9669778
**Summary:** Tested whether the 4-channel Muse (two frontal + two temporoparietal dry channels) yields frequency-domain measures comparable to a 64-channel BioSemi ActiveTwo, using deliberately minimal (<5 min) data collection. With a mastoid-referenced montage, Muse could be used to examine PSD in all frequency bands, individual alpha frequency, and frontal alpha asymmetry, with satisfying internal-consistency reliability for alpha power/asymmetry; IAF-based asymmetry estimation gave no advantage over the whole alpha band.
**Relevance: RQ2** — The key *positive* validation for this project's exact approach: band-power features (not just ERPs) from Muse's four channels are defensible, complementing Krigolson's time-domain result (A3).

### C3. Consumer- vs research-grade spectral comparison including Muse S Gen 2 (2024)
**Citation:** Comparison of EEG signal spectral characteristics obtained with consumer- and research-grade devices. *Sensors, 2024, 24*(24), 8108. https://doi.org/10.3390/s14248108 *(article page: https://www.mdpi.com/1424-8220/24/24/8108)*
**Summary:** Recorded resting-state EEG with three consumer devices (PSBD Headband Pro, PSBD Headphones Lite, **Muse S Gen 2**) and a research-grade Brain Products amplifier with mirroring montages. All consumer devices showed higher mean power in low-frequency bands (characteristic of dry electrodes); PSBD Headband matched the research system most precisely, and **Muse showed the poorest signal quality, with extremely low alignment** with the research-grade system in this comparison. The paper notes the wider Muse validation literature is "rather controversial," citing susceptibility to artefacts and test–retest variability alongside reliable ERP results.
**Relevance: RQ1 + RQ2** — The most Muse S-specific caution found: spectral features from Muse S are the noisiest link in this pipeline, which strengthens the project's contribution (honest LOSO numbers on realistic consumer hardware) but must be acknowledged as a limitation. Note it tested Gen 2 hardware and resting state, not FatigueSet's task recordings.

---

## D. Subject-independent / LOSO evaluation and generalisation gaps

### D1. Lan et al. (2019) — EEG-based cross-subject mental fatigue recognition
**Citation:** Lan, Z., Liu, Y., Cui, J., Sourina, O., & Müller-Wittig, W. (2019). EEG-based cross-subject mental fatigue recognition. In *2019 International Conference on Cyberworlds (CW)*. https://doi.org/10.1109/CW.2019.00048 *(author preprint, Fraunhofer Singapore / NTU)*
**Summary:** LOSO evaluation on a public driving-fatigue EEG dataset (Cao et al.; 11 subjects used), two fatigue levels, comparing logistic regression, transfer component analysis (TCA) + LR, and EEGNet. **LOSO mean accuracies: LR 63.62% (±9.93), TCA+LR 72.70% (±9.42), EEGNet-8,2 63.98% (±11.10), EEGNet-4,2 59.84% (±9.55)** — the deep model did **not** beat the classical baseline cross-subject. The paper explicitly criticises earlier "inter-subject" claims (e.g., 84.38% for a residual CNN) whose experiments mixed samples from all subjects into train and test, noting a genuinely subject-independent 4-level attempt without adaptation reached only 39.80%.
**Relevance: RQ1 + RQ2 (core comparator)** — The closest published analogue to this project's design, and a prior for expectations: LOSO fatigue accuracy in the ~60–73% range, high per-subject variance, and no automatic CNN advantage cross-subject — exactly the RQ2 hypoproject space.

### D2. Lan et al. — Inter-subject transfer learning for EEG mental fatigue recognition
**Citation:** Lan, Z., et al. Inter-subject transfer learning for EEG-based mental fatigue recognition. Fraunhofer Singapore / NTU (conference paper; PDF via Fraunhofer Singapore publications).
**Summary:** Extends D1 with Maximum Independence Domain Adaptation (MIDA) under LOSO on 11 subjects: **MIDA achieved 73.01% with all 30 channels; TCA achieved 68.00% with a single (occipital) selected channel**. Shows channel count can be cut drastically with modest accuracy cost under domain adaptation.
**Relevance: RQ1 + RQ2** — FatigueSet's Muse S has only 4 channels; this work suggests limited channels are not fatal cross-subject, but also that the gains came from *domain adaptation using unlabelled test-subject data* — which this project's strict LOSO protocol deliberately excludes, so its numbers are an upper reference, not a direct comparator.

### D3. AXELLSTM / EfficientLSTM (2026) — Nested-LOSO EEG fatigue framework
**Citation:** An EEG-based fatigue detection framework integrating interpretable feature selection and efficient temporal modeling (AXELLSTM). *Frontiers in Neuroscience, 2026*. https://doi.org/10.3389/fnins.2026.1903252
**Summary:** Used **nested LOSO** with all feature selection, channel selection, imputation and normalisation performed *inside* each fold to avoid leakage. Eight features (incl. beta and gamma power) and 18 channels were stably selected across folds. Under LOSO, EfficientLSTM achieved **accuracy 0.7454 (±0.2147), balanced accuracy 0.7436 (±0.2139), Macro-F1 0.7143 (±0.2452)** — note the very large per-subject standard deviations.
**Relevance: RQ1 + RQ2** — A current, methodologically careful LOSO benchmark: ~0.71–0.75 mean performance with ±0.21–0.25 spread across subjects is the realistic LOSO landscape. Its nested protocol mirrors this project's leakage discipline, and its per-subject variance justifies the project's per-participant reporting.

### D4. Cross-subject driving fatigue with personal spectral reference (2026)
**Citation:** Cross-subject EEG driving fatigue detection: Early personal spectral reference and causal long-sequence residual encoding. *Frontiers in Neuroscience, 2026*. https://doi.org/10.3389/fnins.2026.1997832
**Summary:** Argues fatigue is a slow drift poorly served by second-scale windows, and that inter-subject variability typically exceeds intra-subject fluctuation. Method: estimate a personal spectral reference from a short session-opening window, map later spectra into residual space, and accumulate over a long causal sequence. On LOSO binary classification of the MPD-DF driving-fatigue dataset, **five-seed macro-F1 was 0.655 (±0.005)** vs ≈0.587 for the best matched absolute-spectrum baseline; on SEED-VIG LOSO vigilance regression, RMSE 0.209 (5-band) / 0.200 (2 Hz).
**Relevance: RQ1 + RQ4** — Two lessons: (1) another LOSO fatigue number in the mid-0.60s F1; (2) its "personal reference" uses only a short *unlabelled, session-opening* window from the test subject — relevant to this project's rule that participant-specific adjustment is allowed in LOSO only if estimable without held-out labels/future information. It also motivates the RQ4 concern in reverse: session/session-opening information is powerful, so session labels are dangerous.

---

## E. Data leakage in EEG ML beyond Brookshire

### E1. Lee et al. (2023) — Leakage risk in deep-learning CAD for psychiatric disorders
**Citation:** Lee, H.-T., Cheon, H.-R., Lee, S.-H., Shim, M., & Hwang, H.-J. (2023). Risk of data leakage in estimating the diagnostic performance of a deep-learning-based computer-aided system for psychiatric disorders. *Scientific Reports, 13*, 16633. https://doi.org/10.1038/s41598-023-43542-8
**Summary:** Using resting-state EEG from 77 PTSD patients and 58 controls, augmented by cropping windows, compared four CV schemes: subject-wise (sCV), overlapped subject-wise, trial-wise (tCV) and overlapped trial-wise. Augmentation improved performance under all schemes, but **inflated diagnostic performance was observed under the incorrect trial-wise schemes due to data leakage**; the authors conclude subject-wise CV must be used after cropping augmentation.
**Relevance: RQ1** — Independent confirmation, in a clinical EEG setting, that the specific mechanism in this project — overlapping/cropped windows split at trial level — inflates results. Directly supports the PLAN.md rule that overlapping windows must never cross splits.

### E2. FATE (2026) — Leakage- and confound-aware evaluation for wearable fatigue (uses FatigueSet)
**Citation:** Jeribi, F., & Khubrani, M. (2026). FATE: A novel leakage- and confound-aware evaluation framework for trustworthy wearable mental-fatigue detection. *Frontiers in Public Health, 2026*. https://doi.org/10.3389/fpubh.2026.2011960
**Summary:** Proposes a partition ladder (random samples → held-out participants), participant-level uncertainty, and indices for leakage inflation (LII) and confound dominance, validated by simulation/spike-in and applied to **MEFAR (23 employees) and FatigueSet (12 participants)** with ten models. Re-implementations reproduced two published MEFAR pipelines' accuracies (97.2% and 95.0%), but **every model reaching ROC-AUC 0.96–1.00 under random splits fell to 0.47–0.62 on unseen participants (LII up to 0.49)**. Features predicted time of day (ROC-AUC 0.61–0.71) and task phase (0.67–0.81) at least as well as fatigue; **no model–dataset evaluation reached fatigue-attributable evidence (their Tier 3)**, and 18 of 27 graded published studies lacked participant-independent evidence.
**Relevance: RQ1 + RQ4 (critical — see §F)** — The single most important recent paper for this project: it applies exactly this project's critique *to FatigueSet itself* and finds chance-level unseen-participant performance under its framework. See the gaps section for how the project must position against it.

---

## F. Papers/projects that used FatigueSet

*Searches (20 Nov 2026) across web/GitHub/Zenodo/Kaggle found no public code repository, Kaggle notebook, or Zenodo mirror dedicated to FatigueSet beyond the dataset's own esense.io distribution; the uses below are the published ones found. Absence here means "not found in this search," not "does not exist."*

### F1. Kalanadhabhatta et al. (2022) — Original dataset paper
See **A4**. Labels/splits: the paper is a dataset + preliminary-analysis paper (fatigue ratings and CRT/n-back performance analysed against physical-activity conditions); it does not present a LOSO fatigue classifier. Its key label finding — mental fatigability not significant without considering physical activity — is a design constraint for anyone building labels from it.

### F2. Jeribi & Khubrani (2026) — FATE framework applied to FatigueSet
See **E2**. Labels/splits: evaluated FatigueSet (and MEFAR) across a partition ladder from random splits to held-out participants, with ten models and confound probes (time of day, task phase). **Headline LOSO-relevant numbers: random-split ROC-AUC 0.96–1.00 → 0.47–0.62 on unseen participants.** This is, to date, the only found work reporting held-out-participant results on FatigueSet.

### F3. Ramakrishna et al. (2026) — MSARNet federated fatigue regression on FatigueSet
**Citation:** Ramakrishna, P. N., Johnson, [et al.], Parthiban, & Mohideen. (2026). MSARNet: A privacy-preserving multimodal deep tabular network for federated physiological fatigue regression. *Frontiers in Artificial Intelligence, 2026*. https://doi.org/10.3389/frai.2026.1952954 *(author list as given on the article page)*
**Summary:** Regresses physical and mental fatigue levels from five **non-EEG** modality groups (heart rate, inter-beat interval, accelerometer, EDA, skin temperature) using FatigueSet preprocessed into **2,819 overlapping 30-s windows with 15-s step**. Compares DeepFM/AutoInt/TabNet adaptations under centralised and federated (FedAvg/FedProx/FedAdam) paradigms, where federation is per-participant clients. Best result: MSARNet + FedAdam **RMSE 0.2293, MAE 0.2135**, 5.9% better than the next-best federated model; authors attribute federated gains partly to implicit per-client personalisation.
**Relevance: RQ3 + RQ1** — Useful for RQ3 (which non-EEG modalities carry fatigue signal in FatigueSet, in regression form), but its evaluation is **not LOSO classification**: overlapping windows and per-client personalisation mean its numbers cannot be compared with unseen-participant performance, and its window overlap is precisely what this project's protocol forbids across splits.

### F4. Cui et al. (2025) — **PREPRINT** — Heterogeneous multi-source fatigue detection with modality imputation
**Citation:** Cui, L., Wu, Y., Ying, T., & Li, W. (2025). Enhancing fatigue detection through heterogeneous multi-source data integration and cross-domain modality imputation. *arXiv:2507.16859* [cs.RO]. **PREPRINT** (submitted to IEEE; not peer-reviewed as of this review). https://arxiv.org/abs/2507.16859
**Summary:** Formalises a deployment setting where high-quality sensors are unavailable in the field, and imputes missing target-domain modalities from heterogeneous source datasets. Uses VPFD (field), MEFAR and **FatigueSet** as sources — noting all share HR, PPG, GSR, skin temperature and ACC, with FatigueSet additionally providing EEG and ECG (Muse S for EEG, Zephyr for ECG, Empatica E4 for the rest). Abstract reports consistent gains over baselines in sensor-constrained scenarios and improved generalisation across subjects and domains (specific per-dataset figures are in the full text/tables, not the abstract).
**Relevance: RQ3** — Evidence that FatigueSet's value to others is mainly as a *multimodal source* for transfer/imputation, and a pointer that cross-dataset modality alignment (shared HR/EDA/temp/ACC) is feasible — relevant if the multimodal extension is pursued. Treat quantitative claims cautiously (preprint).

### F5. Kodikara, Wijekoon & Meegahapola (2025) — FatigueSense
**Citation:** Kodikara, C., Wijekoon, S., & Meegahapola, L. (2025). FatigueSense: Multi-device and multimodal wearable sensing for detecting mental fatigue. *ACM Transactions on Computing for Healthcare, 6*(2), Article 14, 1–36. https://doi.org/10.1145/3709363
**Summary:** The main machine-learning follow-up on FatigueSet. Builds two label families, "psychological" fatigue (from the subjective VAS ratings) and "physiological" fatigue, and trains device-specific and multi-device models across all four devices. Reported AUCs are modest: device-specific 0.63–0.69 (psychological) and 0.74–0.80 (physiological); multi-device 0.69–0.74 and 0.81–0.88. Its split strategy could not be verified from the sources available to this project, so no claim is made here about whether its evaluation was subject-wise.
**Relevance: RQ3 + RQ1** — The honest published reference point for what multimodal FatigueSet models score, and a reminder that the published numbers on this dataset were never in the 95%+ range. Its AUCs are not comparable with this project's unseen-participant balanced accuracies (different task construction and evaluation).
*DOI verified 21 Nov 2026 against the ACM Digital Library record for this title (dl.acm.org/doi/10.1145/3709363) and the Crossref-derived index record (ISSN 2637-8051): the correct DOI is 10.1145/3709363. The DOI 10.1145/3797885 recorded in an earlier note belongs to a different article in the same journal (Dani et al., 2026, on self-management technologies in multiple sclerosis).*

---

## G. Subjective fatigue vs fatigability — beyond Kluger

### G1. Enoka & Duchateau (2016) — Translating fatigue to human performance
**Citation:** Enoka, R. M., & Duchateau, J. (2016). Translating fatigue to human performance. *Medicine & Science in Sports & Exercise, 48*(11), 2228–2238. https://doi.org/10.1249/MSS.0000000000000929
**Summary:** Argues for a unified definition: fatigue is a disabling symptom in which physical and cognitive function is limited by interactions between **performance fatigability** (decline in an objective performance measure) and **perceived fatigability** (the sensations that regulate the integrity of the performer). By defining fatigue through fatigability, the reported level is normalised to the demands of the task being performed.
**Relevance: RQ4** — Complements Kluger (A2) with the performance/perceived pairing and the task-demand caveat: the same performance decline means different things under different task loads — pertinent to FatigueSet, where physical-activity condition varies by design (see A4).

### G2. Fatigue and human performance: An updated framework (2022)
**Citation:** Behrens, M., et al. (2022). Fatigue and human performance: An updated framework. *Sports Medicine* (PubMed: 36258141).
*Access note: summarised from the PubMed abstract.*
**Summary:** Narrative review updating Enoka & Duchateau (2016): distinguishes trait fatigue from task-induced state fatigue, adds cognitive performance fatigue and perceived cognitive fatigue to the taxonomy, and adds determinants (effort perception, affective valence, self-regulation). Proposes defining task-induced state fatigue as a psychophysiological condition characterised by decreased motor/cognitive performance **and/or** increased perception of fatigue.
**Relevance: RQ4** — The "and/or" is the point for this project: subjective and performance labels can legitimately disagree, so their agreement analysis is a finding, not a data problem.

---

## H. Gaps our project fills

Grounded only in the sources above:

1. **FatigueSet has almost no subject-independent classification evidence.** Of the found FatigueSet uses, the original paper (A4/F1) reports no LOSO classifier; MSARNet (F3) uses overlapping windows and per-participant federated personalisation rather than unseen-participant evaluation; Cui et al. (F4, preprint) use it as an auxiliary transfer source. Only FATE (E2/F2) reports held-out-participant results on FatigueSet — as part of a general evaluation framework, with ten off-the-shelf models, not as a dedicated EEG modelling study. A focused, reproducible LOSO benchmark of a 1D-CNN vs a band-power baseline on FatigueSet EEG, with per-participant results and 5+ seeds, does not exist in the found literature.

2. **The one LOSO result on FatigueSet says "chance" — but does not close the question.** FATE found unseen-participant ROC-AUC of 0.47–0.62 and no fatigue-attributable evidence, and showed features encode time-of-day/task-phase confounds at least as strongly as fatigue. That makes the project's RQ4 design (separate subjective, performance and diagnostic session labels, all under identical LOSO splits) a direct, necessary follow-up: FATE demonstrates the confound; it does not test whether *label construction* (fatigue vs fatigability, per A2/G1/G2) changes the answer, nor compare a raw-EEG CNN against band-power features head-to-head on this dataset.

3. **Cross-subject EEG fatigue work is mostly driving datasets, not wearable lab data.** The careful LOSO studies found (D1–D4) use driving/simulator EEG with research-grade montages; their LOSO range (~60–73% accuracy; F1 ~0.66; Macro-F1 ~0.71 with ±0.25 subject spread) sets expectations but does not transfer directly to 4-channel consumer Muse S data, whose spectral quality is itself contested (C1–C3, esp. C3's poor Muse S Gen 2 alignment). The project fills the wearable-specific gap — and must report per-participant spread, not just means, given D3's variance.

4. **Leakage critiques rarely compare model families under identical clean splits.** Brookshire (A7) and Lee (E1) show leakage inflates CNNs (and classical models), and Lan (D1) found EEGNet did *not* beat logistic regression cross-subject — but no found study runs a raw-window 1D-CNN and a band-power classical baseline through the *same* within/session-out/LOSO ladder on the *same* dataset with matched label sets. That three-protocol, two-model, multi-label comparison is the project's core methodological contribution (RQ1/RQ2/RQ4 together).

5. **Multimodal LOSO evidence for FatigueSet is missing.** RQ3 is genuinely open: MSARNet (F3) shows non-EEG modalities carry regressable fatigue signal in FatigueSet but under non-LOSO evaluation; no found study tests whether adding HR/EDA to EEG improves *unseen-participant* performance on FatigueSet. Even a null result, cleanly obtained, fills a real gap.

**Positioning caution (from E2):** the project should cite FATE early and explicitly. Its contribution is not "first to notice leakage in FatigueSet" — FATE did that — but the first dedicated, label-aware, model-comparative LOSO study of FatigueSet EEG, testing whether *any* defensible label/model combination survives unseen participants, and quantifying the within→LOSO gap per participant.
