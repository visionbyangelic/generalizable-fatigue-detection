# Chapter map

*Support material for my own writing: how the research questions, methods and results of this project map onto the chapter structure, so I can draft chapter by chapter without re-deciding the structure each time. This is not a draft of the paper.*

*Structure used: the School of Industrial and Applied Sciences structure (docs/school_format.md, structure B), which is the most plausible fit for a Computer Science project in this institution. My school and its exact structure are not yet confirmed; if the structure turns out to be the other one (structure A), the only change is Chapter Three, which then carries the system design and implementation detail (system analysis, architecture, data flow) alongside the research methodology below. Format rules from the same document: minimum 40 pages, Times New Roman 12, double spacing, APA 6th edition references, supervisor's written approval before the defence.*

## Preliminary pages

Title page, certification, dedication, acknowledgement, abstract, table of contents, lists of tables and figures. The abstract is written last, from docs/writing_pack/results_summary.md: one line per research question, each with its locked claim. The six figures in results/figures/ are the results-chapter figures; their captions are drafted in the results summary.

## Chapter One: Introduction

- Background of the study: fatigue detection from wearable EEG, and the field's documented problem that headline accuracies collapse when the person tested was not in the training data (the literature anchors are in docs/literature.md: Brookshire 2024, Lan 2019, the FATE benchmark 2026). The chapter leads with the field's problem; my preliminary single-participant experiment (99.18% fatigued-class recall) enters as motivation and as one documented instance, in the background or the problem statement, never as the headline.
- Statement of the problem: published claims are mostly evaluated on splits that share sessions or participants between training and test, so it is unknown what the models have learned.
- Aim and objectives: the aim (test whether fatigue detection generalises) and the four research questions. RQ1: how large is the generalisation gap between within-session, within-subject and cross-participant evaluation? RQ2: does a 1D-CNN on raw EEG beat a band-power baseline on identical honest splits? RQ3: do heart rate and skin conductance add anything EEG does not, across participants? RQ4: what do session-level labels actually measure?
- Hypotheses, scope and limitations (short form; the full bank is docs/writing_pack/limitations_bank.md), significance of the study, and the operational definitions (fatigue against fatigability, block, session, LOPO).

## Chapter Two: Literature Review

- Conceptual review: fatigue and fatigability as different constructs (the taxonomy point from the label work), subjective scales (KSS, rating scales) against performance measures (PVT reaction time).
- Theoretical framework: EEG band-power markers of fatigue (theta and alpha behaviour), and why cross-participant transfer is the hard case (individual differences, session and device variability).
- Empirical studies: organised by evaluation honesty rather than by model family. Within-session and random-split claims; the collapse studies (Brookshire 2024: over 99% within-subject to about 53% on held-out subjects; FATE 2026: AUC 0.96 to 1.00 in-study, 0.47 to 0.62 across studies); FatigueSense (Kodikara, Wijekoon and Meegahapola 2025, DOI 10.1145/3709363, verified 21 Nov 2026) as the multimodal comparator for RQ3; DROZY and Massoz et al. 2016 as the second dataset's source.
- Summary of the literature and the gap: no published leave-one-participant-out study of FatigueSet EEG that is label-aware and model-comparative on identical splits, and no multimodal LOSO test on it. Source: docs/literature.md §H.

## Chapter Three: Methodology

- Research design: a pre-registered, leakage-controlled evaluation study; label rules, pipeline and the usefulness gate were frozen before the runs they govern (docs/label_rules.md, docs/decision_table_lock.md), and amendments are versioned documents, not silent changes.
- Data: FatigueSet and DROZY, their provenance, terms and cohort sizes (docs/DATA_INVENTORY.md, docs/DROZY_INVENTORY.md).
- Procedures: preprocessing, the Amendment-1 artefact rule, feature sets, the baseline and CNN specifications, the three evaluation tiers, seeds, metrics, bootstrap intervals and null tests. All facts are collected in docs/writing_pack/methods_facts.md.
- Validity and ethics of the design: the leakage clauses (training-fold-only fitting, no shared windows or participants across splits), the positive controls, and the secondary-data position (public datasets, no redistribution of raw data).

## Chapter Four: Results and Discussion

One section per research question, results first and discussion after, each closing on its locked claim (docs/writing_pack/key_sentences.md):

- RQ1 and RQ2 (FatigueSet EEG): the three tiers (Figure 1), the per-participant picture (Figure 2), the CNN degeneracy report and the validation-carve sensitivity, discussed against the collapse literature.
- DROZY: the KSS/RT dissociation (Figure 3), the order-arm negative control and the disagreeing-subset analysis, discussed as evidence about what labels measure; the positive-control verdict is pending and the section is written to state that plainly.
- RQ4: the inclusion-filter test, the day-level arm and session-identity probe, and the preliminary replication with its ladder (Figure 4), discussed as a case study in how a session confound produces a near-perfect score.
- RQ3: the feature-arm comparison (Figure 5), the increments as the headline quantities, the tier inversion, and the descriptive heart-rate table (Figure 6), discussed against FatigueSense and the exercise-physiology confound.

## Chapter Five: Conclusion and Recommendations

- Summary of the work and conclusion: one conclusion per research question, each a locked claim, no new numbers.
- Contribution: a leakage-controlled, label-aware cross-participant evaluation of two public datasets, with the failure modes (mask transfer, session identity, label dissociation, exercise confound) documented as reusable diagnostics.
- Recommendations: for future work (larger cohorts, per-window ground truth, HRV-grade peripherals, CNN training regimes that address the degeneracy) and for practice (evaluation must match the claim: a model sold as detecting fatigue in new users must be tested on new users).
- Limitations: drawn from docs/writing_pack/limitations_bank.md.

## References and appendices

References in APA 6th edition; the FatigueSense DOI (10.1145/3709363) is verified, and the wrong DOI that circulated in my notes (10.1145/3797885, a different article) must not appear. Appendices: the repository and track.md as the audit trail, the pre-registration documents and amendments, and the environment pin (docs/environment_pin.md).
