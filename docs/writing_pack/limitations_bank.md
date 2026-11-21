# Limitations bank

*Support material for my own writing: the limitations of the project as they were established in the project documents, one or two plain sentences each, with the source named so I can check the wording before I use it. This is not a draft of the paper.*

**Small cohorts.** Every cross-participant claim rests on 11 analysed participants (FatigueSet) or 13 (DROZY), so intervals are wide and small-to-moderate effects cannot be excluded in either direction. Source: docs/decision_table_lock.md, docs/cnn_loso_results.md.

**Estimation-only inference.** The pre-registered usefulness margin (delta = 0.05) was smaller than the smallest margin detectable at this sample size (0.093 to 0.100 on FatigueSet, 0.067 to 0.122 on DROZY), so no equivalence or proven-null claim is available anywhere in the project; all results are estimates with intervals. Source: docs/decision_table_lock.md, docs/drozy_results.md.

**Labels are inherited from blocks.** Under the block design, every window in the first block is labelled alert and every window in the last block fatigued, so the labels are confounded with time-on-task and session phase by construction; this is a declared limitation of the label rules, not something the analysis can remove. Source: docs/label_rules.md §1.

**Sparse subjective ratings.** FatigueSet carries only 3 subjective ratings per session (108 in total), so window-level labels rest on a block rule anchored to a coarse rating grid rather than on per-window ground truth. Source: docs/DATA_INVENTORY.md.

**One participant contributes nothing.** P05 has no retained windows in the primary pool (frontal channels survive the artefact rule in only 7.5% of that participant's windows), so the analysed cohort is 11 of the 12 enrolled. Source: docs/week2_diagnostics.md.

**Performance labels were weak on FatigueSet.** The performance-label route depends on a per-participant reaction-time slope that was essentially zero at group level in FatigueSet, which is why performance labels stayed secondary there and were only usable as a primary arm on DROZY. Source: docs/label_rules.md §4.

**The CNN null is weak evidence.** 16 of 55 CNN fits were near-constant, concentrated in two seeds, and constant-output fits score exactly 0.5, which biases the CNN-versus-baseline comparison toward a null; optimisation failure and absence of signal cannot be separated with the logged quantities, so the CNN result bounds only this frozen configuration. Source: docs/cnn_loso_results.md.

**Incomplete CNN within-subject coverage.** The CNN within-subject tier covers 7 of 12 participants (the others lack a converged fit or a two-class test session), so the CNN side of the RQ1 tier table is thinner than the baseline side. Source: docs/cnn_loso_results.md.

**The Subject 01 mechanism is unseparated.** On the preliminary experiment's two sessions, the available data cannot separate electrode placement or contact differences from state-correlated physiology or behaviour (blink, jaw and movement rates all rose in the fatigued session); the claim stops at non-identifiability. Source: docs/rq4_session_labels.md.

**The replication covers one participant.** The preliminary experiment existed only for Subject 01, so the replication and the amplitude-regime finding are established for that participant; the cohort context (Subject 01 is the extreme of the retention-gap distribution) is descriptive. Source: docs/rq4_session_labels.md.

**DROZY session order is not isolated.** KSS-extreme and protocol-order labellings differ for only 2 of 13 DROZY participants, so the KSS result cannot be separated from session order, and the KSS-minus-RT contrast rests on the 4 participants whose arms disagree. Source: docs/drozy_results.md, docs/drozy_exploratory_checks.md.

**DROZY positive control still pending.** The injected-signal recovery control that would license cross-dataset statements about the pipeline had not finished when this pack was written, so the DROZY outcome-table verdict is open. Source: docs/drozy_results.md (Amendment 2).

**RQ3 cannot separate fatigue from exercise.** The FatigueSet protocol varies treadmill intensity across days and time-on-task across blocks, so heart rate and skin conductance can recover the first-block/last-block contrast through exercise physiology alone; the design has no contrast that separates the two explanations. Source: docs/rq3_results.md.

**No heart-rate variability in RQ3.** The inter-beat-interval data were too sparse (about 31% of windows hold 3 or more usable beats), so the heart-rate arm is mean heart rate only and nothing in RQ3 speaks to HRV. Source: docs/RQ3_INVENTORY.md.

**Class-asymmetric peripheral missingness.** Peripheral validity was 87.9% for fatigued windows against 99.8% for alert windows, driven by session-level data loss in two participants (cause unestablished), so a missingness shortcut exists in the RQ3 design and is bounded, not eliminated, by the indicators-only arm. Source: docs/rq3_results.md.

**Two montages, no spatial comparison.** FatigueSet uses a 4-channel consumer headset (AF7, AF8, TP9, TP10) and DROZY a clinical PSG montage, so channel-level or spatial comparisons across the two datasets are not meaningful and none are made. Source: docs/drozy_results.md, docs/RQ3_INVENTORY.md.

**Cross-environment reproducibility is bounded, not exact.** The Week 2 headline numbers were computed on a software stack that was not recorded at the time; re-running under the pinned stack reproduces them within solver tolerance (one boundary window flips), so the stored values are canonical and exact bit-identity is not claimed. Source: docs/environment_pin.md, docs/rq4_session_labels.md.
