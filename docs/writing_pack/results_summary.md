# Results summary

*Support material for my own writing: the headline numbers of the project in one place, each with the claim wording I locked for it and a pointer to the document or table it comes from. This is not a draft of the paper. I write the paper from this, in my own words.*

*Conventions: BA = balanced accuracy. Intervals are 95% participant-bootstrap confidence intervals (10,000 resamples) unless a line says otherwise. Everything in this project is reported as estimates with intervals: the feasibility-gate ruling (docs/decision_table_lock.md) made every phase estimation-only, so no equivalence or proven-null claim appears anywhere below.*

---

## 1. Week 2 baseline: band-power EEG on FatigueSet (RQ1/RQ2 foundation)

Pipeline: locked band-power features (43 features including 3 validity indicators) with logistic regression, primary subjective labels (M1 windows alert, M3 windows fatigued, sessions kept only where the subjective mental-fatigue rating rose, 32 of 36 sessions), Amendment-1 artefact rule. Artefact retention: 1,121 of 1,568 candidate windows (71.5%); primary pool 1,053 windows (564 alert, 489 fatigued), 11 of 12 participants contributing both classes (P05 contributes none).

| Protocol | BA | Source |
|---|---|---|
| Within-subject | 0.484 | docs/week2_diagnostics.md |
| Leave-one-session-out | 0.515 | docs/week2_diagnostics.md |
| LOPO, masked all-channel (primary, P) | 0.508 [0.471, 0.542] | docs/week2_diagnostics.md |
| LOPO, frontal-only (sensitivity, F) | 0.544 [0.500, 0.589] | docs/week2_diagnostics.md |
| LOPO, V1 (uncapped, class-weighted) | 0.503 [0.445, 0.552] | docs/week2_diagnostics.md §18 |
| LOPO, V2 (within-session normalised) | 0.556 [0.480, 0.613] | docs/week2_diagnostics.md §18 |

Supporting diagnostics (all in docs/week2_diagnostics.md):

- Within-session estimate (windows split inside one session): 0.744. Locked reading: temporal drift is decodable within a session; the number is a separability upper bound under a position-confounded split (window position alone ranks the labels perfectly, AUC 1.000 by construction), never a decoding accuracy.
- Block permutation null: 0.508 ± 0.032 against a real score of 0.499, p = 0.612.
- Mask-only model (validity indicators, no EEG), session-out: 0.530, about the full model's level. Corrected locked wording: no feature set transfers above chance; EEG features add nothing beyond the mask pattern; the only systematic cross-session predictions demonstrated (P08's within-subject inversion, 0.225, which returns to 0.467 when the indicator features are removed) were mask-produced.
- Positive control: injected theta/alpha effects are recovered with a graded dose response (full dose BA 0.834, half dose 0.695); minimum detectable effect = theta x1.5 / alpha x1.25. The pipeline can see an effect of that size; it does not see one in the real labels.
- Leakage gap: random window split 0.571 against honest session-out 0.499 on the same features.
- Gate: the usefulness margin was fixed at delta = 0.05 before the LOPO runs; the smallest margin detectable at about 90% power was 0.093 to 0.100, so equivalence at delta = 0.05 was unattainable and the phase is estimation-only. Defensible summary wording from the lock: no detectable transfer; effects up to about 0.56 cannot be excluded at this size.

## 2. CNN cross-participant results: RQ2 answered, RQ1 in tiers

Source: docs/cnn_loso_results.md; tables in results/tables/cnn_loso_* and rq1_generalisation_gap*.

- CNN LOPO: BA 0.493 [0.452, 0.529], AUC 0.517 [0.466, 0.564], MCC 0.000 [-0.081, 0.076]. 55 of 55 folds (11 participants x 5 model seeds, canonical cap-seed-0 set of 806 windows).
- Corrected paired contrast on identical windows, CNN minus baseline: +0.005 [-0.068, +0.071]. The first-reported contrast, -0.014 [-0.075, +0.042], mixed two window sets (baseline 0.508 on the primary pool, 0.488 on the canonical set); the correction and its cause are disclosed in the results document.
- Locked RQ2 claim: there is no evidence that either the frozen-configuration EEGNet CNN or the band-power baseline exceeds chance in cross-participant fatigue classification on FatigueSet (N = 11 analysed of 12 enrolled), and no evidence of an advantage for the CNN.
- Degeneracy: 16 of 55 CNN fits predicted one class for at least 95% of a held-out participant's windows (all under seeds 0 and 3; 43 of 55 fits stopped at epoch 16, the earliest stop the patience allows). Locked reading: the CNN result is a null for this configuration and does not bound what a differently trained network could do; because constant-output fits score exactly 0.5, the CNN-versus-baseline test is biased toward a null and the CNN null is weak evidence. Descriptive only, never a headline: the 39 non-degenerate fits average 0.492 [0.439, 0.538].
- Validation-carve sensitivity (post hoc): carving early-stopping validation by whole training participant gives CNN LOPO 0.509 [0.482, 0.538], a paired movement of +0.016 [-0.012, +0.047]. The original carve search was verified from the code to use training participants only and to compute no performance score.
- RQ1 tiers: same-session estimate 0.744 (position-confounded); honest within-subject at chance (baseline 0.484, CNN 0.493); LOPO at chance (baseline 0.508, CNN 0.493). The pre-registered within-minus-LOPO gap is reported with its intervals (paired: baseline -0.030 [-0.112, +0.053], n = 8; CNN -0.006 [-0.104, +0.085], n = 7) and labelled uninformative: both honest endpoints sit at chance, so the subtraction cannot measure what moving to a new person costs.

## 3. DROZY: the label dissociation

Source: docs/drozy_results.md and docs/drozy_exploratory_checks.md; the frozen FatigueSet pipeline applied unchanged to DROZY (clinical PSG EEG, 13 participants analysed), leave-one-participant-out only.

| Arm | BA [95% CI] | AUC | Swap-null p |
|---|---|---|---|
| KSS (subjective) labels, all-channel | 0.602 [0.563, 0.644] | 0.681 | 0.005 |
| KSS labels, Fz only | 0.543 [0.505, 0.582] | 0.597 | 0.134 |
| RT (performance) labels, all-channel | 0.482 [0.421, 0.549] | 0.477 | 0.726 |
| RT labels, Fz only | 0.467 [0.414, 0.527] | 0.462 | 0.811 |
| Protocol-order labels, all-channel | 0.536 [0.493, 0.572] | (not reported) | 0.229 |

Locked headline: KSS-extreme labels were decodable across participants; RT-extreme labels from the same recordings were not; this contrast is not isolated from session order and rests on a small disagreeing subset.

Supporting numbers: paired KSS minus RT contrast +0.120 [0.062, 0.179], about 60% of it carried by the 4 participants whose two arms pick different fatigued sessions (P01, P02, P04, P05); paired KSS minus order contrast +0.066 [0.031, 0.105], resting on the 2 participants whose labels differ between those arms; declared sensitivities (P11 out, P7 out) move the KSS headline by at most 0.003; the KSS point estimate sits 0.094 above the FatigueSet LOPO estimate, just under the pre-registered 0.10 line for a clearly interpretable cross-dataset contrast, and is reported as a descriptive contrast only. The Fz-only arm is descriptive: no spatial interpretation is made. Framing: an exploratory dissociation from a pre-registered pipeline, reported openly as reversing the pre-registered label prediction.

Pre-declared outcome table: verdict **PENDING**. The Amendment 2 positive control (injected-signal recovery on DROZY, with the paired KSS-minus-order contrast) is still running; no outcome-table row is claimed until it lands and is reviewed.

## 4. RQ4: session labels, and what the preliminary 99.18% was

Source: docs/rq4_session_labels.md; tables in results/tables/rq4_* and subject01_*.

The inclusion-filter question (the RQ4 question as asked): block-phase session labels (M1 alert, M3 fatigued, no subjective filter) against the primary labels on identical windows, features, model and seeds. All-36-sessions LOPO 0.516 [0.482, 0.548] against the primary 0.508; session-out unchanged at 0.515; within-subject 0.461 against 0.484. Cohen's kappa between the two labellings on shared windows is 1.000, which is a tautology (both labellings are functions of block; the rating only filters sessions) and is never reported as a finding. Locked headline: RQ4 tested whether the subjective-rise inclusion filter matters; under honest splits it does not.

Day-level arm (post hoc, estimation-only): whole earliest session alert, whole latest session fatigued. LOPO BA 0.484 [0.353, 0.620], exact swap null p = 0.822. The deliberately leaky random-window split on the same labels scores 0.741. A session-identity probe (which of the 3 sessions a window comes from, within participant) scores 0.821 against 0.333 chance: the recordings carry a strong session signature, and any split that shares recordings between train and test can spend it.

The preliminary replication (Subject 01, the preliminary experiment's own configuration: 1,353 overlapping windows reproduced exactly, chronological 80/20 split, no artefact rejection): logistic regression BA 1.000 (fatigued recall 127/127), CNN BA 0.993 (recall 127/127, all 3 seeds). The preliminary headline reproduces. The mechanism evidence:

- The Amendment-1 mask retains 97.1% of Session 01 windows and 2.5% of Session 03 windows; every fatigued test window is mask-rejected.
- A mask-only model scores 0.993, and the trivial rule "fatigued if and only if not mask-retained" also scores 0.993. Removing AF7, or the whole frontal pair, deflates nothing (LR 0.993, CNN 1.000): the session signature is headset-wide.
- A blocked split (60 s blocks, overlap removed, recordings still shared) barely reduces the scores: LR 0.975, CNN 0.915.
- Session 03 AF7 amplitudes are about 9 times Session 01's (median peak-to-peak 408 vs 46 microvolts), with zero flatline failures in either session: the channel is not dead, and the elevation is present from the first window and robust to a 6x threshold sweep.
- Subject 01 is the cohort extreme for day-level retention gap (-96.4 points; the next is -86.6, and signs vary across the cohort).

Locked claim: "The preliminary result reproduces in its own configuration, but there the fatigued class coincides almost perfectly with a session-wide amplitude regime that fails the pre-declared quality criterion (mask-retained 97.1% of Session-01 windows against 2.5% of Session-03 windows). The score therefore identifies the session and cannot be read as evidence of fatigue decoding. The available data do not separate placement or contact differences from state-correlated physiology or behaviour."

Subject 01 under the fixed design (for the write-up's comparison table): within-subject BA 0.436; LOPO 0.496 (subjective labels) and 0.508 (block-phase labels); day-level LOPO 0.682 on only 2 fatigued windows, reported with that caveat wherever it appears; Session 03 contributes zero retained fatigued windows to the primary pool.

## 5. RQ3: heart rate and skin conductance

Source: docs/rq3_results.md; design in docs/rq3_preregistration.md (Amendment 1 included); tables in results/tables/rq3_*.

Locked claim: peripheral features recover the protocol-defined M1/M3 contrast across participants, and the design cannot separate fatigue from exercise-related physiology.

Keep-all pool, LOPO (the headline tier):

| Arm | BA | 95% CI |
|---|---|---|
| Indicators only (missingness pattern) | 0.539 | [0.500, 0.595] |
| EEG only (stored Week 2) | 0.508 | [0.471, 0.542] |
| Peripheral only (HR + EDA) | 0.741 | [0.633, 0.851] |
| Combined (EEG + peripheral) | 0.737 | [0.636, 0.838] |

Headline increments (paired per participant): peripheral minus indicators +0.202 [0.085, 0.314]; combined minus EEG-only +0.230 [0.128, 0.332]. The restricted pool of 993 complete-peripheral windows agrees (peripheral 0.724; increment +0.224 [0.133, 0.315]), and scoring keep-all-trained models on complete test windows only agrees (peripheral 0.725). The pre-declared outcome row 2 matched: peripheral above indicators in both pools. Secondary tiers: session-out peripheral 0.709; within-subject peripheral 0.549 [0.448, 0.655], at chance imprecisely. Descriptive ablations: HR-only 0.647 [0.556, 0.748], EDA-only 0.640 [0.558, 0.735]. Descriptive heart-rate table: the M3-minus-M1 mean heart-rate difference is -1.3 bpm on low-intensity days, +13.3 on medium, +18.2 on high, which tracks the exercise load of the day. The tier inversion against EEG (peripheral strong across participants, weak within; EEG at chance everywhere) is reported as the more parsimonious reading, a shared exercise response, and as a reading, not a finding.

Data facts that travel with the result: inter-beat-interval data were too sparse for heart-rate variability features (the heart-rate arm is mean HR only); peripheral missingness was class-asymmetric (fatigued validity 87.9% vs alert 99.8%) because of session-level data loss in two participants, cause unestablished.

---

## Figure captions

**Figure 1 (fig_rq1_tiers.png). RQ1 in three tiers.** Balanced accuracy for the band-power baseline and the EEGNet CNN under three evaluations of increasing honesty on FatigueSet: windows split inside the same session, new sessions of known participants, and a wholly unseen participant (LOPO). The same-session value (0.744, baseline only; the CNN has no estimate on this tier) is labelled position-confounded because window position alone ranks the labels perfectly under the block design. Locked reading: the collapse in this project sits between the same-session estimate and every honest split, not between the two honest splits; both honest tiers are at chance for both models, and the within-minus-LOPO gap is reported as uninformative (floor effect), never as a finding.

**Figure 2 (fig_rq2_perparticipant.png). RQ2 per participant.** Per-participant LOPO balanced accuracy for the CNN and the baseline on identical folds (means over 5 seeds; error bars are the seed SD). Locked reading: the values scatter around chance for both models (cohort means 0.493 and 0.508) and the spread is reported as values only; no per-participant pattern, responder story or individual-difference claim is made from it.

**Figure 3 (fig_drozy_dissociation.png). The DROZY dissociation.** LOPO balanced accuracy on DROZY under two label constructions applied to identical recordings: subjective KSS extremes and performance (reaction-time) extremes, each with all-channel and Fz-only features, with participant-bootstrap intervals and within-participant session-label swap-null p-values. Locked reading: KSS-extreme labels were decodable across participants and RT-extreme labels were not; the contrast is not isolated from session order and rests on a small disagreeing subset. The Fz-only bars are descriptive and carry no spatial interpretation.

**Figure 4 (fig_rq4_ladder.png). Subject 01 under stricter evaluation.** Balanced accuracy for Subject 01's two prototype sessions as the evaluation tightens: the preliminary experiment's own chronological configuration (LR 1.000, CNN 0.993), a mask-only model with no EEG (0.993), a blocked split within the same recordings (LR 0.975, CNN 0.915), and this project's fixed design (within-subject 0.436; unseen-person test 0.496). The figure carries the mask-retention annotation (97.1% of Session 01 windows retained against 2.5% of Session 03 windows). Locked reading: the preliminary score reproduces in its own configuration but identifies the session, because the fatigued class coincides almost perfectly with a session-wide amplitude regime that fails the pre-declared quality criterion; it cannot be read as evidence of fatigue decoding.

**Figure 5 (fig_rq3_arms.png). RQ3 by feature arm.** LOPO balanced accuracy on FatigueSet for the protocol-defined M1/M3 contrast under four feature arms (indicators only, EEG only, peripheral only, combined), with participant-bootstrap intervals. The title names the protocol contrast, not fatigue, as the locked RQ3 wording requires. Locked reading: peripheral features recover the protocol-defined M1/M3 contrast across participants (headline quantities are the increments, +0.202 over indicators and +0.230 over EEG-only), and the design cannot separate fatigue from exercise-related physiology.

**Figure 6 (fig_rq3_hr_intensity.png). Heart rate by intensity day (descriptive).** The M3-minus-M1 mean heart-rate difference grouped by the treadmill intensity of the session day (-1.3 bpm low, +13.3 medium, +18.2 high), computed over the primary pool's HR-usable windows. No model is fitted and no interval or test is attached. Locked reading: the table is descriptive evidence that the heart-rate feature tracks the exercise load of the day; it is reported beside the RQ3 result, with no inference drawn from it.
