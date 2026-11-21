# Key sentences: locked claims and the claims I do not make

*Support material for my own writing: the claim wordings that were locked for each result, each paired with the stronger or different claim the same numbers do not support. When I adapt a sentence for the paper, I keep it on the locked side of the line. This is not a draft of the paper.*

## RQ2: CNN against baseline (docs/cnn_loso_results.md)

**Locked claim:** "There is no evidence that either the frozen-configuration EEGNet CNN or the band-power baseline exceeds chance in cross-participant fatigue classification on FatigueSet (N = 11 analysed of 12 enrolled), and no evidence of an advantage for the CNN."

Claims I do not make:

- "The CNN underperforms the baseline" or "the CNN is equivalent to the baseline." The corrected paired contrast is +0.005 [-0.068, +0.071]; equivalence is untestable at this size and inferiority is not shown.
- "CNNs cannot decode fatigue." The claim is scoped to the frozen configuration: this architecture, this training recipe, this dataset. A differently trained network is not bounded by this result.
- "The CNN failed to train" or "the network is broken." 16 of 55 fits were near-constant, and the record supports only: optimisation failure and absence of signal cannot be separated with the logged quantities.
- The first-reported contrast (-0.014) is never quoted without its correction note: it mixed two window sets, and the corrected value on identical windows is +0.005.

## RQ1: the generalisation gap (docs/cnn_loso_results.md, docs/week2_diagnostics.md)

**Locked wording:** the three tiers are reported separately (same-session 0.744, position-confounded; honest within-subject at chance; LOPO at chance), and the within-minus-LOPO gap is labelled uninformative because both honest endpoints sit at chance (a floor effect).

Claims I do not make:

- The within-minus-LOPO subtraction is never presented as a finding or a measured gap.
- The 0.744 is never called a "ceiling" and never a decoding accuracy. Locked terms: "within-session estimate", "temporal drift is decodable", "separability upper bound", always with the position-confound label attached.
- "No generalisation gap exists" is not a finding of this project; the honest tiers cannot show one because neither rises above chance.

## Week 2 baseline (docs/week2_diagnostics.md, docs/decision_table_lock.md)

**Locked claim:** no feature set transfers above chance; EEG features add nothing beyond the mask pattern; the only systematic cross-session predictions demonstrated were mask-produced. Defensible summary under the gate ruling: no detectable transfer; effects up to about 0.56 cannot be excluded at N = 24 sessions.

Claims I do not make:

- "The only thing that transfers is the mask pattern." That sentence was withdrawn as an overstatement: the mask-only interval also covers chance, and a single participant's inversion is the only mask-produced transfer demonstrated.
- "EEG band-power features are at chance, full stop." The honest form carries the interval and the size limit with it.
- "The equivalence test failed." There was no equivalence test: the feasibility gate ruled the delta = 0.05 margin unattainable before the LOPO runs, and the phase was reported estimation-only.

## DROZY (docs/drozy_results.md)

**Locked headline:** "KSS-extreme labels were decodable across participants; RT-extreme labels from the same recordings were not; this contrast is not isolated from session order and rests on a small disagreeing subset."

**Locked control sentence** (settled at review, quoted in full in docs/drozy_positive_control.md and in the results summary): the spectral-injection control detected RT-aligned band-power shifts at dose 0.5 and above on a 0.25-step grid, so the RT result is informative only about shifts of at least that size and does not rule out smaller RT-related band-power changes, and the dose scale is not mapped onto the observed KSS score.

Claims I do not make:

- "Subjective labels transfer" or "KSS labels generalise." The decodability statement is scoped to the extreme sessions of this dataset and this pipeline.
- "Performance labels do not transfer." The RT arm is a null estimate with a wide interval, not a proven absence. The positive control prices it only to within its grid: injected effects at dose 0.5 and above are excluded, smaller RT-related band-power changes are not, and the RT point estimate itself (0.482) is evidence of nothing.
- "The KSS result (0.602) is clinically useful or viable." No viability claim of any kind.
- "The pipeline is validated across datasets." The Amendment 2 positive control has run and its comparability gate passed, but under the amendment's scope rule a passed control upgrades no claim, so this sentence is still not made.
- "Protocol order is ruled out." The only permitted order wording: order alone does not reach significance. KSS and order labels differ for only 2 of 13 participants, so the arms are nearly the same labelling.
- The Fz-only result carries no spatial interpretation (never "the effect is frontal" or "the effect is not frontal").
- The DROZY section is framed as an exploratory dissociation that reversed the pre-registered label prediction; no pre-declared outcome-table row is claimed in full (Row 1's control and interval conditions are met, but the 0.094 cross-dataset gap sits under the pre-registered 0.10 line).

## RQ4 and the preliminary result (docs/rq4_session_labels.md)

**Locked claim:** "The preliminary result reproduces in its own configuration, but there the fatigued class coincides almost perfectly with a session-wide amplitude regime that fails the pre-declared quality criterion (mask-retained 97.1% of Session-01 windows against 2.5% of Session-03 windows). The score therefore identifies the session and cannot be read as evidence of fatigue decoding. The available data do not separate placement or contact differences from state-correlated physiology or behaviour."

**Locked RQ4 headline (the filter question):** "RQ4 tested whether the subjective-rise inclusion filter matters; under honest splits it does not."

Claims I do not make:

- "Recording-quality failure", "bad recording", "dead channel" or any hardware diagnosis. Zero flatline failures were found; the amplitude regime is a description, not a cause. Behaviour (blinks, jaw clenches, movement all rose in Session 03) and placement are both consistent with the data and neither is separable from state-correlated physiology.
- "The preliminary result was wrong" or "an artefact". It reproduces exactly (LR 1.000, CNN 0.993 in its own configuration); what fails is the attribution of the score to fatigue.
- "Leakage explains the 99.18%." The blocked split, which removes the overlap and window-adjacency routes, still scores 0.915 to 0.975; the operative fact is the class-equals-amplitude-regime coincidence, which is a design property (class = session), not a leakage route.
- "The mask caused the score." The mask-only model is a diagnostic that locates the information; the underlying session-wide amplitude regime is the fact, and the mask is only how it becomes visible.
- "There is no fatigue signal in this data." Non-identifiability is the claim: the design cannot separate the candidate explanations.
- Cohen's kappa of 1.000 between the two labellings is never reported as agreement evidence; it is a tautology, because both labellings are functions of block.
- The strict-pipeline 2x2 cell on Subject 01 (24 retained windows, 1 fatigued test window) is never quoted; it was ruled uninformative by design.
- The day-level Subject 01 value (0.682) never appears without its window counts (2 fatigued windows).

## RQ3 (docs/rq3_results.md)

**Locked claim:** "Peripheral features recover the protocol-defined M1/M3 contrast across participants, and the design cannot separate fatigue from exercise-related physiology." The confound is inside the claim, on purpose.

Claims I do not make:

- "Peripheral signals detect fatigue", "heart rate and EDA detect fatigue", or "peripheral features outperform EEG for fatigue detection." The result is about the protocol contrast (first block against last block of a treadmill session), never about fatigue as a state.
- The 0.741 is never reported without the indicators-only 0.539 beside it and the tier named.
- No percentage of the increment is apportioned to missingness, in either direction; the missingness contribution is not identified by this design.
- The missingness itself is described only as "session-level data loss in two participants"; its cause is unestablished and is never attributed to participants removing the device.
- The within-subject tier is described as "at chance, imprecisely", never as a proven null, and the tier inversion against EEG is a reading (the more parsimonious one), never a finding.
- The heart-rate-by-intensity table carries no inference: no model, no interval, no test.
