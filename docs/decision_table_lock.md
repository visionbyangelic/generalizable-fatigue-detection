# Decision Table & Equivalence Lock — Week 2/3 Interpretation Rules

**Status: PANEL AGREEMENT REACHED 20 Nov 2026 (Reviewer 1 lead, Reviewer 2 concurring with counters, Reviewer 1 final synthesis). SESOI value (Δ = 0.05) pending my confirmation — everything else is fixed by rule, not by results. This document must be final before the Week 2 diagnostics are interpreted.**

## Why this document exists
The baseline result (at chance) is already known. If we invent the interpretation rules after seeing the diagnostics too, every conclusion can be accused of being chosen to fit. So the rules below were negotiated between two independent reviewers (Reviewer 1, Reviewer 2 — reviews 6–9 in docs/methods_notes.md) and locked BEFORE the diagnostics conclude. The claim is falsifiable: §4 states what outcome would have shown the opposite.

## 1. Locked diagnostics (both reviewers agreed)
Feature sanity → trivial baselines (window-position, low-frequency-only, mask-only) → positive control (injected synthetic fatigue effect must be detected) → leakage-gap demonstration (same pipeline, random window split) → within-session ceiling check → block permutation (circular shifts within session) → equivalence test → dose-response (descriptive only, no inferential claims).

## 2. Equivalence rule ("at chance" made precise)
- SESOI Δ = **0.05**: balanced accuracy **0.55** is the smallest effect worth calling useful — below that, a 4-channel consumer headband cannot support useful fatigue monitoring. Hard ceiling: Δ may never exceed 0.08.
- Test: one-sided, α = 0.05; inference at SESSION level (bootstrap CI), never window level. "Equivalent to chance" = upper CI limit < 0.50 + Δ.
- **Feasibility gate (computed from label-free information only):** session-level SD estimated from the block-permutation null; if at N≈24 sessions the smallest Δ giving ~90% power to show equivalence (true effect 0) exceeds 0.05, then equivalence is declared UNATTAINABLE up front and the study reports estimation-only (CIs, no null claim). The bound is never widened to rescue a null.
- Variance never feeds into Δ itself (a noisy dataset must not make a null easier to declare).

## 3. Analysis structure + tier table
P = masked all-channel (primary; owns the headline claim). F = frontal-only (key sensitivity; qualifies P, can never rescue a failed P into a positive headline). "Consistent with chance" for F means equivalent by the SAME test and Δ — "not significant" never counts as a null. Superiority claims across P/F get Holm correction. A frontal above-chance result counts as localised signal ONLY if it survives the mask-only baseline and leakage-gap checks (AF7/AF8 carry blink/frontalis artefacts).

| P | F | Locked interpretation |
|---|---|---|
| Equivalent | Equivalent | **Robust null** |
| Equivalent | Straddles | Distributed null; frontal unresolved |
| Above chance | Equivalent | Distributed signal, not frontal-localised |
| Equivalent | Above chance (survives artefact checks) | Localised / channel-sensitive signal |
| Above chance | Above chance | Signal; localisation not distinguishable |
| Straddles | Not above chance | True inconclusive |

("Above chance" = Holm-adjusted significant, lower CI > 0.50. "Straddles" = neither equivalent nor above chance.)

## 4. Falsification statement
The non-decodability claim is FALSE if: the primary analysis is above chance under the honest session-out/LOSO protocols with the positive control passed and the leakage-gap accounted for — i.e. a real, split-honest effect exists. The claim is also void if the positive control FAILS (pipeline blind → no conclusion about the signal) or the mask-only baseline matches the EEG models (the "signal" was missingness, not brain activity).

## 5. Scope discipline (both reviewers)
Claims are scoped to "band-power features + logistic regression (and, separately, the EEGNet CNN) under session-held-out / LOSO evaluation on Muse S" — never "fatigue is not decodable from EEG". Coverage is reported by class alongside accuracy. Per-participant distributions with intervals accompany every mean.
