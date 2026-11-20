# Second dataset — pre-registration (DROZY) — DRAFT for my approval

**Status: PANEL AGREEMENT (Reviewer 1 lead + Reviewer 2 concurring, reviews 13–14, 20 Nov 2026). NOT active until I approves. No DROZY data has been downloaded; the pipeline code must be frozen and hashed before download (Reviewer 1's forking-paths rule).**

## Why a second dataset at all
Not to find a dataset where the model wins (that would be the rescue move both reviewers rejected). DROZY is a **diagnostic**: it changes hardware (clinical wet PSG vs dry Muse), induction (up to ~28 h sleep deprivation vs treadmill protocol) and — decisively — carries **two label types on identical recordings** (subjective KSS + objective reaction times), which permits the attribution FatigueSet alone cannot make.

## The one primary cross-dataset contrast
Leave-one-participant-out balanced accuracy of the locked band-power + logistic-regression pipeline, DROZY vs FatigueSet (0.508 [0.471, 0.542]). Session-out is **not** compared across datasets: in DROZY, session is aliased with deprivation level (the label itself), so session-out folds are degenerate/single-class. Δ stays 0.05; the feasibility gate is recomputed for DROZY (expected ≈ 0.086–0.093 at N = 14 → estimation-only expected; the bound is never relaxed to obtain a verdict).

## DROZY label rules (fixed here, before download)
- Windows: 4 s non-overlapping, same bands (δ/θ/α/β), same rejection thresholds in µV; rejection rates reported per dataset.
- **KSS labels (subjective arm):** within-person contrast — each participant's lowest-KSS session = alert, highest-KSS session = fatigued; the middle session is excluded from the primary (mirrors FatigueSet's M1-vs-M3 block contrast). Windows inherit their session's label. Class prevalence reported.
- **RT labels (performance arm):** identical session contrast, but sessions ranked by PVT lapse rate (reaction times > 500 ms) instead of KSS — same EEG, different label construction. This is the label-artefact test: hardware and data held fixed, only the labels change.
- Frontal-only on DROZY = Fz alone (declared; Muse frontal = AF7/AF8; the montages share no sites — stated wherever compared). Mask-only baseline: **not applicable** on clinical recordings (no dropout pattern) — reported as N/A, never as an empty control.
- EEGNet: kernel lengths scaled by the sampling-rate ratio (512/128) by rule, never tuned. Same declared V1/V2 sensitivities, same seed count, run once.

## Pre-declared outcome table (claims fixed in advance — Reviewer 1, review 14)
| DROZY LOPO result | What the thesis may say |
|---|---|
| Clearly above FatigueSet, CI excludes 0.50, positive control recovered | Transfer is achievable under a strong manipulation + clean hardware; the Muse null is not a generic evaluation limit. Attribution only via the label/channel ablations. |
| Above chance with RT labels, at chance with KSS labels | Label-construction effect (same EEG, same hardware — only the labels differ). |
| At chance, positive control recovered | Consistent with a general limit for this pipeline family at small N. Never "fatigue is undecodable". |
| High only under random-window splits | The leakage gap replicated on independent data. |

## Scope guardrails
One secondary dataset only. No DROZY-specific tuning. SEED-VIG is a separate audit project (not this thesis). STEW is excluded (workload construct; no repeated sessions). Timeline: DROZY work slots into Weeks 3–4 alongside the FatigueSet CNN LOSO; if anything slips, the multimodal extension (RQ3) is cut first, as already planned.
