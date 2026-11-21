# Second dataset — pre-registration (DROZY) — DRAFT for my approval

**Status: LOCKED — panel agreement (Reviewer 1 lead + Reviewer 2 concurring, reviews 13–14) AND APPROVED 20 Nov 2026 ("Approve the DROZY plan — download it and lock the pre-registration"). The pipeline code freeze hash below was recorded BEFORE the DROZY download, per the forking-paths rule.**

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
| DROZY LOPO result | What the project may say |
|---|---|
| Clearly above FatigueSet, CI excludes 0.50, positive control recovered | Transfer is achievable under a strong manipulation + clean hardware; the Muse null is not a generic evaluation limit. Attribution only via the label/channel ablations. |
| Above chance with RT labels, at chance with KSS labels | Label-construction effect (same EEG, same hardware — only the labels differ). |
| At chance, positive control recovered | Consistent with a general limit for this pipeline family at small N. Never "fatigue is undecodable". |
| High only under random-window splits | The leakage gap replicated on independent data. |

## Scope guardrails
One secondary dataset only. No DROZY-specific tuning. SEED-VIG is a separate audit project (not this project). STEW is excluded (workload construct; no repeated sessions). Timeline: DROZY work slots into Weeks 3–4 alongside the FatigueSet CNN LOSO; if anything slips, the multimodal extension (RQ3) is cut first, as already planned.

## Code freeze (recorded before DROZY download)

SHA-256 manifest of the analysis pipeline (`src/` + `configs/`) at freeze time:

```
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  src/__init__.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  src/data/__init__.py
0eff4c4dba7ec5c10e6978143a20ed13c6961156e1e746af84101f3f3df4024b  src/data/inspect_dataset.py
a1c53058063b2ddc7b4eae25e9d807bfb734d9ba83b9c1dfe82c05ad96a59541  src/data/load_session.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  src/evaluation/__init__.py
7343ee5ba6229d7def0e9ef4dca8c1843359aef5812b8b98242ee5c23b0f73d7  src/evaluation/closure_pack.py
f45e902f1e2e86d8af34be1e687e875ce050037ac7beadddca51dcf41d0aa8a1  src/evaluation/diagnostics.py
3db179a59c5f02ef09421131fecd126b54b23953af2f548c088fc2ad896019af  src/evaluation/diagnostics_followup.py
1147fef24f7864580e6d7959b1b2b460a1f027c7c174c8560a7f9e49ab8b408e  src/evaluation/metrics.py
ee225c172694632dac4bf18b7ba154c65fd0326cf5e484a39c368d79ad458029  src/evaluation/panel_additions.py
ba92ab0011fc39bc828153960e0484ecf2e5315b3568284140d93bbc39797742  src/evaluation/precision_variants.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  src/features/__init__.py
209baef5d37e477cb16693eec70519c96b77f799ec0b802ebd052943cca486ea  src/features/bandpower.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  src/labels/__init__.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  src/models/__init__.py
6a624cce77efaf6f5fd9ed28ec8235ea7e034ba595dcc1f708676c11560d34d8  src/models/baseline.py
ec58dc7e516c0fc1d7e38b4310fce68bc21f2f08586f390b4261720b21b5b4d5  src/models/baseline_ablated.py
e986085850a63380d478416c606242ff158708ffaf75aab805903b6d12020356  src/models/cnn.py
12b117ba86e5160a4e3c509d436d3015ddf0d43cc1f694c46ffcb1b202ca3084  src/models/cnn_segval.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  src/preprocessing/__init__.py
0ffa4f0b8528d11c2e432a3b1d500b47991aaa69c5fd8998e258b2ea475d338e  src/preprocessing/build_windows.py
a6584b854d658bc0191bd5e49f02df959f25e5819804d9e3a45340aed5a484ab  src/preprocessing/build_windows_a1.py
6c55bd600e923e0db199e84780aeb23e0aaf55a22ddcc6044fa0da05cad9120e  src/preprocessing/diagnostics_a1.py
6057d219a05613779cd2869eedcc4beba21e466e169d4321f5fcb579e8fdda45  src/preprocessing/psd_diagnostic.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  src/utils/__init__.py
```

The DROZY file-format adapter is the only new code permitted, and it may implement only the rules declared above (windowing, bands, µV rejection, label construction). Any other change after this freeze must be logged as a deviation with its reason, before results are seen.

## Amendment 1 — ADOPTED 20 Nov 2026 (panel decision, at my direction)
**Attestation and disclosure.** This amendment is adopted before any DROZY modelling. What has been seen: the KSS values, session counts, the placeholder below, and per-session lapse rates (computed only to verify the reaction-time arm exists). No EEG features, model outputs, or performance numbers have been computed. The amendment is blind to signal; it is not blind to labels, and it says so.
**Order of operations.** (1) Invalid label values are set to missing. (2) Session availability is counted per participant. (3) Alert/fatigued sessions are selected by the rules below. (4) Exclusions are applied. No step uses signal data.
**Missing-value rule (general).** Any KSS value outside the integers 1–9 is missing. The inventory confirms exactly one such value: Participant 7, test 1 = 0, a provider placeholder for a test that never happened — no EDF or PVT file exists for it. Participant 7 therefore has tests 2–3 in both arms, is kept under the general selection rule, and is flagged: their alert session (test 2) carries a smaller dose gap than other participants'.
**Selection and tie rules (both arms, symmetric).** Alert = the session with the lowest label value; fatigued = the highest; the remaining session is excluded as the middle session. Ties at the highest value break toward the LATER test; ties at the lowest value break toward the EARLIER test; if highest = lowest, no contrast exists and the participant is excluded from that arm. The inventory checked ties at both extremes on both arms: the only tie is Participant 11's KSS maximum (tests 2 and 3, both 7) → test 3 is their fatigued session and test 2 is the excluded middle session.
**Basis for dose ordering.** Per the published protocol (Massoz et al. 2016), the three tests run in a fixed order across a period of prolonged wakefulness. Verified against the recorded PVT header timestamps (test 1 ≈ morning of day 1, test 2 ≈ 03:30 on day 2, test 3 ≈ midday on day 2): test index orders cumulative sleep-deprivation dose for every participant.
**Reaction-time arm label rule.** Within-person contrast on session lapse rate (share of responses slower than 500 ms): lowest lapse-rate session = alert, highest = fatigued, middle excluded; the same tie and missing rules apply; a session with no PVT file is missing. Participants 2 and 13 have equal lapse counts across sessions but distinct lapse rates (different trial counts) — ranking uses the rate, as defined here. A participant's selected sessions may differ between the KSS and RT arms; selections are recorded per arm.
**Participant 12 — excluded from both arms and from all pooled steps** (normalisation, feature selection, tuning, controls). Reason, structural: one session cannot form a within-person contrast in either arm; as a held-out fold it would be single-class (balanced accuracy undefined), and in training its label would be confounded with participant identity — the shortcut this design exists to remove.
**Sensitivities (exactly two; robustness checks, never alternative primaries).** S1: Participant 11 excluded (N = 12). S2: Participant 7 excluded from the KSS arm (N = 12 there). If a conclusion changes under a sensitivity, it is reported as not robust.
**Feasibility gate.** The identical locked computation, on the same bases, with realised N = 13 substituted for 14. The recomputed numbers are logged before any modelling. No expected verdict is written into this lock.
**Realised-N note.** 13 participants → 13 leave-one-participant-out folds and a bootstrap over 13 clusters; wide intervals are expected. Cross-dataset comparison keeps the pre-declared traps in view (different N, label types, channel sets).
**Closure.** The inventory checked every participant and session on both arms for ties at both extremes, missing and out-of-range values, and duplicates; the items above are the only deviations. Any further issue found before modelling gets a new dated amendment — never discretion.
**Version record.** Original lock 20 Nov 2026; this revision 20 Nov 2026 after Reviewer 2's confirmation (review 15) and Reviewer 1's conditional approval (review 16, both in `docs/methods_notes.md`). I could not call this one myself, so the decision went to the panel as I directed — their revised version is adopted as written, 20 Nov 2026.

## Amendment 2 — PROPOSED 21 Nov 2026 (post hoc; NOT adopted; do not run until adopted)
**Status and label.** This amendment is proposed after the DROZY LOPO results were seen, and it is labelled post hoc for that reason. It is not adopted. Adoption follows the same panel route as Amendment 1 (both reviewers, then my decision), and the control it describes must not be run before adoption. Nothing in the locked rules or in Amendment 1 changes if this amendment is never adopted.
**What it proposes: a DROZY positive control.** Spectral injection on the adapter side, mirroring the Week 2 FatigueSet control. Take the DROZY windows of a label arm; for each fatigued-class window, scale the theta (4-8 Hz) and alpha (8-12 Hz) bins of each valid channel's own FFT by the square root of the power gain (phase preserved, alert windows untouched), at the two Week 2 doses: full strength theta power x2.0 and alpha power x1.5; half strength theta power x1.5 and alpha power x1.25. The gains are homogeneous (the same for every fatigued window), as in the Week 2 primary control. Features are then re-extracted from the injected windows and passed through the frozen pipeline untouched: same features, same logistic regression, same leave-one-participant-out folds, same participant bootstrap and swap null. Both doses are run under both label arms (KSS and RT). The injection code lives in the adapter's window space, the only new-code surface the freeze allows, and that code is part of what adoption covers; no frozen file changes.
**Why (two purposes, and only two).** (i) The RT arm's minimum detectable effect under its own label structure. The RT arm scored 0.482, and today I cannot say whether that is a small-or-absent effect or an insensitive test. If the pipeline recovers an injected effect of known size under RT labels, the 0.482 becomes a measurement; if it cannot recover even the full dose under RT labels, the RT result must be described as uninformative about effect size. (ii) A recovery curve on DROZY channels to set beside FatigueSet's. The Week 2 control recovered the injected effect at 0.834 (full dose) and 0.695 (half dose) under session-out evaluation. Whether the same injected physiology produces a comparable recovery curve on clinical channels, through this adapter, is the adapter-faithfulness evidence the cross-dataset reading currently lacks.
**Pass criterion, declared in advance.** The control passes if, under KSS labels, the full-dose recovery balanced accuracy (participant-mean LOPO, reported with its bootstrap interval and swap-null p like every other DROZY number) is at least 0.75 and the half-dose recovery is at least 0.60. These thresholds mirror FatigueSet's recovery pattern (0.834 full, 0.695 half) with headroom for the smaller N. The RT-arm recovery is reported at both doses whatever it shows; it is evidence for purpose (i), not part of the pass/fail.
**Scope rule.** If the control fails, every cross-dataset statement is blocked (no DROZY-versus-FatigueSet reading of any kind) and the RT arm's result is reported as uninformative about effect size. A failure never touches the KSS-arm result: that arm's separation from its own swap null does not depend on the control. If the control passes, the RT result may be described as an effect below the control's demonstrated detection floor, still estimation-only. Passing upgrades no claim, changes no locked wording, and does not alter the estimation-only standing set by the feasibility gate.
**Conduct if adopted.** One run: both doses, both arms, reported in full including any failure, in a run folder per combination following the existing convention, with the recovery numbers written into docs/drozy_results.md alongside a statement of which scope rule applies.
