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

## Amendment 1 — PROPOSED 20 Nov 2026 (pre-results; from the data inventory)
Triggered by `docs/DROZY_INVENTORY.md` findings, before any DROZY modelling:
- **A1 — Tie-break.** If a participant's highest or lowest KSS is tied across sessions, the fatigued (resp. alert) session is the tied session with the greater cumulative sleep-deprivation dose — i.e. the later test in protocol order. Rationale: the protocol orders tests by increasing deprivation, so among equal scores the later session is the stronger dose. The rule uses protocol design only, never signal data. (Affects Participant 11: tests 2 and 3 both KSS 7 → test 3 is the fatigued session.)
- **A2 — Participant 12 excluded from both arms.** Only test 1 has data, so no within-person contrast exists for either label construction. Realised N: 13 participants (26 sessions) per arm with A1 applied. The feasibility gate is recomputed at realised N, as §"primary contrast" prescribes.
- **A3 — Placeholder score.** Test 7-1's KSS of 0 is a provider placeholder for a session that was never recorded; treated as missing, not as a score.
Status: proposed — to be confirmed through the usual review before any DROZY model runs. The locked rules above are unchanged.
