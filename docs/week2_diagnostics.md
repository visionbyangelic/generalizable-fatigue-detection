# Week 2 Diagnostics + CNN Completion — Summary (20 Nov 2026)

**Question under test:** the band-power baseline (Amendment-1 features,
logistic regression, locked discipline) scored at chance — within-subject
balanced accuracy **0.484**, pooled leave-one-session-out **0.515**
(MCC 0.045). Reviewer 2 review 6 ruled this "an unconfirmed checkpoint,
not yet a finding" and required four diagnostics; Reviewer 1 review 7
added three more, corrected the permutation test, and required an
equivalence criterion. This document reports all of them, plus the CNN
within-subject run under segment-grouped validation. All numbers come
from actual runs; run folders follow `experiments/README.md`.

**Decision tree (Reviewer 2, review 6):** within-session separation
works + across-session fails → the signal is session-specific
(drift/impedance/quality), which is itself the project story, properly
evidenced. Both fail → features/labels carry nothing.

**Falsification criterion (stated before these diagnostics ran):** the
null would have been falsified by a session-out balanced accuracy whose
bootstrap CI lower bound exceeded 0.5 together with block-permutation
p < 0.05, or by a positive Δmental dose–response. None occurred.

---

## Part 1 — Baseline diagnostics

### 1. Within-session ceiling check → signal EXISTS inside sessions
Time-blocked folds inside single sessions (3 contiguous chunks per class,
2-window purge; sessions with both classes and ≥20 windows/class, cap
seed 0; 9 sessions). Aggregate over sessions: **balanced accuracy 0.744,
ROC-AUC 0.778, MCC 0.498**. Every eligible session is above 0.59:
P08/S02 0.944, P11/S01 0.875, P11/S02 0.765, P08/S03 0.762, P09/S01 0.750,
P09/S03 0.692, P02/S03 0.690, P12/S02 0.629, P10/S02 0.591.
→ The features DO separate early vs late windows when train and test
come from the same session. (`diagnostics_within_session_ceiling.csv`;
run `2026-11-20_baseline_withinsession_subjective_s0`.)

### 2. Permutation tests → the cross-session score is indistinguishable from no-signal
- **iid within-session shuffle** (200 perms, session-out, seed 0): null
  BA 0.498 ± 0.028; real 0.499 → **p = 0.493**. MCC p = 0.423.
- **Block permutation — circular label shifts within session** (Reviewer 1's
  correction; the citable one): null BA **0.508 ± 0.032**, 95% range
  [0.449, 0.574]; real 0.499 → **p = 0.612**. MCC null 0.013 ± 0.064,
  real 0.009 → p = 0.488. The block null is wider and centred higher
  than the iid null, exactly as predicted (iid is anti-conservative
  under autocorrelation) — and the real score sits *below* its mean.
→ No metric/fold bug is manufacturing the null, and no cross-session
signal hides in it. (`diagnostics_permutation_combined_summary.csv`;
runs `..._permutation_...`, `..._permutation-block_...`.)

### 3. Trivial baselines → the confound is total, and it is not in the EEG features
Session-out, cap seeds 0–4:
- **Window-index only** (normalised position within session): **BA 0.914,
  AUC 1.000, MCC 0.787.** Per-session AUC is exactly 1.0 in all 20
  sessions — necessarily so: block inheritance puts every alert window
  before every fatigued window, so position is a deterministic label
  proxy. This quantifies the declared confound (label_rules.md §1):
  the task as labelled is perfectly solvable by time alone. Position is
  not an EEG feature; this is not pipeline leakage — it is the size of
  the hill any honest EEG model must be judged against.
- **δ-band log-power only** (4 channels): BA 0.544 ± 0.006, AUC 0.580,
  MCC 0.105 — marginally *above* the full 43-feature model, consistent
  with a weak δ component diluted by 39 other features, or noise
  (the permutation null was computed for the full model; treat as a
  hint, not a finding).
(`diagnostics_trivial_baselines.csv`; run `..._baseline_trivial_...`.)

### 4. Δmental dose–response → absent (if anything, negative)
Per held-out session, session-out balanced accuracy vs the session's
Δmental: **Spearman ρ = −0.341, p = 0.141, n = 20** (cap-seed-0 only:
ρ = −0.458, p = 0.042). Sessions where participants reported the largest
fatigue increases were NOT classified better. Per Reviewer 1, at n = 20 this
is descriptive only — but the required direction (positive) is absent.
(`diagnostics_delta_mental.csv`, figure
`results/figures/diagnostics_delta_mental_scatter.png`.)

### 5. P08 deep-dive → domain-shift artefact carried by the mask features
P08's within-subject score (BA 0.225, AUC 0.237 ± 0.032 across 5 cap
seeds; n = 127 test windows) is **not** noise and **not** too-little-data.
Predicted probabilities are confident and inverted — not clustered at
0.5 (≤10% of windows in [0.4, 0.6]): true fatigued windows receive median
P(fatigued) ≈ 0.01–0.08 in all three held-out sessions; true alert
windows in S02/S03 receive median ≈ 0.92–0.96. Fatigued recall: S01 0/130
(pooled seeds), S02 30/135, S03 5/105. The three largest LR coefficients
in P08's within folds are the **mask indicators** — valid_TP10 (+1.03),
both_temporals_masked (−0.91), valid_TP9 (+0.81), all sign-consistent
across folds — while spectral coefficients are weaker and sign-inconsistent
(2/3 folds). Inside P08's own sessions the ceiling check separates at
0.94/0.76. **Verdict: a session-level domain-shift artefact expressed
through session-varying quality/mask patterns — not a stable inverted
physiology.** (`diagnostics_p08_*.csv`.)

### 6. Feature sanity → no degenerate features
NaNs occur only where Amendment 1 designs them: TP9 features masked in
78.5% of windows, TP10 in 75.6%, temporal ratios in 72.5% (imputed with
training-fold medians). No continuous feature is near-constant (min SD
0.038); the only zero-variance-style flags are the binary indicators
themselves. (`diagnostics_feature_sanity.csv`.)

---

## Panel additions (Reviewer 1 review 7)

### 7. Mask-only baseline → the shortcut matches the full model
Labels predicted from the 3 validity indicators + coverage counts ONLY
(no EEG), session-out, seeds 0–4: **BA 0.530, AUC 0.521, MCC 0.034** —
statistically indistinguishable from the full 43-feature EEG model
(0.515 / 0.549 / 0.045). Whatever label information transfers across
sessions in this pipeline is carried at least as well by the masking
pattern as by the spectra. This is the shortcut ceiling every later
model must beat to claim spectral content.
(`diagnostics_maskonly.csv`; run `..._baseline_maskonly_...`.)

### 8. Positive control → the pipeline is NOT blind (dose–response recovered)
Synthetic fatigue-like effect injected into fatigued windows only
(spectral gain on each valid channel's own FFT, phase preserved; alert
windows and masked channels untouched). Manipulation check passed:
observed frontal log-power shifts 0.292/0.181 vs targets 0.301/0.176.
Recovered, session-out, cross-seed means:
- **Full strength** (θ power ×2.0, α ×1.5): **BA 0.834, AUC 0.902, MCC 0.639**
- **Half strength** (θ ×1.5, α ×1.25): **BA 0.695, AUC 0.768, MCC 0.360**
A graded effect produces graded recovery. Had a transferable θ/α effect
of even the half strength existed in the real labels, this pipeline
would have seen it. (`diagnostics_positive_control*.csv`;
run `..._baseline_positivecontrol_...`.)

### 9. Leakage-gap demonstration → the split is worth +0.07 BA here
Identical features/model/data (cap seed 0); only the split changes.
Random stratified window-level 5-fold (pooled across sessions and
participants): **BA 0.571, AUC 0.609, MCC 0.141**. Honest session-out:
0.499 / 0.493 / 0.009. The leakage gap (+0.072 BA, +0.116 AUC, +0.132
MCC) is real but modest on this dataset — and even the leaky number is
nowhere near the 0.9+ figures this project's RQ1/RQ4 critique targets,
because the within-session signal itself is only partial (ceiling 0.744
with purged time-blocked folds). (`diagnostics_leakage_gap.csv`;
run `..._baseline_leakygap_...`.)

### 10. Equivalence → "at chance" is now a supported claim, scoped
Bootstrap (10,000 resamples over units) on the headline balanced
accuracy, against the pre-specified SESOI of 0.58:
- **Session-out:** 0.515, 95% CI **[0.468, 0.555]** — upper bound < 0.58 ✓
- **Within:** 0.484, 95% CI **[0.390, 0.562]** — upper bound < 0.58 ✓
Both protocols are equivalent to chance within the smallest effect of
interest. Scope, per Reviewer 1's warning: this claims non-decodability for
**this feature family + this evaluation on this dataset** — not that
fatigue is undecodable from Muse S in general. Label validity remains
a rival explanation (Δmental is a self-report proxy; §4 of
label_rules.md documents the subjective/performance disagreement).
(`diagnostics_equivalence.csv`; run `..._baseline_equivalence_...`.)

### 11. Coverage by class → retention is class-asymmetric; reported, not hidden
Primary (Δmental > 0) sessions, M1/M3 candidate windows:
| class | candidates | A1-retained | coverage | capped primary | coverage |
|---|---|---|---|---|---|
| alert (M1) | 816 | 564 | 69.1% | 317 | 38.8% |
| fatigued (M3) | 652 | 489 | 75.0% | 489 | 75.0% |
Amendment-1 retention itself favours fatigued windows by ~6 points
(post-outcome selection — one reason the mask pattern carries label
information, cf. §7); the further alert reduction is the by-design M1
cap. Per-participant table in `diagnostics_coverage_by_class.csv`
(run `..._baseline_coverage_...`).

---

## Part 2 — CNN (EEGNet-8,2) within-subject, segment-grouped validation

Protocol as specified by the project lead (Reviewer 2-endorsed with caveats):
test = one held-out session per participant; validation = whole
contiguous run-segments carved from the training sessions with a
2-window purge; temporal channel dropout (p = 0.2) in training;
mask-indicator input planes; train-fold-only normalisation; seeds {0,1};
≤100 epochs, patience 15. **This validation measures segment
generalisation within known sessions, NOT unseen-session transfer** —
that caveat travels with every number below.

**Coverage: 7 of 12 participants** (P02, P03, P04, P06, P08, P09, P11).
Uncovered, with reasons logged in the run notes: P01 (both folds — the
single training session's 18 windows cannot fund a ≥6-window both-class
validation plus a ≥20-window post-purge subtrain), P07 and P05 (no
both-class train/test arrangement exists; P05 has no Amendment-1 windows
at all), P10 and P12 (remaining training sessions are single-class).
Individual folds also uncoverable: P02/S03, P03/S01, P04/S03, P06/S02.
Partial coverage to note when citing: P02 rests on one test session
(n = 6 windows), P03 on one (n = 2 — its 0.750 is uninterpretable and
must not be quoted), P04 on one (n = 10), P06 on one (n = 22).

**Results** (aggregate = mean over covered participants; per-participant
values are cross-seed means):
- Seed 0: BA 0.467, macro-F1 0.335, AUC 0.566, MCC −0.071
- Seed 1: BA 0.519, macro-F1 0.511, AUC 0.513, MCC +0.059
- Per participant: P09 0.594 (MCC 0.188), P02 0.583 (n = 6), P03 0.750
  (n = 2, disregard), P06 0.452, P04 0.429, P11 0.353 (MCC −0.297),
  P08 0.290 (MCC −0.440).

The CNN does not beat the baseline on the same protocol (0.484): it is
at chance on aggregate, with large seed variance, and it reproduces the
baseline's two inverted participants (P08, P11) — the inversion is a
property of the data/sessions, not of the model family. Per Reviewer 2's
anticipation, "CNN ≤ baseline" is itself a reportable Week-2 finding.
(Runs `2026-11-20_cnn_within_subjective_s0/s1`; the stopped
session-grouped attempt is preserved in the `*_STOPPED_sessionval`
folders.)

---

## Interpretation against the decision tree

Reviewer 2's branch 1 is the outcome — **with Reviewer 1's joint-reading caveat
attached**. Within-session separation is real (0.744), cross-session
transfer is absent (inside the block-permutation null; equivalence to
chance within 0.58 supported for both headline protocols), and the
pipeline demonstrably detects graded spectral effects when they exist
(positive control). But the within-session separability cannot be
credited to fatigue physiology: window position alone is a perfect
ranking proxy by construction, and the mask/coverage pattern alone
matches the full EEG model across sessions. The defensible Week-2 claim
is therefore narrow and strong: **for this feature family and this
evaluation, the early-vs-late spectral differences in FatigueSet are
session-bound — they do not survive a change of session, and the part
of the signal that does transfer is indistinguishable from recording-
quality structure.** That is precisely the failure mode the project was
designed to expose in the preliminary 99.18% result (RQ4), now evidenced
rather than asserted.

## Problems encountered (documented, none silently absorbed)
1. First diagnostics run crashed at the Δmental figure
   (`textcoords="offset_points"` is invalid in matplotlib 3.6; must be
   `"offset points"`). Fixed in `diagnostics.py`; sections 4–5 were
   completed by `src/evaluation/diagnostics_followup.py` without
   re-running the 7-minute permutation test.
2. sklearn `SimpleImputer` warnings in some small training segments
   (a temporal feature all-NaN within a segment); the pipeline handles
   train/test consistently — same behaviour as the main baseline.
3. CNN session-grouped validation was structurally infeasible on this
   dataset (prior stopped run, preserved for audit); the segment-grouped
   replacement still leaves 5 participants and 6 folds uncoverable under
   the pre-set feasibility criteria — logged per fold, never forced.

## Ablation: EEG-features-only — the mask-shortcut causal test (Reviewer 2 review 10)

**Why this run exists.** The mask-only baseline (session-out BA 0.530)
matching the full 43-feature model (0.515) showed EEG adds nothing
*beyond* the mask across sessions — but correlation is not causation:
it did not prove the indicators were responsible for any part of the
full model's behaviour. This ablation re-runs the identical baseline
(same Amendment-1 windows, cap seeds 0–4, folds, train-fold-only
imputation/scaling/class weights; code `src/models/baseline_ablated.py`,
which imports the full model's pipeline verbatim) with the 3
mask/validity-indicator features removed — 40 EEG spectral features
only. Masked channels' spectral features remain NaN → training-fold
median, exactly as in the full model; the model is simply no longer
told which channels were masked. Reviewer 2's prediction: session-out
collapses to ~0.50 and P08's inversion disappears.

**Results** (per-seed aggregates, mean ± SD across seeds 0–4; run
folders `2026-11-20_baseline_<protocol>_subjective_ablated_s<seed>`,
summaries `results/tables/baseline_<protocol>_subjective_ablated_summary.csv`):

| Protocol | Model | Bal. acc. | Macro-F1 | ROC-AUC | MCC |
|---|---|---|---|---|---|
| Session-out | Full (43 feat) | 0.515 ± 0.018 | 0.421 ± 0.023 | 0.548 ± 0.047 | 0.045 ± 0.043 |
| Session-out | Mask-only (5 feat) | 0.530 ± 0.006 | 0.386 | 0.521 | 0.034 |
| Session-out | **Ablated (40 feat)** | **0.519 ± 0.009** | **0.426 ± 0.010** | **0.566 ± 0.045** | **0.056 ± 0.023** |
| Within | Full (43 feat) | 0.484 ± 0.011 | 0.440 ± 0.016 | 0.487 ± 0.027 | −0.032 ± 0.024 |
| Within | **Ablated (40 feat)** | **0.509 ± 0.009** | **0.470 ± 0.018** | **0.509 ± 0.032** | **0.019 ± 0.025** |
| Within, P08 only | Full | 0.225 ± 0.022 | — | 0.237 | −0.580 |
| Within, P08 only | **Ablated** | **0.467 ± 0.024** | — | **0.490** | **−0.066** |

**Reading.**
1. **P08: prediction confirmed, decisively.** The confident inversion
   (BA 0.225, AUC 0.237, MCC −0.580, every seed ≤ 0.245) collapses to
   chance the moment the indicators are removed (BA 0.467, AUC 0.490 =
   no ranking signal, MCC −0.066). The inversion was *caused* by the
   mask/validity features — full stop. This also explains the small
   within-subject aggregate rise (0.484 → 0.509): P08's recovery alone
   contributes (0.467 − 0.225)/8 ≈ +0.030, i.e. all of it; for the
   other 7 participants the indicators changed nothing on average.
2. **Session-out aggregate: no collapse — because there was nothing to
   collapse.** Ablated (0.519), full (0.515) and mask-only (0.530) are
   indistinguishable (all well inside the block-permutation null 95%
   range [0.449, 0.574]). The full model was never scoring above the
   EEG-only model, so removing the indicators removes no performance.
   Reviewer 2's "~0.50" prediction lands numerically (0.519 is chance), but
   via equivalence of all three feature sets at chance level, not via
   a drop from the full model.

**Verdict.** The strong aggregate form of "the transferable residue was
the mask shortcut" is *not* supported — there is no above-chance
cross-session residue for any feature set. The refined claim, which
this ablation does establish causally, is stronger where it matters:
**no feature set transfers across sessions above chance; EEG spectral
features and the recording-quality pattern are interchangeable at
chance level (EEG contributes nothing beyond the mask); and the only
systematic cross-session predictions demonstrated anywhere in this
pipeline — P08's confident inversion — were produced by the
mask/validity features and vanish without them.** The mask is not
carrying a hidden positive signal; it is the only component shown to
act across sessions at all, and when it acts, it acts as an artefact.

**Problem hit during this run (documented).** scikit-learn was absent
from this VM's Python (the Week-2 environment had not persisted);
reinstalled scikit-learn 1.9.1 — the same version as the original
runs — and verified faithfulness before ablating: the full model
reproduced its stored seed-0 numbers exactly (session-out BA 0.4991 /
MCC 0.0090; within P08 BA 0.245; within aggregate 0.4970). Only then
was the ablation run.

---

## Closure pack (Reviewer 1 review 11 requirements) — 20 Nov 2026

Consultation 11 accepted the diagnostics above but required six more
items before Week 2 can close, and corrected the reading of §7 and §10
(see the conclusion section below, which supersedes the §10 wording).
All items below were run under the lock-doc rules (Amendment-1 windows,
cap seeds 0–4, train-fold-only discipline); code:
`src/evaluation/closure_pack.py`.

### 12. Frontal-only (F) baseline — the tier table's missing column
Feature set (exact, 20 features): log + relative band powers of AF7 and
AF8 (4 bands each) + the 4 frontal ratios (computed from AF7/AF8 mean
band powers only; no temporal channel enters any of them). No temporal
features, no mask indicators. Results (cross-seed grand means):
- **Within-subject: BA 0.535**, AUC 0.536, MCC 0.072 (8 participants —
  same coverage as P).
- **Pooled session-out: BA 0.553**, AUC 0.605, MCC 0.106 (20 sessions).
F sits slightly above P on both Week-2 protocols (P: 0.484 / 0.515) —
reported as estimates with the LOPO intervals below; no tier-table
verdict is entered, because the equivalence side of the table awaits
my confirmed Δ (§15). (`baseline_*_subjective_frontal_summary.csv`;
runs `2026-11-20_baseline_{within,session-out}_subjective_frontal_s0..s4`.)

### 13. Leave-one-PARTICIPANT-out (LOPO) — the participant-clustered headline
Sessions are nested in 12 participants, so session-out alone overstates
the effective sample. LOPO (test = all sessions of one participant,
train = the other 11), cap seeds 0–4, aggregate = mean over
participants of cross-seed per-participant BA, CI by participant-level
bootstrap (10,000 resamples):
- **P (masked all-channel): BA 0.508, 95% CI [0.471, 0.542]**, AUC 0.541,
  MCC 0.018 (11 participants).
- **F (frontal-only): BA 0.544, 95% CI [0.500, 0.589]**, AUC 0.571,
  MCC 0.091 (11 participants).
Per-participant values are in `baseline_loso_subjective*_a1_summary.csv`
(P range: P10 0.382 … P07 0.582; F range: P10 0.408 … P04 0.680).
**P05 has no LOPO fold** — mechanism documented in §17, item 2.
(Runs `2026-11-20_baseline_loso_subjective(_frontal)_s0..s4`;
`closurepack_lopo_aggregate.csv`.)

### 14. Positive control, hardened — heterogeneous injection + MDE
- **Heterogeneous injection** (per-session random amplitudes centred on
  the half-strength dose: θ gain ~ LogNormal(ln 1.5, 0.55), α gain ~
  LogNormal(ln 1.25, 0.45); ~20% of sessions null-or-opposite by design,
  seed 20261120; realised draw: 26 graded, 3 opposite, 0 null of 29
  sessions — see §17, item 3): recovered **BA 0.657, AUC 0.750,
  MCC 0.317** (cross-seed means), vs 0.695 for the homogeneous
  half-strength dose and 0.515 uninjected. The pipeline recovers a
  realistic, session-varying effect, not only the best case.
  Manipulation check passed (observed vs target mean log10 frontal
  shifts: θ 0.071/0.072, α 0.032/0.027). Per-session recovery vs
  injected log θ gain: Spearman ρ = 0.259 (descriptive).
- **Minimum detectable effect (MDE):** extending the homogeneous dose
  curve downward — θ/α power gains (2.0, 1.5) → BA 0.834, CI
  [0.774, 0.885]; (1.5, 1.25) → 0.695 [0.631, 0.756];
  (1.25, 1.125) → 0.562 [0.509, 0.614]; (1.1, 1.05) → 0.495
  [0.452, 0.534] (session-level bootstrap CIs over cross-seed
  per-session means). Against the block-permutation null band
  (q975 = 0.574), the smallest tested dose whose CI clears the band is
  **θ ×1.5 / α ×1.25** — the MDE for this pipeline/protocol. A
  homogeneous cross-session θ/α effect smaller than that could exist
  undetected; at or above it, it would have been seen.
  (`diagnostics_positive_control_{doses,mde,hetero,hetero_gains,
  hetero_check}.csv`; runs `..._positivecontrol-mde_...`,
  `..._positivecontrol-hetero_...`.)

### 15. Feasibility-gate inputs — NO verdict declared
Per the lock doc §2, Δ is mine to confirm; this section supplies
only the gate's inputs (SEs from label-free sources; Δ_min = (z.95 +
z.90)·SE = 2.9265·SE for ~90% power, one-sided α = 0.05, true effect 0):
| SE basis | SE | Δ_min |
|---|---|---|
| Permutation null SD, session-mean statistic, circular shifts (n = 200) | 0.0319 | 0.093 |
| Permutation null SD, session-mean statistic, block-length-8 (n = 100) | 0.0327 | 0.096 |
| Permutation null SD, participant-mean statistic, circular (n = 200) | 0.0323 | 0.094 |
| Permutation null SD, participant-mean statistic, block-8 (n = 100) | 0.0342 | 0.100 |
| Analytical, per-session BA SD 0.104 / √24 | 0.0213 | 0.062 |
| Analytical, per-session BA SD 0.104 / √20 (observed folds) | 0.0233 | 0.068 |
| Analytical, per-participant LOPO BA SD 0.064 / √12 | 0.0184 | 0.054 |

The permutation SD is robust across block structures (0.032–0.034).
Conditional reading (no verdict): IF Δ = 0.05 were locked, equivalence
would be **unattainable under every basis** (all Δ_min > 0.05); IF
Δ = 0.08, testable under the analytical bases only; IF Δ = 0.10,
testable under all but the block-8 participant basis; the lock doc's
hard ceiling is Δ ≤ 0.08. Full conditional table:
`diagnostics_gate_delta_table.csv`; per-permutation statistics:
`diagnostics_gate_permutation_nulls.csv`. (Run
`2026-11-20_baseline_gate-inputs_subjective_s0`.)

### 16. Leakage gap with CI; with/without P08
- **Leakage gap:** headline aggregate +0.072 BA (pooled leaky OOF
  0.571 − honest mean-of-folds 0.499, seed 0). Paired per-session gap
  (same session under both splits): **+0.052, 95% CI [−0.019, +0.121]**
  (bootstrap over the 20 paired sessions). The interval includes zero:
  at session level the gap is positive on average but not precisely
  estimated at this N. No literature-contrast framing is attached
  (review 11, item 6). (`diagnostics_leakage_gap_{ci,
  paired_sessions}.csv`; run `..._leakygap-ci_...`.)
- **Without P08:** session-out BA 0.515 → **0.530** (17 sessions);
  LOPO P 0.508 → **0.514** [0.474, 0.548]; LOPO F 0.544 → **0.543**
  [0.494, 0.592]. Removing the artefact participant moves every
  headline by ≤ 0.015 — the null reading is not a P08 artefact.
  P08 remains IN all headline numbers. (`closurepack_without_p08.csv`;
  run `..._withoutp08_...`.)
- **Mask-ablation (EEG-only features, parallel run):** removing the 3
  validity indicators (40 spectral features): session-out BA **0.519**
  (vs 0.515 with indicators), within **0.509** (vs 0.484), and P08's
  within-subject BA rises from 0.225 to **0.467** — the inversion
  disappears when the model cannot see the mask, confirming §5's
  mechanism. (`baseline_*_subjective_ablated_summary.csv`; runs
  `2026-11-20_baseline_*_subjective_ablated_s0..s4`.)

---

## Definitions table — every headline number, one canonical definition

All rows: band-power features + logistic regression (C = 1.0,
class_weight = balanced), Amendment-1 windows, subjective labels,
unless the feature-set column says otherwise. "Fold-mean" = mean over
folds of per-fold balanced accuracy; "cross-seed" refers to the M1-cap
subsample seeds 0–4.

| Value | Metric | Protocol | Seeds | Aggregation | Feature set |
|---|---|---|---|---|---|
| 0.499 | BA | session-out | 0 only | fold-mean over 20 held-out sessions | P, 43 feats |
| 0.515 | BA | session-out | 0–4 | grand mean (20 sessions × 5 seeds; = mean over sessions of cross-seed per-session means) | P, 43 feats |
| 0.484 | BA | within-subject | 0–4 | mean over 8 participants of per-participant BA (per participant: BA on concatenated out-of-fold predictions, cross-seed mean) | P, 43 feats |
| 0.535 | BA | within-subject | 0–4 | as 0.484 | F, 20 frontal feats |
| 0.553 | BA | session-out | 0–4 | as 0.515 | F, 20 frontal feats |
| 0.508 | BA | LOPO | 0–4 | mean over 11 participants of cross-seed per-participant BA (test fold = the participant's pooled sessions) | P, 43 feats |
| 0.544 | BA | LOPO | 0–4 | as 0.508 | F, 20 frontal feats |
| 0.530 | BA | session-out | 0–4 | mean over seeds of per-seed fold-means | mask-only (3 indicators + 2 coverage counts, no EEG) |
| 0.519 | BA | session-out | 0–4 | as 0.515 | EEG-only ablation (40 spectral feats, no indicators) |
| 0.571 | BA | random window split (leaky demo) | 0 only | POOLED out-of-fold BA over all 806 windows (not a fold-mean) | P, 43 feats |
| 0.744 | BA | within-session ceiling | 0 only | mean over 9 eligible sessions of per-session out-of-fold BA (time-blocked folds, 2-window purge) | P, 43 feats |
| 0.914 | BA | session-out | 0–4 (identical each seed) | mean over seeds of per-seed fold-means | window-index only (AUC = 1.000) |
| 0.544 | BA | session-out | 0–4 | mean over seeds of per-seed fold-means | δ-band log-power only |
| 0.834 | BA | session-out, positive control | 0–4 | mean over seeds of per-seed fold-means | P, 43 feats; injected θ ×2.0 / α ×1.5 |
| 0.695 | BA | session-out, positive control | 0–4 | as 0.834 | P, 43 feats; injected θ ×1.5 / α ×1.25 |
| 0.657 | BA | session-out, positive control | 0–4 | as 0.834 | P, 43 feats; heterogeneous per-session injection (§14) |
| 0.225 | BA | within-subject, P08 only | 0–4 | cross-seed mean of P08's per-seed out-of-fold BA | P, 43 feats |
| 0.467 / 0.519 | BA | CNN within-subject (segment-grouped validation) | CNN seeds 0 / 1 (not cap seeds) | mean over 7 covered participants of per-participant BA | EEGNet-8,2 on raw windows + mask planes |

Two clarifications the panel required, stated once here:
- **Dose–response (§4):** Δmental DID rise — mean +18.0 VAS points,
  positive in 32 of 36 sessions (Week 1 inspection) — so the labels
  rest on a real reported fatigue increase. What §4 shows is only that
  per-session classification accuracy does not *track the size* of
  that rise. Label validity is therefore not the explanation §4
  supports; it remains a rival explanation only via the coarseness of
  a 3-ratings-per-session self-report (label_rules.md §1).
- **Mask vs time:** the MNAR diagnostic (`mnar_diagnostic.csv`) shows
  temporal-channel failure rates FALL across the session — 89.3% of
  windows in first halves vs 85.1% in second halves (all sessions;
  primary-only: 89.8% vs 85.8%) — and are lower in M3 than M1 blocks.
  The mask pattern is therefore partly a time-on-task proxy (contact
  settling), which is the mechanism behind §3/§5/§7's behaviour.

---

## Conclusion — Week 2 (wording adopted from Reviewer 1 review 11)

Within sessions, temporal drift is decodable (ceiling 0.744) — but
position alone ranks perfectly within sessions, so that separability
is time-on-task structure, not evidence of a fatigue signal. Across
sessions, no model exceeds the block-permutation null: the full
43-feature model (session-out 0.515, LOPO 0.508), the frontal-only
model (0.553, LOPO 0.544), the EEG-only ablation (0.519), and the
mask-only model (0.530) all sit inside or beside the null band
[0.449, 0.574], and the mask-only result does NOT establish a
transferable shortcut — it is itself inside the null; the mask story
is local to P08, whose inversion is a mask-carried domain-shift
artefact (ablated: 0.467). The pipeline is not blind: homogeneous
θ/α effects are recovered with a graded dose–response (MDE
θ ×1.5 / α ×1.25), and a heterogeneous per-session effect is
recovered at 0.657. The CNN shows no evidence of an advantage over
the baseline at this N (underpowered; not "at/below baseline" as a
settled comparison). The defensible Week-2 statement is therefore:
**no detectable cross-session transfer for this feature family and
evaluation; effects up to ~0.56 balanced accuracy cannot be excluded
at N = 24 sessions.** The equivalence verdict itself awaits my
confirmed SESOI Δ plus the feasibility gate (§15): the 0.58-based
equivalence stated in §10 must NOT be cited as null support
(review 11, item 3), and §10 is superseded by this section.

## Problems encountered in the closure pack (documented)
1. First diagnostics figure crash and imputer warnings — see the
   earlier "Problems encountered" section; unchanged.
2. **P05's absence mechanism corrected.** The closure pack's first
   LOPO notes claimed P05's fold was "skipped by the both-classes
   fold rule". Verification against the data showed the real
   mechanism is upstream: P05's only retained session (S01) is
   single-class (5 alert / 0 fatigued), so the per-session M1 cap
   (cap M1 to the retained M3 count = 0) reduces P05 to zero windows
   in EVERY capped dataset — P05 is absent from all capped analyses
   (baseline, diagnostics, LOPO alike), not merely skipped in LOPO.
   The LOPO run notes were corrected in place; the same absence
   applies retroactively to the Week-2 baseline coverage statements.
3. **Heterogeneous-injection realised draw:** with the fixed seed,
   the ~20% null-or-opposite design realised as 3 opposite and
   0 null sessions of 29 (P(no null) ≈ 5% under the design). The
   run stands as drawn (seed fixed in advance); the realised
   composition is reported in §14 and in the gains table rather than
   re-drawn, which would have been seed-shopping.
4. §10's equivalence-at-0.58 statement is superseded by the
   conclusion above (review 11); the §10 text is retained
   unedited for audit, as is the stopped CNN run.

---

## 18. Pre-declared precision variants V1 / V2 — SENSITIVITIES, estimates only (20 Nov 2026)

Declared in advance in track.md (20 Nov 2026) and the lock doc's
status note, after I confirmed Δ = 0.05 ("confirm the 55% line
and try to make it better"), with "better" scoped before any run to
*measuring more precisely under the same honest rules* — the line
does not move and no result below changes the primary (P) result or
the estimation-only standing of the equivalence claim. **Reporting
status is now settled by the lead reviewer's ruling (Reviewer 1,
review 12; lock §6): the gate governs, the primary stays
estimation-only, and a variant could earn the prescribed sensitivity
sentence only if it BOTH attained its gate at Δ = 0.05 AND cleared
0.55. No variant below does either completely** — every variant's
Δ_min exceeds 0.05 on every basis (V1 0.081–0.092; V2 0.102–0.118),
and the only sub-0.55 upper bound (V1-P session-out, 0.533) belongs
to a variant whose gate is unattained. The variant results therefore
stand as reported estimates under R2/R3, and Week 2's equivalence
business is closed. Code:
`src/evaluation/precision_variants.py`. All runs are single
deterministic passes (no cap, no subsampling; LR deterministic;
folder suffix s0 denotes that pass). CIs: bootstrap over units
(10,000 resamples, rng 42) — session-level for session-out,
participant-level for LOPO. Gate permutations: one common set of 200
circular shifts (rng 60000+p) reused across all feature sets and
protocols, so null SDs are paired across variants.

### V1 — uncapped pool + class weighting
Full Amendment-1 retained pool (1,053 windows: 564 alert /
489 fatigued), NO per-session M1 cap; imbalance handled by
training-fold balanced class weights (sklearn `class_weight=
'balanced'`: w_c = n_train / (2 · n_c), fit inside each training
fold — the same Pipeline as the primary). Stated design difference
vs the capped primary: single-class sessions, which the cap reduced
to zero windows, contribute their windows to TRAINING pools here
(they still cannot be test folds: session-out has the same 20
both-class test sessions; LOPO the same 11 both-class participants,
with P05's 5 alert windows now present in other folds' training
data instead of absent entirely).

| Feature set | Protocol | BA [95% CI] | Macro-F1 [CI] | AUC [CI] | MCC [CI] |
|---|---|---|---|---|---|
| P (43) | session-out (20 sess) | 0.495 [0.451, 0.533] | 0.398 [0.352, 0.442] | 0.545 [0.456, 0.639] | −0.004 [−0.097, 0.078] |
| P (43) | LOPO (11 part) | 0.503 [0.445, 0.552] | 0.407 [0.344, 0.469] | 0.552 [0.454, 0.653] | 0.005 [−0.116, 0.110] |
| F (20) | session-out (20 sess) | 0.550 [0.509, 0.605] | 0.450 [0.400, 0.513] | 0.601 [0.523, 0.683] | 0.100 [0.021, 0.201] |
| F (20) | LOPO (11 part) | 0.547 [0.503, 0.596] | 0.449 [0.395, 0.517] | 0.606 [0.538, 0.674] | 0.100 [0.012, 0.201] |

Against the capped primaries (P session-out 0.515, CI [0.468, 0.555]
from §10's table; P LOPO 0.508 [0.471, 0.542]; F session-out 0.553;
F LOPO 0.544 [0.500, 0.589]): V1-P point estimates sit slightly
BELOW the capped ones (0.495 / 0.503), with a marginally narrower
session-out interval (width 0.082 vs 0.087) but a WIDER LOPO
interval (0.107 vs 0.071) — restoring the discarded alert windows
did not tighten the participant-level estimate, because that
interval is driven by between-participant spread, not window count.
V1-F is numerically almost identical to capped F. Per-participant
LOPO values are in the summary tables; the widest single value is
P04 under P features (BA 0.284), against 0.725 for the same
participant under F — reported as an observation, not interpreted
here. (`baseline_{session-out,loso}_subjective_V1uncapped(_frontal)
_summary.csv`; runs `2026-11-20_baseline_{session-out,loso}
_subjective_V1uncapped(-frontal)_s0`.)

### V2 — within-session-normalised features (P, 43)
Each continuous feature (40 of 43) centred/scaled per session with
that session's own windows (nanmean / nanstd, ddof=0; labels never
used); indicator features unchanged; Pipeline imputation/scaling
still training-fold only, on top. **Transductivity, stated as
required wherever V2 is reported: the normalisation uses each
session's own unlabelled windows — for a test session, that
session's own windows are included in its normalisation statistics —
so V2 is transductive at session level by design** (Reviewer 2
review 10 declared sensitivity). Guards fired as documented
in the code: of 30 sessions × 40 continuous features = 1,200
(session, feature) pairs, 224 were all-NaN in their session (left
NaN; training-fold imputation downstream — these are temporal
features in sessions where the channel was masked in every window)
and 96 had SD 0 or undefined (centred only).

| Protocol | BA [95% CI] | Macro-F1 [CI] | AUC [CI] | MCC [CI] |
|---|---|---|---|---|
| session-out (20 sess) | 0.583 [0.498, 0.665] | 0.525 [0.442, 0.610] | 0.627 [0.523, 0.726] | 0.152 [−0.003, 0.307] |
| LOPO (11 part) | 0.556 [0.480, 0.613] | 0.503 [0.429, 0.567] | 0.596 [0.489, 0.701] | 0.099 [−0.046, 0.205] |

V2's point estimates are the highest in this document (session-out
+0.088 over V1-P; LOPO +0.053) and its intervals are also the widest
(session-out width 0.167; LOPO 0.133) — and its permutation null SDs
are the largest of any feature set below (Δ_min 0.102–0.118), i.e.
the same normalisation that lifts the point estimate also widens
the null distribution of the statistic. All three facts are inputs
for the ruling; none is interpreted here. (`baseline_{session-out,
loso}_subjective_V2sessnorm_summary.csv`; runs `..._V2sessnorm_s0`.)

### V2 mask-only arm (Reviewer 1 ruling R3)
The 5 indicator/coverage features of §7's mask-only set (valid_TP9,
valid_TP10, both_temporals_masked, n_valid_temporals,
n_valid_channels), built from the index, NOT session-normalised
(indicators/coverage are not continuous EEG features), on the same
uncapped pool and V2 protocols — so any V2 change can be attributed
between EEG features and the mask shortcut:

| Protocol | BA [95% CI] | Macro-F1 [CI] | AUC [CI] | MCC [CI] |
|---|---|---|---|---|
| session-out (20 sess) | 0.532 [0.477, 0.595] | 0.414 [0.325, 0.506] | 0.537 [0.480, 0.601] | 0.040 [−0.091, 0.181] |
| LOPO (11 part) | 0.544 [0.499, 0.593] | 0.418 [0.310, 0.521] | 0.509 [0.451, 0.561] | 0.080 [−0.036, 0.197] |

Attribution inputs (estimates only): under the same pool and
protocols, the mask-only arm sits at 0.532 / 0.544 against the V2
EEG arm's 0.583 / 0.556 — a point-estimate separation of +0.051
(session-out) and +0.012 (LOPO), with intervals overlapping
throughout; the mask-only arm is itself close to its capped §7
value (0.530, session-out). Two LOPO folds of the mask-only arm
(P06, P07) score exactly 0.500 with MCC 0: within those
participants' retained windows the mask features carry no
class-varying prediction (single-class predictions). (`baseline_
{session-out,loso}_subjective_V2sessnorm_maskonly_summary.csv`;
runs `..._V2sessnorm-maskonly_s0`.)

### Gate inputs per variant (NO verdict)
Circular-shift permutation null SDs (n = 200, common set) and
Δ_min = 2.9265 · SD (lock §2 formula), beside the observed BA
interval and the confirmed SESOI line (upper bound 0.55). These are
inputs for the lead reviewer's ruling; nothing here applies the
tier table.

| Variant | Basis (statistic / folds) | Null SD | Δ_min | Observed BA [CI] | CI upper vs 0.55 |
|---|---|---|---|---|---|
| V1 P | session-mean / session-out | 0.0295 | 0.086 | 0.495 [0.451, 0.533] | 0.533 — below |
| V1 P | participant-mean / session-out | 0.0314 | 0.092 | 0.495 [0.451, 0.533] | 0.533 — below |
| V1 P | participant-mean / LOPO | 0.0307 | 0.090 | 0.503 [0.445, 0.552] | 0.552 — above by 0.002 |
| V1 F | session-mean / session-out | 0.0275 | 0.081 | 0.550 [0.509, 0.605] | 0.605 — above |
| V1 F | participant-mean / session-out | 0.0305 | 0.089 | 0.550 [0.509, 0.605] | 0.605 — above |
| V1 F | participant-mean / LOPO | 0.0287 | 0.084 | 0.547 [0.503, 0.596] | 0.596 — above |
| V2 P | session-mean / session-out | 0.0396 | 0.116 | 0.583 [0.498, 0.665] | 0.665 — above |
| V2 P | participant-mean / session-out | 0.0404 | 0.118 | 0.583 [0.498, 0.665] | 0.665 — above |
| V2 P | participant-mean / LOPO | 0.0348 | 0.102 | 0.556 [0.480, 0.613] | 0.613 — above |
| V2 mask | session-mean / session-out | 0.0232 | 0.068 | 0.532 [0.477, 0.595] | 0.595 — above |
| V2 mask | participant-mean / session-out | 0.0204 | 0.060 | 0.532 [0.477, 0.595] | 0.595 — above |
| V2 mask | participant-mean / LOPO | 0.0324 | 0.095 | 0.544 [0.499, 0.593] | 0.593 — above |

For reference, the capped primary's gate inputs (§15) were Δ_min
0.093–0.100 on the permutation bases. V1's permutation Δ_min values
(0.081–0.092) are slightly smaller than the capped ones; V2's
(0.102–0.118) are larger; every variant's Δ_min on every basis
remains above the confirmed Δ = 0.05. Full table:
`variants_gate_inputs.csv`; per-permutation statistics:
`variants_gate_permutation_nulls.csv`; run folders
`2026-11-20_baseline_gate-inputs_subjective_{V1uncapped,
V1uncapped-frontal,V2sessnorm,V2sessnorm-maskonly}_s0`.

### Problems encountered in the variant runs (documented)
1. **Summary-CSV CI columns dropped on first write.** The first
   version of the summary writer assigned the aggregate row with
   `.loc`, which silently discarded the CI columns (they are not
   among the per-unit columns). The point estimates were unaffected
   and the CIs existed in the gate table; the writer was fixed
   (row concatenation instead of `.loc` assignment) and all eight
   summary tables were regenerated from the stored run metrics —
   the regenerated CIs reproduce the originals exactly (bootstrap
   rng 42), so no analysis was re-run and no number changed.
2. No other problems: all folds, guards and skips behaved as
   documented above; every skipped fold is listed in its run's
   notes.md.
