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
(drift/impedance/quality), which is itself the thesis story, properly
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
nowhere near the 0.9+ figures this thesis's RQ1/RQ4 critique targets,
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
quality structure.** That is precisely the failure mode the thesis was
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
