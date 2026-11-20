# Label Rules — LOCKED (20 Nov 2026)

> ## STATUS: LOCKED — working rules adopted 20 Nov 2026
> Drafted 20 Nov 2026 from the Week 1 inspection (`docs/dataset_notes.md`), the finalised defaults in `docs/how-others-did-it.md` §6, and the Reviewer 2 peer-review addendum to that document. Approved the same day; her response (recorded in track.md): she did not follow the technical detail, instructed that everything be recorded and the work proceed, and that any different results that come in be recorded too. These rules therefore stand as locked from 20 Nov 2026: thresholds are never tuned on test results, and any change requires a dated amendment below with rationale, plus re-running affected experiments.

---

## 0. Shared rules (all label sets)

- **Windowing:** 4 s **non-overlapping** windows (512 samples after 256→128 Hz downsampling), cut **only within continuous EEG runs** — never across a timestamp gap > 100 ms or a `*2.csv` join (inspection §3: mean gap burden 165 s/session).
- **Filtering:** bandpass 1–40 Hz (zero-phase), per continuous run, before windowing. No notch in the primary pipeline (40 Hz low-pass excludes 50 Hz UK mains); 50 Hz-notch variant is a sensitivity check only.
- **Window rejection (artefact):** drop a window if (a) any 1-s sub-window exceeds **±75 µV peak-to-peak** on any channel, or (b) any channel is **flatline** (SD < 0.5 µV over the window), or (c) the window overlaps a `muse_device_fit` = Bad (4) sample on any channel. (Reviewer 2 addendum: ±75 µV replaces the wet-cap ±100 µV norm for dry electrodes.) **Rejection % is reported per block (M1/M3) per session** so class-balance skew is auditable.
- **Block membership:** a window belongs to block M1/M2/M3 if its centre falls inside that block's marker interval, using **per-block marker anchoring** (inspection §7 — mandatory; a global clock origin misaligns 8/36 sessions by ~180 s). Windows in the baseline-rest, treadmill-activity, and task-switch phases carry **no primary label** (activity = motion-saturated; task-switch windows are used only in explicitly-labelled sensitivity analyses).
- **Leakage clauses (PLAN §7, binding):** every threshold, median, scaler, and class weight is computed on **training folds only**. No per-held-out-participant statistics (the rs2lab per-participant median split is therefore **not** used as-is in LOSO). No interpolation of ratings across blocks (look-ahead; MSARNet-style interpolation is excluded from the primary pipeline).

## 1. Subjective-rating labels (PRIMARY)

- **Scale:** VAS 0–100 (`exp_fatigue.csv`, mental + physical; inspection §5: mental range 0.0–77.2, 108 ratings total).
- **Construction (block inheritance, no interpolation):** windows in the **M1 block** inherit the session's M1 **mental** rating → class **alert**; windows in the **M3 block** inherit the M3 mental rating → class **fatigued**. M2-block windows are excluded from the primary binary task (used in sensitivity analyses M1-vs-M2 and M2-vs-M3).
- **Δ filter (Reviewer 2 addendum, adopted):** a session contributes to the primary subjective experiment **only if Δmental = M3 − M1 > 0** for that session. From the inspection, **32/36 sessions qualify**. Excluded: **P07/S01 (−0.1), P07/S02 (0.0), P10/S03 (−15.4), P12/S03 (−5.1)**.
- **Negative control (mandatory):** the 4 excluded sessions form the **low-fatigue control set**. The trained model is also run on their M1-vs-M3 windows: if it separates those blocks as confidently as in Δ > 0 sessions, the "fatigue" signal is substantially time-on-task / electrode-drift / identity, and Results must say so.
- **Declared limitation:** within a session, block labels are perfectly confounded with time-on-task and phase. This is the dataset's own design (block inheritance as in Shuchih-Negi), is identical across models and splits (so RQ1/RQ2 comparisons stay fair), and is interrogated by the negative control above, the performance label set (§2), and the RQ4 diagnostic (§3).
- **Sensitivity analyses (pre-registered):** (i) M1-vs-M2 and M2-vs-M3 block pairs; (ii) excluding P07 entirely (near-zero reporter: all 9 mental ratings ≤ 0.42); (iii) physical-VAS labels in place of mental.
- **Time alignment:** block intervals from `exp_markers.csv` with per-block anchoring (§0); rating *values* are block-level, so no finer alignment is claimed.

## 2. Performance-based labels (CO-PRIMARY)

- **Measures (fields confirmed in inspection §6):** CRT — `responseTime` (trial-onset-referenced; stimulus-locked RT = responseTime − (stimulusStartTime − trialStartTime)), `isCorrectResponse`; 2-back — `responseTime` on responded trials only (~20%; empty = correct rejection), `isCorrectResponse`. Block summaries per session: mean stimulus-locked CRT RT, CRT accuracy, mean n-back RT, n-back accuracy, for M1/M2/M3.
- **Data-quality exclusion:** **P02/S01 is excluded from all CRT-baseline constructions** (M1 accuracy 0.19 — broken baseline; inspection §6). 29/3,888 CRT trials with corrupt/NaN stimulus delays are dropped from RT means.
- **Expectation (stated in advance):** group-level fatigability is ≈ 0 (CRT ΔRT +0.004 s; n-back Δacc +0.015). Labels are therefore constructed per session from Δ vs M1, and a null result for this arm is a legitimate, reportable outcome.
- **Form A — Zero-Calibration (PRIMARY for LOSO):** thresholds come **only from training subjects**. A post-baseline block (M2, M3) of a session is labelled **fatigued** if its combined performance change vs that session's M1 block crosses the training-derived threshold — operationally: ΔCRT-RT (stimulus-locked) above the **training subjects' median ΔCRT-RT**, or CRT/n-back accuracy drop above the training subjects' median drop (either-criterion rule, fixed here). M1 blocks are labelled **alert**. For within-subject/session-out protocols the same thresholds are computed on the training sessions only. *No information from a held-out participant — not even their M1 block — enters the label rule.*
- **Form B — Baseline-Calibrated (VARIANT, clearly labelled):** as Form A, but each participant's Δ is additionally centred using **their own M1 block** (block identity from `exp_markers.csv` = protocol metadata, not labels). Reported separately as *Baseline-Calibrated LOSO*, never merged with Form A numbers; the gap between A and B is itself reported (it prices the value of a personal baseline).
- **Residualised sensitivity (noted, optional):** regress block performance on M1 level across training subjects and label by residual change, guarding against regression-to-the-mean in simple deltas (Reviewer 2 addendum).
- **Time alignment:** windows inherit their block's label (§0 block membership); no within-block temporal grading is claimed.

## 3. Session labels (RQ4 diagnostic ONLY)

- Label = recording session / intensity condition (low/medium/high from `metadata.csv`), replicating the preliminary experiment's design (99.18% fatigued-class recall) under our three protocols for comparison against §1/§2.
- **Never** presented as fatigue ground truth. Purpose: measure how much of the apparent performance is session/intensity/day artefacts (cf. FATE's phase/time probes). Confound probes accompany it: predict (a) task phase and (b) session intensity from the same features with the same baseline model.

## 4. Agreement analysis (from Week 1 inspection — final numbers)

- Mental vs physical VAS: **r = 0.595** (108 paired ratings); Δ sign agreement **28/36** sessions.
- Subjective vs performance: subjective Δmental averages **+18.0** points while CRT ΔRT averages **+0.004 s** and n-back Δaccuracy **+0.015** — the constructs largely **disagree** at group level, replicating the original paper's non-correlation (p > 0.05). The two label sets are therefore analysed and reported separately throughout, and their disagreement is a finding, not a defect to be engineered away.

## Amendments
| Date | Change | Rationale | Runs affected |
|---|---|---|---|
| 2026-11-20 | Initial PROPOSED rule set (this document) | Week 1 inspection numbers + finalised defaults (`how-others-did-it.md` §6) + Reviewer 2 addendum | None yet — no training has run |

---

## Peer-review amendments (Reviewer 2 review of Week 1 findings, 20 Nov 2026 — adopted, still PROPOSED until I locks)

1. **The 4 non-increasing sessions are NOT a clean "alert control".** Renamed: *non-increasing sessions — drift/sensitivity check set*. Non-increasing ratings may reflect M1 ceiling (already fatigued), rating insensitivity, or disengagement. Actions: audit their M1/M3 trajectories; report LOSO results with and without them; report their per-fold distribution (P07×2, P10×1, P12×1 — uneven across folds).
2. **Channel quality is handled per-channel, not whole-window.** TP9/TP10 Bad does not by itself reject a window; channel-wise masking applies, and a **frontal-only (AF7/AF8) vs all-channel** comparison is a pre-registered LOSO variant. Never interpolate across gaps. Missingness will be reported by block (M1 vs M3) to check informative missingness.
3. **Class balance by construction:** per session, M1 windows are capped/subsampled to match the retained M3 window count; retained counts per class/session are reported in the preprocessing log.
4. **Performance labels are a secondary/exploratory target** (group-level performance did not decline: CRT ΔRT +0.004 s, n-back Δacc +0.015). No claim of "performance-defined fatigue detection" unless per-subject effects are shown. Agreement between subjective and performance label sets reported as Cohen's kappa.
5. **Alignment accounting:** windows lost or re-anchored due to the +180 s clock step are counted and reported per affected session.

---

## AMENDMENT 1 — 20 Nov 2026 (my decision; Reviewer 2 peer-reviewed)
**Replaces §0(c)'s whole-window rejection.** Rationale: the original rule retained only 12.7% of windows and zero windows for P04/P05/P06/P07, because TP9/TP10 carry genuine broadband contact noise (SD ≈42 µV) while AF7/AF8 are clean (5–8 µV) — evidence in `docs/week2_deviation.md`. A rule that deletes a third of the participants is a selection-bias risk (potentially missing-not-at-random if contact degrades with fatigue), so the rejection logic — not the labels — is amended. Label definitions (§1–§4) are unchanged.
**New rejection/masking rules:**
- A window is **retained iff both frontal channels (AF7, AF8) pass** criteria (a) amplitude, (b) flatline, (c) fit on those channels.
- A temporal channel (TP9/TP10) failing any criterion in that window is **masked for that window**: excluded from feature computation (baseline aggregates over valid channels) and flagged with an explicit per-window validity indicator for the CNN (masked input + indicator; never silent unflagged zeros).
- **Primary configuration (locked now, before any LOSO scores exist):** all-valid-channels with temporal masking as above. **Pre-registered variant:** frontal-only (AF7/AF8). The primary is not swapped after seeing results.
- **Mandatory diagnostics:** (i) MNAR check — temporal failure rate by block (M1 vs M3) and over time-on-task, reported; (ii) per-subject retention + masking-% table; (iii) threshold-sensitivity curve (retention vs amplitude threshold) in the appendix.
- The 4 non-increasing sessions remain the drift/sensitivity check set under the same masking rules.
