# Dataset Notes — FatigueSet Week 1 Inspection Findings

**Date:** 20 Nov 2026 · **Code:** `src/data/inspect_dataset.py` · **Notebook:** `notebooks/01_data_inspection.ipynb` (executed, outputs embedded)
**Data:** `data/raw/fatigueset/` (extracted from the verified `fatigueset.zip`; see `DATA_INVENTORY.md`) · **Per-session numbers:** `results/tables/inspection_summary.csv`
**Companion docs:** `fatigueset-research.md` (what the literature/documentation claims) — this document is what the data itself says.

---

## 0. Headline verdicts

1. **Structure is complete:** all 36 sessions (12 participants × 3) contain all 39 core files. Nothing is missing at file level.
2. **Δmental > 0 in 32/36 sessions** — the adopted primary subjective-label set (block labels, Δ-filtered) covers 32 sessions; 4 are excluded by the filter (listed in §5).
3. **EEG sampling is as specified** (256 Hz nominal; median 251 samples per wall-clock second) but the recordings are **gap-ridden**: mean 165 s per session sits inside >100 ms timestamp gaps; windowing must stay within continuous runs.
4. **Alignment is exact — but only per-block.** A single global clock origin would misalign later blocks by ~180 s in 8/36 sessions (§7). Per-block anchoring to markers makes every checked offset < 1 s.
5. **Performance fatigability is essentially absent at group level** (CRT ΔRT = +0.004 s; n-back Δaccuracy = +0.015), replicating the original paper's null. The performance-label arm will be weak/noisy and must be reported per-participant.
6. **Frontal fit is good, temporal fit is poor.** AF7/AF8 are Good ~96–97% of the time; TP9/TP10 only ~57–60%. One session (P11/S03) never has all four channels Good simultaneously.

## 1. Structure

- `metadata.csv` (note: UTF-8 BOM — read with `utf-8-sig`) maps each participant's session numbers to intensity by Latin square; verified for all 36 sessions.
- Every session directory holds the canonical 39 files (13 Muse, 6 E4, 6 eSense, 9 Zephyr, 5 experiment). Session totals: mean 21.66 min (SD 3.29, range 17.15–29.00) — matches the paper's 21.24 min.

## 2. Session phases (from `exp_markers.csv`, 17 events/session)

| Phase | Definition (markers) | Mean | Range |
|---|---|---|---|
| Baseline rest | start_baseline → end_baseline | 3.00 min | 3.00–3.00 (fixed) |
| M1 block | end_baseline → start_activity | 4.03 min | 2.92–7.47 |
| Treadmill activity | start_activity → end_activity | 2.95 min | 1.31–3.26 |
| M2 block | end_activity → start_fatigue | 4.72 min | 2.82–10.50 |
| Task-switch (S3) | start_fatigue → end_fatigue | **4.23 min** | 2.40–8.12 |
| M3 block | end_fatigue → end_session | 2.30 min | 2.00–2.73 |

> ⚠️ **Surprise vs the paper:** the paper describes S3 task-switching as "~15 min"; the markers — the authoritative record — put it at **4.23 min on average**. The protocol still works (ratings rise, §5), but any window budget planned around a 15-min S3 must be revised down.

## 3. Muse EEG (`forehead_eeg_raw.csv`)

- **Channels:** TP9, AF7, AF8, TP10 in every session; no NaN cells anywhere (0 in 36 files).
- **Sampling rate:** median samples per wall-clock second = **250.9** (range 243–253 across sessions) against the nominal 256 Hz. Timestamps are BLE-packet-jittered (consecutive-sample diffs: 63% 0 ms, 15% 1 ms, tail to ~13 ms within continuous data; 76.3% of timestamps are duplicates of the previous sample). **Conclusion: sample index, not timestamp deltas, is the time base** — exactly as the research doc predicted.
- **Span-based rate is misleading:** (n−1)/span averages only 234 Hz because of gaps (next point) — do not use it.
- **Gaps:** every session has >100 ms gaps; total gap time averages **165 s/session**. Worst: **P04/S01 (567 s, including the dataset's largest single gap, 547.7 s ≈ 9.1 min)**, P12/S01 (560 s), P06/S01 (515 s). Gaps >1 s occur in 27/36 sessions. → Windows must be cut **within continuous runs only**; a gap map per session is a Week 2 preprocessing deliverable.
- **Coverage vs markers:** EEG start coincides with `start_baseline` (median offset 0.0 s) in most sessions; exceptions: **P02/S01 starts 107.1 s late**, P07/S01 18.1 s late. EEG end coincides with `end_session` (median 0.0 s) except **P09/S02, whose EEG stops 244.2 s (~4.1 min) before the session ends**.
- **P04/S01 anomaly:** `forehead_eeg_raw.csv` is **empty (0 data rows)**; the entire EEG lives in `forehead_eeg_raw2.csv` (269,081 rows) which carries a stray leading index column starting at 33,520. Its timestamps span the session from baseline (start offset ≈ 0 s). Loader rule (concatenate base + `*2`, drop stray index column, sort by timestamp) handles it — verified in the inspection loader.

## 4. Fit quality and artefact events (`muse_device_fit.csv`, 1=Good / 2=Medium / 4=Bad)

| Channel | % Good (mean) | Worst session | % Bad (mean) |
|---|---|---|---|
| TP9 | 56.9% | 13.4% | 17.0% |
| AF7 | 96.0% | 55.3% | 2.3% |
| AF8 | 96.9% | 69.4% | 2.2% |
| TP10 | 59.9% | **0.0%** | 13.7% |
| **All 4 Good simultaneously** | **52.0%** (SD 27.1) | **0.0% (P11/S03)** | — |

Worst sessions by all-channel Good fit: **P11/S03 0.0%** (TP10 Bad 85.7% of the time — treat TP10 as effectively absent there), P05/S02 6.8%, P05/S01 13.4%, P07/S01 14.2%, P06/S01 14.9%. Eleven sessions sit below 50% all-Good. The proposed rejection rule (drop windows overlapping fit=Bad on any channel, `label_rules.md`) will therefore remove very different amounts of data per session — per-block rejection rates must be reported (Reviewer 2 addendum point 6).

Events: blinks mean **994/session** (521–1,561); jaw clenches mean **103/session** (1–683; worst P05/S03 683, P07/S01 533) — jaw-event density tracks the poor-fit sessions, consistent with temporalis/frontalis artefact load.

## 5. Subjective labels (`exp_fatigue.csv` — 3 rows/session, 108 total)

- Mental VAS: mean 26.4, SD 19.7, median 23.1, range **0.0–77.2**. Physical VAS: mean 23.5, SD 18.7, range 0.0–77.2 (the identical maxima are a coincidence of the scale conversion).
- **Agreement:** Pearson r(mental, physical) = **0.595** across the 108 paired ratings; Δ sign agreement in 28/36 sessions — related but not interchangeable constructs.
- **Δmental (M3−M1):** mean +18.0, SD 19.5, range −15.4 to +74.7. **> 0 in 32/36 sessions** → the primary subjective experiment (§label rules) keeps 32 sessions.
- **Excluded by the Δ filter (4):** P07/S01 (−0.1, low), P07/S02 (0.0, medium), P10/S03 (−15.4, low), P12/S03 (−5.1, high). These become the **low-fatigue negative-control** pool.
- By intensity: Δmental mean low +20.3 (10/12 > 0), medium +12.0 (11/12), high +21.8 (11/12) — the mental-fatigue rise is **not** monotone in treadmill intensity, a useful check against "session = intensity" shortcuts.
- **P07 is a near-zero reporter:** mental VAS 0.00–0.42 across all 9 ratings. Two of their three sessions fail the Δ filter; the third (Δ = +0.28) passes on a technically-positive but negligible change — flag for sensitivity analysis.

## 6. Performance labels (`exp_crt.csv`, `exp_nback.csv`)

Trial counts are exact in every session: CRT 36/block, n-back 20/block; blocks present for all 36 sessions.

**CRT.** As recorded, `responseTime` averages ~1.66 s — it is referenced to **trial onset**, and the stimulus appears a median **1.205 s** later. Stimulus-locked RT (responseTime − stimulus delay, after dropping 29/3,888 corrupt or NaN-delay trials): **M1 0.451 s, M2 0.440 s, M3 0.454 s**. ΔRT (M3−M1) = **+0.004 s** (SD 0.072; slower in 19/36 sessions — a coin flip). Accuracy is at ceiling: 0.951 / 0.970 / 0.961; Δaccuracy +0.010 (SD 0.131, inflated by one outlier below).

**Broken baseline:** **P02/S01 CRT M1 accuracy = 0.19** (vs 0.97/0.94 in M2/M3) — the M1 block is not a usable baseline (likely initial key-mapping confusion); its Δaccuracy of +0.75 is an artefact. P02/S01 must be excluded from CRT-baseline constructions (its n-back and subjective data are unaffected).

**2-back.** Accuracy 0.975 / 0.981 / 0.990 — hard ceiling; Δaccuracy = **+0.015** (worse in only 4/36 sessions). RT on responded trials (~20% of trials; the rest are correct rejections with empty `responseTime`): 0.643 / 0.638 / 0.602 s; ΔRT = **−0.041 s** (participants got *faster*). 

**Verdict:** at group level there is no performance fatigability signal to speak of — the same null the original paper reported. Performance labels (label rules §2) are therefore built per-participant from Δ vs M1 with training-only thresholds, and the performance arm's results must be reported per-participant with the null expectation stated up front. The disagreement between subjective (Δmental mean +18) and performance (Δ ≈ 0) labels is itself a result, consistent with the paper's non-correlation finding.

## 7. Alignment — two clocks

- **Clock model verified:** `epoch_ms ≈ start_experiment + relative_s × 1000`, checked on all 108 fatigue submissions and 108 CRT block starts: median offset **0.06–0.07 s**.
- **The catch:** in **8/36 sessions** the relative clock steps by **+176 to +186 s at exactly one block boundary** (between M1→M2 in P11/S01, P12/S01; between M2→M3 in P02/S03, P04/S02, P05/S01, P07/S03, P08/S03, P10/S01). The step ≈ the 3-min activity duration — the relative clock evidently excludes the treadmill period in those sessions, booked at different boundaries. A single-origin alignment would silently misplace whole blocks by ~3 min in those sessions.
- **Rule adopted:** anchor **each measurement block to its own markers** (`submit_survey`, `start_crt`, `start_nback` of that block). With per-block anchoring, every verified offset is < 1 s.
- **Missing markers:** exactly **5** empty timestamps in the whole dataset — the **third block's `start_nback`** in **S01 of P01–P05** (the known P01/S01 defect is a five-participant pattern, not a one-off). Impact is limited: that block's n-back start must be interpolated from neighbouring markers if ever needed; block anchoring for fatigue/CRT is unaffected (their markers are present). No other marker is missing anywhere (17 events in every file).

## 8. Continuation (`*2.csv`) files

65 files across 9 sessions (P02/S01, P02/S02, P04/S01, P05/S02, P06/S01, P06/S02, P09/S02, P10/S01, P10/S03) — mostly earable files. **Only P04/S01 touches EEG** (all six forehead EEG files incl. raw; §3). Eleven of P04/S01's continuations carry the stray leading index column; other sessions' continuations do not. Loader contract: for every base file, append `*2` if present, drop a stray index column if the header starts empty, sort by timestamp, and treat the join as a potential gap (check the timestamp step across the join).

## 9. Quality flags & exclusion recommendations

No session is excluded wholesale at inspection. Session-level handling that the pipeline must encode:

| Issue | Sessions | Handling |
|---|---|---|
| Δmental ≤ 0 | P07/S01, P07/S02, P10/S03, P12/S03 | Excluded from primary subjective set; used as negative controls |
| Broken CRT M1 baseline (acc 0.19) | P02/S01 | Excluded from CRT-baseline label constructions only |
| TP10 effectively absent (fit) | P11/S03 | Keep session; fit-based window rejection will gut TP10 — report per-channel yield |
| All-channel Good fit < 25% | P11/S03, P05/S02, P05/S01, P07/S01, P06/S01, P06/S02, P10/S03, P05/S03 | Keep; rejection rates reported per block (label rules) |
| Heavy EEG gap burden (>200 s) | P04/S01, P12/S01, P06/S01, P07/S01, P02/S01, P05/S01, P01/S01, P08/S01, P09/S02, P10/S01, P12/S03 | Keep; windows only within continuous runs |
| EEG late start / early stop | P02/S01 (+107 s), P09/S02 (−244 s at end) | Keep; window yield per block will reflect it |
| Near-zero subjective reporter | P07 (all sessions) | Sensitivity analysis without P07 |

## 10. Per-session summary

Full machine-readable version: `results/tables/inspection_summary.csv` (also carries physical VAS, blink/jaw counts, n-back ΔRT, CRT M1 accuracy).

| P | S | Intensity | Session min | EEG min | Samples/s (med) | Gap s | Fit Good % | Mental M1 | Mental M2 | Mental M3 | Δmental | CRT ΔRT (s) | N-back Δacc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P01 | S01 | low | 26.85 | 26.30 | 251 | 227 | 87.1 | 14.2 | 11.2 | 45.8 | +31.7 | +0.128 | +0.050 |
| P01 | S02 | medium | 22.70 | 22.34 | 252 | 148 | 78.6 | 33.5 | 48.3 | 34.0 | +0.6 | +0.085 | +0.050 |
| P01 | S03 | high | 22.43 | 22.06 | 252 | 86 | 67.6 | 4.0 | 37.5 | 35.8 | +31.8 | -0.045 | +0.000 |
| P02 | S01 | low | 23.84 | 21.67 | 251 | 354 | 26.6 | 1.0 | 1.5 | 48.5 | +47.5 | -0.049 | +0.000 |
| P02 | S02 | high | 19.90 | 19.37 | 251 | 84 | 65.9 | 2.5 | 15.1 | 77.2 | +74.7 | -0.025 | +0.000 |
| P02 | S03 | medium | 22.39 | 22.05 | 251 | 10 | 81.0 | 33.1 | 36.8 | 53.5 | +20.4 | +0.070 | +0.000 |
| P03 | S01 | high | 20.48 | 20.10 | 251 | 38 | 44.7 | 24.3 | 38.3 | 36.8 | +12.5 | +0.005 | +0.000 |
| P03 | S02 | low | 19.29 | 18.91 | 252 | 115 | 31.3 | 21.7 | 28.6 | 28.5 | +6.8 | +0.180 | +0.100 |
| P03 | S03 | medium | 18.47 | 18.09 | 250 | 8 | 50.4 | 15.7 | 30.4 | 19.6 | +3.9 | -0.003 | -0.050 |
| P04 | S01 | high | 27.21 | 26.77 | 252 | 567 | 36.3 | 11.8 | 11.8 | 31.8 | +20.0 | -0.116 | +0.000 |
| P04 | S02 | medium | 24.58 | 24.20 | 250 | 330 | 57.6 | 9.4 | 8.6 | 18.5 | +9.0 | -0.001 | +0.000 |
| P04 | S03 | low | 18.36 | 17.99 | 253 | 86 | 33.8 | 35.0 | 35.6 | 59.4 | +24.4 | +0.009 | +0.000 |
| P05 | S01 | medium | 29.00 | 28.63 | 252 | 207 | 13.4 | 26.2 | 19.4 | 46.4 | +20.1 | +0.056 | +0.000 |
| P05 | S02 | high | 19.23 | 18.89 | 252 | 148 | 6.8 | 18.9 | 24.6 | 25.7 | +6.8 | -0.060 | +0.000 |
| P05 | S03 | low | 17.84 | 17.48 | 252 | 132 | 19.5 | 4.4 | 6.1 | 18.9 | +14.4 | -0.148 | +0.000 |
| P06 | S01 | medium | 22.53 | 22.17 | 244 | 515 | 14.9 | 6.8 | 2.4 | 22.5 | +15.7 | -0.050 | +0.050 |
| P06 | S02 | low | 18.77 | 18.40 | 252 | 23 | 20.8 | 1.1 | 0.3 | 9.0 | +7.9 | +0.028 | +0.050 |
| P06 | S03 | high | 18.38 | 17.95 | 251 | 88 | 62.0 | 6.2 | 6.9 | 16.7 | +10.4 | -0.040 | +0.050 |
| P07 | S01 | low | 22.01 | 21.35 | 249 | 376 | 14.2 | 0.1 | 0.3 | 0.0 | -0.1 | -0.033 | +0.100 |
| P07 | S02 | medium | 20.98 | 20.58 | 251 | 202 | 64.2 | 0.0 | 0.4 | 0.0 | +0.0 | +0.131 | +0.000 |
| P07 | S03 | high | 23.90 | 23.33 | 252 | 77 | 17.7 | 0.0 | 0.1 | 0.3 | +0.3 | +0.066 | +0.000 |
| P08 | S01 | low | 19.00 | 18.62 | 251 | 227 | 70.7 | 12.1 | 19.6 | 62.2 | +50.1 | +0.022 | +0.050 |
| P08 | S02 | high | 17.15 | 16.63 | 252 | 10 | 72.3 | 18.5 | 57.9 | 70.7 | +52.2 | +0.021 | +0.000 |
| P08 | S03 | medium | 20.95 | 20.58 | 252 | 68 | 71.0 | 17.1 | 27.5 | 35.1 | +18.1 | +0.069 | +0.000 |
| P09 | S01 | high | 26.40 | 26.04 | 252 | 38 | 71.9 | 52.2 | 57.1 | 71.4 | +19.2 | -0.084 | +0.050 |
| P09 | S02 | low | 22.52 | 17.29 | 251 | 219 | 85.4 | 20.7 | 47.1 | 74.4 | +53.8 | +0.020 | +0.150 |
| P09 | S03 | medium | 19.40 | 19.02 | 250 | 15 | 90.1 | 18.6 | 35.8 | 63.9 | +45.3 | -0.019 | +0.000 |
| P10 | S01 | high | 27.77 | 27.27 | 251 | 241 | 74.7 | 4.9 | 17.6 | 32.1 | +27.2 | +0.003 | +0.050 |
| P10 | S02 | medium | 19.67 | 19.29 | 251 | 10 | 62.5 | 35.0 | 38.3 | 38.1 | +3.1 | -0.016 | -0.050 |
| P10 | S03 | low | 17.82 | 17.44 | 251 | 297 | 21.2 | 34.6 | 30.4 | 19.2 | -15.4 | +0.029 | -0.100 |
| P11 | S01 | medium | 26.35 | 25.95 | 252 | 29 | 78.7 | 26.1 | 19.3 | 27.9 | +1.8 | +0.003 | +0.000 |
| P11 | S02 | high | 21.96 | 21.26 | 252 | 2 | 73.5 | 10.7 | 14.0 | 21.9 | +11.2 | -0.004 | -0.050 |
| P11 | S03 | low | 19.27 | 18.89 | 252 | 171 | 0.0 | 13.6 | 22.2 | 23.6 | +10.0 | +0.087 | +0.000 |
| P12 | S01 | medium | 25.48 | 24.89 | 243 | 560 | 51.5 | 67.8 | 62.5 | 74.3 | +6.5 | +0.014 | +0.050 |
| P12 | S02 | low | 18.76 | 18.39 | 251 | 2 | 86.6 | 21.5 | 31.0 | 33.5 | +11.9 | -0.118 | +0.000 |
| P12 | S03 | high | 18.24 | 17.84 | 250 | 214 | 67.6 | 37.6 | 36.8 | 32.5 | -5.1 | -0.017 | +0.000 |

## 11. What this changes (plan impact)

1. **Label filter is viable:** 32/36 sessions pass Δmental > 0 — the primary subjective experiment loses only 4 sessions (PLAN Week 1 assumption holds).
2. **Windowing design must be gap-first:** with a mean 165 s/session inside gaps and one 9.1-min hole, "window within continuous runs" moves from hygiene to a core preprocessing component; per-session window yields will vary a lot.
3. **Per-block marker anchoring is mandatory** in the loader (8 sessions would otherwise be misaligned by ~3 min) — this is now part of the alignment spec, not an optional check.
4. **S3 window budget shrinks:** the task-switch phase is ~4.2 min, not ~15 — M3-block windows (the "fatigued" class) come mostly from the ~2.3-min M3 measurement block and the S3 tail; class counts must be computed before promising balanced classes.
5. **Performance arm expectations reset:** group-level Δ ≈ 0 on both tasks. The performance-label experiment is a per-participant analysis with a plausible null outcome — which, if it holds under LOSO, is itself a publishable contrast with the subjective arm (and with RQ4's session labels).
