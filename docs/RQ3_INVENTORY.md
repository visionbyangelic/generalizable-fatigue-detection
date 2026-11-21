# RQ3 INVENTORY: The Empatica E4 peripheral streams, before any multimodal modelling

**Written:** 21 Nov 2026 · **Scope:** Phase 1 of RQ3 (the optional multimodal extension). This document records only what the wrist streams contain, how well they cover the blocks the primary labels live in, and how certainly they can be aligned to the EEG windows. No peripheral feature has been computed for modelling and no model has seen these signals. The design that this inventory feeds is declared in `docs/rq3_preregistration.md`.

**Why this inventory exists.** RQ3 asks whether heart rate and electrodermal activity add anything to EEG for cross-participant fatigue detection. Before that question can be asked honestly, three things about the peripheral data have to be known: whether the streams are actually there during the M1 and M3 blocks, whether their gaps fall evenly across the two classes, and whether a wrist timestamp and an EEG timestamp mean the same moment. All three are properties of the recordings, not of any model, so they are measured here, label-blind, first.

## The streams

Every one of the 36 sessions contains all six Empatica E4 files (verified on disk; the file-presence claim in `docs/DATA_INVENTORY.md` holds for the extracted tree, not just the archive listing):

| File | Content | Rate (README) | Rate (measured median interval) |
|---|---|---|---|
| `wrist_bvp.csv` | Blood volume pulse (PPG) | 64 Hz | 15.6 ms (64 Hz) |
| `wrist_eda.csv` | Electrodermal activity, µS | 4 Hz | 250 ms |
| `wrist_skin_temperature.csv` | Skin temperature, °C | 4 Hz | 250 ms |
| `wrist_acc.csv` | Acceleration, g | 32 Hz | 31.25 ms |
| `wrist_hr.csv` | Heart rate, average over the last 10 s, bpm | 1 Hz | 1000 ms |
| `wrist_ibi.csv` | Inter-beat intervals, ms, one row per detected beat | per beat | n/a |

Continuation (`*2.csv`) files exist for the wrist streams in exactly one session, P02/S01 (all six files). Nowhere else. The loaders used throughout this project (`src/data/load_session.py`) already concatenate continuations in timestamp order, and every number below is computed on the concatenated streams.

Two properties of the E4's own derived products matter for everything that follows. `wrist_hr` is a complete 1 Hz series wherever the band was recording: the firmware smooths over detected beats and keeps emitting values. `wrist_ibi` is the raw detection list, and it is sparse in a way the HR series is not (Section 4). The two products therefore have completely different missingness, and any design that treats "heart data" as one thing would hide that.

## Coverage of the session span

All sensor files in this dataset publish on one epoch-millisecond axis, the same axis as the experiment markers and the Muse EEG. The high-rate streams (BVP, ACC) start within 0.2 s of the `start_baseline` marker in 35 of 36 sessions, and every stream ends within about a second of the `end_session` marker wherever the recording lasted that long (the worst case outside the exceptions below is 1.6 s). The slower products (HR at 1 Hz, EDA and temperature at 4 Hz) place their first and last samples up to about a second inside the span simply because of their sampling grids. The picture is of streams aligned and trimmed to the session span by the dataset creators. The exceptions are specific and worth naming, because they are the whole missing-data story of RQ3:

| Session | What happens | Consequence |
|---|---|---|
| P09/S02 | Stream starts 31.8 s after `start_baseline` | None for the labelled blocks: M1 begins at `end_baseline`, minutes later. Full M1/M3 coverage. |
| P02/S01 | One internal gap of about 170 s in every wrist stream, at the `*2.csv` split | M1 block coverage about 0.51 (EDA) / 0.48 (HR); M3 complete. The recording stopped and restarted mid-session. |
| P08/S02 | Stream ends 256.8 s before `end_session` | M3 block coverage 0.004 (EDA) / 0.016 (HR): effectively no wrist data in M3. |
| P10/S01 | Stream ends 734.8 s before `end_session` | M3 block coverage 0.003 (EDA) / 0.014 (HR): effectively no wrist data in M3. |
| P11/S03 | Stream ends 163.6 s before `end_session` | M3 block coverage 0.004 (EDA) / 0.015 (HR): effectively no wrist data in M3. |
| P06/S02 | Stream ends 1.6 s early | EDA M3 coverage 0.992. Negligible. |

A clock offset cannot produce "starts on the marker, ends early"; these are genuinely shorter recordings, a band that stopped or came off before the session ended. There are no other internal gaps anywhere in the regularly sampled streams: across the remaining sessions, no inter-sample interval exceeds four times the stream's median interval.

Block spans, for scale: M1 runs 175 to 448 s depending on the session, M3 runs 120 to 164 s. Outside the exceptions above, coverage of both blocks is 1.000 for BVP, EDA, temperature, ACC and HR in every session. The full per-session table is at the end of this document.

## What this means at the level of the primary windows

The primary analysis pool is the 1,053 retained Amendment-1 windows (M1 and M3 blocks of the 32 sessions whose subjective rating rose; `data/processed/windows_a1_uncapped_index.csv`). Each window's epoch-ms span is known exactly: the stored `centre_ms` is the exact midpoint of the window's EEG timestamp span for all 1,053 windows (largest discrepancy 0.0 ms), so a wrist sample belongs to a window by plain timestamp overlap, on the same axis the EEG windows were built on.

Applying three draft validity rules to those windows (the rules themselves are fixed in the pre-registration, informed by these counts):

- **HR valid** (at least 2 in-window `wrist_hr` samples with a positive value): **94.3%** of pool windows. By class: alert 99.8%, fatigued 87.9%.
- **EDA valid** (at least 8 in-window samples, the 4 Hz expectation being 16, and a non-flat trace): **94.4%**. By class: alert 100%, fatigued 87.9%. A count-only rule gives the identical number; the trace-amplitude clause never binds. There is no dead-sensor session: exactly-zero EDA samples are at most 1.3% of any session's samples.
- **IBI valid** (at least 3 in-window beats with durations in [300, 2000] ms): **30.7%**. By class: alert 35.8%, fatigued 24.7%. By participant it runs from 7.2% (P12) and 9.5% (P06) up to 67.7% (P07).

The 60 windows that fail the HR/EDA rules are not scattered. 59 of them are fatigued windows: all 27 of P08/S02's M3 pool windows and all 32 of P10/S01's pool windows (that session contributes only M3 windows to the pool), which is precisely the two early-ending streams. The remaining one is an alert window in P02/S01, whose only wrist gap is the restart gap. Per participant, validity is 80.9% for P08 and 71.2% for P10, and at least 99% for everyone else. P11/S03's early ending touches no pool window, because that session's retained pool windows are three M1 windows only.

So the missingness is whole-session and one-sided: a wristband that stopped before the final block. It is not random window dropout, and it is not symmetric across classes (fatigued windows lose their peripheral data at about 12%, alert windows at about 0.2%). Any RQ3 design that silently dropped windows with missing peripheral data would delete most of two participants' fatigued windows and call the remainder a fair test. The pre-registration's missing-data rule is written against exactly this pattern.

## The IBI stream cannot support window-level heart-rate variability

Every IBI duration the E4 emitted, in all 36 sessions, lies inside [300, 2000] ms (0.0% outside). The values are clean; there are simply too few of them. Beat counts per session run from 63 (P04/S03, in a session of about 27 minutes) to 1,765 (P05/S01). The longest run with no detected beat is about 364 s (P06/S02). Under the 10-second-gap criterion, IBI coverage of M1 ranges from 0.17 to 0.99 across sessions and of M3 from 0.07 to 1.00, with most sessions well below 0.9 in at least one block. A 4-second window needs several consecutive detected beats before a variability statistic means anything, and only about 31% of pool windows have even three usable beats. The pre-registration therefore takes its declared fallback: no IBI-derived feature at window level. Heart rate enters through the complete 1 Hz `wrist_hr` product instead.

## Alignment: how certain is the wrist-to-EEG timing?

Three independent observations, in increasing order of strength:

1. **Shared axis, trimmed spans.** The wrist streams begin within about a second of the `start_baseline` marker in 35 of 36 sessions (the high-rate streams within 0.2 s; the slower products within their sampling grids), and end within about a second of `end_session` wherever the recording lasted that long. The one exception is P09/S02's 31.8 s late start, which falls before any labelled block. Whoever prepared this dataset aligned the streams to the marker axis and trimmed them to the session span.
2. **Exact window centres.** The stored centre of every primary pool window is the exact midpoint of that window's EEG timestamps (0.0 ms maximum discrepancy over 1,053 windows), so window membership for wrist samples is computed on the identical axis, not reconstructed from a second clock.
3. **A motion cross-check.** The wrist and forehead accelerometers record the same body's movement. Cross-correlating their detrended magnitude series over the treadmill span (plus 60 s either side), the peak correlation falls at a lag within ±2 s in all 36 sessions (median +0.3 s), and no session's peak sits anywhere near the ±30 s edge of the search range. The correlations themselves are modest (median peak 0.46), which is expected for two different body sites whose motions are related but not identical, so this check bounds any residual offset to a few seconds rather than proving synchrony sample by sample.

**The plain statement.** Alignment is at phase level and it is solid: a wrist sample and an EEG sample carrying the same epoch-ms timestamp belong to the same moment to within a few seconds at worst, and block membership (M1/M3) is exactly as certain for the wrist as it is for the EEG, because both are read off the same markers on the same axis. The streams are not locked to a shared hardware clock, so no sub-second synchrony claim exists, and RQ3 makes none: its features are summaries over 4-second windows of signals that move slowly (heart rate is already a 10-second trailing average; EDA evolves over seconds to minutes), and the blocks they are averaged within run for minutes. A residual offset of even 2 s could not move a window's peripheral content in any material way, and it cannot move a window between blocks at all.

## The other peripheral sources, for the record

- **Zephyr BioHarness** (`chest_physiology_summary.csv`, 1 Hz): present in all 36 sessions (998 to 1,718 rows), carrying heart rate, breathing rate and an SDNN-type HRV over the last 300 beats, each with reliability flags; the HR flag reads unreliable for up to 19.3% of samples in the worst session (P08/S02). It exists and is usable in principle. RQ3 does not use it: the declared peripheral set is the E4's, because EDA exists only on the E4, and mixing a second device (second clock, second missingness pattern, its own reliability flags) into the heart-rate slot would double the alignment and missing-data surface of a question that is supposed to be about whether peripheral signals help at all. Recorded here so the choice is visible rather than silent.
- **Skin temperature** (`wrist_skin_temperature.csv`, 4 Hz): full block coverage wherever the band recorded, session medians from 25.9 °C (P04/S03, which sits at room temperature throughout and reads as a loose or off-wrist band) to about 33 °C. Excluded from the feature set by design; the reason is given in the pre-registration.
- **eSense ear PPG**: present, not inventoried in depth. It is a third heart-rate source with no EDA; out of RQ3's declared scope for the same reason as the Zephyr.

## Per-session coverage table

Start/end offsets are relative to `start_baseline` / `end_session` (seconds), shown for the EDA stream; the other regularly sampled streams agree to within their sampling grids (in P09/S02 the HR product's first sample lands at +41.8 s, after its first averaging window). "Gaps" counts inter-sample intervals over four times the stream median in the BVP stream (all wrist streams agree). Coverage is the fraction of the block span covered, tolerating gaps up to twice the median sample interval (for IBI, gaps up to 10 s between detected beats).

| Session | Start off (s) | End off (s) | Gaps (total s) | EDA cov M1 | EDA cov M3 | HR cov M1 | HR cov M3 | IBI beats | IBI cov M1 | IBI cov M3 |
|---|---|---|---|---|---|---|---|---|---|---|
| P01/S01 | +0.0 | +0.0 | 0 (0) | 1.000 | 1.000 | 1.000 | 1.000 | 900 | 0.96 | 0.76 |
| P01/S02 | +0.2 | -0.1 | 0 (0) | 1.000 | 1.000 | 1.000 | 1.000 | 865 | 0.95 | 0.89 |
| P01/S03 | +0.1 | -0.0 | 0 (0) | 1.000 | 1.000 | 1.000 | 1.000 | 387 | 0.77 | 0.56 |
| P02/S01 | +0.0 | +0.0 | 1 (170) | 0.509 | 1.000 | 0.484 | 1.000 | 474 | 0.21 | 0.90 |
| P02/S02 | +0.1 | -0.0 | 0 (0) | 1.000 | 1.000 | 1.000 | 1.000 | 1179 | 0.84 | 0.89 |
| P02/S03 | +0.0 | +0.0 | 0 (0) | 1.000 | 1.000 | 1.000 | 1.000 | 1011 | 0.79 | 0.88 |
| P03/S01 | +0.0 | +0.0 | 0 (0) | 1.000 | 1.000 | 1.000 | 1.000 | 236 | 0.52 | 0.08 |
| P03/S02 | +0.2 | -0.2 | 0 (0) | 1.000 | 1.000 | 1.000 | 1.000 | 680 | 0.67 | 0.85 |
| P03/S03 | +0.2 | -0.1 | 0 (0) | 1.000 | 1.000 | 1.000 | 1.000 | 506 | 0.42 | 0.79 |
| P04/S01 | +0.0 | +0.0 | 0 (0) | 1.000 | 1.000 | 1.000 | 1.000 | 593 | 0.41 | 0.36 |
| P04/S02 | +0.0 | +0.0 | 0 (0) | 1.000 | 1.000 | 1.000 | 1.000 | 433 | 0.69 | 0.90 |
| P04/S03 | +0.1 | -0.1 | 0 (0) | 1.000 | 1.000 | 1.000 | 1.000 | 63 | 0.17 | 0.24 |
| P05/S01 | +0.0 | +0.0 | 0 (0) | 1.000 | 1.000 | 1.000 | 1.000 | 1765 | 0.67 | 0.95 |
| P05/S02 | +0.2 | -0.2 | 0 (0) | 1.000 | 1.000 | 1.000 | 1.000 | 875 | 0.70 | 0.94 |
| P05/S03 | +0.1 | -0.1 | 0 (0) | 1.000 | 1.000 | 1.000 | 1.000 | 962 | 0.99 | 0.99 |
| P06/S01 | +0.1 | -0.1 | 0 (0) | 1.000 | 1.000 | 1.000 | 1.000 | 255 | 0.52 | 0.33 |
| P06/S02 | +0.2 | -1.6 | 0 (0) | 1.000 | 0.992 | 1.000 | 1.000 | 240 | 0.44 | 0.53 |
| P06/S03 | +0.2 | -0.1 | 0 (0) | 1.000 | 1.000 | 1.000 | 1.000 | 250 | 0.37 | 0.22 |
| P07/S01 | +0.0 | -0.1 | 0 (0) | 1.000 | 1.000 | 1.000 | 1.000 | 917 | 0.79 | 0.85 |
| P07/S02 | +0.1 | -0.0 | 0 (0) | 1.000 | 1.000 | 1.000 | 1.000 | 581 | 0.73 | 0.54 |
| P07/S03 | +0.0 | +0.0 | 0 (0) | 1.000 | 1.000 | 1.000 | 1.000 | 1264 | 0.65 | 1.00 |
| P08/S01 | +0.0 | -0.1 | 0 (0) | 1.000 | 1.000 | 1.000 | 1.000 | 448 | 0.77 | 0.46 |
| P08/S02 | +0.2 | -256.8 | 0 (0) | 1.000 | 0.004 | 1.000 | 0.016 | 507 | 0.75 | 0.08 |
| P08/S03 | +0.0 | +0.0 | 0 (0) | 1.000 | 1.000 | 1.000 | 1.000 | 784 | 0.93 | 1.00 |
| P09/S01 | +0.2 | -0.2 | 0 (0) | 1.000 | 1.000 | 1.000 | 1.000 | 1430 | 0.89 | 0.88 |
| P09/S02 | +31.8 | -0.1 | 0 (0) | 1.000 | 1.000 | 1.000 | 1.000 | 1230 | 0.73 | 0.90 |
| P09/S03 | +0.1 | -0.0 | 0 (0) | 1.000 | 1.000 | 1.000 | 1.000 | 1155 | 0.91 | 0.99 |
| P10/S01 | +0.0 | -734.8 | 0 (0) | 1.000 | 0.003 | 1.000 | 0.014 | 598 | 0.84 | 0.07 |
| P10/S02 | +0.2 | -0.2 | 0 (0) | 1.000 | 1.000 | 1.000 | 1.000 | 391 | 0.58 | 0.07 |
| P10/S03 | +0.0 | -0.1 | 0 (0) | 1.000 | 1.000 | 1.000 | 1.000 | 633 | 0.77 | 0.43 |
| P11/S01 | +0.0 | +0.0 | 0 (0) | 1.000 | 1.000 | 1.000 | 1.000 | 482 | 0.70 | 0.51 |
| P11/S02 | +0.1 | -0.1 | 0 (0) | 1.000 | 1.000 | 1.000 | 1.000 | 312 | 0.49 | 0.42 |
| P11/S03 | +0.1 | -163.6 | 0 (0) | 1.000 | 0.004 | 1.000 | 0.015 | 335 | 0.80 | 0.07 |
| P12/S01 | +0.0 | +0.0 | 0 (0) | 1.000 | 1.000 | 1.000 | 1.000 | 499 | 0.62 | 0.31 |
| P12/S02 | +0.0 | -0.2 | 0 (0) | 1.000 | 1.000 | 1.000 | 1.000 | 318 | 0.63 | 0.24 |
| P12/S03 | +0.1 | -0.2 | 0 (0) | 1.000 | 1.000 | 1.000 | 1.000 | 237 | 0.43 | 0.08 |
