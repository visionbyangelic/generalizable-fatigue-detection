# Week 2: Phase A outcome and STOP decision (20 Nov 2026)

## What was done
Phase A (preprocessing + window dataset) was implemented **exactly** as written in
`docs/label_rules.md` §0–§1 by `src/preprocessing/build_windows.py`
(loader: `src/data/load_session.py`). Outputs in `data/processed/`:
`windows.npy` (155, 4, 512) seed-0 M1-capped primary set, `windows_uncapped.npy`
(177, 4, 512) full retained pool, `windows_control.npy` (22, 4, 512),
matching index CSVs, and `preprocessing_log.csv` (per session/block accounting).
Sanity check passed: **0 of 177 windows cross a >100 ms gap**; no window crosses
a run boundary (windows are cut inside runs by construction and verified).

## The problem: the locked rejection rules leave almost no data
Of **1,568 candidate M1/M3 windows, only 199 (12.7%) survive** the §0 rejection
rules. Cause breakdown (M1+M3): amplitude criterion (a) rejects ~87% of
candidates; fit criterion (c) is secondary; flatline (b) rejects none.
Diagnosis (verified directly, `P03/S01` M3): after the locked 1–40 Hz bandpass,
the **temporal channels TP9/TP10 carry genuine broadband contact noise**
(filtered SD ≈ 42 µV, median 1-s peak-to-peak ≈ 225 µV) while frontal AF7/AF8
are clean (SD 5–8 µV). This matches the Week 1 fit statistics (TP9/TP10 "Good"
only ~57–60% of the time). It is not a code artefact: clean sessions
(e.g. P09/S01, P01/S01) pass at 93–96%.

Consequences for the locked protocols:
- **Zero retained M1/M3 windows at all: P04, P05, P06, P07.**
- Single-class only: P03 (2 alert / 0 fatigued), P11 (4/0), P12 (3/0).
- Both classes: only P01 (11/20), P02 (32/23), P08 (5/23), P09 (17/28), P10 (3/6).
- The mandatory **negative control is unusable as specified**: the 4
  non-increasing sessions yield 22 windows, all M3, all from P12/S03, no M1
  windows, nothing from P07/S01, P07/S02 or P10/S03.

## Decision
Per the build instruction ("if a rule proves impossible to implement as written,
STOP that part and document why, do not silently deviate"), **Phases B and C
were NOT run**: the within-subject / session-out evaluations in PLAN §6 are
undefined for 7 of 12 participants on the faithfully-built dataset, and any
numbers produced from the 5 remaining participants would not be the registered
protocol. The locked rules were **not** altered. This needs a dated amendment
to `label_rules.md` (my decision) before Week 2 can proceed.

## Tension inside the locked document (flagged, not resolved here)
`label_rules.md` §0(c) mandates whole-window rejection when a window overlaps
fit=Bad on **any** channel, while the same document's peer-review amendment §2
states "TP9/TP10 Bad does not by itself reject a window; channel-wise masking
applies, and a frontal-only (AF7/AF8) vs all-channel comparison is a
pre-registered LOSO variant." Phase A followed §0 as written (per the build
brief). The amendment's channel-wise/frontal-only route is the obvious
candidate fix, but adopting it is an amendment decision, not an implementer's.

## Amendment options for the decision-maker (not implemented)
1. **Channel-wise masking (amendment §2 route):** reject a window only if a
   *frontal* channel fails (a)/(b)/(c); mask/exclude failing temporal channels
   per window (feature-level for baseline; channel-masked input for CNN), with
   a frontal-only vs all-channel comparison as the pre-registered variant.
2. **Frontal-only primary dataset** (AF7/AF8), temporal channels sensitivity-only.
3. **Per-channel rejection counts with a quota** (e.g. window survives if ≥3 of
   4 channels pass), weaker provenance than 1–2 in the literature we reviewed.
Any option must be written into `label_rules.md` as a dated amendment with
rationale, and Phase A re-run, before Phase B/C.

## Implementation notes / interpretations on record
- "±75 µV peak-to-peak" (§0a) implemented as **peak-to-peak > 150 µV** in any
  1-s sub-window (i.e. ±75 µV about the sub-window centre). The stricter
  alternative reading (p2p > 75 µV) would reject even more; per-window max
  p2p is stored in the index CSVs so either threshold can be audited.
- **Block anchoring:** block intervals are taken from each block's own marker
  pair in epoch ms (`end_baseline→start_activity`, `end_activity→start_fatigue`,
  `end_fatigue→end_session`); no global origin + relative-time mapping is used
  anywhere. The documented +180 s step lives in the experiment files'
  relative-seconds clock, which this pipeline never uses for membership.
  Two attempts to verify marker↔EEG alignment via the treadmill motion burst
  were **inconclusive** (in noisy sessions the |ΔEEG| motion proxy is dominated
  by contact noise, not treadmill movement) and are recorded here rather than
  silently dropped. `preprocessing_log.csv` counts windows in post-step blocks
  of the 8 stepped sessions as `windows_reanchored` (their membership depends
  on per-block anchoring); `windows_lost_clockstep` = 0 (all M1/M3 marker
  pairs are present in all 36 sessions; the 5 missing `start_nback` markers
  do not bound any labelled block).
- 2 corrupt EEG cells in the whole dataset were coerced to NaN and their
  samples treated as run breaks.
- Fit values are matched to EEG samples by nearest timestamp (≤1 s); a window
  is fit-rejected if any matched sample in its span is Bad (4) on any channel.
