# DATA INVENTORY — What Actually Arrived

**Downloaded:** 20 Nov 2026 · **Source:** Official Google Drive link from https://www.esense.io/datasets/fatigueset (no form, no sign-in, no personal details required)
**Local file:** `data/raw/fatigueset.zip` (kept out of git) · A copy of this inventory also lives at `data/raw/DATA.md`.

## Archive verification

| Check | Result |
|---|---|
| File size | 641,686,295 bytes (642 MB) |
| MD5 | `a358db31c27a8c7e0a1ef255b7c6ba28` |
| SHA-256 | `2db4c9d785307a309f66a404d90a5dbdbe1a45d0a954f980204dd39e2a7fd8ca` |
| Format | Zip (store/uncompressed entries), `file` verified |
| CRC integrity | `zipfile.testzip()` → **None (all 1,473 files pass)** |
| Entries | 1,522 total (1,473 files + 49 directory entries) |
| Uncompressed total | 3,380.7 MB |
| Published checksum from authors | None exists — hashes above are ours |

Note: the archive was **listed and sampled, not fully extracted** (3.38 GB). Files read directly from the zip for verification: root `README.md`, `metadata.csv`, and for P01/S01 + P01/S03: `exp_fatigue.csv`, `exp_markers.csv`, `exp_crt.csv`, `exp_nback.csv`, `exp_task_switch.csv`, plus heads of `forehead_eeg_raw.csv`, `muse_device_fit.csv`, `wrist_eda.csv` (extracted samples in `data/raw/extracted_sample/`).

## Root files (4)

| File | Size | Contents |
|---|---|---|
| `README.md` | 7,993 B | Full data dictionary (per-file columns, units, sampling rates) |
| `metadata.csv` | 213 B | Participant → low/medium/high session mapping (counterbalancing) |
| `pre_task_survey.xlsx` | 15,054 B | Per-session SSS sleepiness + GVAS vigour/affect |
| `preliminary_questionnaire.xlsx` | 17,769 B | Demographics, BFI-10, MCTQ, Fatigue Severity Scale, IFIS fitness |

## Per-participant inventory

Every participant has **all 3 sessions**, and **every one of the 36 sessions contains all 39 core files** (verified against the archive listing — no missing EEG, no missing experiment files anywhere).

| Participant | Sessions (intensity) | Files | Size (MB) | Extra `*2.csv` continuations |
|---|---|---|---|---|
| 01 | 01 (low), 02 (medium), 03 (high) | 117 | 324.4 | 0 |
| 02 | 01 (low), 02 (high), 03 (medium) | 129 | 280.6 | 12 |
| 03 | 01 (high), 02 (low), 03 (medium) | 117 | 263.4 | 0 |
| 04 | 01 (high), 02 (medium), 03 (low) | 134 | 291.6 | 17 |
| 05 | 01 (medium), 02 (high), 03 (low) | 123 | 291.9 | 6 |
| 06 | 01 (medium), 02 (low), 03 (high) | 129 | 258.7 | 12 |
| 07 | 01 (low), 02 (medium), 03 (high) | 117 | 292.6 | 0 |
| 08 | 01 (low), 02 (high), 03 (medium) | 117 | 255.4 | 0 |
| 09 | 01 (high), 02 (low), 03 (medium) | 123 | 296.1 | 6 |
| 10 | 01 (high), 02 (medium), 03 (low) | 129 | 286.8 | 12 |
| 11 | 01 (medium), 02 (high), 03 (low) | 117 | 274.8 | 0 |
| 12 | 01 (medium), 02 (low), 03 (high) | 117 | 264.3 | 0 |
| **Total** | **36 sessions** | **1,473** | **3,380.7** | **65** |

Intensity mapping is from `metadata.csv` (session number → intensity differs per participant by Latin-square counterbalancing).

## The 39 core files in every session

**Muse S (forehead) — 13:** `forehead_eeg_raw.csv` (256 Hz, TP9/AF7/AF8/TP10, µV), `forehead_eeg_{alpha,beta,delta,gamma,theta}_abs.csv` (10 Hz), `forehead_acc.csv`, `forehead_gyro.csv` (52 Hz), `muse_blinks.csv`, `muse_jaw_clenches.csv`, `muse_device_fit.csv`, `muse_device_battery.csv`, `muse_device_touch.csv`

**Empatica E4 (wrist) — 6:** `wrist_bvp.csv` (64 Hz), `wrist_eda.csv` (4 Hz), `wrist_skin_temperature.csv` (4 Hz), `wrist_acc.csv` (32 Hz), `wrist_hr.csv` (1 Hz), `wrist_ibi.csv`

**eSense earable — 6:** `ear_acc_{left,right}.csv`, `ear_gyro_{left,right}.csv`, `ear_ppg_{left,right}.csv` (all 100 Hz)

**Zephyr BioHarness (chest) — 9:** `chest_raw_ecg.csv` (250 Hz), `chest_raw_breathing.csv` (25 Hz), `chest_raw_acc.csv` (100 Hz), `chest_rr_interval.csv`, `chest_bb_interval.csv`, `chest_physiology_summary.csv`, `chest_sensor_summary.csv`, `zephyr_activity_summary.csv`, `zephyr_device_status.csv` (1 Hz summaries)

**Experiment — 5:** `exp_fatigue.csv` (3 rows: M1/M2/M3 physical + mental VAS 0–100), `exp_markers.csv` (17 phase events, epoch ms), `exp_crt.csv` (108 trials), `exp_nback.csv` (60 trials), `exp_task_switch.csv` (200 trials, induction task)

## Continuation (`*2.csv`) files — loader must handle

65 extra files where a recording evidently split/restarted; the `*2.csv` continues the corresponding base file and must be concatenated in timestamp order:
- Earable files (6 per affected session): P02/S02, P05/S02, P06/S01, P06/S02, P09/S02, P10/S01, P10/S03
- Wrist files (6): P02/S01
- Nearly all files (17, incl. `forehead_eeg_raw2.csv`): **P04/S01**

## Verified content spot-checks

- `exp_fatigue.csv`: exactly 3 rows in **all 36** sessions → **108 ratings total**; mental fatigue scores range **0.0–77.2** (0–100 VAS).
- `exp_crt.csv` (P01/S01): 108 trials = 36 × `measurementBlock` 0/1/2, with `responseTime` and `isCorrectResponse` per trial.
- `forehead_eeg_raw.csv` (P01/S01): header `timestamp,TP9,AF7,AF8,TP10`; timestamps are epoch ms starting exactly at the `start_baseline` marker (1629370423000); consecutive samples share ms timestamps (256 Hz > 1 ms resolution).
- Sensor timestamps (epoch ms) and experiment relative times (seconds in `exp_*` answer/trial times) use **different clocks** — alignment goes through `exp_markers.csv`.
- Known defect found: P01/S01 `exp_markers.csv` has one event (`start_nback`, third block) with an **empty timestamp**.
