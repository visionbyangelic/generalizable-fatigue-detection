# Dataset Notes — FatigueSet

**Full research report:** [`docs/fatigueset-research.md`](fatigueset-research.md) · **Verified inventory of the downloaded copy:** [`docs/DATA_INVENTORY.md`](DATA_INVENTORY.md)

## Summary (20 Nov 2026)

- **Obtained.** `data/raw/fatigueset.zip` (641,686,295 bytes, SHA-256 `2db4c9d785307a309f66a404d90a5dbdbe1a45d0a954f980204dd39e2a7fd8ca`) downloaded from the official esense.io page's public Google Drive link — **no form, licence click-through, or sign-in was required**, so nothing was submitted on my behalf. CRC integrity passed for all 1,473 files; 3.38 GB uncompressed.
- **Contents confirmed:** 12 participants × 3 sessions (low/medium/high treadmill intensity, counterbalanced; mapping in `metadata.csv`) × 39 core CSVs — Muse S EEG raw (256 Hz, TP9/AF7/AF8/TP10) + band powers, E4 wrist, eSense earables, Zephyr chest, plus experiment files (`exp_fatigue`, `exp_markers`, `exp_crt`, `exp_nback`, `exp_task_switch`). No session is missing any core file.
- **Watch-outs for Week 1 inspection (next step):**
  1. Subjective labels are sparse: **3 VAS ratings per session (108 total)** — label rule (interpolation vs block) is the key methods decision; see research doc §7–§8.
  2. Sessions = activity-intensity days, not fatigued/rested — matters for RQ4.
  3. Two clocks: sensor/marker timestamps are epoch ms; experiment answer/trial times are relative seconds.
  4. 65 `*2.csv` continuation files must be concatenated (incl. `forehead_eeg_raw2.csv` for P04/S01).
  5. Artefact gates exist in-data: `muse_device_fit.csv`, `muse_blinks.csv`, `muse_jaw_clenches.csv` — use them.
  6. One known defect: missing `start_nback` timestamp in P01/S01 markers.
  7. No explicit licence — cite the paper; do not redistribute raw data.
