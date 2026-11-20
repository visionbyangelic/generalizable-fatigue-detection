"""FatigueSet session loader (Muse EEG + fit + markers + fatigue ratings).

Handles the dataset quirks documented in docs/dataset_notes.md:
- ``*2.csv`` continuation files (append, sort by timestamp; P04/S01's base
  EEG file is empty and all rows live in the continuation).
- Stray leading index columns in some continuation files.
- Corrupt cells (2 in the whole dataset) coerced to NaN.
- Participant/session folders are zero-padded numbers (``01``..``12``).
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[2] / "data" / "raw" / "fatigueset"
CHANNELS = ["TP9", "AF7", "AF8", "TP10"]
PARTICIPANTS = [f"{i:02d}" for i in range(1, 13)]
SESSIONS = ["01", "02", "03"]

# Sessions whose experiment relative clock steps ~+180 s at a block boundary
# (docs/dataset_notes.md section 7): (participant, session) -> boundary crossed.
CLOCK_STEP = {
    ("11", "01"): "M1->M2", ("12", "01"): "M1->M2",
    ("02", "03"): "M2->M3", ("04", "02"): "M2->M3", ("05", "01"): "M2->M3",
    ("07", "03"): "M2->M3", ("08", "03"): "M2->M3", ("10", "01"): "M2->M3",
}
# Sessions excluded from the primary subjective set (delta-mental <= 0).
NONINCREASING = {("07", "01"), ("07", "02"), ("10", "03"), ("12", "03")}


def _read_sensor_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, low_memory=False)
    if df.columns[0].startswith("Unnamed") or df.columns[0] == "":
        df = df.iloc[:, 1:]
    return df


def load_concat(p: str, s: str, base: str) -> pd.DataFrame:
    """Load base file + ``*2.csv`` continuation, sorted by epoch-ms timestamp."""
    frames = []
    for cand in (base, base.replace(".csv", "2.csv")):
        fp = RAW / p / s / cand
        if fp.exists():
            frames.append(_read_sensor_csv(fp))
    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_numeric(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp"])
    return out.sort_values("timestamp", kind="stable").reset_index(drop=True)


def load_eeg(p: str, s: str):
    """Return (timestamps int64 ms, values float64 (n,4) in CHANNELS order)."""
    df = load_concat(p, s, "forehead_eeg_raw.csv")
    ts = df["timestamp"].to_numpy(dtype=np.int64)
    x = np.column_stack(
        [pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float) for c in CHANNELS]
    )
    return ts, x


def load_fit(p: str, s: str):
    """Return (timestamps int64 ms, fit float64 (n,4)); fit value 4 == Bad."""
    df = load_concat(p, s, "muse_device_fit.csv")
    ts = df["timestamp"].to_numpy(dtype=np.int64)
    fit = np.column_stack(
        [pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float) for c in CHANNELS]
    )
    return ts, fit


def load_markers(p: str, s: str) -> dict:
    """Marker name -> list of epoch-ms times (file order preserved)."""
    df = pd.read_csv(RAW / p / s / "exp_markers.csv", dtype={"eventMarker": str})
    df["utcTime"] = pd.to_numeric(df["utcTime"], errors="coerce")
    out = {}
    for name, t in zip(df["eventMarker"], df["utcTime"]):
        out.setdefault(name, []).append(t)
    return out


def block_intervals(p: str, s: str) -> dict:
    """Block intervals in epoch ms, each from its OWN marker pair
    (per-block anchoring, docs/dataset_notes.md section 7):
    M1 = end_baseline -> start_activity; M2 = end_activity -> start_fatigue;
    M3 = end_fatigue -> end_session."""
    mk = load_markers(p, s)

    def t(name):
        vals = [v for v in mk.get(name, []) if pd.notna(v)]
        return float(vals[0]) if vals else np.nan

    return {
        "M1": (t("end_baseline"), t("start_activity")),
        "M2": (t("end_activity"), t("start_fatigue")),
        "M3": (t("end_fatigue"), t("end_session")),
    }


def load_fatigue(p: str, s: str) -> pd.DataFrame:
    return pd.read_csv(RAW / p / s / "exp_fatigue.csv")


def delta_mental(p: str, s: str) -> float:
    f = load_fatigue(p, s)
    return float(f["mentalFatigueScore"].iloc[2] - f["mentalFatigueScore"].iloc[0])
