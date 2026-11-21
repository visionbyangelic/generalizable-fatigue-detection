"""RQ3 descriptive table — mean heart rate by block and treadmill day.

Status: DESCRIPTIVE ONLY. Added after the RQ3 results were reviewed
(docs/methods_notes.md, reviews 37-38) as the agreed non-modelled
companion to the results: no model is fitted, no interval or test is
computed, and no inferential language attaches to the numbers. The
table simply records, over the primary-pool windows, the mean of the
E4 heart-rate feature on M1 windows and on M3 windows, grouped by the
treadmill intensity of the session the windows come from, with the
M3-minus-M1 difference per intensity.

Pool: the 1,053 primary-pool windows (windows_a1_uncapped_index +
peripheral_features_a1), uncapped. Means are taken over the windows
whose heart rate is usable (hr_valid = 1); the counts are in the
table. Session intensity per participant is from the dataset's own
metadata.csv (Latin-square counterbalancing), the same mapping as
docs/DATA_INVENTORY.md.

Output: results/tables/rq3_hr_by_intensity.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
TAB = ROOT / "results" / "tables"


def intensity_map():
    meta = pd.read_csv(ROOT / "data" / "raw" / "fatigueset" / "metadata.csv",
                       encoding="utf-8-sig", dtype=str)
    out = {}
    for _, r in meta.iterrows():
        part = f"P{int(r.participant_id):02d}"
        for level in ("low", "medium", "high"):
            out[(part, f"S{int(r[f'{level}_session']):02d}")] = level
    return out


def main():
    idx = pd.read_csv(PROC / "windows_a1_uncapped_index.csv").reset_index(drop=True)
    per = pd.read_csv(PROC / "peripheral_features_a1.csv")
    key = ["participant", "session", "start_sample"]
    assert (per[key].values == idx[key].values).all()
    df = idx[["participant", "session", "block", "label"]].copy()
    df["hr_mean"] = per["hr_mean"]
    df["hr_valid"] = per["hr_valid"]
    imap = intensity_map()
    df["intensity"] = [imap[(p, s)] for p, s in
                       zip(df.participant, df.session)]
    usable = df[df.hr_valid == 1]
    rows = []
    for level in ("low", "medium", "high"):
        g = usable[usable.intensity == level]
        m1 = g[g.block == "M1"].hr_mean
        m3 = g[g.block == "M3"].hr_mean
        rows.append({
            "intensity": level,
            "n_windows_hr_usable_m1": len(m1),
            "hr_mean_m1_bpm": float(m1.mean()),
            "n_windows_hr_usable_m3": len(m3),
            "hr_mean_m3_bpm": float(m3.mean()),
            "hr_m3_minus_m1_bpm": float(m3.mean() - m1.mean()),
        })
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "rq3_hr_by_intensity.csv", index=False)
    print(out.round(2).to_string(index=False))
    print(f"pool windows: {len(df)}; HR-usable: {len(usable)}")


if __name__ == "__main__":
    main()
