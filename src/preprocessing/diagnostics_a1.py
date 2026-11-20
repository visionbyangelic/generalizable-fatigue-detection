"""Amendment 1 mandatory diagnostics (docs/label_rules.md, Amendment 1).

Inputs: data/processed/candidates_a1.csv (every candidate window with
per-channel pass/fail at the locked +/-75 uV threshold) and the A1 indexes.

Outputs (results/tables/):
  retention_amendment1.csv   per participant x set (primary / check-set):
                             candidates, retained, retention %, retained per
                             class, TP9/TP10 masking % among retained.
  mnar_diagnostic.csv        temporal-channel failure rates (any temporal
                             channel failing any criterion) by block (M1 vs
                             M3) and by session time (first vs second half of
                             each session's candidate windows), with counts,
                             rate difference and chi-square p-value.
  threshold_sensitivity.csv  retention % under Amendment-1 gating at
                             amplitude p2p limits 50/75/100/125/150 uV.
                             NOTE: the locked operating point "+/-75 uV" was
                             implemented (build_windows.py, deviation doc) as
                             p2p > 150 uV, i.e. the 150 row below IS the
                             locked operating point; the amp_threshold column
                             also gives the equivalent +/- amplitude.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
TAB = ROOT / "results" / "tables"
FLAT_SD = 0.5


def retention_table(cand):
    idx_u = pd.read_csv(PROC / "windows_a1_uncapped_index.csv")
    idx_c = pd.read_csv(PROC / "windows_a1_control_index.csv")
    rows = []
    lab = cand[cand.block.isin(["M1", "M3"])]
    for (part, is_ctrl), g in lab.groupby(["participant", "control_session"]):
        ret = g[g.retained_a1]
        src = idx_c if is_ctrl else idx_u
        src = src[src.participant == part]
        rows.append({
            "participant": part,
            "set": "check" if is_ctrl else "primary",
            "candidate_M1M3": len(g),
            "retained": len(ret),
            "retention_pct": round(100 * len(ret) / len(g), 1) if len(g) else 0.0,
            "retained_alert_M1": int((src.label == 0).sum()),
            "retained_fatigued_M3": int((src.label == 1).sum()),
            "masked_TP9_pct_of_retained": round(100 * (~ret.valid_TP9).mean(), 1) if len(ret) else 0.0,
            "masked_TP10_pct_of_retained": round(100 * (~ret.valid_TP10).mean(), 1) if len(ret) else 0.0,
            "both_temporals_masked_pct": round(
                100 * ((~ret.valid_TP9) & (~ret.valid_TP10)).mean(), 1) if len(ret) else 0.0,
        })
    out = pd.DataFrame(rows).sort_values(["participant", "set"])
    out.to_csv(TAB / "retention_amendment1.csv", index=False)
    return out


def _rate_test(df_a, df_b, name_a, name_b):
    a_fail = int(df_a["temporal_fail"].sum()); a_n = len(df_a)
    b_fail = int(df_b["temporal_fail"].sum()); b_n = len(df_b)
    table = np.array([[a_fail, a_n - a_fail], [b_fail, b_n - b_fail]])
    p = chi2_contingency(table)[1] if table.min() >= 0 and a_n and b_n else np.nan
    return {
        "group_a": name_a, "n_a": a_n, "temporal_fail_a": a_fail,
        "fail_rate_a_pct": round(100 * a_fail / a_n, 2) if a_n else np.nan,
        "group_b": name_b, "n_b": b_n, "temporal_fail_b": b_fail,
        "fail_rate_b_pct": round(100 * b_fail / b_n, 2) if b_n else np.nan,
        "rate_diff_pp_b_minus_a": round(100 * b_fail / b_n - 100 * a_fail / a_n, 2) if a_n and b_n else np.nan,
        "chi2_p": p,
    }


def mnar_table(cand):
    lab = cand[cand.block.isin(["M1", "M3"])].copy()
    lab["temporal_fail"] = ~(lab.valid_TP9 & lab.valid_TP10)
    rows = []
    for scope, d in [("all_sessions", lab),
                     ("primary_sessions_only", lab[~lab.control_session]),
                     ("check_sessions_only", lab[lab.control_session])]:
        rows.append({"comparison": "block_M1_vs_M3", "scope": scope,
                     **_rate_test(d[d.block == "M1"], d[d.block == "M3"], "M1", "M3")})
        # Session time: first vs second half of each session's candidates.
        d = d.sort_values(["participant", "session", "centre_ms"]).copy()
        d["half"] = d.groupby(["participant", "session"]).cumcount()
        d["half"] = d["half"] / d.groupby(["participant", "session"])["half"].transform("max")
        rows.append({"comparison": "session_time_first_vs_second_half", "scope": scope,
                     **_rate_test(d[d.half < 0.5], d[d.half >= 0.5], "first_half", "second_half")})
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "mnar_diagnostic.csv", index=False)
    return out


def threshold_table(cand):
    lab = cand[cand.block.isin(["M1", "M3"])].copy()
    rows = []
    for lim in [50, 75, 100, 125, 150]:
        front_ok = ((lab.max_p2p_AF7 <= lim) & (lab.max_p2p_AF8 <= lim)
                    & (lab.sd_AF7 >= FLAT_SD) & (lab.sd_AF8 >= FLAT_SD)
                    & ~(lab.fail_fit_AF7 | lab.fail_fit_AF8))
        rows.append({
            "p2p_limit_uv": lim,
            "equiv_pm_amplitude_uv": lim / 2,
            "is_locked_operating_point": lim == 150,
            "candidate_M1M3": len(lab),
            "retained": int(front_ok.sum()),
            "retention_pct": round(100 * front_ok.mean(), 1),
        })
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "threshold_sensitivity.csv", index=False)
    return out


def main():
    cand = pd.read_csv(PROC / "candidates_a1.csv")
    print("=== Retention (Amendment 1) ===")
    print(retention_table(cand).to_string(index=False))
    print("\n=== MNAR diagnostic ===")
    print(mnar_table(cand).to_string(index=False))
    print("\n=== Threshold sensitivity ===")
    print(threshold_table(cand).to_string(index=False))


if __name__ == "__main__":
    main()
