"""Exploratory checks on the DROZY LOPO results, run after the
headline numbers were already reported (21 Nov 2026).

These are post-results checks, not new primaries. They exist because
the headline pattern (KSS labels decodable across participants, RT
labels not, on identical recordings) has three obvious soft spots
that I want measured rather than argued about:

  audit     Are the reported bootstrap intervals really resampling
            participants? Are the swap nulls centred where they
            should be? And why does the Fz-only interval exclude 0.5
            while its permutation p-value is 0.134?
  order     A protocol-order arm: labels from session order alone
            (earliest session = alert, latest = fatigued). If this
            scores like the KSS arm, the KSS result may be session
            order wearing a label's clothes.
  contrast  The disagreement accounting between the KSS and RT label
            arms, a paired bootstrap on BA(KSS) - BA(RT), and the
            accuracy split between participants whose two arms agree
            on the fatigued session and those whose arms disagree.

Everything here reuses the frozen fold machinery and the DROZY
baseline wiring (src/evaluation/drozy_baseline.py); the only new
computation is the order arm's labels and its own swap null, built
with the same functions the gate used.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.closure_pack import lopo_folds, participant_boot_ci
from src.evaluation.drozy_baseline import (arm_frame, load_all,
                                           lopo_statistic, swapped_labels,
                                           write_run)
from src.models.baseline import run_folds, summarise

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "results" / "tables"
DATE = "2026-11-21"

# The four primary combinations, named as in drozy_baseline.COMBOS and
# in results/tables/drozy_lopo_aggregate.csv.
PRIMARY = ["kss_all", "kss_fz", "rt_all", "rt_fz"]


# ------------------------------------------------------------------ audit
def audit():
    """Recompute every reported interval from the stored per-
    participant values and compare with the aggregate table, then
    take the swap nulls apart to explain the Fz-only discrepancy.

    The point of recomputing instead of trusting the printout: the
    interval function takes an array of numbers, and whether those
    numbers are participants or windows is decided by what the caller
    passes. Reading the caller says participants; recomputing from
    the stored per-participant summaries proves the stored numbers
    are the ones that came out of that call."""
    agg = pd.read_csv(TAB / "drozy_lopo_aggregate.csv")
    agg = agg.set_index("combo")
    nulls = pd.read_csv(TAB / "drozy_gate_permutation_nulls.csv")
    rows = []
    for combo in PRIMARY + [f"{c}{s}" for c in ("kss_all", "kss_fz")
                            for s in ("-sensS1", "-sensS2")]:
        per = pd.read_csv(TAB / f"baseline_loso_drozy_{combo}_summary.csv")
        vals = per.balanced_accuracy.values
        pt, lo, hi, n = participant_boot_ci(vals)
        a = agg.loc[combo]
        row = {"combo": combo, "n_participants": n,
               "ba_recomputed": pt, "ba_reported": a.balanced_accuracy,
               "ci_lo_recomputed": lo, "ci_lo_reported": a.ba_ci95_low,
               "ci_hi_recomputed": hi, "ci_hi_reported": a.ba_ci95_high,
               "max_abs_diff": max(abs(pt - a.balanced_accuracy),
                                   abs(lo - a.ba_ci95_low),
                                   abs(hi - a.ba_ci95_high)),
               "per_participant_ba_sd": float(vals.std(ddof=1)),
               "bootstrap_se": float(vals.std(ddof=1) / np.sqrt(n))}
        if combo in PRIMARY:
            nv = nulls[(nulls.combo == combo)
                       & (nulls.variant == "swap_A")]
            stats = nv.ba_participant_mean.values
            row.update({
                "null_mean_stored": float(stats.mean()),
                "null_sd_stored": float(stats.std(ddof=1)),
                "null_per_participant_sd_mean":
                    float(nv.per_participant_ba_sd.mean()),
                # what the SD of the mean would be if the 13 fold
                # scores were independent draws with the null's
                # per-participant spread
                "null_sd_if_independent":
                    float(nv.per_participant_ba_sd.mean() / np.sqrt(n)),
                "p_recount": float((1 + (stats >= pt).sum())
                                   / (len(stats) + 1)),
                "p_reported": float(a.p_value_ge_real)})
            # Var(mean) = sd_pp^2 / n * (1 + (n-1) rho); solve for the
            # average pairwise correlation rho between fold scores.
            var_ratio = (row["null_sd_stored"] ** 2) / (
                row["null_per_participant_sd_mean"] ** 2 / n)
            row["implied_fold_correlation"] = (var_ratio - 1) / (n - 1)
        rows.append(row)
        print(f"audit {combo}: n={n} BA {pt:.4f} [{lo:.4f}, {hi:.4f}] "
              f"max|diff| vs reported {row['max_abs_diff']:.2e}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "drozy_ci_audit.csv", index=False)
    return out


# ------------------------------------------------------------------ order
def order_table(tables):
    """Protocol-order label table: per participant, the earliest
    session with data is 'alert', the latest is 'fatigued', any
    session between them is excluded. Participant 12 has one session,
    so no contrast exists and they stay excluded, as in both arms.
    Session availability is taken from the KSS label table (both arms
    recorded the same sessions_with_data)."""
    src = tables["kss"][["participant", "sessions_with_data"]]
    rows = []
    for r in src.itertuples():
        tests = [int(t) for t in str(r.sessions_with_data).split(",")]
        if len(tests) < 2:
            rows.append({"participant": r.participant,
                         "sessions_with_data": r.sessions_with_data,
                         "alert_test": np.nan, "fatigued_test": np.nan,
                         "middle_test": np.nan, "alert_value": np.nan,
                         "fatigued_value": np.nan, "included": False,
                         "note": "fewer than two usable sessions; "
                                 "excluded: no within-person contrast "
                                 "exists"})
            continue
        mid = tests[1:-1]
        rows.append({"participant": r.participant,
                     "sessions_with_data": r.sessions_with_data,
                     "alert_test": float(tests[0]),
                     "fatigued_test": float(tests[-1]),
                     "middle_test": float(mid[0]) if mid else np.nan,
                     "alert_value": float(tests[0]),
                     "fatigued_value": float(tests[-1]),
                     "included": True,
                     "note": "labels from session order only: earliest "
                             "session with data vs latest"})
    table = pd.DataFrame(rows)
    table.to_csv(TAB / "drozy_labels_order.csv", index=False)
    return table


def order():
    idx, feats, tables = load_all()
    table = order_table(tables)
    from src.data.drozy_adapter import FEATURE_NAMES, FZ_FEATURES
    null_rows, per_rows = [], []
    for name, cols in (("order_all", FEATURE_NAMES),
                       ("order_fz", FZ_FEATURES)):
        df, fc = arm_frame(idx, feats, table, cols)
        preds, skipped = run_folds(df, fc, lopo_folds(df))
        per = summarise(preds, "participant")
        pt, lo, hi, n = participant_boot_ci(per.balanced_accuracy.values)
        # the order arm needs its own swap null: the gate nulls were
        # built on the KSS/RT label frames, and a null is a property
        # of the label structure it permutes. Same swap function,
        # same seed convention as the gate's swap_A set.
        stats_p = []
        for p in range(200):
            rng = np.random.default_rng(20_000 + p)
            stat, per_p = lopo_statistic(df, fc, y=swapped_labels(df, rng))
            stats_p.append(stat)
            null_rows.append({"combo": name, "variant": "swap_A",
                              "perm": p, "ba_participant_mean": stat,
                              "per_participant_ba_sd":
                                  per_p.balanced_accuracy.std(ddof=1)})
        stats_p = np.array(stats_p)
        pval = float((1 + (stats_p >= pt).sum()) / (len(stats_p) + 1))
        fset = "all53" if name.endswith("all") else "fz12"
        folder = f"{DATE}_baseline_loso_protocol-order_drozy-{name}_s0"
        notes = f"""# {folder}

Split definition: leave-one-participant-out, as in the DROZY label
arms. Labels come from session order only: each participant's
earliest session with data is labelled alert, their latest session
fatigued, any middle session excluded (results/tables/
drozy_labels_order.csv). Participant 12 stays excluded (one session,
no contrast). {n} folds; skipped folds: {skipped if skipped else 'none'}.

Why this run exists: it is an exploratory check written after the
DROZY headline results were seen. The KSS arm's fatigued session is
the latest session for most participants, so a model that only
learned 'later session' could in principle reproduce the KSS score
without reading sleepiness at all. This arm measures exactly that.

Result: balanced accuracy {pt:.3f}, 95% CI [{lo:.3f}, {hi:.3f}] by
participant bootstrap (10,000 resamples over per-participant values,
rng 42). Permutation null (200 within-participant session-label
swaps, computed for this arm): mean {stats_p.mean():.3f}, SD
{stats_p.std(ddof=1):.3f}, p = {pval:.3f}.

Leakage checklist:
- [x] Normalisation (StandardScaler) fit on training fold only (Pipeline, per fold)
- [x] Median imputation for masked-channel NaNs fit on training fold only
- [x] Class weights computed from training fold only (class_weight='balanced')
- [x] No overlapping windows across splits (tumbling windows; splits at participant level)
- [x] Held-out participant contributes nothing to fitting, imputation, scaling or tuning
- [x] No tuning on test data (C=1.0 locked default)
- [x] Label construction fixed in this file's docstring and the checks document before the run

Environment: local VM (CPU), scikit-learn 1.9.1, MNE 1.13.2.
"""
        write_run(folder, name, per, preds, skipped,
                  {"features": f"DROZY {'all-channel (53)' if fset == 'all53' else 'Fz-only (12)'}",
                   "labels": "protocol order (earliest vs latest session "
                             "with data), exploratory"},
                  notes)
        per_out = per.copy()
        per_out["combo"] = name
        per_out.to_csv(TAB / f"baseline_loso_drozy_{name}_summary.csv",
                       index=False)
        per_rows.append({"combo": name, "n_participants": n,
                         "balanced_accuracy": pt, "ba_ci95_low": lo,
                         "ba_ci95_high": hi,
                         "perm_null_mean": float(stats_p.mean()),
                         "perm_null_sd": float(stats_p.std(ddof=1)),
                         "p_value_ge_real": pval})
        print(f"{name}: BA {pt:.3f} [{lo:.3f}, {hi:.3f}] n={n} "
              f"null mean {stats_p.mean():.3f} SD "
              f"{stats_p.std(ddof=1):.3f} p={pval:.3f}", flush=True)
    pd.DataFrame(null_rows).to_csv(
        TAB / "drozy_order_permutation_nulls.csv", index=False)
    pd.DataFrame(per_rows).to_csv(TAB / "drozy_order_arm.csv", index=False)


# ------------------------------------------------------------------ contrast
def contrast():
    kss = pd.read_csv(TAB / "drozy_labels_kss.csv")
    rt = pd.read_csv(TAB / "drozy_labels_rt.csv")
    inc = kss[kss.included].set_index("participant")
    inc_rt = rt[rt.included].set_index("participant")

    # --- disagreement accounting -------------------------------------
    # 'Later session' means the highest test index that participant
    # actually has data for, read off sessions_with_data.
    rows = []
    for p in inc.index:
        tests = [int(t) for t in str(inc.loc[p, "sessions_with_data"])
                 .split(",")]
        rows.append({
            "participant": p, "latest_test_with_data": max(tests),
            "kss_fatigued_test": int(inc.loc[p, "fatigued_test"]),
            "rt_fatigued_test": int(inc_rt.loc[p, "fatigued_test"]),
            "kss_alert_test": int(inc.loc[p, "alert_test"]),
            "rt_alert_test": int(inc_rt.loc[p, "alert_test"]),
            "kss_fatigued_is_latest":
                int(inc.loc[p, "fatigued_test"]) == max(tests),
            "rt_fatigued_is_latest":
                int(inc_rt.loc[p, "fatigued_test"]) == max(tests),
            "fatigued_agrees": int(inc.loc[p, "fatigued_test"])
                               == int(inc_rt.loc[p, "fatigued_test"]),
            "alert_agrees": int(inc.loc[p, "alert_test"])
                            == int(inc_rt.loc[p, "alert_test"])})
    acct = pd.DataFrame(rows)
    acct.to_csv(TAB / "drozy_label_agreement.csv", index=False)
    print(f"accounting: KSS fatigued = latest session for "
          f"{acct.kss_fatigued_is_latest.sum()}/13; RT fatigued = "
          f"latest for {acct.rt_fatigued_is_latest.sum()}/13; arms "
          f"disagree on the fatigued session for "
          f"{(~acct.fatigued_agrees).sum()}/13: "
          f"{list(acct.loc[~acct.fatigued_agrees, 'participant'])}",
          flush=True)

    # --- paired contrast, all-channel --------------------------------
    a = pd.read_csv(TAB / "baseline_loso_drozy_kss_all_summary.csv") \
        .set_index("participant").balanced_accuracy
    b = pd.read_csv(TAB / "baseline_loso_drozy_rt_all_summary.csv") \
        .set_index("participant").balanced_accuracy
    diff = (a - b).dropna()
    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(diff), size=(10_000, len(diff)))
    means = diff.values[idx].mean(axis=1)
    paired = pd.DataFrame([{
        "contrast": "BA(KSS all-channel) - BA(RT all-channel), paired "
                    "by participant",
        "n_participants": len(diff), "mean_difference": diff.mean(),
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
        "bootstrap_share_above_zero": float((means > 0).mean())}])
    paired.to_csv(TAB / "drozy_paired_contrast.csv", index=False)
    print(f"paired contrast: mean diff {diff.mean():.3f} "
          f"[{np.percentile(means, 2.5):.3f}, "
          f"{np.percentile(means, 97.5):.3f}] n={len(diff)}", flush=True)

    # --- agree/disagree split ----------------------------------------
    # Each participant's fold score already exists per arm; the split
    # just averages those scores inside the two groups. The groups
    # are small (the disagreeing group especially), so the counts are
    # printed and saved alongside every mean.
    split_rows = []
    for combo in ("kss_all", "kss_fz", "rt_all", "rt_fz"):
        per = pd.read_csv(TAB / f"baseline_loso_drozy_{combo}_summary.csv") \
            .set_index("participant").balanced_accuracy
        for group, mask in (("agree", acct.fatigued_agrees),
                            ("disagree", ~acct.fatigued_agrees)):
            members = acct.loc[mask, "participant"]
            vals = per.reindex(members).dropna()
            split_rows.append({"combo": combo, "group": group,
                               "n_participants": len(vals),
                               "participants": ",".join(members),
                               "mean_balanced_accuracy": vals.mean(),
                               "per_participant_ba_sd":
                                   vals.std(ddof=1) if len(vals) > 1
                                   else np.nan})
    split = pd.DataFrame(split_rows)
    split.to_csv(TAB / "drozy_agree_disagree_split.csv", index=False)
    print(split.to_string(index=False), flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "audit"
    {"audit": audit, "order": order, "contrast": contrast}[cmd]()
