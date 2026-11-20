"""Follow-up to src/evaluation/diagnostics.py: sections 4 (delta-mental)
and 5 (P08 deep-dive) only.

Why a separate script: the first full diagnostics run (20 Nov 2026)
completed feature sanity, the within-session ceiling check, the
permutation test and the trivial baselines, then crashed in section 4 at
the matplotlib annotation (textcoords="offset_points" is not a valid
coordinate name in matplotlib 3.6 - it must be "offset points") before
writing the figure and running section 5. Rather than re-running the
~7-minute permutation test, this script reuses the module's helpers and
executes only the two missing sections. The fix is also applied in
diagnostics.py itself, so a future full re-run works end to end.
"""
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.diagnostics import (AGG, DATE, FIG, ROOT, RUNS, TAB,
                                        load_capped, p08_deep_dive)


def delta_mental():
    insp = pd.read_csv(ROOT / "results" / "tables" / "inspection_summary.csv",
                       dtype={"participant": str, "session": str})
    insp["heldout_session"] = ("P" + insp.participant + "/S" + insp.session)
    summ = pd.read_csv(TAB / "baseline_session-out_subjective_a1_summary.csv")
    s0 = pd.read_csv(RUNS / f"{DATE}_baseline_session-out_subjective_s0"
                     / "metrics.csv")
    s0 = s0[s0.heldout_session != AGG][["heldout_session", "balanced_accuracy",
                                        "mcc"]]
    s0.columns = ["heldout_session", "balanced_accuracy_seed0", "mcc_seed0"]
    dm = (summ[["heldout_session", "balanced_accuracy_mean", "mcc_mean",
                "roc_auc_mean", "n_mean"]]
          .merge(insp[["heldout_session", "delta_mental", "intensity"]],
                 on="heldout_session")
          .merge(s0, on="heldout_session"))
    rho, p_rho = stats.spearmanr(dm.delta_mental, dm.balanced_accuracy_mean)
    rho0, p_rho0 = stats.spearmanr(dm.delta_mental, dm.balanced_accuracy_seed0)
    dm["spearman_rho_BA_vs_delta"] = rho
    dm["spearman_p"] = p_rho
    dm.to_csv(TAB / "diagnostics_delta_mental.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(dm.delta_mental, dm.balanced_accuracy_mean, zorder=3)
    for _, r in dm.iterrows():
        ax.annotate(r.heldout_session, (r.delta_mental,
                                        r.balanced_accuracy_mean),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.axhline(0.5, color="grey", lw=1, ls="--")
    ax.set_xlabel("Session delta-mental (M3 - M1 VAS)")
    ax.set_ylabel("Session-out balanced accuracy (mean over cap seeds 0-4)")
    ax.set_title(f"Per-session accuracy vs delta-mental\n"
                 f"Spearman rho = {rho:.2f}, p = {p_rho:.3f}, n = {len(dm)} sessions")
    fig.tight_layout()
    fig.savefig(FIG / "diagnostics_delta_mental_scatter.png", dpi=150)
    plt.close(fig)
    print(f"delta-mental: Spearman rho={rho:.3f} p={p_rho:.3f} n={len(dm)} "
          f"(seed0: rho={rho0:.3f} p={p_rho0:.3f})", flush=True)
    return dm


def main():
    delta_mental()
    print("loading cap seed 0 for P08 ...", flush=True)
    df0, feats0 = load_capped(0)
    coef_df, top, preds = p08_deep_dive(df0, feats0)
    print("P08 top features:\n", top.to_string(index=False), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
