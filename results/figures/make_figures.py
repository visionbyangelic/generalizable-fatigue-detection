"""Build the results figures for the writing pack from stored tables and
documented values.

Every number plotted here comes from a stored results table or from a
results document in docs/; the source is named beside each value in the
code. Nothing is recomputed from raw data and nothing is estimated.
Run with the pinned environment (docs/environment_pin.md):

    PYTHONPATH=~/workspace/.pylibs python3 results/figures/make_figures.py

Outputs (150 dpi PNG, written next to this script):
    fig_rq1_tiers.png
    fig_rq2_perparticipant.png
    fig_drozy_dissociation.png
    fig_rq4_ladder.png
    fig_rq3_arms.png
    fig_rq3_hr_intensity.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TABLES = HERE.parent / "tables"

# Colourblind-safe palette (Okabe-Ito).
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILLION = "#D55E00"
GREY = "#7F7F7F"
LIGHT_GREY = "#BFBFBF"

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "figure.dpi": 150,
        "savefig.dpi": 150,
    }
)

CHANCE = 0.5
USEFUL = 0.55  # the 0.55 usefulness line adopted by the project's gate ruling


def reference_lines(ax, useful=True, label=True):
    ax.axhline(CHANCE, color=GREY, linestyle="--", linewidth=1.0, zorder=0)
    bbox = {"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.0}
    if label:
        ax.text(
            ax.get_xlim()[1], CHANCE + 0.008, "chance (0.50) ",
            ha="right", va="bottom", fontsize=8, color=GREY, bbox=bbox, zorder=6,
        )
    if useful:
        ax.axhline(USEFUL, color="#444444", linestyle=":", linewidth=1.0, zorder=0)
        if label:
            ax.text(
                ax.get_xlim()[1], USEFUL + 0.008, "usefulness line (0.55) ",
                ha="right", va="bottom", fontsize=8, color="#444444", bbox=bbox, zorder=6,
            )


def finish(ax, path, ylim=None):
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"wrote {path.name}")


# ----------------------------------------------------------------------
# Figure 1: RQ1 tiers.
# Values: docs/cnn_loso_results.md, RQ1 section. Same-session 0.744 is the
# band-power baseline's within-session estimate (Week 2 diagnostics) and is
# labelled position-confounded; the CNN has no same-session estimate.
# LOPO intervals are the participant-bootstrap CIs from the same document
# (baseline [0.471, 0.542], CNN [0.452, 0.529]); error bars are drawn on
# the LOPO tier only, as stated in the figure note.
# ----------------------------------------------------------------------
def fig_rq1_tiers():
    tiers = ["Same session\n(within-session split)", "Within-subject\n(new sessions)", "Cross-participant\n(LOPO)"]
    baseline = [0.744, 0.484, 0.508]
    cnn = [np.nan, 0.493, 0.493]
    lopo_ci = {"baseline": (0.471, 0.542), "cnn": (0.452, 0.529)}

    x = np.arange(len(tiers))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    b1 = ax.bar(x - w / 2, baseline, w, color=BLUE, label="Band-power baseline")
    cnn_x = [x[1] + w / 2, x[2] + w / 2]
    b2 = ax.bar(cnn_x, cnn[1:], w, color=ORANGE, label="1D-CNN (EEGNet)")
    for bars in (b1, b2):
        for rect in bars:
            ax.annotate(
                f"{rect.get_height():.3f}",
                (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                ha="center", va="bottom", fontsize=9,
            )
    # LOPO error bars.
    for xi, model, val in [
        (x[2] - w / 2, "baseline", baseline[2]),
        (x[2] + w / 2, "cnn", cnn[2]),
    ]:
        lo, hi = lopo_ci[model]
        ax.errorbar(xi, val, yerr=[[val - lo], [hi - val]], color="#222222",
                    capsize=4, linewidth=1.1, zorder=5)
    ax.annotate(
        "position-confounded: window position\nalone ranks the labels perfectly",
        (x[0] + 0.16, baseline[0] + 0.035), ha="center", fontsize=8.5, color=VERMILLION,
    )
    ax.annotate(
        "no CNN estimate\non this tier",
        (x[0] + 0.05, 0.40), ha="left", fontsize=8, color=GREY,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(tiers)
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("RQ1: three tiers of evaluation, FatigueSet EEG (primary labels)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.56, 0.99))
    reference_lines(ax, label=False)
    ax.text(
        0.0, -0.30,
        "Dashed line: chance (0.50). Dotted line: the 0.55 usefulness line. Error bars (LOPO tier): 95%\n"
        "participant-bootstrap CI. Both honest tiers sit at chance for both models.",
        transform=ax.transAxes, fontsize=8.5, color="#333333",
    )
    finish(ax, HERE / "fig_rq1_tiers.png", ylim=(0.30, 0.86))


# ----------------------------------------------------------------------
# Figure 2: RQ2 per-participant LOPO balanced accuracy.
# Values: results/tables/cnn_loso_subjective_a1_summary.csv and
# results/tables/baseline_loso_subjective_a1_summary.csv
# (per-participant mean over 5 seeds; error bars are the seed SD).
# ----------------------------------------------------------------------
def fig_rq2_perparticipant():
    cnn = pd.read_csv(TABLES / "cnn_loso_subjective_a1_summary.csv")
    base = pd.read_csv(TABLES / "baseline_loso_subjective_a1_summary.csv")
    df = base[["participant", "balanced_accuracy_mean", "balanced_accuracy_std"]].merge(
        cnn[["participant", "balanced_accuracy_mean", "balanced_accuracy_std"]],
        on="participant", suffixes=("_base", "_cnn"),
    ).sort_values("participant")

    x = np.arange(len(df))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.bar(
        x - w / 2, df["balanced_accuracy_mean_base"], w, color=BLUE,
        yerr=df["balanced_accuracy_std_base"], capsize=3,
        error_kw={"linewidth": 0.9}, label="Band-power baseline",
    )
    ax.bar(
        x + w / 2, df["balanced_accuracy_mean_cnn"], w, color=ORANGE,
        yerr=df["balanced_accuracy_std_cnn"], capsize=3,
        error_kw={"linewidth": 0.9}, label="1D-CNN (EEGNet)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(df["participant"])
    ax.set_xlabel("Held-out participant")
    ax.set_ylabel("Balanced accuracy (mean over 5 seeds)")
    ax.set_title("RQ2: per-participant LOPO balanced accuracy, CNN vs baseline")
    ax.legend(frameon=True, loc="lower left", facecolor="white", framealpha=0.92,
              edgecolor="none")
    reference_lines(ax)
    ax.text(
        0.0, -0.24,
        "Error bars: SD across the 5 seeds. Cohort means: baseline 0.508, CNN 0.493. P05 has no retained windows and no fold.",
        transform=ax.transAxes, fontsize=8.5, color="#333333",
    )
    finish(ax, HERE / "fig_rq2_perparticipant.png", ylim=(0.15, 0.72))


# ----------------------------------------------------------------------
# Figure 3: DROZY dissociation.
# Values: docs/drozy_results.md headline table (balanced accuracy,
# participant-bootstrap 95% CI, within-participant session-label swap
# null p, 200 swaps).
# ----------------------------------------------------------------------
def fig_drozy_dissociation():
    labels = ["KSS labels,\nall-channel", "KSS labels,\nFz only", "RT labels,\nall-channel", "RT labels,\nFz only"]
    ba = [0.602, 0.543, 0.482, 0.467]
    lo = [0.563, 0.505, 0.421, 0.414]
    hi = [0.644, 0.582, 0.549, 0.527]
    pvals = ["swap-null p = 0.005", "swap-null p = 0.134", "swap-null p = 0.726", "swap-null p = 0.811"]
    colors = [BLUE, "#56B4E9", VERMILLION, "#E8A598"]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    bars = ax.bar(x, ba, 0.55, color=colors)
    ax.errorbar(
        x, ba, yerr=[np.array(ba) - np.array(lo), np.array(hi) - np.array(ba)],
        color="#222222", capsize=4, linewidth=1.1, fmt="none", zorder=5,
    )
    for rect, v, top, p in zip(bars, ba, hi, pvals):
        ax.annotate(f"{v:.3f}", (rect.get_x() + rect.get_width() / 2, top + 0.012),
                    ha="center", fontsize=9.5)
        ax.annotate(p, (rect.get_x() + rect.get_width() / 2, 0.345),
                    ha="center", fontsize=8.5, color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Balanced accuracy (LOPO, N = 13)")
    ax.set_title("DROZY: subjective (KSS) vs performance (RT) labels on identical EEG")
    reference_lines(ax)
    finish(ax, HERE / "fig_drozy_dissociation.png", ylim=(0.32, 0.72))


# ----------------------------------------------------------------------
# Figure 4: Subject 01 ladder.
# Prototype-setup and blocked-split values:
# results/tables/rq4_replication_proto_summary.csv (chronological:
# LR 1.000, CNN 0.993; blocked: LR 0.975, CNN 0.915) and
# results/tables/rq4_check2_route_location.csv (mask-only LR 0.993).
# Fixed-design values: results/tables/subject01_fixed_design.csv,
# seed-mean rows (within-subject 0.436, LOPO 0.496).
# ----------------------------------------------------------------------
def fig_rq4_ladder():
    fixed = pd.read_csv(TABLES / "subject01_fixed_design.csv")
    means = fixed[(fixed["participant"] == "P01") & (fixed["seed"] == "mean_of_s0-s4")]
    within = float(means.loc[means["protocol"].str.startswith("within"), "balanced_accuracy"].iloc[0])
    lopo = float(means.loc[means["protocol"].str.startswith("LOPO"), "balanced_accuracy"].iloc[0])

    steps = [
        ("Prototype\nsetup (LR)", 1.000, BLUE),
        ("Prototype\nsetup (CNN)", 0.993, ORANGE),
        ("Mask only\n(LR, no EEG)", 0.993, VERMILLION),
        ("Blocked split\n(LR)", 0.975, BLUE),
        ("Blocked split\n(CNN)", 0.915, ORANGE),
        ("Fixed design\nwithin-subject", round(within, 3), GREEN),
        ("Fixed design\nunseen person", round(lopo, 3), GREEN),
    ]
    x = np.arange(len(steps))
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    bars = ax.bar(x, [s[1] for s in steps], 0.58, color=[s[2] for s in steps])
    for rect, (_, v, _) in zip(bars, steps):
        ax.annotate(f"{v:.3f}", (rect.get_x() + rect.get_width() / 2, v + 0.02),
                    ha="center", fontsize=9.5)
    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in steps], fontsize=8.5)
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("RQ4: Subject 01, the same recordings under stricter evaluation")
    reference_lines(ax, useful=False, label=False)
    ax.text(
        0.0, -0.30,
        "Dashed line: chance (0.50). Prototype setup = the preliminary experiment's own configuration (chronological split);\n"
        "blocked split = 60 s blocks within the same recordings; fixed design = this project's pipeline and labels.\n"
        "Amendment-1 mask retention in the prototype's two sessions: 97.1% of Session 01 (alert) windows retained,\n"
        "2.5% of Session 03 (fatigued) windows retained. Every fatigued test window in the prototype setup is mask-rejected.",
        transform=ax.transAxes, fontsize=8.5, color="#333333",
    )
    finish(ax, HERE / "fig_rq4_ladder.png", ylim=(0.30, 1.10))


# ----------------------------------------------------------------------
# Figure 5: RQ3 arms, keep-all pool, LOPO.
# Values: results/tables/rq3_arm_summary.csv (pool = keepall,
# protocol = lopo). Title names the protocol-defined contrast, as the
# locked RQ3 wording requires.
# ----------------------------------------------------------------------
def fig_rq3_arms():
    arms = pd.read_csv(TABLES / "rq3_arm_summary.csv")
    keep = arms[(arms["pool"] == "keepall") & (arms["protocol"] == "lopo")].set_index("arm")
    order = [
        ("indicators", "Indicators only\n(missingness pattern)", GREY),
        ("eeg", "EEG only\n(Week 2 baseline)", BLUE),
        ("peripheral", "Peripheral only\n(HR + EDA)", GREEN),
        ("combined", "Combined\n(EEG + peripheral)", VERMILLION),
    ]
    x = np.arange(len(order))
    vals = [keep.loc[k, "balanced_accuracy_mean"] for k, _, _ in order]
    lo = [keep.loc[k, "balanced_accuracy_lo"] for k, _, _ in order]
    hi = [keep.loc[k, "balanced_accuracy_hi"] for k, _, _ in order]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    bars = ax.bar(x, vals, 0.55, color=[c for _, _, c in order])
    ax.errorbar(x, vals, yerr=[np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)],
                color="#222222", capsize=4, linewidth=1.1, fmt="none", zorder=5)
    for rect, v, top in zip(bars, vals, hi):
        ax.annotate(f"{v:.3f}", (rect.get_x() + rect.get_width() / 2, top + 0.015),
                    ha="center", fontsize=9.5)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label, _ in order])
    ax.set_ylabel("Balanced accuracy (LOPO, N = 11)")
    ax.set_title("RQ3: protocol-defined M1/M3 contrast by feature arm (FatigueSet)")
    reference_lines(ax)
    ax.text(
        0.0, -0.26,
        "Error bars: 95% participant-bootstrap CI. Headline quantities are the increments: peripheral minus indicators\n"
        "+0.202 [0.085, 0.314]; combined minus EEG-only +0.230 [0.128, 0.332].",
        transform=ax.transAxes, fontsize=8.5, color="#333333",
    )
    finish(ax, HERE / "fig_rq3_arms.png", ylim=(0.35, 0.95))


# ----------------------------------------------------------------------
# Figure 6: RQ3 descriptive heart-rate difference by intensity day.
# Values: results/tables/rq3_hr_by_intensity.csv. Descriptive only:
# no model, no interval, no test, as the RQ3 document states.
# ----------------------------------------------------------------------
def fig_rq3_hr_intensity():
    hr = pd.read_csv(TABLES / "rq3_hr_by_intensity.csv")
    hr["intensity"] = pd.Categorical(hr["intensity"], ["low", "medium", "high"], ordered=True)
    hr = hr.sort_values("intensity")
    diffs = hr["hr_m3_minus_m1_bpm"].to_numpy()
    colors = [GREY if d < 0 else GREEN for d in diffs]
    x = np.arange(len(hr))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bars = ax.bar(x, diffs, 0.5, color=colors)
    for rect, v in zip(bars, diffs):
        ax.annotate(f"{v:+.1f}", (rect.get_x() + rect.get_width() / 2, v + (0.7 if v >= 0 else -0.7)),
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=10)
    ax.axhline(0, color="#333333", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(["Low intensity day", "Medium intensity day", "High intensity day"])
    ax.set_ylabel("Mean heart rate, M3 minus M1 (bpm)")
    ax.set_title("RQ3 (descriptive): heart-rate block difference by intensity day")
    ax.text(
        0.0, -0.26,
        "Descriptive table only: no model fitted, no interval or test\n"
        "attached. Means over the primary pool's HR-usable windows:\n"
        "low 119 M1 / 88 M3; medium 234 / 200; high 210 / 142.",
        transform=ax.transAxes, fontsize=8.5, color="#333333",
    )
    finish(ax, HERE / "fig_rq3_hr_intensity.png", ylim=(-4.5, 22))


if __name__ == "__main__":
    fig_rq1_tiers()
    fig_rq2_perparticipant()
    fig_drozy_dissociation()
    fig_rq4_ladder()
    fig_rq3_arms()
    fig_rq3_hr_intensity()
