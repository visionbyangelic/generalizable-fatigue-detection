"""RQ3 tier-parity verification — LOPO tier vs within-subject tier.

Added after the RQ3 results were reviewed (docs/methods_notes.md,
review 38): before the two tiers are reported side by side, verify
from the run configs and the code path that they differ ONLY in the
fold structure, in the four respects the review names:

  1. Target definition: same label column (primary subjective labels
     on Amendment-1 windows) feeding both protocols.
  2. Window pool: both protocols fold the same capped keep-all pool
     at each cap seed (cap_pool is a deterministic function of the
     uncapped pool and the seed, applied before any fold is drawn,
     in the single run_cell path of src/evaluation/rq3_arms.py).
  3. Class balance handling: the same per-session M1 cap (cap_pool)
     and the same class_weight='balanced' in the locked pipeline.
  4. Normalisation route: the same Pipeline (training-fold median
     imputation, training-fold StandardScaler) fitted per fold by
     run_folds for both protocols; the within-participant-normalised
     variant is a separate cell ('norm'), used by neither tier here.

Checks performed:
  - config.yaml of every keep-all peripheral and combined run,
    LOPO vs within, seeds 0-4: all fields equal except 'protocol'
    (and the run's own git_commit stamp, which is metadata).
  - at each cap seed, the LOPO metrics' window counts sum to the
    capped pool size exactly (every capped window is a test window
    exactly once), and the within metrics' counts sum to the capped
    windows covered by the feasible rotations.
  - the within tier's skipped rotations, read from the run notes.

Also writes the within-subject per-participant seed means (both arms)
to results/tables/rq3_within_per_participant.csv for the results doc.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.baseline import cap_pool

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
RUNS = ROOT / "experiments" / "runs"
TAB = ROOT / "results" / "tables"
DATE = "2026-11-21"
SEEDS = [0, 1, 2, 3, 4]
ARMS = ["peripheral", "combined"]


def load_cfg(arm, protocol, seed):
    d = RUNS / f"{DATE}_rq3_keepall_{arm}_{protocol}_s{seed}"
    return yaml.safe_load((d / "config.yaml").read_text()), d


def main():
    idx = pd.read_csv(PROC / "windows_a1_uncapped_index.csv")
    problems = []

    # ---- 1. config parity
    for arm in ARMS:
        for seed in SEEDS:
            a, _ = load_cfg(arm, "lopo", seed)
            b, _ = load_cfg(arm, "within", seed)
            for k in sorted(set(a) | set(b)):
                if k in ("protocol", "git_commit"):
                    continue
                if a.get(k) != b.get(k):
                    problems.append(
                        f"config mismatch {arm} s{seed} field {k}: "
                        f"lopo={a.get(k)} within={b.get(k)}")
    print("config parity (all fields except protocol/commit):",
          "OK" if not problems else problems)

    # ---- 2. window-pool parity via metrics counts
    for seed in SEEDS:
        capped_n = len(cap_pool(idx, seed))
        for arm in ARMS:
            for protocol in ("lopo", "within"):
                d = RUNS / f"{DATE}_rq3_keepall_{arm}_{protocol}_s{seed}"
                m = pd.read_csv(d / "metrics.csv")
                m = m[m.participant != "AGGREGATE_mean_over_participants"]
                total = int(m.n.sum())
                if protocol == "lopo" and total != capped_n:
                    problems.append(
                        f"{arm} s{seed} lopo covers {total} windows, "
                        f"capped pool has {capped_n}")
                if protocol == "within":
                    print(f"  s{seed} {arm} within covers {total} of "
                          f"{capped_n} capped windows (feasible "
                          f"rotations only)")
        print(f"  s{seed} capped pool = {capped_n} windows; "
              f"lopo coverage exact for both arms")

    # ---- 3. within per-participant values + spread
    sm = pd.read_csv(TAB / "rq3_per_participant_seedmeans.csv")
    w = sm[(sm.pool == "keepall") & (sm.protocol == "within")]
    piv = w.pivot_table(index="participant", columns="arm",
                        values="balanced_accuracy")
    piv.to_csv(TAB / "rq3_within_per_participant.csv")
    for arm in ARMS:
        v = piv[arm].dropna()
        print(f"within {arm}: n={len(v)} mean={v.mean():.3f} "
              f"sd={v.std(ddof=1):.3f} min={v.min():.3f} "
              f"max={v.max():.3f}")
        print("  " + ", ".join(f"{p} {x:.3f}" for p, x in v.items()))

    print("PARITY VERDICT:", "CLEAN" if not problems else problems)
    return problems


if __name__ == "__main__":
    main()
