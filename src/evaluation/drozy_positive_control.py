"""DROZY positive control, run under Amendment 2 to
docs/dataset2_preregistration.md (adopted 21 Nov 2026, post hoc).

The question this answers is narrow: if I take the DROZY windows and
paint a known theta/alpha increase onto the windows one label arm
calls fatigued, does the frozen pipeline recover it, and how big
does the painted effect have to be before the reaction-time arm can
see it? Two purposes, both declared in the amendment: the RT arm's
minimum detectable effect under its own label structure (is its
0.482 a small effect or an insensitive test?), and a recovery curve
on DROZY channels to set beside FatigueSet's (0.834 at full dose,
0.695 at half, from the Week 2 control in
src/evaluation/panel_additions.py).

Design, following the amendment:

  identity   Dose 0 (theta x1.0, alpha x1.0) pushed through the
             injection code path on the true labels. It must
             reproduce the reported uninjected scores exactly (KSS
             all-channel 0.602, RT all-channel 0.482). If it does
             not, everything stops: a mismatch is a finding about
             the adapter, not something to debug around silently.
  permuted   The primary control. Each repeat first swaps each
             participant's two session labels at random (coin flip
             per participant), so any natural label-signal link is
             null by construction; the injection is then applied to
             the permuted-fatigued windows and the full LOPO runs
             under the permuted labels. Whatever separation appears
             can only be the injected effect. Five repeats per dose.
  companion  The same dose ladder on the true labels, reported as
             the increment over the arm's uninjected score, with a
             swap-null p-value on the increment. The injection is
             deterministic on true labels, so there is one value per
             dose rather than five.

Injection mirrors Week 2 exactly: for each fatigued-labelled window,
the theta (4-8 Hz) and alpha (8-12 Hz) bins of each valid channel's
own FFT are scaled by sqrt(power gain), phase preserved, alert
windows untouched, masked channels skipped (they are stored as
zeros and carry no features anyway). The gains are power gains.
Doses interpolate linearly between dose 0 (theta x1.0, alpha x1.0)
and full dose (theta x2.0, alpha x1.5), so dose 0.5 is theta x1.5 /
alpha x1.25, the Week 2 half dose. Channels: all five PSG channels.

Detection (the RT arm's reading, and reported for both arms): a
swap-null p-value per repeat, 100 within-participant session-label
swaps planned (the gate's second basis used 100), p = (1 + count of
nulls >= observed) / (swaps + 1). Computing a null statistic means
13 logistic-regression fits, so the null stops early once 5 nulls
reach the observed value: the detection verdict (p < 0.05) is then
already decided, because a completed 100-swap count could only be
larger, putting p at 6/101 or above. Counts and swaps actually run
are recorded for every null.

Everything downstream of the injection is the frozen pipeline:
features re-extracted by the adapter, folds and model from
src/evaluation/drozy_baseline.py and src/models/baseline.py,
untouched. Results are appended to the summary tables as each
combination finishes, and combinations already present in a table
are skipped, so an interrupted run resumes where it stopped.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.fft import irfft, rfft, rfftfreq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.drozy_adapter import CHANNELS, FEATURE_NAMES, FS_OUT, \
    extract_features
from src.evaluation.closure_pack import lopo_folds, participant_boot_ci
from src.evaluation.drozy_baseline import load_all, lopo_statistic, \
    swapped_labels, write_run
from src.models.baseline import run_folds, summarise

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
TAB = ROOT / "results" / "tables"
DATE = "2026-11-21"

DOSES = [0.0, 0.25, 0.5, 0.75, 1.0]
REPEATS = 5
PERM_SEED0 = 70_000      # repeat s uses rng 70000 + s for its label swap
NULL_SEED0 = 20_000      # swap p of every null uses rng 20000 + p (gate rule)
NULL_SWAPS = 100         # planned swaps per null (the gate's second basis)
NULL_STOP = 5            # early stop: detection verdict decided at count 5
# Reported uninjected participant-mean balanced accuracies (stored
# precision in results/tables/drozy_lopo_aggregate.csv); the dose-0
# identity check compares against these.
EXPECTED = {"kss": 0.6022, "rt": 0.4822}

LEAKAGE = """Leakage checklist:
- [x] Normalisation (StandardScaler) fit on training fold only (Pipeline, per fold)
- [x] Median imputation for masked-channel NaNs fit on training fold only
- [x] Class weights computed from training fold only (class_weight='balanced')
- [x] No overlapping windows across splits (tumbling windows; splits at participant level)
- [x] Held-out participant contributes nothing to fitting, imputation, scaling or tuning
- [x] No tuning on test data (C=1.0 locked default)
- [x] Injection is the declared Amendment 2 manipulation: it uses the run's own labels by design (a synthetic effect painted on fatigued-labelled windows), applied to the whole arm before splitting, exactly as in the Week 2 FatigueSet control; it moves no information from any test fold into fitting beyond that declared manipulation

Environment: local VM (CPU), scikit-learn 1.9.1, MNE 1.13.2.
"""


def dose_gains(dose):
    """(theta, alpha) power gains at a ladder dose, by linear
    interpolation between dose 0 (x1.0, x1.0) and full dose
    (x2.0, x1.5)."""
    return 1.0 + dose * (2.0 - 1.0), 1.0 + dose * (1.5 - 1.0)


def arm_frame_pos(idx, table):
    """The DROZY baseline's arm frame (same selection as
    src/evaluation/drozy_baseline.arm_frame: each included
    participant's alert session labelled 0, fatigued session 1),
    with one extra column, pos: the row number of each window in
    data/processed/drozy_windows.npy, so the window arrays can be
    pulled for injection. The dose-0 identity check is what proves
    this frame and the baseline's frame describe the same windows:
    if they did not, the identity scores would not reproduce."""
    parts = []
    for r in table[table.included].itertuples():
        for test, lab in ((int(r.alert_test), 0), (int(r.fatigued_test), 1)):
            m = (idx.participant == r.participant) & (idx.test == test)
            sel = idx[m].copy()
            sel["label"] = lab
            sel["pos"] = np.where(m)[0]
            parts.append(sel)
    df = pd.concat(parts).reset_index(drop=True)
    df["session"] = "S" + df.test.astype(str)
    return df


def inject(windows, df, theta_gain, alpha_gain):
    """Spectral-gain injection into the windows df labels fatigued,
    valid channels only, mirroring the Week 2 control. windows is
    the (n, 5, 512) float32 array aligned row by row with df."""
    out = windows.copy()
    freqs = rfftfreq(windows.shape[2], d=1.0 / FS_OUT)
    th = (freqs >= 4) & (freqs < 8)
    al = (freqs >= 8) & (freqs < 12)
    for i in np.where(df.label.values == 1)[0]:
        for ci, c in enumerate(CHANNELS):
            if not bool(df[f"valid_{c}"].iloc[i]):
                continue
            X = rfft(out[i, ci].astype(np.float64))
            X[th] *= np.sqrt(theta_gain)
            X[al] *= np.sqrt(alpha_gain)
            out[i, ci] = irfft(X, n=windows.shape[2]).astype(out.dtype)
    return out


def run_lopo(df, feats):
    """One LOPO pass through the frozen fold machinery. Returns
    (per-participant table, predictions, skipped folds, participant-
    mean BA, CI low, CI high, n)."""
    preds, skipped = run_folds(df, feats, lopo_folds(df))
    per = summarise(preds, "participant")
    pt, lo, hi, n = participant_boot_ci(per.balanced_accuracy.values)
    return per, preds, skipped, pt, lo, hi, n


def swap_null(df, feats, observed, paired_feats=None):
    """Swap-null for a LOPO statistic on (df, feats).

    With paired_feats None, the null statistics are the LOPO
    participant-mean BA of feats under within-participant session-
    label swaps. With paired_feats given, each null statistic is the
    increment: BA of feats minus BA of paired_feats under the SAME
    swap pattern (the companion's null for the increment).

    Early stopping per the amendment: the loop stops once NULL_STOP
    nulls reach the observed value, since the detection verdict is
    decided there. Returns a dict with the counts actually run; the
    p-value is exact only when all planned swaps ran, otherwise a
    lower bound (6/101 for a stop at count 5) is reported and the
    verdict is 'not detected'."""
    count, done, stats = 0, 0, []
    for p in range(NULL_SWAPS):
        rng = np.random.default_rng(NULL_SEED0 + p)
        y = swapped_labels(df, rng)
        stat, _ = lopo_statistic(df, feats, y=y)
        if paired_feats is not None:
            base, _ = lopo_statistic(df, paired_feats, y=y)
            stat = stat - base
        stats.append(stat)
        done += 1
        if stat >= observed:
            count += 1
        if count >= NULL_STOP:
            break
    exact = done == NULL_SWAPS
    return {"null_swaps_planned": NULL_SWAPS, "null_swaps_done": done,
            "null_count_ge": count,
            "p_value": float((1 + count) / (NULL_SWAPS + 1)) if exact
                       else None,
            "p_lower_bound": float((1 + count) / (NULL_SWAPS + 1)),
            "detected": bool(exact and (1 + count) / (NULL_SWAPS + 1)
                             < 0.05),
            "null_stats": stats}


def band_shift(feats_inj, feats_base, df):
    """Manipulation check, as in Week 2: mean log10 band-power shift
    in the injected (fatigued-labelled) windows, per band, averaged
    over the five channels' log-power features. Should land close to
    log10(power gain) for theta and alpha."""
    fat = df.label.values == 1
    out = {}
    for band in ("theta", "alpha"):
        cols = [f"logpow_{c}_{band}" for c in CHANNELS]
        out[band] = float((feats_inj.loc[fat, cols]
                           - feats_base.loc[fat, cols]).mean().mean())
    return out


def append_row(path, row, key_cols):
    """Append one summary row unless its key is already in the file
    (resume rule). Returns True if the row was written."""
    if path.exists():
        old = pd.read_csv(path)
        dup = old
        for c in key_cols:
            dup = dup[dup[c] == row[c]]
        if len(dup):
            return False
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(),
                               index=False)
    return True


def load_arm(arm):
    idx, _, tables = load_all()
    arr = np.load(PROC / "drozy_windows.npy")
    df = arm_frame_pos(idx, tables[arm])
    windows = np.ascontiguousarray(arr[df.pos.values])
    return df, windows


def base_features(arm, df, windows):
    """Dose-0 features through the injection code path (gains 1.0):
    the reference feature set for the identity check, for the
    permuted ladder's dose 0, and for the companion's uninjected
    statistic."""
    inj = inject(windows, df, 1.0, 1.0)
    return extract_features(inj, df)


# ---------------------------------------------------------------- identity
def identity():
    agg = pd.read_csv(TAB / "drozy_lopo_aggregate.csv").set_index("combo")
    rows, ok_all = [], True
    for arm in ("kss", "rt"):
        df, windows = load_arm(arm)
        feats0 = base_features(arm, df, windows)
        per, preds, skipped, pt, lo, hi, n = run_lopo(df, feats0)
        stored = float(agg.loc[f"{arm}_all", "balanced_accuracy"])
        diff = abs(pt - stored)
        ok = diff <= 1e-9
        ok_all &= ok
        rows.append({"arm": arm, "n_participants": n,
                     "ba_dose0_injection_path": pt,
                     "ba_reported": stored, "abs_diff": diff,
                     "identity_ok": ok})
        folder = f"{DATE}_poscontrol_dose0-identity_drozy-{arm}_all_s0"
        notes = f"""# {folder}

Amendment 2 dose-0 identity check, {arm.upper()} arm, all-channel
features. Dose 0 (theta power x1.0, alpha power x1.0) was pushed
through the injection code path on the true labels and the frozen
LOPO run repeated. The point is adapter validation, not a result:
if this number is not the reported uninjected score, the injection
path (or the window alignment behind it) is not faithful and the
control does not proceed.

Result: balanced accuracy {pt:.4f} against the reported {stored:.4f}
(abs diff {diff:.2e}); identity {'OK' if ok else 'FAILED'}.
Skipped folds: {skipped if skipped else 'none'}.

{LEAKAGE}"""
        write_run(folder, folder, per, preds, skipped,
                  {"control": "Amendment 2 dose-0 identity check",
                   "arm": arm, "dose": 0.0, "theta_gain": 1.0,
                   "alpha_gain": 1.0, "labels": "true labels"},
                  notes)
        print(f"identity {arm}: BA {pt:.6f} vs reported {stored:.6f} "
              f"diff {diff:.2e} -> {'OK' if ok else 'FAILED'}",
              flush=True)
    pd.DataFrame(rows).to_csv(TAB / "drozy_poscontrol_dose0.csv",
                              index=False)
    if not ok_all:
        print("IDENTITY CHECK FAILED - stopping before any control "
              "run, per Amendment 2.", flush=True)
    return ok_all


# ---------------------------------------------------------------- permuted
def permuted():
    summary_path = TAB / "drozy_poscontrol_permuted.csv"
    nulls_path = TAB / "drozy_poscontrol_nulls.csv"
    for arm in ("kss", "rt"):
        df, windows = load_arm(arm)
        feats0 = base_features(arm, df, windows)
        for dose in DOSES:
            g_th, g_al = dose_gains(dose)
            for rep in range(REPEATS):
                key = {"arm": arm, "dose": dose, "repeat": rep}
                if summary_path.exists():
                    old = pd.read_csv(summary_path)
                    done = old[(old.arm == arm) & (old.dose == dose)
                               & (old.repeat == rep)]
                    if len(done):
                        print(f"permuted {arm} dose {dose} rep {rep}: "
                              f"already recorded, skipping", flush=True)
                        continue
                rng = np.random.default_rng(PERM_SEED0 + rep)
                df_p = df.copy()
                df_p["label"] = swapped_labels(df, rng)
                if dose == 0.0:
                    feats_d = feats0
                else:
                    inj = inject(windows, df_p, g_th, g_al)
                    feats_d = extract_features(inj, df_p)
                per, preds, skipped, pt, lo, hi, n = run_lopo(df_p,
                                                              feats_d)
                null = swap_null(df_p, feats_d, pt)
                for s_i, s_v in enumerate(null.pop("null_stats")):
                    append_row(nulls_path,
                               {"control": "permuted", **key,
                                "swap": s_i, "stat": s_v},
                               ["control", "arm", "dose", "repeat",
                                "swap"])
                shift = band_shift(feats_d, feats0, df_p)
                row = {**key, "theta_gain": g_th, "alpha_gain": g_al,
                       "perm_seed": PERM_SEED0 + rep,
                       "n_participants": n, "recovery_ba": pt,
                       "ba_ci95_low": lo, "ba_ci95_high": hi,
                       "theta_log10_shift": shift["theta"],
                       "alpha_log10_shift": shift["alpha"],
                       **{k: null[k] for k in
                          ("null_swaps_planned", "null_swaps_done",
                           "null_count_ge", "p_value", "p_lower_bound",
                           "detected")}}
                append_row(summary_path, row, ["arm", "dose", "repeat"])
                folder = (f"{DATE}_poscontrol_permuted_drozy-{arm}_all"
                          f"_dose{dose:g}_r{rep}_s0")
                notes = f"""# {folder}

Amendment 2 primary (permuted-label) positive control, {arm.upper()}
arm, dose {dose:g} (theta power x{g_th:.3f}, alpha power
x{g_al:.3f}), repeat {rep}. The repeat's labels are a within-
participant session-label swap of the true labels (rng
{PERM_SEED0 + rep}), so any natural label-signal link is null by
construction; the injection was applied to the permuted-fatigued
windows and the frozen LOPO ran under the permuted labels. Recovery
here can only be the injected effect.

Result: recovery balanced accuracy {pt:.3f} [{lo:.3f}, {hi:.3f}]
(participant bootstrap). Swap null on this repeat's injected
features: {null['null_swaps_done']} of {NULL_SWAPS} planned swaps
run, {null['null_count_ge']} reached the observed value,
{'p = ' + format(null['p_value'], '.3f') if null['p_value'] is not None else 'p >= ' + format(null['p_lower_bound'], '.3f') + ' (early stop; detection verdict decided)'}.
Manipulation check (mean log10 shift in injected windows): theta
{shift['theta']:.3f} (target {np.log10(g_th):.3f}), alpha
{shift['alpha']:.3f} (target {np.log10(g_al):.3f}).
Skipped folds: {skipped if skipped else 'none'}.

{LEAKAGE}"""
                write_run(folder, folder, per, preds, skipped,
                          {"control": "Amendment 2 permuted-label "
                                      "positive control",
                           "arm": arm, "dose": dose,
                           "theta_gain": g_th, "alpha_gain": g_al,
                           "repeat": rep,
                           "perm_seed": PERM_SEED0 + rep,
                           "labels": "within-participant swapped "
                                     "(permuted)"},
                          notes)
                print(f"permuted {arm} dose {dose} rep {rep}: "
                      f"recovery {pt:.3f} detected={null['detected']} "
                      f"(null count {null['null_count_ge']} in "
                      f"{null['null_swaps_done']} swaps)", flush=True)


# --------------------------------------------------------------- companion
def companion():
    summary_path = TAB / "drozy_poscontrol_companion.csv"
    nulls_path = TAB / "drozy_poscontrol_nulls.csv"
    for arm in ("kss", "rt"):
        df, windows = load_arm(arm)
        feats0 = base_features(arm, df, windows)
        _, _, _, base_pt, _, _, _ = run_lopo(df, feats0)
        for dose in DOSES:
            if dose == 0.0:
                continue  # increment is 0 by construction at dose 0
            key = {"arm": arm, "dose": dose}
            if summary_path.exists():
                old = pd.read_csv(summary_path)
                if len(old[(old.arm == arm) & (old.dose == dose)]):
                    print(f"companion {arm} dose {dose}: already "
                          f"recorded, skipping", flush=True)
                    continue
            g_th, g_al = dose_gains(dose)
            inj = inject(windows, df, g_th, g_al)
            feats_d = extract_features(inj, df)
            per, preds, skipped, pt, lo, hi, n = run_lopo(df, feats_d)
            inc = pt - base_pt
            null = swap_null(df, feats_d, inc, paired_feats=feats0)
            for s_i, s_v in enumerate(null.pop("null_stats")):
                # repeat = -1: the companion has no repeats, but the
                # nulls file is shared with the permuted control and
                # its columns must line up.
                append_row(nulls_path,
                           {"control": "companion_increment", **key,
                            "repeat": -1, "swap": s_i, "stat": s_v},
                           ["control", "arm", "dose", "repeat", "swap"])
            shift = band_shift(feats_d, feats0, df)
            row = {**key, "theta_gain": g_th, "alpha_gain": g_al,
                   "n_participants": n, "ba_injected": pt,
                   "ba_uninjected": base_pt, "increment": inc,
                   "theta_log10_shift": shift["theta"],
                   "alpha_log10_shift": shift["alpha"],
                   **{k: null[k] for k in
                      ("null_swaps_planned", "null_swaps_done",
                       "null_count_ge", "p_value", "p_lower_bound",
                       "detected")}}
            append_row(summary_path, row, ["arm", "dose"])
            folder = (f"{DATE}_poscontrol_truelabel_drozy-{arm}_all"
                      f"_dose{dose:g}_s0")
            notes = f"""# {folder}

Amendment 2 companion (true-label) positive control, {arm.upper()}
arm, dose {dose:g} (theta power x{g_th:.3f}, alpha power
x{g_al:.3f}). The injection was applied to the truly fatigued
windows and the frozen LOPO ran under the true labels. Reported as
the increment over the arm's uninjected score, because on true
labels the natural signal and the injected effect are mixed.

Result: balanced accuracy {pt:.3f} against the uninjected
{base_pt:.3f}, increment {inc:+.3f}. Swap null on the increment
(injected and uninjected statistics computed under the same swap
pattern): {null['null_swaps_done']} of {NULL_SWAPS} planned swaps
run, {null['null_count_ge']} reached the observed increment,
{'p = ' + format(null['p_value'], '.3f') if null['p_value'] is not None else 'p >= ' + format(null['p_lower_bound'], '.3f') + ' (early stop; verdict decided)'}.
Manipulation check (mean log10 shift in injected windows): theta
{shift['theta']:.3f} (target {np.log10(g_th):.3f}), alpha
{shift['alpha']:.3f} (target {np.log10(g_al):.3f}).
Skipped folds: {skipped if skipped else 'none'}.

{LEAKAGE}"""
            write_run(folder, folder, per, preds, skipped,
                      {"control": "Amendment 2 true-label companion",
                       "arm": arm, "dose": dose, "theta_gain": g_th,
                       "alpha_gain": g_al, "labels": "true labels"},
                      notes)
            print(f"companion {arm} dose {dose}: BA {pt:.3f} "
                  f"increment {inc:+.3f} detected={null['detected']} "
                  f"(null count {null['null_count_ge']} in "
                  f"{null['null_swaps_done']} swaps)", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "identity":
        sys.exit(0 if identity() else 1)
    elif cmd == "permuted":
        permuted()
    elif cmd == "companion":
        companion()
    else:
        if not identity():
            sys.exit(1)
        permuted()
        companion()
        print("CONTROL RUN COMPLETE", flush=True)
