"""Pre-declared precision variants V1 / V2 (track.md 20 Nov 2026;
docs/decision_table_lock.md status note; I: "confirm the 55% line
and try to make it better" - "better" scoped IN ADVANCE to measuring
more precisely under the same honest rules).

Both variants are SENSITIVITIES. Nothing in this module declares a
verdict under the lock doc's tier table; outputs are estimates + CIs +
gate inputs only. What may be claimed from them is decided by the
lock's tier table, not by anything computed here, and until that is
settled they are reported as estimates and nothing more.

  V1  UNCAPPED + CLASS-WEIGHTED. Full Amendment-1 retained pool
      (1,053 windows: 564 alert / 489 fatigued), NO per-session M1 cap.
      Imbalance handled by class weights: sklearn
      class_weight='balanced', w_c = n_train / (2 * n_c) with n_c the
      TRAINING-fold class count, fit inside each training fold (same
      Pipeline as the primary). Feature sets: P (43) and F (frontal
      20). Protocols: pooled leave-one-session-out + LOPO.
      Design note vs the capped primary: single-class sessions (which
      the cap reduced to zero windows) contribute their windows to the
      TRAINING pool here; they still cannot be test folds.

  V2  WITHIN-SESSION-NORMALISED FEATURES (declared sensitivity,
      fixed in advance). Each continuous feature is centred/scaled
      per session using that session's OWN windows (nanmean / nanstd,
      ddof=0; labels never used). This is TRANSDUCTIVE at session
      level: a test session's windows are normalised with statistics
      that include that session's own unlabelled windows. Indicator
      features (valid_TP9, valid_TP10, both_temporals_masked) are
      unchanged. Feature set: P (43). Protocols as V1.
      Guard: if a session-feature SD is 0 or undefined, the feature is
      centred only; if the session mean is undefined (all-NaN), the
      values stay NaN (training-fold median imputation downstream).

  V2-MASK  Mask-only arm under the V2 protocol. The decision-table
      lock requires V2 to carry its own mask-only arm and requires
      both variants to be reported together, failures included, so
      this arm is not optional: without it, a V2 change could be
      credited to the EEG features when the mask produced it. It uses
      the 5 indicator/coverage features of panel_additions.mask_only
      (valid_TP9, valid_TP10, both_temporals_masked, n_valid_temporals,
      n_valid_channels), built from the index, NOT session-normalised
      (they are indicators/coverage, not continuous EEG features),
      uncapped pool, same protocols - so any V2 change can be
      attributed between EEG features and the mask shortcut.

Seeds: there is no cap and no subsampling anywhere in these variants;
logistic regression is deterministic given the data, so each analysis
is a SINGLE deterministic pass (run folders carry the s0 suffix for
that pass). The only randomness is in resampling/permutation, with
fixed, stated generators: bootstrap CIs 10,000 resamples, rng 42;
permutations: ONE common set of 200 circular-shift label permutations
(rng 60000+p) reused across all feature sets and both protocols, so
the null SDs are paired across variants.

Gate inputs per feature set: block-permutation (circular-shift) null
SD of (a) the session-mean BA statistic and (b) the participant-mean
BA statistic from session-out folds, and (c) the participant-mean BA
statistic from LOPO folds; Delta_min = (z.95 + z.90) * SD = 2.9265 * SD
(lock doc section 2 formula, ~90% power, one-sided alpha 0.05, true
effect 0). Permutation fits use a mathematically identical shortcut:
the median imputer and StandardScaler are label-free, so they are fit
once per fold on the real features and only the LR (with its
training-fold balanced class weights) is refit per permutation.

Outputs: run folders per experiments/README.md, summary tables in
results/tables/, gate tables (variants_gate_inputs.csv,
variants_gate_permutation_nulls.csv).
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.metrics import binary_metrics
from src.features.bandpower import FEATURE_NAMES, extract_features
from src.models.baseline import (METRIC_COLS, make_model, run_folds,
                                 sessionout_folds, summarise)

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
RUNS = ROOT / "experiments" / "runs"
TAB = ROOT / "results" / "tables"
DATE = "2026-11-20"
SESOI_UPPER = 0.55
Z_GATE = float(stats.norm.ppf(0.95) + stats.norm.ppf(0.90))  # 2.9265
N_PERM = 200
PERM_RNG_BASE = 60000
BOOT_N = 10000
BOOT_RNG = 42
INDICATORS = ["valid_TP9", "valid_TP10", "both_temporals_masked"]
FRONTAL_FEATURES = (
    [f"logpow_{c}_{b}" for c in ["AF7", "AF8"]
     for b in ["delta", "theta", "alpha", "beta"]]
    + [f"relpow_{c}_{b}" for c in ["AF7", "AF8"]
       for b in ["delta", "theta", "alpha", "beta"]]
    + [f"frontal_ratio_{r}" for r in ["th_al_over_beta", "al_over_beta",
                                      "th_al_over_al_beta", "th_over_beta"]])
CI_METRICS = ["balanced_accuracy", "macro_f1", "roc_auc", "mcc"]


def git_commit():
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()


def lopo_folds(df):
    keys = []
    for part in df.participant.unique():
        te = (df.participant == part).values
        keys.append((part, ~te, te))
    return keys


def boot_ci(vals, n_boot=BOOT_N, seed=BOOT_RNG):
    """Bootstrap 95% CI for the mean of per-unit values."""
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    means = vals[idx].mean(axis=1)
    return float(vals.mean()), float(np.percentile(means, 2.5)), \
        float(np.percentile(means, 97.5)), len(vals)


def mask_only_frame(idx):
    """Identical 5-feature set to panel_additions.mask_only."""
    vt = idx[["valid_TP9", "valid_TP10"]].astype(float)
    return pd.DataFrame({
        "valid_TP9": vt.valid_TP9, "valid_TP10": vt.valid_TP10,
        "both_temporals_masked": ((vt.valid_TP9 == 0)
                                  & (vt.valid_TP10 == 0)).astype(float),
        "n_valid_temporals": vt.valid_TP9 + vt.valid_TP10,
        "n_valid_channels": vt.valid_TP9 + vt.valid_TP10
                            + idx.valid_AF7.astype(float)
                            + idx.valid_AF8.astype(float)},
        index=idx.index)


def session_normalise(feats, idx):
    """V2: centre/scale each continuous feature within its session.
    Returns (normalised DataFrame, guard-count dict)."""
    cont = [c for c in FEATURE_NAMES if c not in INDICATORS]
    out = feats.copy()
    guards = {"centre_only_sd0_or_undefined": 0, "all_nan_left_nan": 0}
    keys = list(zip(idx.participant, idx.session))
    frame = pd.DataFrame({"key": keys})
    for col in cont:
        x = feats[col].astype(float)
        tmp = pd.DataFrame({"key": keys, "x": x.values})
        g = tmp.groupby("key")["x"]
        mean = g.transform(lambda s: np.nanmean(s) if s.notna().any()
                           else np.nan)
        sd = g.transform(lambda s: np.nanstd(s) if s.notna().any()
                         else np.nan)
        # guard accounting per (session, feature)
        per = tmp.groupby("key")["x"].agg(
            lambda s: (np.nanmean(s) if s.notna().any() else np.nan,
                       np.nanstd(s) if s.notna().any() else np.nan))
        for m_, s_ in per:
            if np.isnan(m_):
                guards["all_nan_left_nan"] += 1
            elif np.isnan(s_) or s_ == 0:
                guards["centre_only_sd0_or_undefined"] += 1
        z = (x.values - mean.values) / np.where(
            (sd.values > 0) & np.isfinite(sd.values), sd.values, 1.0)
        out[col] = np.where(np.isnan(mean.values), np.nan, z)
    return out, guards


# ------------------------------------------------------------------ runs
def write_variant_run(name, variant_label, protocol, per_group, preds,
                      skipped, group_col, feature_desc, split_desc,
                      extra_cfg, extra_notes):
    d = RUNS / name
    d.mkdir(parents=True, exist_ok=True)
    metrics = per_group.copy()
    agg = {c: per_group[c].mean() for c in METRIC_COLS if c in per_group}
    metrics.loc[len(metrics)] = {group_col: "AGGREGATE_mean_over_groups",
                                 **agg}
    metrics.to_csv(d / "metrics.csv", index=False)
    preds.to_csv(d / "predictions.csv", index=False)
    cfg = {"model": "logistic_regression", "C": 1.0,
           "class_weight": "balanced: w_c = n_train / (2 * n_c), "
                           "computed on the training fold only",
           "features": feature_desc,
           "imputation": "training-fold median",
           "scaling": "training-fold StandardScaler (Pipeline, per fold)",
           "protocol": protocol, "labels": "subjective (Amendment 1 windows)",
           "variant": variant_label,
           "data": "windows_a1_uncapped FULL retained pool (1053 windows), "
                   "NO per-session M1 cap",
           "seed": "s0 = single deterministic pass (no cap, no "
                   "subsampling; LR deterministic given the data); "
                   "bootstrap rng 42 where used",
           "status": "PRE-DECLARED SENSITIVITY (track.md 20 Nov 2026; "
                     "decision_table_lock.md). Estimates + CIs only; no "
                     "verdict under the lock doc tier table is declared. "
                     "Reporting status awaits the lead reviewer's ruling.",
           "git_commit": git_commit()}
    cfg.update(extra_cfg)
    (d / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (d / "notes.md").write_text(f"""# {name}

PRE-DECLARED SENSITIVITY - estimates and intervals only. No verdict
under the decision-table lock is declared in this run; reporting
status awaits the lead reviewer's ruling.

Split definition: {split_desc}. A fold counts only if train and test
both contain both classes; skipped folds: {skipped if skipped else 'none'}.

{extra_notes}
Leakage checklist:
- [x] Normalisation (StandardScaler) fit on training fold only (Pipeline, per fold)
- [x] Median imputation fit on training fold only
- [x] Class weights from training fold only (class_weight='balanced')
- [x] No overlapping windows across splits (tumbling windows; splits at session/participant level)
- [x] No tuning on test data (C=1.0 locked default)
- [x] Label rules locked (docs/label_rules.md incl. Amendment 1) before any training
- [x] Variant declared in advance (track.md, 20 Nov 2026) - not chosen from results

Environment: local VM (CPU), scikit-learn 1.9.1.
""")
    return agg


def run_protocol(df, feats, protocol, fold_fn, group_col):
    preds, skipped = run_folds(df, feats, fold_fn(df))
    if group_col == "heldout_session":
        per = summarise(preds, "fold").rename(
            columns={"fold": "heldout_session"})
    else:
        per = summarise(preds, "participant")
    return per, preds, skipped


def write_summary(per, group_col, out_name):
    """Per-unit rows + AGGREGATE row carrying bootstrap CIs (rng 42)."""
    summ = per.copy()
    agg_row = {group_col: "AGGREGATE_bootstrap_over_units"}
    for c in METRIC_COLS:
        if c in per:
            agg_row[c] = per[c].mean()
    for m in CI_METRICS:
        _, lo, hi, n = boot_ci(per[m].values)
        agg_row[f"{m}_ci95_low"] = lo
        agg_row[f"{m}_ci95_high"] = hi
    agg_row["n_units_bootstrapped"] = n
    summ = pd.concat([summ, pd.DataFrame([agg_row])], ignore_index=True)
    summ.to_csv(TAB / out_name, index=False)
    return agg_row


VARIANT_NOTES = {
    "V1": """Variant V1 - UNCAPPED + CLASS-WEIGHTED: the full Amendment-1
retained pool (1,053 windows: 564 alert / 489 fatigued) with NO
per-session M1 cap. Class imbalance is handled by the training-fold
balanced class weights instead of by discarding alert windows (the
cap discarded ~44% of retained alert windows in the primary). Design
difference vs the capped primary, stated openly: single-class
sessions, which the cap reduced to zero windows, contribute their
windows to TRAINING pools here; they still cannot be test folds.
""",
    "V2": """Variant V2 - WITHIN-SESSION-NORMALISED FEATURES: each
continuous feature was centred/scaled per session using that
session's own windows (nanmean / nanstd ddof=0 over the session's
retained windows; labels never used). TRANSDUCTIVITY, stated as
required: this normalisation uses each session's own UNLABELLED
windows - including, for a test session, that test session's own
windows - so it is transductive at session level by design (Reviewer 2
review 10 declared sensitivity). It is NOT a substitute for the
primary's train-fold-only normalisation; the Pipeline's imputation
and StandardScaler still fit on training folds only, on top of the
session-normalised features. Indicator features are unchanged.
""",
    "V2MASK": """Variant V2-MASK - mask-only arm under the V2 protocol
(Reviewer 1 ruling R3): the 5 indicator/coverage features of
panel_additions.mask_only (valid_TP9, valid_TP10,
both_temporals_masked, n_valid_temporals, n_valid_channels), built
from the window index - NO EEG features. These features are
indicators/coverage counts, not continuous EEG features, so the V2
session normalisation does not apply to them (unchanged, exactly as
the indicator features are unchanged inside the V2 EEG feature set).
Same uncapped pool and protocols as V2, so any V2 change can be
attributed between EEG features and the mask shortcut. TRANSDUCTIVITY
note: the V2 normalisation statement does not apply to this arm (no
normalised features); it shares V2's pool and splits only.
""",
}

SPLIT_DESCS = {
    "session-out": "pooled: hold out one (participant, session); train "
                   "on all remaining windows of all participants "
                   "(uncapped pool, so single-class sessions contribute "
                   "training windows but cannot be test folds)",
    "loso": "leave-one-PARTICIPANT-out: hold out ALL windows of one "
            "participant; train on the other 11 participants (uncapped "
            "pool). P05 cannot be a test fold (its retained windows "
            "are single-class: 5 alert / 0 fatigued) but its windows "
            "are in the training pools of the other folds - unlike "
            "the capped primary, where the cap removed them entirely",
}


def analyse(df, feats, tag, variant_key, feature_desc, out_prefix):
    """Run session-out + LOPO for one feature set; write runs + summaries.
    Returns dict of aggregate rows keyed by protocol."""
    results = {}
    for protocol, fold_fn, gcol in [
            ("session-out", sessionout_folds, "heldout_session"),
            ("loso", lopo_folds, "participant")]:
        per, preds, skipped = run_protocol(df, feats, protocol, fold_fn,
                                           gcol)
        name = (f"{DATE}_baseline_{protocol}_subjective_{tag}_s0")
        write_variant_run(name, variant_key, protocol, per, preds,
                          skipped, gcol, feature_desc,
                          SPLIT_DESCS[protocol], {}, VARIANT_NOTES[variant_key])
        agg_row = write_summary(
            per, gcol, f"baseline_{protocol}_subjective_{out_prefix}_summary.csv")
        results[protocol] = {"per": per, "agg_row": agg_row,
                             "skipped": skipped}
        print(f"{tag} {protocol}: units={len(per)} skipped={skipped} "
              f"BA={agg_row['balanced_accuracy']:.3f} "
              f"[{agg_row['balanced_accuracy_ci95_low']:.3f}, "
              f"{agg_row['balanced_accuracy_ci95_high']:.3f}] "
              f"AUC={agg_row['roc_auc']:.3f} MCC={agg_row['mcc']:.3f} "
              f"F1={agg_row['macro_f1']:.3f}", flush=True)
    return results


# ------------------------------------------------------------------ gate
def permuted_labels(y, sess_key):
    """One common set of N_PERM circular-shift label permutations."""
    perms = []
    for p in range(N_PERM):
        rng = np.random.default_rng(PERM_RNG_BASE + p)
        yp = y.copy()
        for s in np.unique(sess_key):
            msk = np.where(sess_key == s)[0]
            yp[msk] = np.roll(yp[msk], rng.integers(1, len(msk)))
        perms.append(yp)
    return perms


def _fold_transforms(df, feats, fold_keys):
    """Per fold: label-free imputer+scaler fit on the training rows;
    returns transformed X for all rows + the fold keys."""
    X = feats.values
    out = []
    for fid, tr, te in fold_keys:
        pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler())])
        Xt = pipe.fit_transform(X[tr])
        Xall = np.empty_like(X, dtype=float)
        Xall[tr] = Xt
        Xall[te] = pipe.transform(X[te])
        rest = ~(tr | te)
        if rest.any():
            Xall[rest] = pipe.transform(X[rest])
        out.append((fid, tr, te, Xall))
    return out


def _perm_stats(df, feats, perms, fold_fn, kind):
    """kind: 'session-out' -> session-mean + participant-mean stats;
    'loso' -> participant-mean stat. BA only (the gate statistic)."""
    folds = _fold_transforms(df, feats, fold_fn(df))
    y = df.label.values
    rows = []
    for pi, yp in enumerate(perms):
        per_fold = []
        for fid, tr, te, Xall in folds:
            ytr, yte = yp[tr], yp[te]
            if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
                continue
            m = LogisticRegression(C=1.0, class_weight="balanced",
                                   max_iter=1000)
            m.fit(Xall[tr], ytr)
            ba = binary_metrics(yte, m.predict(Xall[te]))[
                "balanced_accuracy"]
            per_fold.append({"fold": fid,
                             "participant": str(fid).split("/")[0],
                             "balanced_accuracy": ba})
        pf = pd.DataFrame(per_fold)
        row = {"perm_index": pi}
        if kind == "session-out":
            row["ba_session_mean"] = pf.balanced_accuracy.mean()
            row["ba_participant_mean"] = pf.groupby(
                "participant").balanced_accuracy.mean().mean()
        else:
            row["ba_participant_mean"] = pf.balanced_accuracy.mean()
        rows.append(row)
    return pd.DataFrame(rows)


def gate_for(tag, variant_key, df, feats, obs):
    """obs: analyse() results dict. Writes the gate run folder and
    returns (gate_rows, null_rows)."""
    y = df.label.values.copy()
    sess_key = (df.participant + "/" + df.session).to_numpy()
    perms = permuted_labels(y, sess_key)
    null_rows, gate_rows = [], []
    specs = [("session-out", sessionout_folds,
              [("session_mean", "ba_session_mean", "session-out"),
               ("participant_mean", "ba_participant_mean", "session-out")]),
             ("loso", lopo_folds,
              [("participant_mean", "ba_participant_mean", "loso")])]
    for kind, fold_fn, stat_specs in specs:
        st = _perm_stats(df, feats, perms, fold_fn, kind)
        st.insert(0, "variant", tag)
        st.insert(1, "protocol", kind)
        null_rows.append(st)
        for stat_name, col, proto in stat_specs:
            sd = float(st[col].std(ddof=1))
            agg = obs[proto]["agg_row"]
            gate_rows.append({
                "variant": tag, "feature_set_protocol": proto,
                "statistic": stat_name,
                "n_permutations": len(st),
                "permutation": "circular shifts within session, common "
                               f"set of {N_PERM}, rng {PERM_RNG_BASE}+p",
                "null_sd": sd, "delta_min_90pct_power": Z_GATE * sd,
                "observed_balanced_accuracy":
                    agg["balanced_accuracy"],
                "observed_ba_ci95_low": agg["balanced_accuracy_ci95_low"],
                "observed_ba_ci95_high":
                    agg["balanced_accuracy_ci95_high"],
                "sesoi_upper_line": SESOI_UPPER,
                "note": "gate INPUTS only - no verdict declared; "
                        "reporting status awaits the lead reviewer's "
                        "ruling"})
        print(f"gate {tag} {kind}: "
              + "; ".join(f"{c} SD={st[c].std(ddof=1):.4f} "
                          f"dmin={Z_GATE * st[c].std(ddof=1):.3f}"
                          for _, c, _ in stat_specs), flush=True)
    nulls = pd.concat(null_rows, ignore_index=True)
    grows = pd.DataFrame(gate_rows)
    d = RUNS / f"{DATE}_baseline_gate-inputs_subjective_{tag}_s0"
    d.mkdir(parents=True, exist_ok=True)
    grows.to_csv(d / "metrics.csv", index=False)
    cfg = {"model": "logistic_regression (permutation reruns)",
           "variant": variant_key,
           "permutations": f"circular shifts within session x{N_PERM} "
                           f"(rng {PERM_RNG_BASE}+p), one common set "
                           "reused across feature sets and protocols",
           "power_formula": "delta_min = (z_0.95 + z_0.90) * SD = "
                            "2.9265 * SD (lock doc section 2)",
           "labels": "subjective (Amendment 1)",
           "data": "windows_a1_uncapped FULL retained pool, no cap",
           "status": "gate INPUTS only - no verdict declared",
           "git_commit": git_commit()}
    (d / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (d / "notes.md").write_text(f"""# Gate inputs - {tag}

Feasibility-gate INPUTS for the {variant_key} variant (lock doc
section 2 supplies the formula; the bound itself was confirmed by
I at Delta = 0.05 and does not move). This run declares NO
verdict: it reports the label-free permutation null SDs, the implied
Delta_min at ~90% power, and the observed estimates/intervals next to
the 0.55 line, as inputs for the lead reviewer's ruling.

Permutation scheme: circular shifts of labels within session
(preserves per-session class counts and the within-session label
block structure), {N_PERM} permutations, rng {PERM_RNG_BASE}+p - one
common set reused across all variant feature sets and both protocols,
so the null SDs are paired across variants. Per-permutation fits use
the identical model; the median imputer and StandardScaler are
label-free and are fit once per fold on the real features (only the
LR, with its training-fold balanced class weights, is refit per
permutation - mathematically identical to refitting the Pipeline).

Gate inputs:
{grows.to_string(index=False)}

Environment: local VM (CPU), scikit-learn 1.9.1.
""")
    return grows, nulls


def main():
    arr = np.load(PROC / "windows_a1_uncapped.npy")
    idx = pd.read_csv(PROC / "windows_a1_uncapped_index.csv") \
        .reset_index(drop=True)
    df = idx
    print(f"pool: {len(df)} windows, alert={(df.label == 0).sum()}, "
          f"fatigued={(df.label == 1).sum()}", flush=True)
    print("extracting features ...", flush=True)
    feats = extract_features(arr, idx)

    all_gate, all_nulls = [], []

    # ---- V1, P (43) and F (20)
    res_v1p = analyse(df, feats, "V1uncapped", "V1",
                      "bandpower_43_amendment1 (P, masked all-channel)",
                      "V1uncapped")
    g, n = gate_for("V1uncapped", "V1", df, feats, res_v1p)
    all_gate.append(g); all_nulls.append(n)

    fidx = [FEATURE_NAMES.index(c) for c in FRONTAL_FEATURES]
    feats_f = feats.iloc[:, fidx]
    assert list(feats_f.columns) == FRONTAL_FEATURES
    res_v1f = analyse(df, feats_f, "V1uncapped-frontal", "V1",
                      "FRONTAL-ONLY (F): 20 features = logpow + relpow "
                      "for AF7/AF8 (4 bands each) + 4 frontal ratios "
                      "(AF7/AF8 means only). No temporals, no indicators.",
                      "V1uncapped_frontal")
    g, n = gate_for("V1uncapped-frontal", "V1", df, feats_f, res_v1f)
    all_gate.append(g); all_nulls.append(n)

    # ---- V2, P (43, session-normalised)
    feats_v2, guards = session_normalise(feats, idx)
    print(f"V2 normalisation guards: {guards}", flush=True)
    res_v2 = analyse(df, feats_v2, "V2sessnorm", "V2",
                     "bandpower_43_amendment1 (P) with WITHIN-SESSION "
                     "normalisation of the 40 continuous features "
                     "(per-session nanmean/nanstd, labels unused; "
                     "transductive at session level); indicators "
                     "unchanged",
                     "V2sessnorm")
    g, n = gate_for("V2sessnorm", "V2", df, feats_v2, res_v2)
    all_gate.append(g); all_nulls.append(n)

    # ---- V2 mask-only arm
    feats_mask = mask_only_frame(idx)
    res_vm = analyse(df, feats_mask, "V2sessnorm-maskonly", "V2MASK",
                     "MASK-ONLY: 5 indicator/coverage features "
                     "(valid_TP9, valid_TP10, both_temporals_masked, "
                     "n_valid_temporals, n_valid_channels) - identical "
                     "set to panel_additions.mask_only; no EEG features; "
                     "not session-normalised",
                     "V2sessnorm_maskonly")
    g, n = gate_for("V2sessnorm-maskonly", "V2MASK", df, feats_mask, res_vm)
    all_gate.append(g); all_nulls.append(n)

    pd.concat(all_gate, ignore_index=True).to_csv(
        TAB / "variants_gate_inputs.csv", index=False)
    pd.concat(all_nulls, ignore_index=True).to_csv(
        TAB / "variants_gate_permutation_nulls.csv", index=False)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
