# Positive control - HETEROGENEOUS per-session injection

Reviewer 1 review 11: the homogeneous injection is the best case -
a real fatigue effect would vary in size (and possibly sign) across
sessions. This run injects per-session random-amplitude theta/alpha
effects centred on the homogeneous half-strength dose (theta x1.5,
alpha x1.25), with ~20% of sessions null or opposite-sign BY DESIGN
(seed 20261120; exact scheme in src/evaluation/closure_pack.py
hetero_gains(); realised per-session gains in
results/tables/diagnostics_positive_control_hetero_gains.csv;
realised session kinds: {'graded': 26, 'opposite': 3, 'null': 1}). Manipulation check in
results/tables/diagnostics_positive_control_hetero_check.csv.

Recovered (session-out, cap seeds 0-4): cross-seed mean BA
0.657, AUC 0.750,
MCC 0.317 (uninjected: 0.515 / 0.549 / 0.045;
homogeneous half-strength: 0.695 / 0.768 / 0.360). Per-session recovery
vs injected log theta gain: Spearman rho = 0.259 (p = 0.271,
n = 20 sessions) - descriptive.

Leakage checklist:
- [x] Injection is label-dependent BY DESIGN (sensitivity control);
      gains drawn per session from a fixed seed, before splitting
- [x] Scaling/imputation fit on training folds only
- [x] No tuning on test data (locked C=1.0)

Environment: local VM (CPU), scikit-learn.
