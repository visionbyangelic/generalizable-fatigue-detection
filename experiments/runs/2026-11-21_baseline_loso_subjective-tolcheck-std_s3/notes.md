# Baseline LOPO, subjective labels, solver check (std), cap seed 3

RQ4 audit support run (docs/rq4_session_labels.md, bit-identity audit).
Identical windows, features, folds and seeds to the Week 2 primary
LOPO; the only change is the solver setting: standard solver (locked defaults: tol=1e-4, max_iter=1000). Purpose: record
the solver's convergence behaviour (n_iter_, warnings), which the
Week 2 folders did not store, and measure how much of the small
probability gap between the stored Week 2 numbers and the current
stack is solver slack.

Solver record this seed: n_iter_ per fold: [69, 73, 71, 72, 71, 74, 61, 67, 72, 70, 72];
ConvergenceWarnings: 0.

Leakage checklist:
- [x] Normalisation (StandardScaler) fit on training fold only (Pipeline, per fold)
- [x] Median imputation fit on training fold only
- [x] Class weights computed from training fold only (class_weight='balanced')
- [x] No overlapping windows across splits (tumbling windows; split at participant level)
- [x] No tuning on test data (C=1.0 locked; tolerance is the studied variable, fixed before the run)

Environment: pinned current stack (docs/environment_pin.md).
