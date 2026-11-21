# RQ4 2x2 replication, band-power logistic regression, chronological-reversed split, deterministic split (s0 by convention)

Post hoc arm, panel-specified (review 24). Subject 01 only (the
preliminary experiment's participant), day-level labels: whole
Session 01 = alert, whole Session 03 = fatigued. Retained
Amendment-1 windows: 24 (22 alert, 2 fatigued), so every split
trains on 16 windows (15 alert, 1 fatigued) and tests on 8
(7 alert, 1 fatigued); fatigued recall is therefore a 0-or-1
quantity in this arm and is reported with its counts.

Split definition: within each session, LAST floor(0.7*n) windows = train, first windows = test (direction sensitivity).

Leakage checklist (honest version):
- [x] Preprocessing statistics fit on training windows only (LR scaling/imputation; CNN normalisation and class weights)
- [ ] NO session sharing between train and test: VIOLATED BY DESIGN (both sessions, and the participant, appear on both sides; the violation is the object of the run)
- [x] No tuning on test data (all hyperparameters are the frozen pipeline's)
- [x] Label definition and split rule fixed in the driver docstring before the run (post hoc arm, panel-specified)

Environment: pinned current stack (docs/environment_pin.md).
