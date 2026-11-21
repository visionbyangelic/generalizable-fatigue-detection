# RQ4 2x2 replication, EEGNet CNN (frozen cnn.py configuration), random split, split seed 3, model seeds [0, 1, 2]

Post hoc arm, panel-specified (review 24). Subject 01 only (the
preliminary experiment's participant), day-level labels: whole
Session 01 = alert, whole Session 03 = fatigued. Retained
Amendment-1 windows: 24 (22 alert, 2 fatigued), so every split
trains on 16 windows (15 alert, 1 fatigued) and tests on 8
(7 alert, 1 fatigued); fatigued recall is therefore a 0-or-1
quantity in this arm and is reported with its counts.

Split definition: stratified random window split, per class n_test = max(1, round(0.3*n_class)), split seed as in the folder name; the same split is used for the LR and CNN cells.
CNN adaptation (forced, stated in the driver): no both-class validation carve exists at these counts (one fatigued training window), so the frozen early stopping is replaced by a fixed 100-epoch budget on all training windows; architecture, optimiser, class weights and normalisation are the frozen cnn.py configuration.

Leakage checklist (honest version):
- [x] Preprocessing statistics fit on training windows only (LR scaling/imputation; CNN normalisation and class weights)
- [ ] NO session sharing between train and test: VIOLATED BY DESIGN (both sessions, and the participant, appear on both sides; the violation is the object of the run)
- [x] No tuning on test data (all hyperparameters are the frozen pipeline's)
- [x] Label definition and split rule fixed in the driver docstring before the run (post hoc arm, panel-specified)

Environment: pinned current stack (docs/environment_pin.md).
