# Baseline, DAY-LEVEL labels, LEAKY random-window split (RQ4, POST HOC), seed 2

RQ4 diagnostic run, post hoc arm. Same day-level labels, same
features, same model as the day-level LOPO run; only the split
changes. Windows from all 11 participants are pooled and split at
random into 5 stratified folds, so windows of every session appear
in both train and test.

THIS SPLIT IS DELIBERATELY LEAKY. It is run because it is the design
family of the preliminary experiment (whole-session labels, train
and test sharing sessions): it lets the model use session-specific
patterns (drift, contact quality, day effects) as if they were
fatigue. The number it produces is an upper bound manufactured by
the design, not a performance estimate, and it is never reported as
one. Preprocessing (imputation, scaling, class weights) is still fit
on training folds only, so the split is the single leak, matching
the Week 2 leakage-gap demo's discipline.

Leakage checklist (honest version):
- [x] Normalisation (StandardScaler) fit on training fold only (Pipeline, per fold)
- [x] Median imputation fit on training fold only
- [x] Class weights computed from training fold only (class_weight='balanced')
- [ ] NO session sharing between train and test: VIOLATED BY DESIGN (the violation is the object of the run)
- [x] No tuning on test data (C=1.0 locked default)
- [x] Label definition fixed in the driver docstring before the run (post hoc arm, labelled as such)

Environment: local VM (CPU), scikit-learn. Seed = split shuffle
seed 2.
