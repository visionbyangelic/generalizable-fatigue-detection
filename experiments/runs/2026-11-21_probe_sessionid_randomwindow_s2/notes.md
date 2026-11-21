# Session-identification probe, random-window split (RQ4, POST HOC), seed 2

RQ4 diagnostic probe, post hoc arm. Question: from the same 43
band-power features, can a model tell which recording session a
window came from, within the same participant? If sessions carry a
strong feature fingerprint, any evaluation that shares sessions
between train and test can reward recognising the session instead
of the labelled state.

Split definition: per participant, stratified random-window
cross-validation over that participant's retained windows (all
three sessions), folds = min(5, smallest session count) so every
session appears in every training fold. Participants are probed
only if all three sessions have at least 2 retained windows: P05
(one session with windows) and P10 (latest session has a single
window) are excluded and the exclusion is recorded in the summary
table. The random-window split is again deliberate: the probe
measures how much session information the features carry when the
model is allowed to see the same sessions, which is the quantity
that matters for the fingerprinting argument.

Leakage checklist (honest version):
- [x] Normalisation (StandardScaler) fit on training fold only (Pipeline, per fold)
- [x] Median imputation fit on training fold only
- [x] Class weights computed from training fold only (class_weight='balanced')
- [ ] NO session sharing between train and test: VIOLATED BY DESIGN (the probe's object is session information under session sharing)
- [x] No tuning on test data (C=1.0 locked default)

Environment: local VM (CPU), scikit-learn. Seed = split shuffle
seed 2.
