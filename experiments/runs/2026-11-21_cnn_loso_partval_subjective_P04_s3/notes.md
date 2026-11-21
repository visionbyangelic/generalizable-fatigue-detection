# CNN (EEGNet) LOPO, PARTICIPANT-CARVED validation - POST HOC sensitivity, held-out P04, model seed 3

Split definition: leave-one-participant-out. Test = all 43
windows of P04. Validation = ALL windows of P06 (the next
participant after P04 in sorted order, wrapping; a fixed rule that
uses no scores). Subtraining = the remaining participants' windows
from the canonical cap-seed-0 Amendment-1 set (806 windows, 11
participants; P05 has no Amendment-1 windows and has no fold).

**This run is a POST HOC sensitivity, labelled as such everywhere it
is cited.** It changes exactly one thing from the headline CNN LOPO
run (src/models/cnn_loso_segval.py): the early-stopping validation is
carved by whole training participant instead of by run-segments
inside the training participants. The change answers a review point:
segment-carved validation shares sessions with subtraining, which can
reward session fingerprinting and stop training late, biasing the
CNN downward. Everything else (window set, folds, seeds, network,
training constants, channel dropout, patience, improvement
threshold) is identical to the headline run. The training loop
mirrors the frozen train_fold_segval loop; the frozen files are not
modified. No purge is applied at the validation boundary because
validation and subtraining windows belong to different participants
and the windows are non-overlapping.
Carve detail: val_participant_windows=69, subtrain_n=694, epochs_run=16, best_val_loss=0.6718
Wall time for this fold (training + prediction): 26.0 s on the
local VM (CPU), sharing the machine with other project work.

**Interpretation boundary (state wherever cited): the validation
measures generalisation to one known-but-unseen-in-training
participant, not to the test participant; the held-out test
participant is never touched during fitting.**

Mitigations: temporal CHANNEL DROPOUT during training (p = 0.2 per
sample, EEG zeroed + indicator plane zeroed); masked temporals are
zero-filled AND flagged via indicator planes (Amendment 1 - never
silent zeros).

Leakage checklist:
- [x] Held-out participant contributes no window to subtrain, validation, normalisation statistics or class weights
- [x] Validation participant is a training participant, chosen by a fixed score-free rule, and is excluded from subtraining
- [x] Normalisation statistics from the subtraining fold only
- [x] Class weights from the subtraining fold only
- [x] Early stopping never touches the test fold (validation is a whole training participant)
- [x] No overlapping windows across splits (tumbling windows; participant-level splits throughout)
- [x] Label rules locked (docs/label_rules.md incl. Amendment 1) before training

Environment: local VM, CPU, torch 2.15.0+cpu (pinned stack,
docs/environment_pin.md). Model seeds 0-4 on the fixed cap-seed-0
window set; one run folder per fold so an interrupted run resumes.
