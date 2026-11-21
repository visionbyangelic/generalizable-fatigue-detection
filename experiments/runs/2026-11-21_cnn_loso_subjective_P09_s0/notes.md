# CNN (EEGNet) LOPO - subjective labels, held-out P09, model seed 0

Split definition: leave-one-participant-out. Test = all 112
windows of P09; training = the other 10 participants' windows from
the canonical cap-seed-0 Amendment-1 set (806 windows, 11 participants;
P05 has no Amendment-1 windows and has no fold).
Early stopping: validation = whole contiguous RUN-SEGMENTS carved from
the training participants only (run_id groups; every run is
single-class), with a 2-window purge of subtraining windows adjacent to
validation segments in the same recording's time order; best-val-loss
weights restored. The carve received a composite participant/session
key so runs group per recording (driver docstring explains why).
Carve detail: strategy=last1, val_runs=[('P12/S02', 13), ('P12/S02', 20)], val_n=35, subtrain_n=657 (post-purge), epochs_run=16, best_val_loss=0.7150
Wall time for this fold (training + prediction): 18.0 s on the
local VM (CPU), sharing the machine with other project work.

**Interpretation boundary (state wherever cited): the validation
measures segment generalisation inside known training participants,
NOT unseen-participant transfer. The held-out participant is the test
fold and is never touched during fitting.**

Mitigations: temporal CHANNEL DROPOUT during training (p = 0.2 per
sample, EEG zeroed + indicator plane zeroed - the mask-as-fingerprint
mitigation carried over from the Week 2 within run); masked temporals
are zero-filled AND flagged via indicator planes (Amendment 1 - never
silent zeros).

Leakage checklist:
- [x] Held-out participant contributes no window to subtrain, validation, normalisation statistics or class weights
- [x] Normalisation statistics from the subtraining fold only
- [x] Class weights from the subtraining fold only
- [x] Early stopping never touches the test fold (segment-grouped validation from training participants only, with purge)
- [x] No overlapping windows across splits (tumbling windows; participant-level split; purge at validation boundaries)
- [x] Label rules locked (docs/label_rules.md incl. Amendment 1) before training

Environment: local VM, CPU, torch 2.15.0+cpu (pinned stack,
docs/environment_pin.md). Model seeds 0-4 on the fixed cap-seed-0
window set; one run folder per fold so an interrupted run resumes.
