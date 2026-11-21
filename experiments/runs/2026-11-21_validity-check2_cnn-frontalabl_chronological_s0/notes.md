# RQ4 validity Check 2, EEGNet CNN (frozen cnn.py train_fold), chronological split, model seed 0

Post hoc check owed by panel review 28 (docs/methods_notes.md). Same
window set, labels and split as the prototype-faithful replication
(src/evaluation/rq4_replication_proto.py); only the model's inputs
change, which is the point: the replication's models never received
the mask, so where the score's information lives was an inference.
This run prices one input change at a time.
Ablation adaptation: the frozen architecture fixes 6 input planes, so the AF7, AF8 EEG plane(s) are zeroed rather than removed; the network receives no signal from that electrode/those electrodes. Recovery run: the first execution of these checks was interrupted by an environment restart during this CNN stage, after the LR stage's folders were written; this stage was completed afterwards with the specification unchanged, reusing the driver's own functions. Interval columns are empty in this per-seed folder: intervals are computed for the seed-mean statistic and reported in results/tables/rq4_check2_route_location.csv. CNN carve (frozen train_fold, tail-quintile groups): val_group=4, subtrain_n=866, val_n=216, epochs_run=53, best_val_loss=0.0012.

Leakage checklist (honest version):
- [x] Preprocessing statistics fit on training windows only (LR imputation/scaling; CNN normalisation and class weights from the subtraining windows)
- [ ] NO session sharing between train and test: VIOLATED BY DESIGN (the prototype's configuration, inherited from the replication; the violation is the object of the run)
- [ ] NO overlapping windows across the split: VIOLATED BY DESIGN (50% overlap is the prototype's windowing; the test window after the cut overlaps the last training window by 2 s)
- [x] No tuning on test data (both models are the frozen pipeline's; the ablation removes inputs, it does not tune anything)
- [x] Specification written before the run (panel review 28, docs/methods_notes.md, owed these checks by name)

Environment: pinned current stack (docs/environment_pin.md).
