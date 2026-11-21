# RQ4 prototype-faithful replication, EEGNet CNN (frozen cnn.py train_fold), chronological split, model seed 1

Post hoc arm, panel-owed (reviews 25 and 26 in docs/methods_notes.md);
specification and outcome branches were written into
docs/rq4_session_labels.md before the run. Subject 01 only, the
preliminary experiment's own participant and configuration as far as
this project's frozen models allow: whole sessions epoched as one
continuous stream (720 + 633 = 1,353 windows, the preliminary's own
count), 4 s windows at a 2 s stride, whole-session labels, no
artefact rejection or masking. Named departures from the
preliminary's documented chain: this pipeline's 1 to 40 Hz filter
and 128 Hz models in place of its 1 to 50 Hz at 256 Hz, and this
project's two models in place of its Keras 1D-CNN. The Amendment-1
mask is computed for every window and used only to slice the test
predictions into retained and rejected subsets (metrics.csv rows
a1_retained / a1_rejected).

Split definition: chronological 80/20 within each session (first floor(0.8*n) windows in stream order = train, remainder = test, no purge at the cut): the prototype audit notebook's configuration.
CNN carve (frozen train_fold, tail-quintile groups): val_group=4, subtrain_n=866, val_n=216, epochs_run=44, best_val_loss=0.0001. Interval columns are empty in this per-seed folder: intervals are computed for the seed-mean statistic and reported in results/tables/rq4_replication_proto_summary.csv.
Leakage checklist (honest version):
- [x] Preprocessing statistics fit on training windows only (LR imputation/scaling; CNN normalisation and class weights from the subtraining windows)
- [ ] NO session sharing between train and test: VIOLATED BY DESIGN (the prototype's configuration; the violation is the object of the run)
- [ ] NO overlapping windows across the split: VIOLATED BY DESIGN (50% overlap is the prototype's windowing; the test window after the cut overlaps the last training window by 2 s)
- [x] No artefact rejection or masking anywhere (the Amendment-1 mask is computed only to slice test predictions afterwards)
- [x] No tuning on test data (both models are the frozen pipeline's)
- [x] Specification and outcome branches written into docs/rq4_session_labels.md before the run (post hoc arm, panel-owed)

Environment: pinned current stack (docs/environment_pin.md).
