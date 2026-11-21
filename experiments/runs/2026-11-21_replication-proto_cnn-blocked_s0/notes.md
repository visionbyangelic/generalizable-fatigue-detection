# RQ4 prototype-faithful replication, EEGNet CNN (frozen cnn.py train_fold), blocked split, model seed 0

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

Split definition: blocked split: consecutive 30-window blocks within each session, every fifth block = test, the rest train, with training windows that share samples with a test window purged (overlap removed, recording still shared).
CNN carve (frozen train_fold, tail-quintile groups): val_group=4, subtrain_n=878, val_n=219, epochs_run=46, best_val_loss=0.0005. Interval columns are empty in this per-seed folder: intervals are computed for the seed-mean statistic and reported in results/tables/rq4_replication_proto_summary.csv. Blocked-split purge: 16 training windows removed for sharing samples with a test window. Recovery note: the first execution of the driver was interrupted by an environment restart during this stage; this stage was completed by a recovery run reusing the driver's own functions and the saved window set, with the specification unchanged.
Leakage checklist (honest version):
- [x] Preprocessing statistics fit on training windows only (LR imputation/scaling; CNN normalisation and class weights from the subtraining windows)
- [ ] NO session sharing between train and test: VIOLATED BY DESIGN (train and test come from the same two recordings; this split removes window overlap only and is reported as "overlap removed, recording still shared", never as an honest fatigue evaluation)
- [x] No overlapping windows across the split (blocks assigned wholly; boundary training windows purged, count in config)
- [x] No artefact rejection or masking anywhere (the Amendment-1 mask is computed only to slice test predictions afterwards)
- [x] No tuning on test data (both models are the frozen pipeline's)
- [x] Specification written into docs/rq4_session_labels.md before the run (post hoc arm, panel-owed)

Environment: pinned current stack (docs/environment_pin.md).
