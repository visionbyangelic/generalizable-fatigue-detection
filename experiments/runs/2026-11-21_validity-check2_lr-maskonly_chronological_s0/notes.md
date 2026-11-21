# RQ4 validity Check 2, band-power logistic regression (frozen baseline), chronological split, deterministic (s0 by convention)

Post hoc check owed by panel review 28 (docs/methods_notes.md). Same
window set, labels and split as the prototype-faithful replication
(src/evaluation/rq4_replication_proto.py); only the model's inputs
change, which is the point: the replication's models never received
the mask, so where the score's information lives was an inference.
This run prices one input change at a time.
The model sees the Amendment-1 validity pattern of each window and nothing else. The trivial rule (predict fatigued iff the window is not mask-retained) is reported alongside in results/tables/rq4_check2_route_location.csv; it needs no fitting, so it has no run folder of its own.

Leakage checklist (honest version):
- [x] Preprocessing statistics fit on training windows only (LR imputation/scaling; CNN normalisation and class weights from the subtraining windows)
- [ ] NO session sharing between train and test: VIOLATED BY DESIGN (the prototype's configuration, inherited from the replication; the violation is the object of the run)
- [ ] NO overlapping windows across the split: VIOLATED BY DESIGN (50% overlap is the prototype's windowing; the test window after the cut overlaps the last training window by 2 s)
- [x] No tuning on test data (both models are the frozen pipeline's; the ablation removes inputs, it does not tune anything)
- [x] Specification written before the run (panel review 28, docs/methods_notes.md, owed these checks by name)

Environment: pinned current stack (docs/environment_pin.md).
