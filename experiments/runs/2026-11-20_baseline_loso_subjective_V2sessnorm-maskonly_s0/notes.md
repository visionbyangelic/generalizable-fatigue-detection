# 2026-11-20_baseline_loso_subjective_V2sessnorm-maskonly_s0

PRE-DECLARED SENSITIVITY - estimates and intervals only. No verdict
under the decision-table lock is declared in this run; reporting
status awaits the lead reviewer's ruling.

Split definition: leave-one-PARTICIPANT-out: hold out ALL windows of one participant; train on the other 11 participants (uncapped pool). P05 cannot be a test fold (its retained windows are single-class: 5 alert / 0 fatigued) but its windows are in the training pools of the other folds - unlike the capped primary, where the cap removed them entirely. A fold counts only if train and test
both contain both classes; skipped folds: ['P05'].

Variant V2-MASK - mask-only arm under the V2 protocol
(Reviewer 1 ruling R3): the 5 indicator/coverage features of
panel_additions.mask_only (valid_TP9, valid_TP10,
both_temporals_masked, n_valid_temporals, n_valid_channels), built
from the window index - NO EEG features. These features are
indicators/coverage counts, not continuous EEG features, so the V2
session normalisation does not apply to them (unchanged, exactly as
the indicator features are unchanged inside the V2 EEG feature set).
Same uncapped pool and protocols as V2, so any V2 change can be
attributed between EEG features and the mask shortcut. TRANSDUCTIVITY
note: the V2 normalisation statement does not apply to this arm (no
normalised features); it shares V2's pool and splits only.

Leakage checklist:
- [x] Normalisation (StandardScaler) fit on training fold only (Pipeline, per fold)
- [x] Median imputation fit on training fold only
- [x] Class weights from training fold only (class_weight='balanced')
- [x] No overlapping windows across splits (tumbling windows; splits at session/participant level)
- [x] No tuning on test data (C=1.0 locked default)
- [x] Label rules locked (docs/label_rules.md incl. Amendment 1) before any training
- [x] Variant declared in advance (track.md, 20 Nov 2026) - not chosen from results

Environment: local VM (CPU), scikit-learn 1.9.1.
