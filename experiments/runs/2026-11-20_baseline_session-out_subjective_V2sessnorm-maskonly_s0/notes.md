# 2026-11-20_baseline_session-out_subjective_V2sessnorm-maskonly_s0

PRE-DECLARED SENSITIVITY - estimates and intervals only. No verdict
under the decision-table lock is declared in this run; reporting
status awaits the lead reviewer's ruling.

Split definition: pooled: hold out one (participant, session); train on all remaining windows of all participants (uncapped pool, so single-class sessions contribute training windows but cannot be test folds). A fold counts only if train and test
both contain both classes; skipped folds: ['P01/S03', 'P02/S01', 'P03/S03', 'P04/S02', 'P05/S01', 'P06/S01', 'P09/S02', 'P10/S01', 'P11/S03', 'P12/S01'].

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
