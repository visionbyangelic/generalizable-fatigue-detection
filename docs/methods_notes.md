# Methods Notes — Decisions & Consultations

## 2026-11-20 — Consultation: assigning sparse subjective ratings to EEG windows
**Asked:** Reviewer 2 (via my Google account, free tier, at her instruction to consult reviewers when unsure). Cross-check against docs/fatigueset-research.md and docs/literature.md before citing — an AI answer is a lead, not a source.
**Question:** FatigueSet records subjective ratings only 3× per session (M1 baseline, M2 post-physical, M3 post-cognitive). How do researchers assign those ratings as labels to EEG windows, and what are the leakage risks?

**Methods named:**
1. **Backward assignment** — windows before rating t(k) take rating t(k) (e.g. M3 applied to the preceding cognitive-task block). Reviewer 2's suggested starting point: simple, transparent.
2. **Forward / carry-forward** — windows after t(k) take rating t(k) until the next rating.
3. **Interpolation** (linear/spline/step) across M1–M3 onto window midpoints — continuous targets or binned classes.
4. **Stage/delta proxies** — protocol stage, activity tier, or Δ = M3 − M1 as coarse labels.
5. **Thresholded binarisation** — fixed cutoffs, per-session medians, or per-subject z-scores.

**Leakage risks named:**
- Overlapping windows in random splits share raw samples/artefacts → group by subject/session (LOSO/GroupKFold), temporal buffers within session.
- Interpolation = look-ahead: labels for early windows encode future ratings. Prefer causal/backward assignment for the primary pipeline.
- Thresholds/normalisation computed on full session/subject leak label distribution + baseline physiology → compute inside training folds only. (Matches our leakage discipline in PLAN.md §7.)
- Protocol/motion confounding: ratings pair with distinct phases (incl. treadmill) → model may learn motion/EMG, not fatigue. Check: can ACC/artefact features alone predict the labels?

**Status:** Input for the Week 1 label-rules decision (docs/label_rules.md). Decision itself still to be made + locked with I; literature anchors (MSARNet, FatigueSense block approaches) take precedence where they differ.

## 2026-11-20 — Consultation 3: Reviewer 2 peer review of Week 1 inspection + proposed label rules
**Verdict:** challenged (1) the "negative control" framing for the 4 non-increasing sessions (ceiling/insensitivity/disengagement are likelier than clean alertness; uneven fold distribution), (2) whole-window rejection on TP9/TP10 Bad (starves data; use channel-wise masking + frontal-only comparison), (3) M3's small window budget (~63 windows max/session) creating duration/count imbalance (cap M1 to match M3 per session), (4) performance labels given the null group effect (demote to exploratory; report Cohen's kappa between label sets), (5) clock-step accounting (report windows lost/re-anchored per session).
**Adopted:** all five, as amendments in docs/label_rules.md. Status of the rules remains PROPOSED pending my lock.
