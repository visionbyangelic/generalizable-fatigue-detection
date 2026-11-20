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

## Reviewer 2 review 4 — Week 2 rejection-rule amendment (20 Nov 2026)
**Question (one, free tier):** whole-window rejection keeps only 12.7% of windows and zero windows for 4/12 participants (TP9/TP10 contact noise, SD ≈42 µV vs 5–8 µV frontal); choose among (1) channel-wise masking + frontal-gated rejection, (2) frontal-only primary, (3) ≥3-of-4 quota.
**Verdict: Option 1 is most defensible** — IF masking is explicit (validity indicator / feature-level exclusion), never silent zeros the model could learn as a fatigue shortcut. Key points:
- Whole-window rejection risks **missing-not-at-random** bias: if contact degrades as participants tire (slump/sweat), dropping windows biases the labels themselves. Examiners push on survivorship bias (12→8 participants) and post-hoc rule changes; Option 1 is defensible because it follows a clear physical mechanism (Muse S temporal electrodes sit behind the ear — poor contact vs frontal).
- Option 2 risks: frontal-only leans on blink/EOG proxies; narrows the neural claims; sensitivity analysis can look like an afterthought.
- Option 3 rejected: channels aren't interchangeable slots; heterogeneous channel subsets across LOSO folds.
- **Additions adopted as conditions:** (a) run the MNAR diagnostic — does TP failure rate rise with time-on-task / differ M1 vs M3? — and report it; (b) per-subject retention + masking-% table; (c) lock which configuration is PRIMARY before seeing LOSO scores (no picking the better one afterwards); (d) threshold-sensitivity curve (survival vs threshold) for the appendix; (e) for the baseline, compute band features per channel and aggregate over valid channels.
*Lead to cross-check, not a citable source. Full answer in the Reviewer 2 conversation of 20 Nov 2026.*

## Reviewer 1 review 5 — cross-check of the Week 2 amendment (20 Nov 2026)
**Same question as Reviewer 2 review 4, asked at my suggestion.** Reviewer 1 signed in via Google (free plan). **Verdict: Option 1 agreed — with one structural difference from Reviewer 2:** Reviewer 1 recommends making **frontal-only the primary/confirmatory analysis** and the masked all-channel pipeline the pre-specified secondary (fewer researcher degrees of freedom, cleanest LOSO story). Reviewer 2 had endorsed masked-all-channel as primary with frontal-only as the comparison. I to arbitrate (presented in chat 20 Nov 2026).
**Points adopted regardless of that choice:**
- **Framing:** the amendment is a *documented protocol deviation* — post hoc, driven by label-blind signal-quality diagnostics, made before any model performance was examined. Do NOT call it "pre-registered" unqualified. The strict-rule yield (12.7%, 4/12 empty) is reported as a finding.
- **Verify the noise diagnosis with a PSD** of a bad temporal session (broadband+unstable = contact; >20–30 Hz elevation = temporalis EMG; spike = mains). Note for the methods text: our 1–40 Hz bandpass already excludes 50 Hz mains; no separate notch in the primary pipeline (locked §0).
- **Mask-as-fingerprint risk:** contact quality is a participant property — the CNN must use channel dropout in training, and an indicator-removed ablation is owed in the LOSO phase.
- **Coverage reporting:** in LOSO, report accuracy alongside coverage (fraction of windows the system can score) — rejection-at-test-time isn't deployable; this turns the data-quality problem into a contribution.
- **Blink caveat:** frontal amplitude rejection may remove blink-heavy windows, and slow blinks are a fatigue cue — the M1-vs-M3 failure-rate diagnostic must include frontal channels, not only temporals.
- Statistics for Weeks 3–4: per-subject metrics, subject-level bootstrap CIs, label-permutation baseline, nested subject-wise tuning.
*Lead to cross-check, not a citable source. Full answer in the Reviewer 1 conversation of 20 Nov 2026.*
