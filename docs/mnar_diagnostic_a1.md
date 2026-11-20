# MNAR diagnostic — Amendment 1 mandatory check (20 Nov 2026)

**Question (Reviewer 2 review 4 condition):** does temporal-channel
(TP9/TP10) failure correlate with the fatigued block (M3) or with time-on-task?
If contact degraded as participants fatigued, any window-dropping rule would
preferentially delete fatigued data (missing-not-at-random bias against the
target class).

**Method:** all 1,568 candidate M1/M3 windows (`data/processed/candidates_a1.csv`).
Temporal failure = TP9 or TP10 failing any Amendment-1 criterion (amplitude /
flatline / fit) at the locked threshold. Rates compared by block (M1 vs M3)
and by session time (first vs second half of each session's candidate
windows), chi-square test. Full numbers: `results/tables/mnar_diagnostic.csv`.

## Verdict
**The feared mechanism is NOT supported — the association runs the other
way.** Temporal failure is significantly *higher* in M1 than in M3
(all sessions: 91.2% vs 81.9%, −9.3 pp, p ≈ 7×10⁻⁸; primary sessions only:
90.4% vs 84.5%, p ≈ 7×10⁻⁴) and *lower* in the second half of sessions than
the first (85.1% vs 89.3%, p ≈ 0.015). Contact quality, if anything, improves
as sessions proceed (settling-in), it does not degrade with fatigue.

**Caveats, stated plainly:**
- Missingness is therefore *not* MCAR — it is block-associated — which is
  exactly why Amendment 1 masks with explicit indicators instead of dropping
  windows, and why per-block masking rates are reported
  (`results/tables/retention_amendment1.csv`).
- Pooled rates are participant-confounded: the extreme check-set contrast
  (M1 100% vs M3 21.4%) is driven by P07's sessions (temporals fail almost
  everywhere) vs P12/S03 (clean temporals, 25 retained M3 windows). Read the
  primary-sessions-only rows as the headline.
- Masking is heavy overall: both temporals are masked in >90% of retained
  windows for P03, P06, P07, P11 and P12. The primary configuration is in
  practice close to frontal-driven for much of the data; the pre-registered
  frontal-only variant (Weeks 3–4) quantifies how much the temporals add.
