# DROZY positive control: dose ladder under Amendment 2

Run 21 Nov 2026 under Amendment 2 to docs/dataset2_preregistration.md
(adopted post hoc, panel specification in docs/methods_notes.md
reviews 19-20). Code: src/evaluation/drozy_positive_control.py,
unchanged from the hash recorded in the amendment. Tables:
results/tables/drozy_poscontrol_dose0.csv (identity check),
results/tables/drozy_poscontrol_permuted.csv (the ladder),
results/tables/drozy_poscontrol_companion.csv (the true-label
companion ladder),
results/tables/drozy_poscontrol_nulls.csv (every null statistic).
One run folder per repeat under experiments/runs/, named
2026-11-21_poscontrol_*.

## Why this control exists

The DROZY results (docs/drozy_results.md) left two questions that
the headline numbers cannot answer on their own. First, the
reaction-time arm scored 0.482, at chance: is that a small or absent
effect, or is the test on that label structure simply insensitive
at N = 13? Second, the cross-dataset reading sets DROZY beside
FatigueSet, and FatigueSet has a recovered positive control while
DROZY had none: whether the same injected physiology produces a
comparable recovery on clinical channels through this adapter was
untested. The control answers those two questions and no others.
It changes no locked wording, and the sleepiness result never
depended on it.

## What was done

The injection mirrors the Week 2 FatigueSet control. For each
window labelled fatigued under the arm being run, the theta (4-8 Hz)
and alpha (8-12 Hz) bins of each valid channel's own FFT are scaled
by the square root of the power gain: phase preserved, background
shape preserved, alert windows untouched, artefact mask unchanged,
all five PSG channels. Doses interpolate linearly between dose 0
(theta power x1.0, alpha power x1.0) and full dose (theta power
x2.0, alpha power x1.5), so dose 0.5 is theta x1.5, alpha x1.25,
the Week 2 half dose. Features are re-extracted by the adapter and
passed through the frozen pipeline untouched.

The primary control is the label-permuted ladder. Each repeat first
swaps each participant's two session labels at random (coin flip
per participant, seeded per repeat), so any natural label-signal
relationship is null by construction; the injection is then applied
to the permuted-fatigued windows and the full
leave-one-participant-out run is made under the permuted labels.
Recovery at a dose can then only be the injected effect. Five
repeats per dose, five doses, both arms: 50 runs.

Detection for a repeat is a swap-null p-value: 100
within-participant session-label swaps planned, p = (1 + count of
nulls at or above the observed value) / 101. A null stops early
once 5 nulls reach the observed value, because the detection
verdict is decided there; an early-stopped null reports its p as a
lower bound (0.059), never as an exact value. Detection at a dose
means p < 0.05 in at least 4 of the 5 repeats.

Manipulation check: the mean log10 band-power shift in the injected
windows lands within about 0.01 of the log10 of the power gain at
every dose, in both arms (dose 1.0: theta +0.291 against a +0.301
target, alpha +0.181 against +0.176). The injection moved the
features by the amount it was supposed to move them.

## Dose-0 identity check: passed exactly

Before any ladder run, dose 0 was pushed through the injection code
path on the true labels. It reproduces the reported uninjected
scores at full stored precision, with a difference of exactly 0.0
in both arms: KSS all-channel 0.6022, RT all-channel 0.4824. The
injection path and the window alignment behind it are faithful to
the baseline runs, so the ladder numbers below are measured through
the same code that produced the headline numbers.

## Sleepiness arm: comparability gate passed

Permuted-label recovery, mean and range across the five repeats,
with the number of repeats detected:

| Dose | Theta / alpha power gains | Mean recovery [range] | Detected |
|---|---|---|---|
| 0 | x1.00 / x1.00 | 0.489 [0.453, 0.530] | 0 of 5 |
| 0.25 | x1.25 / x1.125 | 0.575 [0.520, 0.596] | 3 of 5 |
| 0.5 | x1.50 / x1.25 | 0.672 [0.640, 0.691] | 5 of 5 |
| 0.75 | x1.75 / x1.375 | 0.744 [0.725, 0.759] | 5 of 5 |
| 1.0 | x2.00 / x1.50 | 0.798 [0.778, 0.817] | 5 of 5 |

The gate set in the amendment applies the FatigueSet recovery marks
to this clean recovery: full dose at least 0.75, half dose at least
0.60. The full-dose mean is 0.798 and the half-dose mean is 0.672,
so the gate is met on both marks. Set beside FatigueSet's own
recovery (0.834 at full dose, 0.695 at half, against its
chance-level baseline), the DROZY curve has the same shape at a
slightly lower level: the adapter and the clinical channels carry
an injected effect of this kind about as well as the Muse data
does. Dose 0 sits at 0.489, where a permuted-label run with no
injection belongs.

## Reaction-time arm: the ladder and the minimum detectable effect

| Dose | Theta / alpha power gains | Mean recovery [range] | Detected |
|---|---|---|---|
| 0 | x1.00 / x1.00 | 0.505 [0.460, 0.536] | 0 of 5 |
| 0.25 | x1.25 / x1.125 | 0.583 [0.538, 0.615] | 3 of 5 |
| 0.5 | x1.50 / x1.25 | 0.660 [0.620, 0.690] | 5 of 5 |
| 0.75 | x1.75 / x1.375 | 0.731 [0.703, 0.755] | 5 of 5 |
| 1.0 | x2.00 / x1.50 | 0.783 [0.762, 0.806] | 5 of 5 |

Detection at a dose needs at least 4 of 5 repeats. Dose 0.25
reaches 3 of 5, so it does not meet the rule; dose 0.5 reaches 5 of
5. The reaction-time arm's minimum detectable effect is therefore
dose 0.5: an injected theta power gain of x1.5 with an alpha gain
of x1.25, painted on the windows that arm calls fatigued.

The pre-declared reading then applies to the RT null of 0.482: it
excludes effects of the size that gives at least 80% detection at
N = 13, that is, injected effects at dose 0.5 and above, under this
arm's own label structure. It does not exclude smaller effects: at
dose 0.25 the same test detects the injection in only 3 of 5
repeats, so an effect of that size could be present in the RT
labels and still produce the observed null. The two arms' ladders
are close throughout (the RT means sit within about 0.015 of the
KSS means at every dose), so the dissociation in the headline
results is not a sensitivity difference between the label
structures: both arms' tests can see an effect of dose-0.5 size,
and only the sleepiness labels carry one on the real data.

## Companion ladder: true-label increments

The amendment also specified a companion: the same dose ladder on
the true labels, reported as the increment over each arm's
uninjected score, with a swap-null p-value on the increment (the
injected and uninjected statistics computed under the same swap
pattern, so the null is a paired test on the increment itself).
It was the one part of the amendment left unrun when the rest of
this control was written up. Reviewer 1 ruled that it is owed
because the amendment as adopted includes it (review 39), so it
was run on 21 Nov 2026 and is reported here exactly as it came
out. The run completed in a single pass with no restarts, under
the same unchanged code.

Sleepiness arm, uninjected score 0.602:

| Dose | Injected BA | Increment | Null: swaps run, count at or above | p | Detected |
|---|---|---|---|---|---|
| 0.25 | 0.682 | +0.080 | 100 of 100, 0 | 0.010 | yes |
| 0.5 | 0.746 | +0.144 | 100 of 100, 0 | 0.010 | yes |
| 0.75 | 0.807 | +0.205 | 100 of 100, 0 | 0.010 | yes |
| 1.0 | 0.851 | +0.249 | 100 of 100, 0 | 0.010 | yes |

Reaction-time arm, uninjected score 0.482:

| Dose | Injected BA | Increment | Null: swaps run, count at or above | p | Detected |
|---|---|---|---|---|---|
| 0.25 | 0.568 | +0.086 | 100 of 100, 0 | 0.010 | yes |
| 0.5 | 0.650 | +0.168 | 100 of 100, 0 | 0.010 | yes |
| 0.75 | 0.720 | +0.238 | 100 of 100, 0 | 0.010 | yes |
| 1.0 | 0.779 | +0.297 | 100 of 100, 0 | 0.010 | yes |

On the true labels, the injected effect adds a rising increment in
both arms, and every increment clears its paired swap null: no
null statistic in any of the eight combinations reached the
observed increment. The manipulation check agrees with the
permuted ladder (the band shifts land on their log10 targets, for
example theta +0.093 against +0.097 at dose 0.25 and +0.291
against +0.301 at full dose). One difference from the permuted
ladder is worth stating plainly rather than smoothing over: the
dose-0.25 increments are detected here, in both arms, while the
permuted ladder's detection rule put the reaction-time arm's
minimum detectable effect at dose 0.5. These are different
statistics. The companion tests the lift over the arm's own
uninjected score with a paired null, which removes the variation
the two scores share; the minimum detectable effect rests on the
permuted recovery test, where the whole score must separate from
a null label structure. The pre-declared reading stands as
written: the RT null excludes injected effects at dose 0.5 and
above, and the companion does not revise it.

The companion changes no verdict and no decision rule. The
comparability gate and the minimum detectable effect rest on the
permuted control, exactly as the amendment specified, and the
locked wording everywhere else is untouched. With this run,
Amendment 2 is complete as adopted: every component it specified
has been run and reported.

## Paired sleepiness-minus-order contrast

This contrast was owed since the exploratory checks (it is Check 6
in docs/drozy_exploratory_checks.md, computed there and recorded
here because the control document is where the session-order
question closes). BA(KSS) minus BA(protocol order), paired by
participant, all-channel: 0.066, 95% interval [0.031, 0.105],
n = 13, with every one of the 10,000 resampled means above zero
(results/tables/drozy_paired_contrast_kss_order.csv). Protocol
order alone does not reach significance against its own null (0.536
[0.493, 0.572], p = 0.229), and on the same participants the
sleepiness labels score higher than the order labels. That
separation rests on the 2 participants whose labels differ between
the arms (P02 and P13); for the other 11 the two arms use identical
labels. Session order is therefore not ruled out as a contributor,
and the locked claim wording keeps saying so.

## Scope: what this control settles and what it does not

Under the amendment's scope rule, a failed gate would have blocked
every cross-dataset statement and left the RT null uninformative
about effect size. The gate passed, so neither consequence applies:
the cross-dataset reading in docs/drozy_results.md stands as
written, and the RT null now carries the effect-size statement
above. Passing upgrades no claim. The locked summary sentence is
unchanged, the estimation-only standing set by the feasibility gate
is unchanged, and the sleepiness result (0.602) never depended on
this control either way: its separation from its own swap null was
established before the control was designed.

The amendment's companion ladder on the true labels was the last
component to run. It is reported in its own section above, it was
run because Reviewer 1 ruled it owed under the amendment as
adopted (review 39), and its results are recorded whatever they
showed. It remains what the amendment made it: a reporting extra
rather than a gate. Both verdicts in this document (the
comparability gate and the minimum detectable effect) rest on the
permuted control, and nothing in the companion section revises
them.

## Environment and provenance

Local VM, pinned stack (scikit-learn 1.9.1, MNE 1.13.2, numpy
2.5.5, scipy 1.18.1, pandas 3.1.1, from the persistent library
folder). The run was interrupted twice by environment restarts and
resumed from the summary tables each time: the driver records each
repeat as it finishes and skips recorded combinations, so no repeat
was ever run twice and at most the one repeat in progress was lost.
Permutation seeds are 70000 plus the repeat index; null swaps use
seeds 20000 plus the swap index, the gate's convention.
