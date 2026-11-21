# Environment pin (recorded 21 Nov 2026)

Every number reported from 21 Nov 2026 onward was computed on this
stack, on the local VM (CPU only). It exists because the Week 2 runs
(20 Nov 2026) were computed before a VM restart on a stack whose
package versions were never recorded; the RQ4 reproduction audit
(docs/rq4_session_labels.md) attributes the small probability
differences between the stored Week 2 numbers and their re-runs to
solver-level differences between that unrecorded stack and this one,
by elimination. Pinning the current stack is what stops that
situation from recurring: any future re-run can now be checked
against an exact environment, not a description of one.

| Component | Version |
|---|---|
| Python | 3.12.3 |
| NumPy | 2.5.5 |
| SciPy | 1.18.1 |
| scikit-learn | 1.9.1 |
| pandas | 3.1.1 |
| MNE | 1.13.2 |
| PyTorch | 2.15.0+cpu (the version recorded in the Week 2 CNN run configs; reinstalled 21 Nov 2026 for the RQ4 2x2 arm) |
| BLAS / LAPACK | scipy-openblas 0.3.34.237.2 (as reported by NumPy's build configuration) |

Packages live in a persistent folder on the VM (they were lost once
to a restart and reinstalled at these versions; the Week 2 seed-0
baseline numbers were reproduced exactly after the reinstall, which
is recorded in track.md).

Canonical-figures rule that follows from the audit: the stored
Week 2 primary numbers remain the canonical values for the primary
results (they are what the locked Week 2 tables report), and all
later arms (DROZY, RQ4, the 2x2 replication) report figures computed
on this pinned stack. The two stacks' Week 2 headlines agree to
three decimal places on every protocol (within-subject 0.484,
session-out 0.515, LOPO 0.508), so no conclusion in the project
depends on which stack's figures are used.
