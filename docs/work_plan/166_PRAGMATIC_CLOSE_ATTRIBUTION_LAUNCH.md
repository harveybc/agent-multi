# Frozen close attribution running

2026-10-30. Immutable runtime `de91e051`, native engine `e0a1c2b`.
Full identity `726ebc8a21d7c268493a6765566865bef1868de3a8b4689a63d039f9d2dfed7e`.
[Manifest](../audits/evidence/pragmatic_close_launch_20261030/MANIFEST.json)
and [runtime tests](../audits/evidence/pragmatic_close_launch_20261030/TESTS.json).

The persistent service started at13:09 UTC /08:09 Bogota.258 runtime tests
passed before replay. Four single-threaded CPU workers, CPU quota400%,
memory high4GiB and maximum6GiB, bounded failure restart, per-case durable
receipts and automatic stage progression. No GPU or training workload.

The ideal stage completed all72 baselines and305 one-tick interventions.
Numbers below measure original CLOSE minus suppressing it for one tick and
then continuing the same policy; they are not additive attainable profits.

| Policy | Closes | Account profit helped | Account profit harmed | CLOSE dominates | HOLD dominates | Return/DD tradeoff |
|---|---:|---:|---:|---:|---:|---:|
| 59a75d73 | 157 | 79 | 78 | 60 | 55 | 42 |
| 280977a9 | 148 | 78 | 70 | 64 | 51 | 33 |

The remaining noise0.5/noise1/noise2/causal stages proceed automatically.
Full scope:360 baseline accounts and3,940 actual early closes, derived from
the frozen163 archive, not a selectively chosen sample. Full-account results
are separate from the original passive-position close classification.

## Concrete follow-up already designed

[Plan167](167_PRAGMATIC_TEMPORAL_RESOLUTION_CONTRAST.md) records two native
counterexamples: the finite planner omitted waiting1h/2h while evaluating
only3/6/12/24h delayed closes. Those omitted actions beat its own chosen score
with equal predicted DD. They explain part, not all, of the observed harms.

The follow-up keeps coefficients frozen and changes only the close-offset
library, using fresh paired seed banks and the unchanged simple/proxy
controls. It is a separate run after this attribution finishes, so this
diagnostic and every source receipt remain intact. No intermediate owner
approval is needed. Do not promote the repaired library before its contrast.
