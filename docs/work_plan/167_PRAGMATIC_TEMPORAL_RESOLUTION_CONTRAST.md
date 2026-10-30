# One-variable temporal resolution contrast

2026-10-30. Follow-up to164, automatically after close attribution completes.

## Measured structural omission

Frozen sequential policy59a75d73, original confirmation world000, ideal
reports, has harmful closes at bars50 and62. Its allowed delayed closes are
3/6/12/24h, although it makes decisions hourly. The omitted1h and2h choices
improve its own score without increasing predicted account drawdown:

| Tick | Original plan | Original score | Omitted plan | Omitted score | DD both % |
|---|---|---:|---|---:|---:|
| 50 | CLOSE then BUY24 | 2.061786873 | HOLD1h then BUY24 | 2.314863208 | 0.288878454 |
| 62 | CLOSE then BUY6 | 2.246120076 | HOLD2h then BUY6 | 2.433199152 | 0.288878454 |

The original CLOSE loses25.750968 and7.725146 USD respectively against
one-tick HOLD followed by its own policy, with no realized DD improvement.
This is a concrete omitted-action witness, not evidence that adding these
choices fixes every close. At tick95 the same additions do not beat its
chosen score, despite the close hurting the account. Replanning/continuation
approximation remains a separate hypothesis to investigate.

## Frozen comparison, not another tuning loop

Keep both sequential finalists' coefficients unchanged. Compare each with
close_offsets=[1,2,3,6,12,24] versus its original[3,6,12,24]. Keep the frozen
simple and predictive-proxy representatives unchanged as controls. Six
policies total. Do not rerank or select coefficients on the old confirmation.

Use new seed banks960000 and970000,36 worlds each, declared before running.
Fresh calibration-noise1 is descriptive, not used to tune these six policies;
all six proceed to ideal/noise0.5/noise1/noise2/causal confirmation regardless
of positive or negative calibration performance.216 calibration and1,080
confirmation native accounts. This is a matched diagnostic contrast, not an
independent final holdout claim after unrestricted research.

## Acceptance and automation

- TR01: regression reproduces both historical omitted-action witnesses on
  the native engine without modifying the original policy's actions.
- TR02: six configurations, two exact pairs differing only in close_offsets;
  controls and coefficients remain unchanged.
- TR03: fresh banks differ from the prior study, same worlds per policy,
  no selection based on confirmation and no performance launch gate.
- TR04: wait for the authenticated164 completion; if its integrity fails,
  record failure instead of launching a dependent comparison.
- TR05: interrupted units resume only by authenticated identity; native
  ledger, costs, direction, early closes and local regret remain reported.

Tests and smoke precede deployment. A persistent bounded four-CPU service
waits for164, authenticates it, runs the full contrast and writes tables and
completion hashes without asking the owner to return. No production policy
is silently replaced. No financial training or floating selection starts.

## Implemented preflight

The two historical omitted-action witnesses are regression tests, and the
configuration test proves only close_offsets differs within each pair.
**261 tests pass**, with zero skips/failures/errors. The smoke completed18
native accounts and authenticated completed resume without rerunning them.
This smoke verifies execution, not which strategy to select.

Smoke identity `70851ab2b57f55e7597a6872e5f711a3ca673e9a6899102b1599cedabe14d1e1`.
[Manifest](../audits/evidence/pragmatic_resolution_smoke_20261030/MANIFEST.json),
[tests](../audits/evidence/pragmatic_resolution_smoke_20261030/TESTS.json),
[completion](../audits/evidence/pragmatic_resolution_smoke_20261030/COMPLETE.json).
The initial preflight was followed by an explicit dependency-hash regression:
completed resume now rejects a missing or changed upstream completion rather
than treating it as a fresh wait. The evidence linked here is the corrected
second preflight; neither historical campaign was changed.

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python tools/run_pragmatic_resolution.py \
  --source <completed-contrast-root> --dependency <full-attribution-root> \
  --dependency-identity <full-attribution-identity> \
  --engine-root <pinned-native-engine> --output <new-resolution-root> --workers 4
```

The full run waits for the authenticated attribution automatically. It writes
STATE.json while waiting and during each stage; the complete metrics are in
REPORT.json/RESULTS.md and their hashes in COMPLETE.json. There is no return
to an approval prompt between the two experiments.

## Full dependent service deployed

Immutable runtime `5e352281`, native engine `e0a1c2b`.
Identity `51a1f17bdfa184f43a2f1e8f9808fa4a3d8c2bb9dea962c348036f3291f84316`.
[Manifest](../audits/evidence/pragmatic_resolution_launch_20261030/MANIFEST.json)
and [runtime tests](../audits/evidence/pragmatic_resolution_launch_20261030/TESTS.json).
The actual deployed runtime passed261 tests and entered WAIT_DEPENDENCY on
the full attribution identity726ebc8a. The service is enabled across session
restart, checks every30s and begins simulation automatically after verifying
the dependency's completion and all artifacts. Four CPU workers, quota400%,
memory high4GiB/maximum6GiB, bounded retries and authenticated unit reuse.
At the first check both services were active with zero restarts; attribution
had130/360 baselines and822 interventions complete. The dependent service
used about67MiB while waiting. This state is normal sequencing, not an owner
approval or auditor-response dependency.
