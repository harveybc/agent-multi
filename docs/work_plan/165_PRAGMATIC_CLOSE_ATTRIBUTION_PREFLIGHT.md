# Close attribution preflight

2026-10-30. [Plan164](164_PRAGMATIC_CLOSE_ATTRIBUTION_PLAN.md).

The exact native replay and diagnostic queue passed **258 tests**, zero
skips/failures/errors. Archive-subset smoke completed four baseline accounts
and44 single-close interventions. All four baseline accounts matched their
sealed163 counterparts exactly, including traces, metrics and ledgers.
Completed resume authenticated the receipts without repeating the work.

Smoke identity:
`b12f7c02f49bad5726da4c9a7f6062d6f3824e17a046e361252f15aafe2dc2dd`.
[Manifest](../audits/evidence/pragmatic_close_smoke_20261030/MANIFEST.json),
[tests](../audits/evidence/pragmatic_close_smoke_20261030/TESTS.json),
[completion](../audits/evidence/pragmatic_close_smoke_20261030/COMPLETE.json),
[summary](../audits/evidence/pragmatic_close_smoke_20261030/REPORT.json).

The smoke includes ideal and noise1 reports for the first archived world.
It is wiring evidence, not a representative estimate or a selection bank.
In that ideal world, three closes of280977a9 harm the passive position but
improve whole-account profit; two closes of59a75d73 harm both. The diagnostic
therefore distinguishes both behaviors rather than automatically treating
all local sacrifices as justified or all local losses as account failures.

The full queue covers every actual early close of the two sequential
finalists across all36 worlds and allfive reporters:360 baseline accounts.
Per-intervention receipts survive restart. A CLOSE is suppressed for one tick
only; controller memory is not reset. Policy continuation may close the next
tick, trade in either direction later, or remain flat. Native SL/TP and the
calendar still have priority. Every result retains the alternative ledger,
compact action trace and full planning-trace digest; baseline planning traces
are already sealed in163.

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python tools/attribute_pragmatic_closes.py \
  --source <completed-contrast-root> --engine-root <pinned-native-engine> \
  --output <new-attribution-root> --workers 4
```

Tests -> ideal -> noise0.5 -> noise1 -> noise2 -> causal -> final report,
without approval between stages. The same command resumes verified units.
STATE.json gives completed baselines and a measured ETA for the current
reporter stage; slower noise stages must not inherit the ideal-stage ETA.
REPORT.json and RESULTS.md update after each stage; COMPLETE.json seals the
entire diagnostic. No optimization, financial training or feature-selection
release is performed by this queue.
