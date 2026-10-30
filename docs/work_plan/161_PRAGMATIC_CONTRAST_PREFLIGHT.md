# Pragmatic contrast preflight

2026-10-30. [Plan160](160_PRAGMATIC_CONTROLLER_CONTRAST_PLAN.md) implemented.

## Verified before launch

- 247 tests passed, zero failures/errors/skips in the actual sequential runner.
- Three smoke configurations, two calibration worlds and six confirmation
world/reporter units completed:24 native policy accounts in total.
- Completed resume authenticated every artifact and performed no new work.
- Scoped Ruff and whitespace checks pass.
- Native runner parity against the previous unchanged arbiter passes for
trades, actions, cash, return, DD, costs and directions.
- Both-direction favorable/adverse close fixtures; hidden-future perturbation;
native two-short-orders versus long-hold witness; expect-before-select versus
sample-wise oracle; cache invalidation after cost changes; hostile resume.

Smoke identity:
`04ce2bffe723c4d34c1132a3f98865ebe4b9e0bccf5afed59fb4a38cec0935b6`.
[Test receipt](../audits/evidence/pragmatic_contrast_smoke_20261030/TESTS.json),
[manifest](../audits/evidence/pragmatic_contrast_smoke_20261030/MANIFEST.json),
[completion](../audits/evidence/pragmatic_contrast_smoke_20261030/COMPLETE.json).
These are plumbing/correctness checks, not winners or financial results.

## Defect caught before the full grid

The native simulator's ISO-week denominator counts a Sunday reopening and
the following Monday as different weeks. Our synthetic account spans one
complete market session. Its orders/week must therefore equal orders, not
orders/2. The first smoke exposed that mismatch. The corrected run uses
market-session weeks for activity eligibility and preserves
`native_iso_orders_per_week` in every account. Regression covers both values.
No completed historical experiment was overwritten or silently reclassified.

## Automated phases

1. Recheck immutable runtime tests and source/native/config hashes.
2. 36 paired noisy calibration worlds x24 configs =864 accounts.
3. Freeze at most two eligible non-dominated configurations per controller,
   using calibration alone. Ineligible fallback is explicitly diagnostic.
4. 36 disjoint worlds x5 reporter settings x up to6 representatives:
   at most1,080 confirmation accounts. Ideal, noise.5/1/2 and causal prefix.
5. Persist complete native traces, plan vectors, local-close alternatives,
   costs and statistics; emit `REPORT.json`, `RESULTS.md`, `COMPLETE.json`.

Four bounded CPU workers, no GPU or model training. Per-unit atomic sealed
receipts survive process restart; exclusive queue lock prevents duplication.
Integrity failure is explicit, never a fabricated metric or silent refit.
The service is restartable without changing its configuration or rerunning
authenticated completed units. Runtime and elapsed estimates are recorded.

The finite first grid varies time exponent, report dispersion penalty and
close persistence independently. Other coefficients remain configurable but
fixed for this contrast. It is not claimed to optimize all strategy parameters.
No automatic financial learning or floating feature selection follows.

## Older lanes

Both causal-history20x300x3 searches and their collector completed. The zero
lane has16,069 candidate receipts, the median lane15,472. Those are preserved
comparators, not a release of the new controller. The new grid uses disjoint
scenario seed families and does not tune on either old confirmation panel.
