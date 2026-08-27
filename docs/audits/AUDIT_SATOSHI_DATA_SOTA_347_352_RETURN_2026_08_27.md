# Audit: DATA-SOTA-347..352 Return

Date: 2026-08-27 America/Bogota
Auditor: General Musashi
Tips: `agent-multi@1207df84`, `lts@b6bef6c`

## Verdict

**ACCEPT 347--352 AS CORRECTED; REVISE BEFORE TRANSFER LOADER.**

The submitted counterexamples are real and the requested corrections govern
the executing paths. Independent evidence:

- 77 focused tests passed; five optional real-data tests initially skipped.
- Tier-A repeated with the real ETH H4 CSV: **6/6 passed**.
- LTS quote scheduler: **22/22 passed** against its real SQLite implementation.
- No GPU job or transfer loader was run.

Four integration findings remain. They do not reopen 347--352, but they block
calling the partitions leakage-free, activating the quote collector as
training-grade, or implementing the transfer loader.

## Findings

### DATA-SOTA-353 (S2): no purge between train, calibration and monitor

The split uses adjacent step ranges: train ends at 8107, calibration begins at
8108; calibration ends at 8707, monitor begins at 8708. Quantile targets extend
to horizon 12, so labels from the preceding partition cross into the next one.
The rolling observation window also shares context across the boundary.

Shared past context may be allowed and declared, but target overlap is not.
Apply a purge of at least `max(horizons)` between partitions and declare whether
an additional embargo is used. Evidence must bind input-context range and target
range separately.

### DATA-SOTA-354 (S2): canonical quote conflicts are silently ignored and OLAP is stale

`INSERT OR IGNORE` treats a second quote with the same
`(venue,symbol,broker_time)` but different payload as an ordinary duplicate.
That hides source revisions or corruption. Additionally,
`alpaca_quote_summary_olap` still reads legacy `quote_observations`, while the
new scheduler writes `quote_canonical`; collected data is invisible to the
existing analytical surface.

Exact duplicate payloads may be idempotent. Conflicting payloads must refuse or
enter an explicit revision table. Migrate the OLAP view to canonical facts and
retain a separately named legacy view if historical compatibility is needed.

### DATA-SOTA-355 (S3): TRM manifest rename is not directory-durable

The temporary manifest file is flushed and fsynced, then renamed, but the parent
directory is not fsynced. A power loss can therefore lose the acknowledged
directory entry. Add parent-directory fsync and an injected failure regression.

### DATA-SOTA-356 (S3): later origin can skip the immediate predecessor

The decision verifier proves only `earlier_start < this_start`. It does not bind
the immediately preceding declared origin, so a much older decision can mint a
later origin while skipping an unresolved intermediate origin.

Bind `predecessor_origin_id` in the origin plan and require exact equality in
the decision artifact and contract. First-origin exemption remains explicit.

## Accepted Facts

- Dates now parse strictly and origin artifacts are loaded and digest-verified.
- The five branches form a complete ordered 83/83 feature partition.
- Objective balancing uses calibration, not monitor.
- The transferred encoder receives runtime-domain inputs; objective heads stay
  outside transferred state.
- Quote terminal states and ordinary retry idempotency are now honest.
- TRM future-effective observations are unavailable before their validity date.
- The o2022 v3 smoke remains mechanics-only and makes no economic claim.

## Gate

Corrections 353--356 are CPU-only. Do not implement or run the transfer loader,
activate the scheduler, or launch GPU pretraining until their independent
reproduction. Live trading remains untouched.
