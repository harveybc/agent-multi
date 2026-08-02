# Six Owner-Approved Improvements: First Evidence-Packet Audit

Audit ID: `AUDIT-GEN-20260801-SIX-01`
Date: 2026-08-01
Auditor: Musashi, temporary independent auditor
Implementation owner: Satoshi, temporary technical lead
Implementation reviewed: `agent-multi@dcfad4c5`
Request reviewed: `agent-multi@a9ec4029`
Runtime mutation: none

## Disposition

| Criterion | Claimed state | Audit disposition |
| --- | --- | --- |
| 1. Consolidated status | ready | `reported_changes_required`; live values reproduce but contract counterexamples remain |
| 2. Critical path | partial | acknowledged; not audited |
| 3. Calibration | design | acknowledged; not audited |
| 4. Queue taxonomy | ready | `reported_changes_required`; live queue is coherent but validator counterexamples remain |
| 5. Role-swap metrics | handback | acknowledged; initial Musashi metrics remain provisional |
| 6. Event-driven audit | partial | acknowledged; measurable trigger log still absent |

## Findings

### AUD-GEN-20260801-035 — S3 — Empty alert set can falsely report zero orders

**Reproduced:** a synthetic watchdog packet contained Alpaca 3 open orders,
IBKR 2 and MT5 1 while `active_event_keys=[]`. `collect()` emitted
`orders_anywhere.value=0`. The field therefore reports an exposure fact from
the absence of alerts, even though direct venue counts are present.

**Impact:** a consolidated safety surface can show zero orders while six are
reported by its own source. Current live counts are genuinely zero, so no
live exposure was concealed during this audit; the defect is prospective but
material.

**Smallest correction:** derive per-venue and aggregate counts directly from
all available venue payloads. Aggregate state is `unavailable`, not zero, if
any required venue count is missing. Keep alerts as a separate field.

**Regression required:** the six-order fixture above; one missing-venue-count
fixture; direct live-count reconciliation.

### AUD-GEN-20260801-036 — S3 — Queue validator accepts contradictory semantics and maps failure to materialized

**Reproduced:** all four fixtures below passed validation:

1. `dependency_blocked` plus `owner_blocked_reason`;
2. `owner_blocked` plus `dependency`;
3. `running` plus `dependency` and `plan_sha256="x"`;
4. `materialized` plus `config_sha256="not-a-sha"`.

A mocked supervisor job with `status="failed"` was silently classified as
`materialized` by `collect()`.

**Impact:** the packet can represent a contradictory or failed item as ready,
defeating the taxonomy's purpose during transition and recovery.

**Smallest correction:** enforce allowed and forbidden fields per state;
validate SHA-256 syntax; explicitly map only known supervisor states; reject
unknown/failed states or expose them outside the executable queue as
unavailable/error history. Completed work should not masquerade as
materialized work.

**Regression required:** the five counterexamples above plus the current live
queue fixture.

### AUD-GEN-20260801-037 — S4 — Provenance and partial-payload honesty are incomplete

**Reproduced:** `/api/network` supplies chain coherence and queue facts but
`sources` registers only `/api/status`; `distinct_unfinalized_tips` and anchor
height lack explicit units/horizons. A valid but incomplete watchdog emitted
`None` values without field-level `unavailable` entries and still reported
zero orders. A syntactically valid `[{}]` snapshot crashed with
`AttributeError` instead of becoming unavailable.

**Smallest correction:** register the network endpoint independently; add
unit/horizon metadata to numeric chain facts; validate source payload type and
required subfields; convert wrong-type or partial payloads to explicit
field-level unavailability.

**Regression required:** partial-object and wrong-type JSON fixtures plus a
source-to-field provenance assertion.

## Verified Non-Findings

- Independent packet SHA-256:
  `a0590c72a0ae2c11f1e5c786d67260567318d2b729854c0e4de5888e4d8210d2`.
- Direct reconstruction materially matched the live packet: generation 7,
  5/20 evaluated at collection time, one finalized anchor, fitness
  `0.0006247008569073586`; Alpaca 823 sessions, IBKR 440, MT5 heartbeat age
  11.35 seconds; social 96 runs/1782 posts/0 drafts; audit snapshot hash
  `af98e560...`.
- The live queue classification was coherent with runtime facts: job 0
  running, job 1 dependency-blocked, M3 dependency-blocked and Darwinex
  owner-blocked.
- Focused suite: 9 passed. Full suite: 414 passed with two pre-existing
  scikit-learn convergence warnings. The request stated 413; this one-test
  count delta is recorded but is not itself a failure.
- Missing top-level sources become unavailable in the author's existing test.

## Bounded Response Requested from Satoshi

Implement only the three smallest corrections above, add the demanded
regressions, and return: commit, test command/count, a fresh packet hash and
the exact direct-source reconstruction. Do not alter criteria 2/3/5/6 merely
to make this audit appear complete. Neither participant closes these findings;
closure follows the role-swap independence rules.

## Improvements Proposed After the Findings

These are non-blocking proposals and do not expand the correction packet:

1. Publish a JSON Schema in which queue entries are a discriminated union by
   canonical state. This makes incompatible fields structurally impossible
   for Python and non-Python consumers.
2. Give every consolidated metric a `source_ref` into the packet's source
   registry and record collection start/end timestamps. This makes cross-front
   temporal skew and provenance mechanically auditable.
3. Add per-source freshness budgets and a derived `fresh/stale/unavailable`
   state. Raw age remains visible; the budget supplies operational meaning
   without pretending one threshold fits all sources.
