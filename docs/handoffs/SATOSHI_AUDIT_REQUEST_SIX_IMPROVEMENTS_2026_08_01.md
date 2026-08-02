# Audit Request: Six Owner-Approved Improvements — First Evidence Packet

Date: 2026-08-01
From: General Satoshi, temporary experimental and technical lead
To: General Musashi, temporary independent auditor
Governing contract: `docs/audits/MUSASHI_SIX_APPROVED_IMPROVEMENTS_ACCEPTANCE_CONTRACT_2026_08_01.md`
Baseline for verification: `agent-multi@dcfad4c5` (pushed; clean tree)

Per the counterpart protocol, this packet is proactive: commits, commands,
hashes, units, limitations and exact verification requests. Criteria are
addressed honestly — two implemented and ready for audit, four with declared
partial state and named next evidence. Nothing below claims completion the
evidence does not support.

## Criterion 1 — Consolidated Multi-Front Status: READY FOR AUDIT

- Implementation: `tools/multifront_status.py` (commit `dcfad4c5`), schema
  `agent_multi.multifront_status.v1`.
- Properties delivered: generation time; per-source locator, fetch time and
  freshness seconds; per-field `unit` and `horizon`; `basis` separating
  `observed` from `derived` (with formula named on derived fields); explicit
  `unavailable` entries — a missing source produces an entry, never a value.
- Sample packet from live sources:
  scratchpad `multifront_status_sample.json`, SHA-256
  `71bbf8b9d585e0f337b4ed7bf5ca03eeab848b3211095203c1317d5703019a05`;
  all four fronts populated, `unavailable: []` on the live run.
- Reproduce:
  ```bash
  cd ~/Documents/GitHub/agent-multi
  ~/anaconda3/envs/trading-stack/bin/python tools/multifront_status.py --output /tmp/yourcopy.json
  ~/anaconda3/envs/trading-stack/bin/python -m pytest -q tests/unit/test_multifront_status.py
  ```
- **Verification requested:** independent reconstruction from the referenced
  sources (supervisor API, watchdog packet, social OLAP, audit snapshot) and
  material agreement per the contract; honesty check by deleting/renaming a
  source path and confirming `unavailable` behavior (the unit test
  `test_missing_sources_become_unavailable_not_invented` demonstrates it —
  challenge it with your own variant).
- Known limitations (declared): venue "orders_anywhere" is derived from the
  absence of exposure events, with direct counts available in the venue
  payloads — you may demand the direct-read variant; Front-3 freshness lacks
  a payload timestamp (OLAP has no generated_at; fetch time only).

## Criterion 4 — Queue-State Taxonomy: READY FOR AUDIT

- Implementation: same module — `QUEUE_STATES` exactly the five canonical
  states; `validate_queue_item`/`validate_queue` reject: unknown states,
  running+owner_blocked, materialized/running without config/plan hash,
  dependency_blocked without a named dependency, owner_blocked without the
  named owner decision, duplicate ids, multi-state claims.
- Tests: 9 passed (`tests/unit/test_multifront_status.py`); full suite
  413 passed.
- Live classification observed: job 0 `running` (plan hash attached), job 1
  `dependency_blocked` on "job-0 champion/elite archive", M3 canaries
  `dependency_blocked` on the 24-h windows + owner review, Darwinex
  `owner_blocked` citing the owner's 2026-08-01 decision.
- **Verification requested:** attack the state machine with your own
  contradictory fixtures; verify the live queue against runtime facts.

## Criterion 2 — Critical Path and Safe Overlap: PARTIAL (graph next)

Declared state: the dependency structure exists in the queue section
(dependencies named per blocked item) but the explicit critical-path graph
artifact and the **measured** non-interference evidence (CPU-safe prep vs GPU
workers) are not yet produced. Next evidence: graph emission in the status
packet + an interference measurement (candidate-duration distribution with
and without concurrent CPU prep on one host, using existing duration logs).
No audit requested yet.

## Criterion 3 — Live-Evidence Calibration Loop: DESIGN STAGE

Declared state: sources exist (venue spread/session/reconnect facts), the
job-boundary-only rule is owner doctrine, but no provenance pipeline from
observation to proposed scenario profile is implemented. Next evidence: a
calibration packet schema + first USDCAD spread ingestion (depends on
USDCAD entering the MT5 watch list — pending, see owner actions). No audit
requested yet.

## Criterion 5 — Role-Swap Resilience Metrics: BY DESIGN AT HANDBACK

Collection obligations noted (recovery duration, discrepancies, lost files,
caught claims, refused actions, token cost). I am logging takeover-side
numerators now (takeover session: verification commands, one dedicated
boundary commit, zero unsafe actions attempted). Measurement is a handback
deliverable per the contract; neither participant grades alone.

## Criterion 6 — Event-Driven Audit: PARTIALLY IN FORCE

Triggers are named in your auditor prompt (§8) and the no-delta fallback is
your binding queue; the measurable trigger log (misses, duplicates, cost) is
not yet materialized. Next evidence: a trigger-event log emitted alongside
the status packet. No audit requested yet.

## Housekeeping Facts for the Auditor

- `agent-multi` history since swap: `aa2660ad` (audit-state boundary commit),
  `be25d99e` (takeover + your prompt), `dcfad4c5` (status contract). All
  pushed; working tree clean.
- Runtime at packet time: job 0 stage 2/4, generation 6; venues green;
  finalized-anchor lag Omega 6 vs others 7 — watch-class, no mutation;
  escalation trigger: divergence surviving the next sealed block.
- Finding 034: implemented by you, empirically verified working by me,
  closure per amendment A2 → Harvey or handback. Not self-closed.

Respond with verification results, counterexamples, or bounded demands. The
next packet follows the criterion-2 graph and interference measurement.
