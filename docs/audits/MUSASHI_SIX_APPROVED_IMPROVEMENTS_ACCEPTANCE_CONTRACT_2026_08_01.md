# Six Owner-Approved Improvements: Audit Acceptance Contract

Date: 2026-08-01
Authority: owner-approved by Harvey
Audit author: Musashi during the temporary auditor role
Implementation owner: Satoshi during the temporary technical-lead role
Runtime mutation by this document: none

This document converts the owner's approval into falsifiable audit criteria.
It does not prescribe Satoshi's implementation and does not claim completion.

## 1. Consolidated Multi-Front Status

Acceptance requires a machine-readable record with generation time, source
freshness, provenance and explicit units/horizons for Fronts 1-4. It must
separate direct observations from derived estimates and expose unavailable
fields rather than inventing values. An independent reconstruction from the
referenced sources must materially agree.

## 2. Critical Path and Safe Work Overlap

Acceptance requires a dependency graph that identifies the active critical
path, dependency-blocked work and preparatory tasks that can run on idle CPU
without touching DOIN chain state or degrading GPU workers. Non-interference
must be measured; a graph alone is insufficient.

## 3. Live-Evidence Calibration Loop

Acceptance requires immutable provenance from observed spreads, sessions,
disconnect/reconnect behavior and runtime feature availability to a proposed
scenario profile. Calibration may enter optimization only at a new job/domain
boundary with a new hash. Live evidence may constrain feasibility and costs;
it may not leak protected outcomes into model or asset selection.

## 4. Queue-State Taxonomy

The canonical states are `running`, `materialized`, `dependency_blocked`,
`proposed` and `owner_blocked`. Acceptance requires explicit transition rules
and tests rejecting contradictory states such as running plus owner-blocked or
materialized without its required artifact/configuration hashes.

## 5. Role-Swap Resilience Metrics

At handback measure: cold-start recovery duration; material discrepancies
against the baseline; lost or undiscoverable files; unsupported claims caught;
unsafe actions attempted or correctly refused; and model/token cost. Report
numerators, denominators and collection gaps. Neither participant grades its
own success alone.

## 6. Event-Driven Audit and Academic Work

Acceptance requires named runtime triggers for material transitions and
incidents, plus a low-frequency fallback cadence. In periods without a delta,
auditor capacity moves to bounded citation verification, collision searches
and evidence matrices rather than polling. Trigger misses, duplicate audits
and audit cost must be measurable.

## Independence and Closure

Musashi verifies evidence but cannot independently close criteria he authored.
Satoshi implements but cannot close his implementation. Dual-party disputes
go to Harvey. Finding 034 remains outside this contract and still requires the
owner or post-handback independent disposition.
