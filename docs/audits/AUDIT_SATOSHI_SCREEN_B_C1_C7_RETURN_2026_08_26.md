# Audit: Screen B C1-C7 correction return

Date: 2026-08-26
Auditor: General Musashi
Audited tips: `agent-multi@a91455fc`, `gym-fx@b3e78fe5`
Verdict: **REVISE — B4 DISPATCH REMAINS REFUSED**

## Findings

### AUD-F1-20260826-324 (S2) — the simulated entry is unprotected for one H4 bar

`shared_execution_envelope` submits the parent first and defers both protective
children until the next `apply_action`. The source explicitly declares that the
protection arms one bar after the fill. This is not equivalent to the Demo seat,
which submits SL/TP with the risk-increasing request, and it can miss a stop or
target touched during the entry bar. Finding 318 therefore remains open.

The problem is not merely theoretical: the accepted B3 primary traces contain
67 `envelope_residual_sweep` events (24/22/21 by origin). A supposed exceptional
guard firing repeatedly proves order lifecycle/size races remain in the normal
path. No result containing such sweeps is G1-authoritative.

### AUD-F1-20260826-325 (S2) — the cost canon mixes venues and mistakes Paper omission for business cost

Spread is sourced from Alpaca ETH/USD quotes, while commission is taken from 103
generic L0 lifecycle reports that contain neither symbol nor venue. The query
does not prove those rows are ETH, Alpaca or MT5 broker facts. More importantly,
a Paper fill reporting zero commission does not establish the fee paid by the
corresponding real product. The proposed primary therefore combines unrelated
surfaces and understates Alpaca crypto economics; it also omits MT5 CFD-specific
spread/financing. Finding 320 remains open.

### AUD-F1-20260826-326 (S2) — B4 still does not bind the shared envelope or cost contract

The return packet acknowledges that B4 must bind the strategy and costs “at
dispatch”; the materialized B4 packet does not contain either. Contract identity
must exist before review, not be injected at launch. Otherwise B4 and B0-B3 are
not the same experiment. Findings 318/320/322 continue to block GPU use.

### AUD-F1-20260826-327 (S2) — missing observation declarations remain accepted

The pipeline seam validates a supplied v2 contract, but an undeclared contract
is still a recorded no-op. The correction order required missing declarations
to refuse on every B4 construction/resume path. A caller can therefore bypass
the authority by omitting the object. Finding 322 is only partially corrected.

### AUD-F1-20260826-328 (S2) — the 1%/2% envelope cannot govern the signal comparison

The corrected CPU run established a useful negative result: 330-545 H4 stops
per year and catastrophic degradation across every directional baseline. This
means the screen measures an evidently mis-scaled envelope rather than signal
quality. “Same harness” does not require spending 47-97 GPU-hours under a known
bad harness. Calibrate envelope geometry causally on fit/monitor data first,
then freeze it before each scored origin. The current 1%/2% result remains a
named diagnostic arm, not the primary G1 economy.

## Accepted work

- Findings 316/317 remain independently verified corrected.
- Finding 319 is corrected at the requested-exposure conversion layer: sizing
  is equity/price based, lagged and capped; the traces expose requested and
  realized exposure.
- Finding 321 is corrected: deterministic trial IDs, conflict refusal, run
  identity, timing and statistical-input exclusion are present.
- The v2 observation contract produces the proven 2660-wide actor when present.
- Evidence clobbered at the historical WP4 path was restored and future output
  defaults to its run directory.
- Focused independent tests: 61 agent-multi and 25 gym-fx tests passed.

## Security disposition

Finding 323 is re-triaged as accepted residual S4 exposure. GPU UUIDs are stable
topology identifiers, not credentials. Sanitize current operational examples
and future evidence where practical, but do not rewrite public Git history and
invalidate the audit chain. Deleting the zero-unique-commit branch alone has no
security value.

## Decision

Do not ratify the 5.53 bp “primary” canon and do not launch B4. Preserve all 45
runs as `DIAGNOSTIC_NOT_G1_AUTHORITY`. Correct the order lifecycle, build
venue-specific economic contracts, causally calibrate the envelope, then rerun
B0-B3 before a bounded B4 preflight is reconsidered.

