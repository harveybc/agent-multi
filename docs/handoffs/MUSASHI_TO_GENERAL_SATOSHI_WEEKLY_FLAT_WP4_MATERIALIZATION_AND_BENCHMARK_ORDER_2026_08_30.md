# Musashi to General Satoshi: weekly-flat WP4 materialization and benchmark

Date: 2026-08-30
Authorization: `IMPLEMENTATION_MATERIALIZATION_BOUNDED_CPU_ONLY`

WP3 is independently accepted in its effect-free scope. Execute WP4 from the accepted weekly-flat authority and work-plan 42; do not reinterpret the policy in a driver.

## WP4.0. Freeze identities

Materialize the reviewed code, policy, calendar/session contract, observation contract, cost envelope, runtime identity and WP3 executor identity by digest. Validate the held lock tuple before release (`pid/holder`, epoch and scope); a malformed internal handle must refuse before writing release intent.

## WP4.1. Materialize W0-W2

Produce the three predeclared experiment cells from work-plan 42, with one factor changed at a time:

- W0: weekly exposure overlay disabled, baseline behavior;
- W1: weekly flatten/wind-down enabled under the frozen calendar contract;
- W2: W1 plus causal reopen blackout/evidence gate.

If the existing authoritative plan defines these labels differently, stop and return the conflict; do not silently remap them.

All cells must share data, observation identity, architecture, seed, optimizer, update budget, cost envelope, action mapping, stop contract and evaluation endpoints. Persist the effective config and its digest before execution.

## WP4.2. Training semantics

- Historical closure intervals contain no synthesized tradable bars.
- Wind-down, forced flatten and reopen blackout use the same accepted core authority as evaluation.
- Forced closes flow through the shared execution envelope and carry costs and one authoritative close event.
- No reward suppression hides account changes.
- Weekly handling and retraining cadence remain orthogonal factors; do not mix weekend retraining into W0-W2.
- Calendar/session evidence is causal and origin-specific; no future closure or reopening evidence may enter an earlier decision.

## WP4.3. Preflight and bounded benchmark

Implement the materializer and driver, then run only:

1. schema/identity/dry-run tests with no model effects;
2. a bounded CPU end-to-end smoke for each W0-W2 cell;
3. a CPU benchmark measuring wall time, bars, updates, memory, state counts, forced closes, cancellations, reopen-blocked decisions, trade conservation and costs.

The benchmark establishes mechanics and estimates later compute only. It has zero economic authority and may not select a model or policy.

## WP4.4. Acceptance evidence

Return PRE/POST reproducers, exact state trajectories around at least two historical weekend gaps and one adjacent holiday, paired identity proof, trade/PnL/cost conservation, deterministic replay, interruption/resume behavior, and an estimated GPU budget derived from measured CPU work.

Include negative tests for stale/missing calendar evidence, bars inside closure, premature reopen, pending entry during wind-down, failed cancel, failed flatten, unresolved recovery, and treatment leakage into W0.

## Hard boundary

Do not deploy, install, restart services, connect to a venue, send commands, touch existing positions, launch long GPU training, promote a checkpoint, or activate weekly-flat live. Any later GPU campaign and any live window require separate independent acceptance and explicit dispatch.

