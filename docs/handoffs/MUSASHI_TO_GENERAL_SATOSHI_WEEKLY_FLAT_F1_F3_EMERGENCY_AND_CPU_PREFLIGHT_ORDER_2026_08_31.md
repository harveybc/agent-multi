# Musashi to General Satoshi: weekly-flat F1-F3 emergency correction and CPU preflight

Date: 2026-08-31
Owner disposition: immediate execution approved.
Priority: P0 safety correction, then exactly one bounded CPU SAC throughput preflight.

The owner asked to deploy immediately. Live deployment is refused at this moment because the accepted mechanism demonstrably permits exposure to cross governed closures and the flatten custody can elect two winners. Execute the following without waiting; no live service or position may be touched.

## F1: closure safety outranks lingering reopen blackout

Reproduce both the holiday-cluster and simple-weekend counterexamples before editing. Define and implement one explicit precedence contract in the shared authority:

1. during a known closure: `EXPECTED_MARKET_CLOSED`, with any exposure reported as a critical unexpected-exposure/recovery fact;
2. before the next closure: `FORCED_FLATTEN` then `WIND_DOWN` outrank a blackout inherited from the prior reopen for exposure reduction and forced close;
3. `REOPEN_BLACKOUT` continues to block risk-increasing entries but may never suppress cancel/close/flatten duties;
4. `NORMAL_TRADING` only when no higher-priority restriction applies.

Prove long, short, pending-entry and carried-position cases. At every governed closure, enabled arms must reach zero positions and zero pending entries or terminate in a typed critical failure. No state may preserve exposure merely because blackout is active.

## F2: eliminate one-bar entry windows and blackout oscillation

Reproduce the one-bar NORMAL blip that admits an entry before blackout returns. Add a session-scoped, causal release latch with hysteresis/probation:

- entry permission begins only after all minimum time/bars/stability predicates have remained satisfied for the declared release sequence and the release is committed at a closed-bar boundary;
- a single transient pass cannot authorize an entry;
- renewed instability blocks new risk immediately but never blocks reductions or the next closure's wind-down/flatten;
- restart deterministically reconstructs the same latch from bound evidence and cannot mint an extra tradable bar.

Predeclare the probation/hysteresis semantics; do not tune them from the reproduced counterexample. Test noisy spread/gap/volatility, missing quotes, exact boundaries, restart and adjacent closures.

## F3: one flatten-custody transition winner

Reproduce the pristine race distribution with real synchronized processes. Correct the transition election so the complete read/verify/expected-state/write/ack transaction is serialized by one durable generation-bound claim. Exactly one process may win `in_flight`; every loser must observe or recover the winner without overwriting it.

Test at least 200 synchronized process races per transition on fresh roots, plus injected crash/fsync boundaries, stale locks, ABA generations, symlinks, wrong modes and fresh-process recovery. Zero double winners is required. A flaky pass is failure.

## C9-C14 acceptance evidence

Preserve the corrected C9-C14 work, but regenerate eligibility after F1-F3. Festive and simple-weekend enabled representatives must be flat and fully reconciled. `cancel_submitted` is not terminal; require the broker's terminal cancellation verdict. Remove all private filesystem topology from public artifacts.

EURUSD remains mechanics-only. It carries no ETH/MT5 economic or calendar authority. Economic campaigns remain blocked by `VENUE_SESSION_HISTORY_UNAVAILABLE` until the ETH/MT5 evidence contract is satisfied.

## Automatic bounded CPU SAC preflight

After F1-F3 pass the focused and full suites, implement and execute exactly one preflight without another owner phrase:

- CPU only, one seed, one corrected overlay-enabled cell;
- at most 20,000 environment steps and 20,000 SAC updates;
- hard wall-clock limit 2 hours and explicit process/phase/progress/ETA telemetry;
- config, code, manifest, data and action/genesis identities persisted before execution;
- observable and externally stoppable;
- report environment throughput and SAC-update throughput separately, peak memory, state counts, closure compliance, conservation and any refusal;
- `MECHANICS_AND_THROUGHPUT_ONLY`, zero economic authority and no checkpoint promotion.

If any F1-F3 test fails or the preflight encounters unresolved exposure/custody state, stop fail-closed and return evidence. Do not weaken a gate to finish the run.

## Still prohibited

No GPU campaign, complete economic grid, deployment, installation, service restart, venue connection, order command, live position change, checkpoint promotion or live activation. A separate independent audit of F1-F3 and the preflight is required before any live window.

