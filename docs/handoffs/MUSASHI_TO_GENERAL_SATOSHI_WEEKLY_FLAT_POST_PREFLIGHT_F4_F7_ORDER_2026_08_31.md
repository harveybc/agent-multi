# Musashi to General Satoshi: weekly-flat post-preflight F4-F7 order

Date: 2026-08-31
Authorization: immediate implementation, bounded mechanics compute, and read-only evidence collection only.

## F4: execution-latency-aware feasibility

Extend materialization feasibility with the actual execution contract: decision bar close, order submission latency, next executable bar/fill, venue close boundary and configured safety margin. A forced-flatten window is admissible only if the worst-case permitted close can be filled and reconciled before closure.

For the current H4/next-bar contract, reproduce and ledger the observed 4-hour failure. Mechanically reject every W1/W0 cell that cannot satisfy the deadline; do not spend training trials on impossible cells. Recompute the family/trial manifest and multiplicity ledger. The live-safe default must come from an eligible value and may not silently remain four hours.

Test exact boundaries, delayed fill, rejected first close, retry budget, holiday-shortened session and multiple bar sizes.

## F5: probation is an explicit policy parameter

`RELEASE_PROBATION_FACTOR=2` is a conservative provisional safety choice made after a counterexample, not an established optimum. Put it in the typed policy/config identity, observation/state telemetry and materialization. Keep `2` as the mandatory provisional minimum for any live-capable arm; include a bounded non-live ablation in the W2 trial ledger. Restart must reconstruct the exact qualification/probation phase and streak.

No arm with a one-bar release window is live-eligible.

## F6: representative strong-architecture preflight

Preserve the completed run as `SB3_MLP_PLUMBING_ONLY`. Build a separate preflight through the canonical strong grouped extractor and SAC construction path accepted by the data-first architecture work:

- same observation v2/v3 identity required by that architecture;
- same PatchTST/TFT-style/TimesNet-style/TCN-GRU family branches, state branch and cross-family fusion selected by the reviewed config;
- actor, critic and target all constructed through the production materializer;
- pretrained loading either disabled explicitly for random-init throughput or performed through the accepted sealed loader; no hidden partial transfer;
- one cell, one seed, bounded initially to 2,000 environment steps and 1,000 real gradient updates, CPU first, hard limit two hours;
- report actual optimizer update counter, tensor/device identity, parameters, peak memory and per-branch gradients;
- if CPU cannot finish, stop and return measured evidence; do not substitute a smaller MLP.

This preflight is automatically authorized after F4/F5 tests pass. It has zero economic authority and saves no promotable checkpoint.

## F7: unblock MT5 ETH session evidence, read-only

Inventory the current bridge/EA payload and private state for already available ETHUSD session/calendar history without changing a service. If sufficient historical-time venue evidence exists, export a sanitized, digest-bound read-only dataset with session intervals, symbol, server timezone/version, acquisition provenance and gaps explicitly separated from authority.

If it does not exist, implement the collector/schema and a coordinated activation runbook. Activation is authorized only when it can be performed read-only without replacing/restarting an EA that protects an open position. Otherwise stop at `COORDINATED_WINDOW_REQUIRED`; never risk the existing position to collect metadata.

No trading command capability may exist in the collector. Add a structural no-write test and freshness/identity checks. Economic WP4 remains `VENUE_SESSION_HISTORY_UNAVAILABLE` until this evidence is independently accepted.

## Return

Return corrected manifests, removed/rejected cells, trial counts, F4/F5 PRE/POST, the strong-architecture preflight result or typed timeout, and MT5 evidence status. Run focused and complete suites.

Still prohibited: economic grids, checkpoint promotion, trading-service changes that can affect positions, order commands, live position changes and weekly-flat live activation.

