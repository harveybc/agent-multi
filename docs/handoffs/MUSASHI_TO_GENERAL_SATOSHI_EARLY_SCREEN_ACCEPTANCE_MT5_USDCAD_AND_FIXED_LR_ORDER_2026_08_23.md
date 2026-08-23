# Musashi to General Satoshi: early-screen acceptance and next execution order

Date: 2026-08-23 America/Bogota
Audited delivery: `agent-multi@31622476`
Owner priority: Alpaca Paper and OANDA MT5 Demo are the only active venues;
IBKR Paper is suspended.

## 1. Independent verdict

The sealed early-intervention screen is **accepted**.

- Focused aggregator suite: 23 passed.
- Independent aggregation from the eight committed reports: exit 0.
- Independently regenerated canonical JSON: byte-identical to the committed
  aggregation (`sha256 b9d56d5e285e9c909d610a5381fde97270856f33fd00eda9aa3dd1f731a3c1f5`).
- Outcome: `SHORT_SCREEN_SIGNAL_AGAINST`.
- Primary deltas plateau minus fixed: seed 101 `0.0`, 202 `-0.0183947`,
  303 `-0.0219084`, 404 `-0.0198121`; median `-0.0191034`;
  zero positive and three negative seeds.

Accepted interpretation is narrow: early ReduceLROnPlateau with the tested
specification harmed this bounded ETH experiment. No checkpoint is promoted;
the tested scheduler specification is not a DOIN gene candidate. This is not
a universal rejection of every adaptive learning-rate method.

The three fail-closed corrections made during aggregation are accepted for
this packet, subject to permanent regression coverage. Record them in the
findings register with implementer and independent verifier separated.

## 2. P0: replace IBKR USD.CAD with MT5 USDCAD

Implement without disturbing the active ETHUSD route.

1. Treat IBKR as `suspended_by_owner`. Do not start TWS, its runner, observer
   or continuity monitor. Preserve its ledgers and 12 closed exposures.
2. Confirm direct MT5 account facts already observed: `USDCAD` exists,
   `trade_mode=4`, minimum volume `0.01`.
3. Generalize the MT5 backend and EA routing for two simultaneous instruments:
   `ETHUSD` and `USDCAD`. Commands, nonces, magic numbers, reservations,
   exposure accounting, idempotency and evidence must be symbol-scoped.
4. Keep the existing ETHUSD EA and runner uninterrupted. A USDCAD addition
   must not reset its session, ledger or position state.
5. Create a separate `USDCAD H4` MT5 profile and runner. Reuse the historical
   USD.CAD model only after proving feature schema, bar timing, scaling,
   artifact hash and action semantics are compatible with MT5 CopyRates data.
   Refuse rather than silently adapt incompatible inputs.
6. Use Demo only, volume `0.01`, native SL/TP on entry, model-controlled early
   close, a unique magic number and the existing daily risk ceiling. No live
   capital and no foreign-position adoption.
7. Add socket-free tests for cross-symbol command theft, nonce replay,
   wrong-symbol acknowledgement, mixed magic numbers, duplicate fills,
   restart idempotency and one-symbol failure while the other remains healthy.
8. Deliver an install/update instruction only for the minimum human MT5 action
   that cannot be performed from Linux. Do not ask the owner to touch ETHUSD.
9. Publish a preflight packet before the first USDCAD order: direct symbol
   facts, zero USDCAD positions/orders, effective profile, hashes, service
   units, rollback and expected first-decision timing.

Historical comparison baseline to preserve:

- MT5 ETHUSD: 23 completed round trips;
- Alpaca SPY: 10 completed exposures and one currently open at last audit;
- IBKR USD.CAD: 12 completed exposures, now suspended.

## 3. P1: main fixed-LR L1 curriculum experiment

Prepare in parallel with P0. Do not reuse ReduceLROnPlateau.

The experiment must answer whether easy pretraining improves subsequent normal
training, not merely whether an easy arm can trade.

### Paired arms

For each seed, materialize both:

- **Control N:** normal-only training from the declared cold start.
- **Treatment EN:** easy training, then normal training using the exact same
  model and learned tensors. Do not reinitialize actor, critic or learned
  normalization state at handoff. Explicitly declare replay-buffer and
  optimizer-state continuity and test the chosen semantics.

Both arms use fixed LR `3e-4` during the normal phase. The normal phase gets
the same data, update budget, early-stopping contract and evaluation surfaces
in both arms. Report the additional easy-phase compute separately; do not hide
it through equal-wall-clock truncation.

### Training contract

- Maximum 2,000 epochs per phase.
- Early-stopping patience 60, inactive before epoch 40.
- Activity/economic episodic objective already accepted; terminal zero-trade
  penalty applies only to an episode with zero trades, never to intra-episode
  NOP actions.
- Easy may remain economically negative while learning activity; do not stop
  or reject it solely for negative profit.
- Handoff requires an eligible easy checkpoint, preserved tensors and at least
  two mapped normal decision crossings.
- Train/monitor/inner/outer/sealed roles remain isolated. The sealed test is
  not inspected for stopping, selection or configuration changes.
- At least four paired seeds, counterbalanced dispatch, immutable manifests,
  durable per-epoch histories and explicit GPU identity.

Primary comparison: paired normal-phase best eligible monitor score EN minus N.
Secondary facts: return, drawdown, Sharpe, trades, exposure, action diversity,
selected/stopping epoch, actor/critic movement and compute cost. Predeclare the
direction rule before any terminal arm exists.

### Dispatch sequence

1. Materialize typed contracts and a deterministic CPU smoke.
2. Demonstrate one genuine easy-to-normal handoff and one normal-only control
   without opening the sealed test.
3. Return the config, manifests, tests and exact four-GPU launch command for
   independent review.
4. Continue P0 CPU/code work while review runs. Do not spend the fleet on an
   unreviewed configuration.

## 4. P2: documentation and status truth

- Update the work plan to identify Alpaca + MT5 as active and IBKR as preserved
  but owner-suspended.
- Remove all operator prompts asking for TWS login while suspended.
- Record the early scheduler screen as accepted negative evidence.
- Do not call the bounded screen statistically conclusive; it is a directional
  four-seed screening result.
- Return one packet covering P0/P1 independently, with separate commits and
  explicit residual doubts. Audit does not block useful implementation work.

No additional owner phrase is required for implementation, CPU tests or MT5
preflight. The first USDCAD Demo order and the four-GPU curriculum dispatch
wait for independent verification of their respective packets.
