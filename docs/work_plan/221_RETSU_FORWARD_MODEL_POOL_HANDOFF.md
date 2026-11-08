# Retsu Handoff: Hourly Forward Model Pool and Paper/Demo Succession

Date: 2026-11-08. Status: implementation assignment and acceptance design;
no broker action or service change performed by this handoff.

## 1. Assignment and Master-Plan Position

Owner request: when a verified model has nonnegative TRAIN and VALIDATION
performance, preserve it and begin collecting forward evidence rather than wait
for research to finish. Run the current pretrained candidate, subsequently add
refined/retrained candidates, and compare them hourly under the same market
conditions. One selected model executes in an authorized Paper/Demo account;
the others maintain independent virtual accounts. Never use real capital.

- Retsu owns implementation, offline tests, restartable automation and deployment
  receipts for this track. Reuse existing infrastructure; do not build another
  trading framework or parallel authoritative operational database.
- Musashi owns checkpoint/recipe export, exact inference and strategy parity,
  refinement diagnosis, and review of the candidate-admission boundary.
- The owner controls account/route authorizations under existing capability
  rules. This document does not mint credentials, release holds or overwrite a
  running service. Implement everything offline through deployment-ready scripts;
  report the exact existing activation requirement if a new route needs one.

This is a parallel event track within master work-plan chapters22,29,32 and
the current generalization/refinement subplan. Historical promoted-paper gates
in chapter32 remain intact. A research-demo candidate is not falsely labeled
as having passed that chapter's independent promotion panel. Do not impose its
26-week statistical gate on starting a no-order shadow account either.

## 2. What Already Exists and What Is Incompatible

Read-only discovery of `lts` at `12bce5f2c4fe2ab0e6a7433d7208d9e05a9cffaa`:

| Existing component | Reuse or limitation |
|---|---|
| `app/live_model_selection.py` | Hash-pinned model pointer, route checks and execution tiers; reuse pattern, not its linear policy loader |
| `app/live_sac_selection.py` | SB3 continuous SAC with action thresholds; NOT the current categorical policy |
| `app/project3_sac_observation.py` | ETH recipe requires rolling z-score and agent state; NOT the selected70 recipe |
| `app/multi_venue_shadow.py` | No-order portfolio marking; not proof that independent policy-driven accounts already exist |
| `app/ibkr_l1_journal.py`, outbox/recovery modules | Existing durable decision/effect and recovery patterns |
| `app/model_position_control.py`, `live_flatten_custody.py` | Inspect ownership and close-management contracts before integration |
| `tools/rolling_evidence_report.py`, `tools/live_sim_replay.py` | Existing read-only report/replay paths; extend where compatible |

Respect each repository's AGENTS instructions. Live ledgers are read-only during
discovery; operational tools, credentials and service restarts are not smoke tests.
The old ETH routes are not the new EURUSD route. Verify instrument, shorting,
leverage, protective-order and data capabilities separately for MT5 and Alpaca.
If a venue cannot implement EURUSD's contract, record incompatibility and keep
that route disabled; do not silently substitute crypto or another instrument.

## 3. Initial Candidate and Eligibility Event

Initial reference: selected70, 28 groups, 13 observations at stride2,
fixed TRAIN scaler, no account-state inputs, categorical NOP/BUY/SELL.
Source: [weight-preserving transfer211](211_SELECTED_WEIGHT_PRESERVING_REFINEMENT.md)
and [policy calibration215](215_FROZEN_POLICY_RETURN_CALIBRATION_RESULTS.md).

| Historical reference | TRAIN2020-2023 | VALIDATION2024 |
|---|---:|---:|
| Net return, percent | 11.51453 | 13.80849 |
| Orders | 748 | 206 |
| Orders/week | 3.57895 | 3.81481 |

These are offline reference results, not forward results. Epoch0 model SHA256:
`079bef0ea6f3b9327772f4204e5d5dbbd4be69a5ab58c32b424aa3cb7c686404`.
Musashi supplies the private artifact location; public docs contain no account
identifiers, private locations or machine topology.

Admission event requires authenticated artifacts, finite nonnegative net return
in both periods, the existing activity rule (strictly more than two orders/week
in both), business-contract validity and a reproducible inference bundle.
No-trade zero return is not an eligible success. Deduplicate on immutable
model/recipe/strategy identity. Event states:

`REGISTERED -> PARITY_VERIFIED -> SHADOW_ACTIVE -> EXECUTION_ELIGIBLE -> EXECUTOR`

Registration starts work; it does not itself grant broker authority. Retain
ineligible candidates with their reasons, without executing them.

## 4. Runtime Architecture

```text
Governed feed -> completed-bar snapshot + availability watermark
                        |
             per-recipe causal feature materialization
                        |
         immutable candidates, deterministic model adapters
                        |
          independent strategy + virtual account per model
                        |
             common-window reports / weekly selector
                        |
       route authority + single-writer lease + flat handover
                        |
             authorized Paper/Demo broker adapter
                        |
            fills / protection / reconciliation events
                        |
               simulator divergence replay
```

Ownership: `agent-multi` exports models and admission evidence; `lts` owns
pool orchestration and execution; `trading-contracts` owns shared schemas;
the existing governed data/feature producers own availability and transforms;
`gym-fx` owns simulation. Keep JSON configuration and established plugin loaders.
Run deterministic hourly inference on CPU first; reserve GPUs for training
unless measured latency requires otherwise. Never train in the order loop.

### Model and data boundary

Persist weights, architecture, column order, group layout, scaler constants,
input/output semantics, strategy/calendar/risk config, code and dependency
identities. Reload the exact bundle, not a separately reconstructed approximation.
Use greedy categorical inference, evaluation mode and the native tie policy.
Stochastic sampling is a different policy; its historical behavior differs
materially and must not be substituted. Do not convert continuous thresholds
into the three actions. Position state belongs to the controller, not the network.

All candidates in a comparable cohort consume the same hourly snapshot ID and
availability cutoff. Materialize identical recipes once, not once per candidate.
Use the latest finalized hourly bar and only features published by decision time;
`now-1h` is not proof of finalization or availability. Preserve timestamp/DST
semantics, cadence and gaps. No new rolling normalization, clipping, interpolation
or synthetic fill. If a required feature is absent, suppress new entries and
record the reason while maintaining protective/forced-close management.

Training through recent data uses the same materialization path with a frozen
as-of watermark. A newly fitted scaler or model gets a new identity and shadow
account. It never overwrites the incumbent in place. Original validation-period
results do not remain an out-of-training evaluation for a successor trained on
that period; keep old results attached to the original artifact.

### Independent accounts and action semantics

Each virtual account has its own balance, position, pending effects and costs.
Shared market data does not mean shared position or profit. Flat BUY/SELL opens;
with a position, the native opposite-direction action closes, while NOP/same
direction holds. Preserve the native rule for same-tick reentry; do not add an
implicit reversal. Include SL, TP, financing, spread and calendar exits.

Inherit the authenticated strategy's full business contract. Explicitly verify
one open position, mandatory SL/TP, leverage1:50, opening embargo12h after weekly
open, forced flat12h before weekly close and size:

`allocated_margin = q * rel_volume * current_balance`, `q in [0,1]`, `rel_volume=0.1`.

Apply rel_volume once; convert margin to instrument units through broker metadata
and leverage. Broker minimum size must not round a small permitted allocation
above its cap. The user removed the old one-entry/day limit: do not reintroduce
it. Export the actual checkpoint contract's entry limits; a disagreement must
be resolved explicitly, not silently changed in deployment.

Protective orders, stale-feed supervision and calendar deadlines are not limited
to hourly inference. Use existing native protection/watchdog behavior. Never
take custody of an unrelated pre-existing account position.

### Selection and succession

Proposal: rank weekly on completed common forward windows, not after each lucky
trade. Default research ranking is net return after costs among candidates
meeting the unchanged activity and risk contract; break ties by lower drawdown,
then retain the incumbent. Record Sharpe, uncertainty and trade counts as well.
Freeze the JSON ranking rule before the first selection; any change is versioned.
Do not select on TRAIN profit or raw critic Q. Higher-return models violating risk
limits are not winners. Apply the authorized route's stronger promotion gates
where required; a research ranking cannot override them.

Require one complete common week and valid coverage for the weekly research
comparison; if insufficient, retain the incumbent and report insufficient support.
Report both lifetime and same-window results. New arrivals start their actual
forward clock when admitted. Historical replay may aid comparison but is not
backdated forward evidence. Keep the initial reference as a permanent shadow.

Only one writer per account/route. Queue a successor until flat; the opening
model retains close responsibility for its existing position. Shared risk and
forced-close controls always override. Switching cannot silently close a trade,
reset the account, or allow two controllers to send orders. Record tenure and
actual broker cash/equity at handover; shadow ledgers continue uninterrupted.

## 5. Required Evidence and Tests

Design acceptance tests first, then component/unit tests and implementation.
Every row below requires a named automated test and evidence in Retsu's return.

| ID | Requirement and falsification test |
|---|---|
| P01 | Full saved observation/action/account replay matches the selected70 baseline; wrong columns, scaler, action mapping or stochastic mode is rejected |
| P02 | Incomplete bars, DST changes, revisions and future-feature perturbation cannot alter earlier decisions; stale input cannot create an entry |
| P03 | Identical models on identical independent initial ledgers produce identical actions and balances; one candidate's trades cannot mutate another's account |
| P04 | Restart before/after send/ack and repeated bars do not duplicate effects; unknown broker outcome reconciles before retry |
| P05 | Concurrent writers, mismatched account tier/asset and unauthorized routes cannot submit; real-capital paths remain disabled |
| P06 | Switch requested during an open position preserves its owner and protection; successor acts only after reconciled flat state |
| P07 | Size, minimum lot, leverage, SL/TP, early close and weekly boundaries match native simulation for long and short cases |
| P08 | Zero orders cannot satisfy eligibility; missing data and non-finite metrics are not zeros; candidates use the same forward comparison window |
| P09 | Broker disconnect preserves native protection; unknown position or stale authority prevents new entries, not unsafe custody changes |
| P10 | Restart resumes persisted cohort, snapshots and last processed bar; feature warm-up is reconstructed without submitting historical catch-up orders |
| P11 | Replay with recorded bid/ask, fees, swaps and execution latency reconciles cash and explains remaining simulator-versus-broker differences |
| P12 | New training cutoff/model/scaler creates a new identity; no mutable artifact changes an active or historical account |

Record per decision input/config/model hashes, signal, intent, controller outcome
and refusal reasons. Record bid/ask, quote source/time, fill/slippage, fees/swap,
rejection/partial-fill and protective-order events. Separate modeled shadow fills
from observed broker fills. Calibrate simulator changes on one captured interval
and verify on later capture; do not silently rewrite earlier accounts.

Per-model reports: net cash/P&L/return, mark-to-market equity, daily-equity Sharpe
with declared annualization, max drawdown, orders/week, buy/sell counts and win
rates, costs, holding time, exposure, early-close/SL/TP/forced-close counts,
coverage, refusals and divergence. Undefined metrics remain null with reasons.
Use existing OLAP/ledger paths for compact facts and content-addressed files for
bulky tensors/plots. Public reports exclude account and private infrastructure IDs.

## 6. Work Packages and Return Contract

1. Inventory existing routes and components read-only; publish capability matrix
   and explicit gaps. Verify the selected70 bundle; no legacy recipe substitution.
2. Implement offline model adapter, common snapshot schema and P01/P02/P07/P12.
3. Implement registry, independent policy accounts and scheduler with P03/P08/P10.
4. Add reports, deterministic weekly ranking, lease/handover and P04/P05/P06/P09.
5. Integrate recorded-feed simulator comparison and P11; package restartable
   services with isolated paths, locks, bounded retries and failure reason codes.
6. Deliver offline end-to-end receipts and a dry-run deployment plan. Activate
   only through the existing authorized Paper/Demo procedure. A first route's
   required owner capability is not replaced by a script bypass.

For each package return commit, effective JSON, test names/results, artifact
hashes and a measured duration. No speculative GPU ETA: this work is primarily
engineering and hourly inference, not a training sweep. Resume completed stages
by identity, never duplicate work because an observer restarts. A failed
candidate may be quarantined while healthy shadows continue; shared feed or
account-authority failures must stop new execution.

Report separately: CODE_VERIFIED, SHADOW_OBSERVING and PAPER_DEMO_EXECUTING.
A document, unit test or running process alone is not an execution receipt.
Retsu's first return should establish exact-model parity and the first captured
hourly shadow snapshot; Musashi continues the independent refinement diagnosis.
