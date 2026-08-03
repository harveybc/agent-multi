# Musashi to Lieutenant Satoshi II: Immediate L1 Execution Order

Date: 2026-08-02 America/Bogota
From: General Musashi, temporary independent auditor
To: Lieutenant Satoshi II, temporary technical lead
Owner priority: active demo trading and business knowledge first

Read first:

1. [L0 acceptance and MT5 security audit](/home/harveybc/Documents/GitHub/agent-multi/docs/audits/AUDIT_L0_ACCEPTANCE_AND_MT5_AT_F2_006_2026_08_02.md)
2. [Continuous demo operations](/home/harveybc/Documents/GitHub/agent-multi/docs/work_plan/29_CONTINUOUS_DEMO_TRADING_OPERATIONS.md)
3. [Your 053-058 return](/home/harveybc/Documents/GitHub/agent-multi/docs/handoffs/SATOSHI_II_RETURN_053_058_2026_08_02.md)

## Ruling You Must Internalize

L0 passed. Stop treating it as the current blocker.

`AT-F2-006` does not make MT5 trade-ready by itself. The present EA is
read-only and has no command path. Do not tell the owner that an EA source
review is the final lock on MT5 trading. It is one gate plus a missing
implementation.

The shortest executable route is IBKR Paper L1. MT5 write work follows in
parallel only when it cannot delay IBKR.

## P0 Build: IBKR Paper L1

Act as a senior trading-systems engineer, distributed-systems engineer,
security engineer and machine-learning platform engineer. Reuse the accepted
L0 contracts and service. Do not create competing DTOs or a second risk
engine.

Deliver:

1. An IBKR Paper adapter behind the same sink interface as
   `ZeroNetworkSink`, disabled by default and impossible to instantiate
   without the exact L1 profile and single-use owner authorization.
2. A three-order bracket builder using the official TWS transmission order:
   parent `Transmit=false`, take-profit child `Transmit=false`, stop-loss
   child `Transmit=true` last.
3. A preflight which obtains current contract details, minimum quantity,
   quantity increment, price increment, margin estimate, market state,
   account fingerprint, client ID and a fresh open-order/position snapshot.
4. Selection of `EUR.USD` unless fresh evidence proves `USD.CAD` eligible.
   Never substitute an unobserved asset silently.
5. A single-use sequence: one minimum-size long bracket -> flat and exact
   reconciliation -> one minimum-size short bracket -> flat and exact
   reconciliation.
6. Broker acknowledgement of parent, SL and TP as a hard post-submit
   condition. Missing or ambiguous protection causes cancel/flatten and a
   global hold. No new risk while any acknowledgement is unknown.
7. Crash-safe idempotency and an effects journal extending the accepted 055
   pattern across broker side effects.
8. Direct OLAP facts for request, broker IDs, statuses, fills, SL/TP child
   state, cancel/flatten, commission, spread, slippage, latency and every
   rejected alternative. Never infer a zero or success from missing alerts.
9. An owner-visible heartbeat and Telegram completion/problem alert using
   deterministic facts only.
10. A one-command deployment and rollback path. TWS authentication remains a
    human action; the service must fail closed when TWS is stale or offline.

Official IBKR implementation references:

- https://interactivebrokers.github.io/tws-api/bracket_order.html
- https://interactivebrokers.github.io/tws-api/order_submission.html

## Mandatory Adversarial Tests

- duplicate activation and duplicate intent;
- concurrent identical intents;
- parent accepted with one child rejected;
- child acknowledgement before parent acknowledgement;
- partial fill before all acknowledgements;
- disconnect after parent submission and before final child submission;
- restart with unknown parent/child state;
- stale capability, stale quote and future quote;
- wrong account, client ID, venue, asset or instrument;
- minimum size breaching any cap;
- invalid tick/quantity rounding;
- long and short wrong-side SL/TP;
- owner kill during every lifecycle state;
- existing manual order or position in the target account;
- no sockets when profile is L0 or authorization is absent.

The full suite and a zero-submit TWS-connected preflight must pass before
returning for audit. Do not submit the canary yourself; return the exact
activation packet and wait for owner authorization after independent review.

## P1 Parallel Build: MT5

Correct findings 060-062 and design a separate demo-only execution EA. The
read-only EA remains read-only. The execution EA must not be a boolean mode of
the observation EA.

Required MT5 design properties:

- signed, expiring, account-bound commands with persistent replay defense;
- source and compiled EX5 hashes in deployment evidence;
- mandatory SL and TP in the initial `MqlTradeRequest`;
- broker result and subsequent terminal-state reconciliation;
- no command for live accounts;
- strict non-empty account/server allowlists;
- bounded request/response payloads;
- idempotent cancel/flatten and restart recovery;
- firewall evidence captured without exposing secrets.

## Deferred

P20 instrumentation, paper expansion and cosmetic dashboards wait until the
IBKR adapter packet is back for audit. CPU-only work is not an excuse to delay
the business-reality loop.

## Return Packet

Return exact commits, changed files, full commands and outputs, fixture names,
TWS capability facts, zero-submit preflight evidence, deployment/rollback
commands, unresolved risks and the exact proposed owner activation packet.
Use clickable absolute Markdown paths. Do not claim readiness that is not
directly reproduced.
