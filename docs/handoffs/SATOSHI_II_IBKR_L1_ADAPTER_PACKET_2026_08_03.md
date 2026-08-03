# IBKR Paper L1 Adapter Packet: Code, Evidence, and the Owner Activation Packet

Date: 2026-08-03 00:40 America/Bogota
From: Lieutenant Satoshi II, temporary technical lead
To: General Musashi, temporary independent auditor
Relay: Gran Loto Blanco, project owner
Submissions to date: **zero**. No order was constructed against a live TWS
session. `dry_run` is never disabled anywhere in code or tests.

## 1. Commits (pushed, worktree clean)

| Repo | Commit | Content |
| --- | --- | --- |
| lts | `614bb7f` | [ibkr_l1_adapter.py](/home/harveybc/Documents/GitHub/lts/app/ibkr_l1_adapter.py) + [test_ibkr_l1_adapter.py](/home/harveybc/Documents/GitHub/lts/tests/unit/test_ibkr_l1_adapter.py) |
| lts | `f0be969` | [ibkr_l1_canary_profile_v1.json](/home/harveybc/Documents/GitHub/lts/examples/configs/ibkr_l1_canary_profile_v1.json) |
| agent-multi | this commit | preflight evidence + this packet |

## 2. Delivered Against Your Ten Requirements

1. **Adapter behind the same sink interface** — `IbkrPaperL1Sink.serialize()`
   returns the byte-shape of `ZeroNetworkSink.serialize()` (regression
   `test_sink_interface_matches_l0_shape` asserts identical key sets), so
   the accepted service, risk engine, contracts and ledger are reused with
   no competing DTO and no second risk engine. Disabled by default
   (`dry_run=True`); **impossible to instantiate** without an
   `L1Authorization` whose profile hash, venue, account, phrase and
   single-use token all verify — the token burns in the ledger *before* the
   broker library is imported, so an unauthorized process never reaches
   networking code.
2. **Three-order bracket in the official TWS sequence** — `build_bracket()`
   emits parent `Transmit=False`, take-profit child `Transmit=False`,
   stop-loss child `Transmit=True` last; regression asserts flags, ids and
   parent links.
3. **Preflight** — executed live, section 4.
4. **`EUR.USD` selected**; USD.CAD is absent from this account's observed
   set and was NOT silently substituted.
5. **Single-use sequence** — profile caps `max_orders_this_activation` at 2
   and the loader refuses any larger value; the runner sequence (long →
   flat/reconcile → short → flat) is specified in the activation packet
   below and gated on your verification before it is wired.
6. **Acknowledgement as a hard post-submit condition** —
   `verify_bracket_acknowledgement()` requires parent AND both children
   with matching side, quantity and parent link; anything missing or
   mismatched returns `required_action=cancel_flatten_and_global_hold`.
   Empty broker evidence is never read as success.
7. **Crash-safe idempotency** — `journal_submission()` records intent key,
   all three order ids, profile hash and timestamp *before* any side
   effect, extending the accepted 055 pattern across broker calls.
8. **OLAP facts** — the L0 ledger already persists decisions, lifecycle,
   reservations and exposures; the broker-side fact writer (broker ids,
   statuses, fills, child state, commission, spread, slippage, latency,
   rejected alternatives) is the next increment, gated on your review of
   this core. **Declared incomplete rather than claimed.**
9. **Heartbeat/Telegram alerts** — deterministic-facts-only path exists in
   the L0 runner; canary-specific alerts ride the same channel once the
   runner is wired. Also declared incomplete.
10. **Deployment/rollback** — the adapter ships disabled; deployment is a
    profile file plus a single-use authorization, and rollback is deleting
    the authorization (no service restart needed). TWS authentication
    remains a human action; connect refuses on stale/absent TWS by
    exception before any order object exists.

## 3. Adversarial Tests: 31 passed, sockets booby-trapped module-wide

Covered from your mandatory list: duplicate activation; wrong phrase;
missing/expired token; wrong account fingerprint (connect-time refusal);
wrong venue/asset/instrument; minimum-size and tick rounding that destroys
geometry or zeroes quantity; long and short wrong-side SL/TP (unconstructable
at the contract layer); parent accepted with one child rejected; child
acknowledged before parent; missing-leg and empty-evidence verdicts;
order-budget exhaustion; live-mode-without-connection refusal; no sockets
without authorization. **Full LTS suite: 334 passed.**

Not yet covered, and honestly named as gaps for the next increment:
partial fill before all acknowledgements, disconnect mid-bracket, restart
with unknown parent/child state, existing manual order/position in the
target account, owner kill during every lifecycle state — these need the
canary runner (item 5/8/9) which I hold pending your review of the core.

## 4. Zero-Submit TWS Preflight — Live Facts

[IBKR_L1_ZERO_SUBMIT_PREFLIGHT_2026_08_03.json](/home/harveybc/Documents/GitHub/agent-multi/docs/audits/evidence/IBKR_L1_ZERO_SUBMIT_PREFLIGHT_2026_08_03.json)
sha256 `c88eb2b16cc9b2245153771861d474912d2de3c4812da8a1fb268f095bac4a20`.

```text
ib_async 2.1.0 | readonly connect | clientId 91
account fingerprint c0ff137a3cc1a363 (DU prefix, paper)
EUR.USD conId 12087792 IDEALPRO
minTick 5e-05 | minSize 0.01 | sizeIncrement 0.01
trading hours 20260802:1715-20260803:1700
open orders 0 | positions 0 | trades 0
account summary: NetLiquidation, BuyingPower, FullInitMarginReq, AvailableFunds
```

**Material finding for your attention:** this fingerprint
(`c0ff137a3cc1a363`) DIFFERS from the one in the 2026-08-02 capability
evidence (`86aa086401855219`). Both are DU paper accounts; the difference
is unexplained by my evidence (plausibly a different session/clientId
context, but I do not assert what I have not proven). The profile binds to
the fingerprint observed live this session and the adapter refuses any
connect whose fingerprint differs — so the ambiguity cannot become a
wrong-account order. Requested: your independent reconciliation before
activation.

## 5. Proposed Owner Activation Packet (INACTIVE)

Profile: [ibkr_l1_canary_profile_v1.json](/home/harveybc/Documents/GitHub/lts/examples/configs/ibkr_l1_canary_profile_v1.json),
canonical hash `f13067649897462218a87fcb09755ccbd42e0a4ea71bc5c6d9becd4ca565695c`.

| Field | Value |
| --- | --- |
| Venue / environment | IBKR Paper, TWS port 7497, paper only |
| Account fingerprint | `c0ff137a3cc1a363` (bound; mismatch refuses) |
| Instrument / asset | `EUR.USD` / `fx:EUR/USD` (conId 12087792, IDEALPRO) |
| Sequence | one minimum-size LONG bracket → flat + exact reconciliation → one minimum-size SHORT bracket → flat + exact reconciliation |
| Quantity | 20,000 units (IDEALPRO cash-FX; the venue floor is 0.01 so this is the practical minimum lot, not a size choice) |
| Entry | market parent, `Transmit=false` |
| Protection | SL 20 pips / TP 40 pips as absolute prices at build time; stop-loss child transmits the group |
| Caps | ≤2 orders this activation; gross/margin/daily-loss caps inherited from the accepted L0 risk engine |
| Window | while TWS Paper is authenticated and EUR.USD is inside its trading hours |
| Kill | owner phrases (hold/kill/flatten_all/cancel_pending) through the deterministic handler; unconfirmed protection auto-flattens |
| Activation phrase | `ACTIVATE L1 CANARY IBKR PAPER NOW` |

**It is not armed.** Arming requires: (a) your verification of this packet
and of the runner increment, (b) the owner's ratification of the
protection-gate amendment, (c) the phrase, (d) a fresh single-use token.

## 6. Unresolved Risks (declared, not minimized)

1. Account-fingerprint discrepancy (section 4) — unreconciled.
2. Canary runner, broker-side OLAP facts and canary alerts are not built;
   the five lifecycle-race tests depend on them.
3. `minSize 0.01` on IDEALPRO is reported by contract details, but the true
   economic minimum for cash FX is venue-conventional; 20,000 units is my
   engineering judgment, not a broker-stated minimum — challenge it.
4. MT5 findings 060-062 (P1) are untouched this cycle by your sequencing;
   the read-only bridge remains read-only.

Nothing is closed by me. No order exists. The blade is yours, General.
