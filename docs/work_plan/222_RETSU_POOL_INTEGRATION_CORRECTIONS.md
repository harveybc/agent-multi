# Retsu: Complete the Model-to-Demo Path

Date: 2026-11-08. Reviewer: Musashi. Scope: focused integration review and
execution guidance following the owner's request. This is not a completed
audit of every LTS component or a broker activation receipt.

Reviewed `lts` `ea340f8f11b5548a6c63ad408dc9a6714813c638` and the audit request
in `agent-multi` `f4aa3ab0a330c8ce0cd7a7e66ea34e115d787e98`.
Do not confuse a functioning fixture harness with an operating model pool.

## Immediate Disposition

The existing MT5 VM is running and its guest agent answered a read-only ping.
No VM restart is needed. This proves guest responsiveness, not a fresh MT5
login, EA snapshot, EURUSD feed or broker execution. The current MT5 runner
still reports the ETH linear model and `snapshot_stale`; the current Alpaca
runner reports the SPY linear model and `no_execution`. Neither is selected70.
No service, capability, broker order or live ledger was modified in this review.

Retsu's 13 offline tests reproduce successfully (0.58 seconds on the review
environment). Four additional executable counterexamples fail. Verdict:
**useful offline scaffolding; REVISE before forward execution**.

## Reproduced Findings

Reproducer: `tools/audit_forward_pool_handoff.py`, writes only temporary fixture
state and an explicitly supplied report; no broker connection or credentials.

```bash
python tools/audit_forward_pool_handoff.py \
  --lts-root <reviewed-lts-checkout> --output <scratch>/counterexamples.json
```

Exit1 means a violated expectation was reproduced, not a tool startup failure.

| ID | Location in reviewed LTS | Measurement and correction |
|---|---|---|
| F01 | `cohort.py:seed_selected70`, `_step`; `snapshot.py:materialize` | Registers one column instead of70; defaults to fixed signal0. `_step` never invokes the loaded network. Replace fixture-only defaults in the production path with the authenticated bundle, actual70 columns, frozen scaler and13x70 stride2 temporal tensor. Missing inference must refuse, never become silent NOP. |
| F02 | `cohort.py:_save`, `_load`; `accounts.py:RouteBook.submit` | Persisted one unknown effect and halt become zero effects and no halt on reload. Journal-before-send must be durable, not an in-memory dictionary. Restore complete effect/authority/position/close-history state; reconcile an unknown outcome before any retry. Test actual process death around submission. |
| F03 | `cohort.py:_mirror_route` | Executor cash8500 becomes shadow cash12000 with zero broker-fill events. Broker accounting and shadow accounting must remain independent across all handovers. Use reconciled executor fills/cash, never copy a shadow's historical wealth or position into the route. |
| F04 | `accounts.py:VirtualAccount.apply`, `_intrabar`; `snapshot.py:HourlyBar` | On a finalized bar, a newly opened long closes at SL using that already completed bar's earlier low. Cash becomes9968.1878 without any post-entry market event. Separate observation bars from subsequent execution events. A pre-entry low/high cannot stop or profit a new position. Preserve next-event execution ordering and bid/ask semantics. |

Further integration gaps verified by source inspection:

- The production risk/entry planner is missing. A global `q=1` and caller-supplied
  shared SL/TP are not the native selected70 strategy. Its pinned
  `env_plugins/discrete_curve_native.py:entry_plan` calls `propose` with
  direction-specific references, decision sigma, frozen genes, financing reserve,
  costs, lot/tick rules and account balance. Reuse this computation through an
  adapter; do not invent replacement distances or volume.
- The service can apply account actions before `PARITY_VERIFIED`/`SHADOW_ACTIVE`;
  only `INELIGIBLE` is excluded. Enforce state gates in the actual step path,
  not just helper methods.
- `rank()` does not perform a guarded handover, and service steps do not persist
  divergence reports. Finish those integrations after the first model/feed slice.
- Do not declare packages1-6 complete on the basis of helper tests. The request
  itself correctly discloses the missing service connections. Preserve that
  distinction in the next receipt.

## Decisions Answered Now

1. Preserve native first-index argmax. An exact BUY/SELL tie above NOP selects
   BUY in this reference; an all-class tie selects NOP. Do not replace it with
   a different tie policy while claiming parity.
2. Offline admission activity is strictly more than two orders/week in both
   periods, as handoff221 specifies. Zero-order zero-return is not eligible.
3. Native `three_signal_native.command_for` maps the opposite signal to command3,
   close without simultaneous reentry. Network signals remain0/1/2.
4. Use the native absolute-order validator and risk planner; do not infer all
   semantics from the older recipe strings `CLOSE_AND_OPEN` or `clip`.
5. CPU/GPU floating-point differences do not require bitwise logits. The32-row
   no-action-disagreement probe is preliminary evidence, not full-service parity.
   Compare full saved support, near-tie cases, positions, SL/TP/q, fills, costs and
   account results on the intended inference backend. Record tolerances before
   the run; never increase them merely to accept a failed replay.
6. Retsu owns wiring and running the complete replay; Musashi supplies source
   artifacts and independently reviews the comparison. Do not wait for Musashi
   to implement Retsu's missing production code.
7. MT5 demo/EURUSD is the first route to qualify. Inspect existing symbol,
   account leverage, protection, calendar and authorization facts read-only.
   Keep SPY/ETH routes separate. An Alpaca SPY route is not a EURUSD route;
   asset support must be established from the venue's asset/capability API, not
   assumed from the fact that its service is running. Reference:
   https://docs.alpaca.markets/us/reference/get-v2-assets-symbol_or_asset_id
8. `SHADOW_OBSERVING=false` is correct until a real governed snapshot reaches the
   network and its virtual account. The next task is to implement that path,
   not to wait for permission to continue offline integration.
9. Yes: executor cash, position custody, unknown effects and close history must
   survive restart. These are requirements, not optional design decisions.
10. Charging a fee on entry and another on exit can be correct per-side charging.
    Do not remove either blindly. Declare entry/exit fee units and reconcile net
    P&L with both, including open-position financing and broker lot conventions.

## Exact Work Order

### A. Complete one real vertical slice first

Load the exported70-column recipe and weights from the existing authenticated
bundle. Build a temporal observation from recorded governed bars using the exact
production feature producers and fixed TRAIN constants. Call the actual model
in eval/inference mode, apply native categorical control and native risk sizing,
then execute against the native replay engine. No fixed signals in this path.

Compare every saved TRAIN and VALIDATION decision and completed trade against
the frozen reference. This is replay verification, not retraining or selection.
Export disagreements by first timestamp and layer (input, logit, action, plan,
fill, accounting), not just an aggregate hash. Use the same confirmed runtime
and source bundle as handoff221; do not alter the selected checkpoint.

### B. Fix persistence and accounting with failing tests first

Turn F01-F04 into regressions, including long/short, gaps, forced-flat,
opposite-signal close, both barriers touched and a restart with an unknown order.
Kill/restart the fixture subprocess at each durable boundary. Add an actual
handover test in which executor and shadow have deliberately different balances.
Reuse the existing outbox/reconciliation abstractions where possible instead of
maintaining an unsafe second route engine. Recheck the full service, not only
individual methods. Do not edit or stop existing broker services for these tests.

### C. Attach the real hourly input path

Produce a70-column availability manifest: source, exact producer/config hash,
last observation/publication time, required warm-up, fixed scaling identity and
refresh cadence. Price-only MT5 bars do not reconstruct external or decomposed
features automatically. Test prefix invariance and hourly batch/incremental
parity; never fill missing features with zeros or switch to the old ETH recipe.

Once parity passes, run a new isolated no-order shadow process on the actual
governed feed. Heartbeat must show selected checkpoint hash,70 actual columns,
window13/stride2, fixed_train, last finalized bar, last inference and last ledger
event. Its historical seed data must be marked replay, not forward observation.
Respect session closures when assessing quote freshness, but never turn stale
data into a fresh decision. Market closure is not permission to fabricate bars.

### D. Qualify and connect MT5 demo

Inspect the existing MT5 bridge and demo mandate metadata through approved
read-only interfaces. Confirm EURUSD symbol and EA routing, independent model
identity, allowed operations, margin/protection rules and existing position
ownership. Repair feed/EA continuity only under the route's operational rules;
do not reset the VM or remove another strategy's state as a shortcut.

Correct the dry-run: `tools/mint_paper_capability.py` is explicitly a single-use
**IBKR** capability tool, not a generic MT5 activation procedure. Do not ask the
owner to run it for this route. Reuse the actual MT5 demo authorization path.
If existing authorization does not cover EURUSD or an owner-only login/approval
is required, identify that exact fact and the smallest user action. Do not
bypass account authority or manufacture a signature.

The old USDCAD runbook includes a human chart/EA step. Do not generalize its
historical "cannot be done from Linux" statement into a present technical fact:
the guest agent responds today. Establish what supported guest automation and
current terminal setup allow before asking for manual UI work. Never expose
credentials through guest-command output or public reports.

After verified shadow and broker preflight, provide the deployment receipt and
use the approved demo activation procedure with flat-account/custody checks,
native SL/TP and an independently reversible service/config change. Then finish
automatic weekly ranking and post-trade simulator divergence. No real capital.

## What Is Needed From the Owner

Nothing for A-C or offline corrections. The VM is already running. Do not ask
for another reboot, new model choice or permission to finish the assigned code.
For D, only report a specific unavoidable account/login/route approval after
checking the existing MT5 path. A generic "mint a capability" request is not
acceptable when it names the wrong venue's tool.

## Next Return

Return corrected commits, F01-F04 regression receipts, full native replay
comparison,70-source availability inventory and the first actual shadow snapshot
with selected model identity. Keep the three statuses separate:
CODE_VERIFIED / SHADOW_OBSERVING / PAPER_DEMO_EXECUTING. None is inferred merely
from a process being active. No additional intermediate research decision is
required from the owner to complete this work order.
