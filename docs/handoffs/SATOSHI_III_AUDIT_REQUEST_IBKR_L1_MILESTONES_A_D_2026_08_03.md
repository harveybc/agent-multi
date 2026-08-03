# Audit Request: IBKR L1 Corrections, Milestones A-E (Findings 063-068)

Date: 2026-08-03 America/Bogota (v1.1.0 — extended from A-D to A-E and
amended after Codebase Memory MCP adoption; v1.0.0 was `agent-multi@16f28c78`)
From: Satoshi III (Mujuro Utsutsu), successor technical lead in bootstrap
To: General Musashi, temporary independent auditor
Owner authorization: Milestone A ordered begun 2026-08-03; CLI capability
concept accepted with constraints (privileged authority separation, fixed
protected storage, one bracket per capability, short expiry, atomic
consumption in the existing L0 ledger, no broker connectivity)
Broker submissions in this work: **orders_submitted = 0** (see §7)

I implemented findings 063-068; per protocol I close none of them. This
packet requests your independent reproduction of Milestones A-E and your
dispositions on the questions in §8. Milestone F is NOT claimed.

## 1. Exact Commits and State

All repositories clean and pushed at request time.

| Repo | Commit | Content |
| --- | --- | --- |
| lts | `0c51844` | Milestone A: exact fake-broker effects contract |
| lts | `e0a4f2c` | Milestone B: capability + strict profile v2 |
| lts | `d003501` | Milestone C: exact acknowledgement + executed recovery |
| lts | `5f10a84` | Milestone D: outbox consumer behind accepted L0 |
| lts | `be6019a` | Milestone E: disabled-by-default runner + heartbeat; D semantic corrections |
| agent-multi | `524877c5` | takeover report (context) |
| agent-multi | `16f28c78` | audit request v1.0.0 (A-D) + ledger row |
| agent-multi | this commit | audit request v1.1.0 (A-E, MCP adoption) |

Baseline was `lts@f0be9698`; the accepted L0 service
(`app/demo_execution_service.py`) is **byte-identical** to the audited
revision — `git diff f0be9698..be6019a -- app/demo_execution_service.py`
is empty. The L1 tables live in the same SQLite ledger via the
`L1ExecutionOlap(DemoExecutionOlap)` subclass; no parallel truth.

Consumer corrections landed inside `be6019a` (found while wiring the
runner; audit them as part of D):

- transient quote problems (missing/stale/future/inverted/wide) now
  DEFER a pending decision instead of terminally rejecting it; the
  decision survives for a fresh quote;
- a decision whose own quote evidence has aged beyond
  `max_decision_age_seconds` (default 300 s) terminally rejects — dead
  evidence never executes late;
- `sync_parent_fill` is idempotent via the `parent_fill_applied` journal
  fact (a second sync can no longer violate lifecycle continuity).

## 2. Changed Files and Why

New, in [lts/app](/home/harveybc/Documents/GitHub/lts/app):

- [ibkr_l1_journal.py](/home/harveybc/Documents/GitHub/lts/app/ibkr_l1_journal.py)
  — durable effect state machine (`journaled_pending → effect_unknown /
  submitted_pending_ack → acknowledged → terminal_*`, illegal promotions
  refused), append-only broker facts, single-use capability burn
  (UNIQUE digest + UNIQUE nonce hash), decisions-outbox queries. One
  database with L0.
- [ibkr_l1_broker.py](/home/harveybc/Documents/GitHub/lts/app/ibkr_l1_broker.py)
  — narrow `IbkrClientProtocol`; translation of the audited `BracketPlan`
  dicts into REAL `ib_async` `Contract`/`Order` objects with exact fields
  and re-checked invariants; recording `FakeIbkrClient` with failure
  injection (disconnect at call N, cancel failure, child-before-parent
  refusal, auto-fill realism for flatten reconciliation).
- [ibkr_l1_executor.py](/home/harveybc/Documents/GitHub/lts/app/ibkr_l1_executor.py)
  — finding 063 correction: journal before every broker call; capability
  burned atomically with the FIRST durable effect record (064); partial or
  failed call sequences durably become `effect_unknown`; the best possible
  return is `submitted_pending_ack` — no code path returns success without
  the corresponding calls journaled; duplicates and restarts replay.
- [ibkr_l1_capability.py](/home/harveybc/Documents/GitHub/lts/app/ibkr_l1_capability.py)
  — strict capability schema v1 bound to profile hash/schema, fingerprint
  algorithm+value, venue/host/port, instrument/asset, risk and quantity
  ceilings, `max_entries=1`, expiry ≤ 1 h; fail-closed gate over the fixed
  protected store (0700/0600 enforced, ambiguity refused, consumed digests
  refused); durable status classification (issued / consumed-before-effect
  / effect-unknown / acknowledged / terminal).
- [ibkr_l1_recovery.py](/home/harveybc/Documents/GitHub/lts/app/ibkr_l1_recovery.py)
  — finding 065 correction: `verify_bracket_exact` requires status,
  account, contract, side, quantity, type, price, TIF and parent-link
  agreement from direct facts; missing facts are failures. Recovery is
  EXECUTED and journaled: hold persisted first in the L0 `halt` key (kill
  never downgraded), idempotent cancels, real opposite-side flatten,
  mandatory position reconciliation; unproven outcomes stay
  `effect_unknown` with the hold in place.
- [ibkr_l1_outbox.py](/home/harveybc/Documents/GitHub/lts/app/ibkr_l1_outbox.py)
  — finding 066 correction: the outbox IS the L0 decisions table; entry
  quantity is the L0 `plan_units` result; profile/capability ceilings
  refuse and never resize; `live_observed` evidence required (ruling R3);
  canary gating makes a new entry impossible while any effect is
  non-terminal; flatten consumption cancels orphan children, reconciles
  to zero and closes the L0 exposure; crash-resume re-acknowledges.

Changed:

- [ibkr_l1_adapter.py](/home/harveybc/Documents/GitHub/lts/app/ibkr_l1_adapter.py)
  — strict profile v2 (067): exact key set, loopback-only, labeled
  `account_id_sha256_16` fingerprint algorithm (068), bounded client id,
  1-2 entry budget, positive finite ceilings with canary sanity bounds,
  every retained field enforced (map in the class docstring). REMOVED:
  the repository activation phrase, `L1Authorization` (064 root), the
  lying `submit_bracket` (063 root) and the presence-only acknowledgement
  verifier (065 root). Connection helper is `connect_readonly()` only.
- `examples/configs/ibkr_l1_canary_profile_v1.json` removed (it carried
  the phrase); `ibkr_l1_canary_profile_v2.json` added.
- New tool [tools/mint_paper_capability.py](/home/harveybc/Documents/GitHub/lts/tools/mint_paper_capability.py)
  — TTY-only, confirmation-phrase-gated, offline mint; the only writer of
  the store; no broker imports.
- Ledger row `MULTI-VENUE-PAPER-001` updated in
  [doc 13](/home/harveybc/Documents/GitHub/agent-multi/docs/work_plan/13_IMPLEMENTATION_STATUS_AND_TASK_LEDGER.md).

## 3. Test Evidence

Environment: `/home/harveybc/anaconda3/envs/trading-stack/bin/python`
(3.12.13, ib_async 2.1.0). Commands and results:

```text
pytest tests/unit/test_ibkr_l1_effects.py     -> 33 passed   (Milestone A)
pytest tests/unit/test_ibkr_l1_capability.py  -> 35 passed   (Milestone B)
pytest tests/unit/test_ibkr_l1_adapter.py     -> 43 passed   (Milestone B)
pytest tests/unit/test_ibkr_l1_recovery.py    -> 28 passed   (Milestone C)
pytest tests/unit/test_ibkr_l1_outbox.py      -> 15 passed   (Milestone D)
pytest tests/unit/test_ibkr_l1_runner.py      -> 10 passed   (Milestone E)
pytest tests/                                  -> 467 passed  (full suite)
```

Every L1 test module booby-traps `socket.socket`/`create_connection`;
`ib_async` is used for object construction only. Your reproducer
`IBKR_L1_ADAPTER_REPRO_2026_08_03.py` no longer applies verbatim (the
functions it drove were removed/replaced); its four counterexample
conditions are each re-expressed as named tests below.

## 4. Failure-Mode Fixture Map (§9 of the cold start)

- 063 exact: `test_state_never_claims_submission_before_any_broker_call`,
  `test_partial_call_sequence_is_unknown_never_success`,
  `test_submission_invokes_parent_tp_sl_in_exact_sequence`
- 064 exact: `test_gate_refuses_already_consumed_capability`,
  `test_consumed_capability_cannot_authorize_a_second_bracket`,
  `test_mint_cli_refuses_without_interactive_terminal`,
  `test_capability_refuses_any_deviation` (19 parametrized bindings)
- 065 exact: `test_audit_065_reproducer_scenario_now_fails_closed`,
  `test_any_identity_or_status_deviation_is_not_protected` (13 params)
- 066: `test_full_canary_long_flat_short_flat`,
  `test_quantity_above_profile_ceiling_is_rejected_not_resized`
- 067: strict-profile rejections in `test_ibkr_l1_adapter.py` (17 tests)
- 068: `test_profile_rejects_wrong_fingerprint_algorithm`; algorithm
  labeled in profile and capability schemas
- duplicate activation/intent/outbox: `test_duplicate_intent_replays_and_makes_no_new_calls`,
  `test_write_capability_never_overwrites`, outbox idempotency via UNIQUE
- concurrent identical intents: `test_concurrent_identical_intents_yield_one_effect`
- crash before effect / after each call / before ack persistence:
  `test_resume_distinguishes_pre_effect_from_unknown`,
  `test_partial_call_sequence_is_unknown_never_success` (calls 1,2,3),
  `test_crash_before_ack_resumes_through_exact_acknowledgement`
- parent accepted with child missing/rejected/cancelled/inactive:
  recovery parametrizations + `test_missing_stop_loss_triggers_hold_cancel_and_terminal`
- wrong type/price/account/contract/side/quantity/parent: 065 params
- partial fill before protection: `test_partial_fill_before_protection_flattens_and_reconciles`
- disconnect between submission/acknowledgement points:
  `test_disconnect_mid_submission_reconciles_via_acknowledge`
- restart with unknown state, no duplicated exposure:
  `test_restart_never_repeats_an_acknowledged_effect`, consumer `resume`
- stale/future quote and stale capability:
  `test_bad_quotes_are_durable_refusals`, `test_expired_capability_refuses`,
  `test_future_issued_capability_refuses`
- rounding breaching caps: `test_quantity_rounding_to_zero_rejects`,
  `test_rounding_that_destroys_geometry_rejects`,
  `test_geometry_beyond_profile_distance_ceiling_is_refused`
- owner hold/kill in every state: `test_global_hold_blocks_new_risk_before_capability_burn`,
  `test_owner_kill_is_honored_and_never_downgraded`,
  `test_halt_defers_entry_consumption`
- recovery-action failure and retry: `test_cancel_failure_journals_unknown_then_retry_completes`,
  `test_unreconciled_flatten_stays_unknown_and_held`
- no socket without pre-issued capability:
  `test_no_capability_defers_without_any_socket_or_effect`
- no broker submission anywhere: module-wide socket booby traps, all 456

## 5. OLAP / Restart / Reconciliation Evidence

The journal persists, per effect: creation before any call, per-leg
`call_attempt`/`call_result`/`call_failure`, acknowledgement snapshots and
verdicts, every recovery attempt/result, reconciliation outcomes and
terminal classification — inspect `l1_effects`, `l1_broker_facts`,
`l1_capabilities` next to the accepted `decisions`/`reservations`/
`exposures`/`lifecycle_events` in one SQLite file. Restart classification
is exercised by the A/D resume tests; capability lifecycle by
`test_capability_status_distinguishes_lifecycle`.

## 5b. Milestone E Deliverables (added in v1.1.0)

- [app/ibkr_l1_runner.py](/home/harveybc/Documents/GitHub/lts/app/ibkr_l1_runner.py)
  — strict config (unknown keys refused); `enabled: false` default never
  constructs a broker client; the Milestone-F-less default client factory
  degrades deterministically to a `client_unavailable` alert while the
  heartbeat keeps flowing; every tick: crash resume → fresh read-only
  quote (accepted `QuoteSource` reuse) → entries → idempotent fill sync →
  flattens → ledger-derived alerts/events → atomic heartbeat write.
  Telegram delivery remains with the deployed watchdog; no token here.
- [examples/systemd/lts-ibkr-l1-canary.service](/home/harveybc/Documents/GitHub/lts/examples/systemd/lts-ibkr-l1-canary.service)
  — disabled by default, double-switched (unit AND config flag), rollback
  commands documented in the unit itself.
- [examples/configs/ibkr_l1_runner.example.json](/home/harveybc/Documents/GitHub/lts/examples/configs/ibkr_l1_runner.example.json)
  — ships `"enabled": false`; command phrases are owner-set outside Git.
- Fixtures: disabled-runner-never-touches-a-client, degraded-no-client,
  full heartbeat-driven entry→fill→flatten cycle, stale/missing quote
  deferral with alert, halt alert, no-capability alert, atomic heartbeat,
  run-once exit.

## 6. What Is NOT Claimed (remaining work)

- **Milestone F:** connected zero-submit preflight and the real
  `IbkrClientProtocol` implementation over `ib_async` (fact mapping from
  `Trade`/`orderStatus`). Not started; no TWS connection was opened in
  this entire work session.
- Commission/spread/slippage/latency capture from real fills pending the
  Milestone F client (spread and quote age are already journaled).

## 7. orders_submitted = 0

Derivation: (a) no TWS/broker connection was opened at any point in this
session — every test runs with sockets booby-trapped and the only
connection helper is `connect_readonly()`, which was never invoked;
(b) the standing read-only watchdog evidence at 2026-08-03T06:47:35Z
(multifront snapshot sha256 `1ffd61d0…947a`) reports zero open orders and
zero positions on IBKR/Alpaca/MT5 with cumulative `orders_submitted = 0`.
A fresh connected preflight is deliberately deferred to Milestone F.

## 8. Questions, Doubts and Dispositions Requested

1. **Capability separation limit (residual risk, declared).** On a
   single-user machine the mint/executor separation is structural (only
   the TTY CLI writes the store; the executor has no minting code) and
   procedural, not cryptographic — a hostile local process with the
   owner's uid could write a file the gate would accept. Disposition
   requested: require an owner-held offline Ed25519 signing key (e.g. via
   `cryptography`) with the public key pinned in the profile, or accept
   the structural separation for the Paper canary tier?
2. **L0 emergency semantics for risk-reducing fills.** The accepted
   `protection_covers_filled` treats ANY filled parent report without
   protection legs as unprotected exposure. Correct for entries; wrong for
   flatten fills — feeding a flatten fill through
   `apply_execution_event` sets hold and re-emits flatten intents. My
   consumer therefore appends flatten lifecycle events directly to the
   same chained ledger with the same continuity rule
   (`ibkr_l1_outbox._consume_flatten`, commented). Since L0 is accepted
   and dual-authority, I did not touch it. Disposition requested: keep the
   workaround, or open a finding to make the L0 emergency check
   intent-class-aware?
3. **Acknowledgement status strictness vs TWS reality.** I accept
   children only in `{PreSubmitted, Submitted}` and the parent also in
   `Filled`. Real TWS transits `PendingSubmit` briefly; a strict
   single-read verify may false-negative into recovery. Proposed for
   Milestone F: a bounded re-poll (N reads, M ms apart) BEFORE declaring
   unprotected, all reads journaled. Disposition requested.
4. **Flatten under kill.** Recovery and flatten run while halt is
   `hold`/`kill` (risk-reducing), and never clear the halt. Confirm this
   matches your intended owner-command semantics.
5. **The five proposals from my
   [resumption report](/home/harveybc/Documents/GitHub/agent-multi/docs/handoffs/SATOSHI_III_TECHNICAL_LEAD_RESUMPTION_2026_08_03.md)
   §5** (GA diversity telemetry in `multifront_status.py`; spread/latency
   cost priors from read-only preflights into the existing OLAP;
   event-sourced effects journal — now implemented as part of B, review
   it there; Hypothesis property-based bracket invariants; anchor-history
   ledger). The owner directs that your dispositions govern all five and
   that none may create a parallel source of truth. I have implemented
   none except the journal (which findings 063/064 required anyway).
6. **Reproducer refresh.** Your `IBKR_L1_ADAPTER_REPRO_2026_08_03.py`
   targets removed symbols. If useful I can draft a v2 reproducer against
   the new surface for your independent run — or you may prefer to write
   it yourself for independence. Your call.
7. **Decision-age policy (new in v1.1.0).** I changed quote-problem
   handling from durable rejection to deferral, and added a terminal
   `decision_stale` rejection anchored to the decision's own quote
   evidence (`quote_time`, default ceiling 300 s). Two sub-questions:
   is 300 s the right ceiling for a 4h-bar strategy (it is deliberately
   conservative), and do you accept `quote_time` as the age anchor
   (deterministic, fixture-stable) rather than wall-clock `decided_at`?

## 8b. Codebase Memory MCP Adoption Report (activation order 2026-08-03)

Adopted under your operating specification; graph used as discovery only.
Index verification: all ten canonical projects answer `list_projects`;
`lts` is indexed at head `5f10a84` (fresh for the traced path; Milestone E
landed after — I will not reindex until the next task depends on the
changed call graph, per §7 of the spec). `agent-multi`'s index is one
docs-only commit behind; docs are excluded from the graph, so no reindex.

Demonstration trace (per §6 high-risk claim rule):

- **Project / head:** `lts` @ `5f10a84` (graph base_sha identical).
- **Traced path:** `L1OutboxConsumer.consume_entries →
  _consume_entry → {halt gate → CapabilityGate.load →
  BracketExecutor.submit_bracket → [one atomic unit: create_effect +
  consume_capability] → IbkrClientProtocol.place_order ×3 →
  BracketLifecycleController.acknowledge → verify_bracket_exact →
  (recover) → DemoExecutionService.apply_execution_event}`.
- **Direct source confirmation:** `get_code_snippet` for
  `lts.app.ibkr_l1_executor.BracketExecutor.submit_bracket` (lines
  73-177) matches current source byte-for-byte; the capability burn is
  inside the same `atomic_unit` as effect creation. Status: observed and
  reproduced (focused tests above).
- **Test evidence:** the §3/§4 fixtures cover every hop of the traced
  path; full suite 467.
- **Graph blind spots and artifacts found (report per spec §4/§6):**
  1. *Protocol dispatch is invisible*: callee edges terminate at
     `IbkrClientProtocol.*`; the runtime implementation (FakeIbkrClient
     today, the TWS client in Milestone F) never appears as a callee.
     Broker-effect claims must always come from journal/test evidence.
  2. *False callee edges from name-based resolution*: the trace lists
     `social_trading_lab.SocialPlatformRegistry.get`,
     `SocialTradingLabError` and `oanda_practice_lab.PracticeOlap.get_state`
     as callees of `_consume_entry`; `ibkr_l1_outbox.py` imports neither
     module (its `.get`/`get_state` bind to `dict.get` and
     `DemoExecutionOlap.get_state`). Confirmed by direct import
     inspection.
  3. *Mis-attributed method resolution*: `self.gate.load(...)` is
     resolved by the graph to `L1Profile.load`; the actual target is
     `CapabilityGate.load`. Confirmed in source.
  4. `tools/` is excluded from the index: the mint CLI — a
     security-relevant authority boundary — is graph-invisible, as are
     systemd units and configs. Direct file reads remain mandatory there.
- **Tool question for you:** given artifacts 2-3 (same-name method
  over-resolution), should the operating spec add a rule that any
  graph-derived *caller/callee claim* crossing module boundaries must
  cite the import statement or a direct source line before it may appear
  in a finding? I already follow that discipline; codifying it would
  protect future sessions.

## 9. Owner Decisions Required

None immediately. The first real capability mint, the Milestone F
connection window and the canary activation remain future explicit owner
actions after your verification, per the unchanged sequence in cold start
§10.

---

*Ritsurei.* General: the wounds you enumerated are sutured and the sutures
are tested, but a surgeon does not certify his own operation. The field is
yours.

— Satoshi III (Mujuro Utsutsu), successor technical lead in bootstrap
