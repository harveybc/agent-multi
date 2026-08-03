# Independent Verification: Multi-Venue Continuity (Findings 075-078)

Date: 2026-08-03 America/Bogota
Verifier: Satoshi III (Mujuro Utsutsu), acting as independent read-mostly
verifier under `MUSASHI_TO_SATOSHI_III_MULTI_VENUE_CONTINUITY_AUDIT_2026_08_03.md`
Implementer of the audited changes: General Musashi (role-swap inversion for
this packet; two-party separation preserved)
Order authority exercised: none — no broker order was placed, cancelled or
altered; no TWS/Alpaca/MT5 session was opened by this verification
Scope verified: `lts@f5016dd..8b67235`; `prediction_provider@78f0af5` (head
confirmed; content review shallow, see §6)
This report closes nothing. Dispositions are recommended to the owner.

## 1. Finding 075 — reproduced on parent, correction verified

**Reproduction:** in an isolated git worktree at the parent commit
`c98a008`, the three reconnect fixtures from
[test_ibkr_l1_tws.py](/home/harveybc/Documents/GitHub/lts/tests/unit/test_ibkr_l1_tws.py)
FAIL (3/3) — the permanent-id reconstruction does not exist there. At
`cffdc13`/HEAD the same fixtures pass (3/3). Worktree removed after use.

**Correction verified in source**
([ibkr_l1_tws.py:121-168](/home/harveybc/Documents/GitHub/lts/app/ibkr_l1_tws.py#L121-L168)):
`_completed_execution_facts` joins completed orders to executions strictly
by `permId` (both sides must carry a positive permanent id; ambiguous
multi-perm matches are dropped); the reconstructed `filled` derives ONLY
from direct execution `cumQty`; a completed order without a matching
execution produces NO fact — missing evidence stays missing, and a
synthetic filled parent cannot be fabricated from status or requested
quantity.

**Observation O-1 (safe direction, review recommended):** the fact's
`totalQuantity` is overridden with `max(cumQty, filledQuantity)`; for a
partially-filled completed order this understates the REQUESTED quantity,
which drives exact verification into recovery. Fail-closed, therefore
safe — but the field name no longer means "requested" on that path.
Recommended disposition: Musashi confirms intent or renames the field.

## 2. Focused Fixtures and Complete Suite

```text
focused (tws, model_runner, model_authority, live_model_selection,
         f0, demo_execution_service)         -> 95 passed
complete LTS suite at 8b67235               -> 530 passed
```

Matches the implementation claim. Findings 076-078 reproduce through their
fixtures: `test_quote_unavailable_is_degraded_without_restart_or_order`
(076 — [ibkr_model_runner.py:222](/home/harveybc/Documents/GitHub/lts/app/ibkr_model_runner.py#L222)
emits `waiting_for_quote` inside the tick, zero submissions, no exception,
therefore no systemd restart churn); the F0-file canonicalization fixtures
(077 — terminal cumulative within 1e-9 canonicalized, negative cumulative
refused); and the venue-labeled serializer (078 —
[demo_execution_service.py:156](/home/harveybc/Documents/GitHub/lts/app/demo_execution_service.py#L156)
emits `f"{intent.venue}.protected_order.v1"` for FUTURE payloads only;
historical append-only rows are not rewritten).

## 3. Runner Heartbeats and Stop Paths — verified

- Shared helper
  [model_runner_heartbeat.py](/home/harveybc/Documents/GitHub/lts/app/model_runner_heartbeat.py)
  writes tmp+`replace` (atomic), covered by
  `test_runner_heartbeat_creates_parent_and_replaces_atomically`.
- All three runners (Alpaca/IBKR/MT5) wait on `threading.Event.wait(
  loop_seconds)` with SIGTERM/SIGINT setting the event — stop returns
  immediately, never waiting out a polling interval
  ([ibkr_model_runner.py:296-321](/home/harveybc/Documents/GitHub/lts/app/ibkr_model_runner.py#L296-L321)).
- Defects write a `degraded_error` heartbeat before propagating.
- Direct runtime sample at 2026-08-03T21:33Z: both local services
  `active`; IBKR heartbeat state `decided`, Alpaca `replayed_signal`,
  both minutes-fresh.

## 4. Model-Manifest Fail-Closure — verified via fixtures

`test_demo_selector_hot_reloads_only_a_fully_verified_pointer`,
`test_selection_tier_is_explicit_and_fail_closed`,
`test_old_model_must_be_drained_before_a_different_session_activates`,
`test_config_change_creates_a_distinct_model_session_with_new_balance`
all pass: invalid hashes cannot add exposure; a changed model drains prior
exposure first; the replacement session seeds from actual post-close
broker cash/equity.

## 5. Profile / Mandate Agreement — verified, one advisory

Profile
([ibkr_usdcad_model_profile_v1.json](/home/harveybc/Documents/GitHub/lts/examples/configs/ibkr_usdcad_model_profile_v1.json)):
`quantity_ceiling` 25000, pinned `contract_con_id` 15016062, USD.CAD,
paper/loopback/7497 enforced. Runner config: `route.minimum_units` 25000
(IDEALPRO minimum honored), gross and margin fractions 0.04, sizing target
risk 0.005% below the ceiling. Mandate metadata (read WITHOUT exposing the
nonce): `quantity_ceiling` 25000, `max_risk_fraction_at_stop` 6.25e-05 =
**0.00625%** — profile, route, mandate and order all agree. Arithmetic
closes: 25,000 × (0.0015 × ~1.40) ≈ $52 ≈ 0.00625% of Paper equity.

**Observation O-2:** `tools/mint_ibkr_model_paper_mandate.py` defaults
`--quantity-ceiling` to 20000 — below the 25,000 IDEALPRO minimum this
route now requires; a default-flag mint would recreate the odd-lot
warning. Recommended disposition: change the default to 25000 or force
the flag to be explicit.

**Observation O-3 (doctrine delta for owner awareness):** the continuous
mandate expires 2026-09-02 (~30 days, multiple entries), a deliberate
departure from the F0-era one-hour single-bracket capability doctrine.
Owner-authorized and enforced by the new authority module
(`test_continuous_paper_gate_derives_one_bound_capability_per_intent`,
`test_continuous_paper_gate_refuses_expiry_and_permissive_file`), but the
delta should be recorded in doctrine, not only in code.

## 6. MT5 Execution EA — source audited; runtime is an explicit unknown

Source review of
[LtsMt5ModelBridge.mq5](/home/harveybc/Documents/GitHub/lts/mt5/MQL5/Experts/LtsMt5ModelBridge.mq5)
(827 lines): `ADAPTER_VERSION = lts.mt5.ea.execution.v2`;
`InpExecutionEnabled` defaults **false**; init refuses any account whose
`ACCOUNT_TRADE_MODE` is not DEMO (line 749); SL and TP are set in the
initial `MqlTradeRequest` (lines 422-423) with minimum-distance validation
(408-409); volume and deviation are bounded by inputs; the bridge secret
is an input, not a hardcode; environment is labeled `demo`.

**Explicit unknowns (evidence gate NOT satisfiable yet):** the old
read-only EA remains attached on Dragon; bridge version handshake, ≥51
closed H4 bars, command acknowledgement, direct native SL+TP position
evidence and restart idempotency CANNOT be verified until the owner
installs the execution EA. Dragon-side service state was not re-sampled by
this session. The prior audit's five-point evidence gate stands unmet.

## 7. Authority Boundaries — verified within stated limits

No Hermes/LLM reference exists in any decision-path module
(`ibkr_model_runner`, `alpaca_model_runner`, `mt5_model_runner`,
`ibkr_model_authority`). Execution derives solely from hash-bound
selected-model manifests through the fail-closed selector; profiles
hard-enforce paper/demo venues, loopback and fingerprints; the mandate
gate refuses expiry and permissive file modes. No Live account path was
found in the reviewed modules; I did not exhaustively audit every
transitive import, and state that plainly.

## 8. Summary of Recommended Dispositions (I close nothing)

- 075, 076, 077, 078: **verified as corrected by independent
  reproduction**; recommend owner closure on Musashi's concurrence.
- O-1 (totalQuantity override), O-2 (mint default 20000), O-3 (mandate
  durability doctrine delta): recommend Musashi triage; none blocks
  continuous Paper operation in my judgment.
- MT5 runtime acceptance: blocked on the owner's EA installation; the
  five-point evidence gate from the implementation audit remains the
  binding checklist.

*Ritsurei.* The General's steel held under my glass: the defect he cut out
stays out, and his corrections survive his own counterexamples re-run by
another hand. What no one can yet see — the MT5 runtime — is named, not
assumed.

— Satoshi III (Mujuro Utsutsu), independent verifier for this packet
