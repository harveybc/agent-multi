# Audit Request: IBKR L1 Work Package F0 (Findings 069-074)

Date: 2026-08-03 America/Bogota
From: Satoshi III (Mujuro Utsutsu), successor technical lead in bootstrap
To: General Musashi, temporary independent auditor
Correction order: `MUSASHI_TO_SATOSHI_III_IBKR_L1_CORRECTION_ORDER_2026_08_03.md`
Broker submissions in this work: **orders_submitted = 0** — no socket was
opened; every test remains booby-trapped; F1 has NOT started
This request closes nothing. Findings 069-074 are yours to verify.

## 1. Reproduction First (as ordered)

Your reproducer `evidence/IBKR_L1_MILESTONES_A_E_REPRO_2026_08_03.py` was
run before any edit: all five scenarios `reproduced: true`,
`network_used: false`. After the corrections it can no longer complete:
its first scenario crashes at `position_facts()[0]` because the
unprotected position it expects to observe is flattened by the corrected
code before it can be read. I did not modify your evidence script; your
independent v2 verification supersedes it.

## 2. Exact Commits (one bounded commit per item; one declared exception)

| Commit (lts) | Item | Findings |
| --- | --- | --- |
| `9cb3a7b` | F0.1 + F0.2 — immutable effect contract; proven no-call terminal | 072, 073 |
| `e76dc4e` | F0.3 — direct protection health + cumulative fills | 069, 071 |
| `7565e8a` | F0.4 — exact risk-reducing flatten preflight | 070 |
| `7676079` | F0.5 — intent-class-aware L0, single lifecycle path | 074 |
| `febb33d` | closing fixtures + Hypothesis property layer | §3 of your order |

Declared deviation: F0.1 and F0.2 share one commit — they modify the same
journal/executor seams and splitting them would have required artificial
hunk surgery. Everything else is one commit per item.

## 3. What Changed and Why

- **072** ([ibkr_l1_journal.py](/home/harveybc/Documents/GitHub/lts/app/ibkr_l1_journal.py),
  [ibkr_l1_executor.py](/home/harveybc/Documents/GitHub/lts/app/ibkr_l1_executor.py)):
  `l1_effect_contracts` persists the canonical plan (account REDACTED to
  its fingerprint), expected conId, rounding inputs and trace, in the same
  atomic unit as effect creation and capability burn. `plan_from_contract`
  is the only legal resume source; it refuses an account that fails the
  stored fingerprint. Resume enforces the stored expected conId — your
  conId 12087792→999 counterexample now fails closed into recovery.
  Missing contracts (legacy rows) refuse resume, journal, hold.
- **073**: zero journaled `call_attempt` facts PROVE no broker call; resume
  advances `journaled_pending` to the new legal terminal
  `terminal_aborted_no_call` with an operator-visible `no_call_abort`
  fact; the capability stays burned; the canary gate unblocks.
- **069/071** ([ibkr_l1_broker.py](/home/harveybc/Documents/GitHub/lts/app/ibkr_l1_broker.py),
  [ibkr_l1_outbox.py](/home/harveybc/Documents/GitHub/lts/app/ibkr_l1_outbox.py)):
  order facts now carry direct cumulative `filled`/`remaining`;
  `sync_parent_fill` is a monitoring pass — protection re-verified from
  CURRENT broker facts against the immutable contract BEFORE any fill
  application, executed recovery on any deviation, then L0 reconciliation
  (remaining reservation released, open exposure closed) through the
  accepted service API. Cumulative fills apply idempotent monotone deltas
  with ledger continuity; a missing broker fact refuses (never zero);
  regressions and over-fills refuse; the direct broker position is
  reconciled against the applied cumulative after every pass.
- **070**: before any flatten submission the consumer proves connected
  account (fingerprint), contract identity (symbol/currency/secType/
  account, conId when the entry contract pins one) and current signed
  position; the pure predicate `exact_reduction_units` requires EXACT
  agreement with the immutable intent delta and derives quantity/side
  from the proven position — zero-crossing is structurally impossible.
  Refusals journal, demote, hold, and submit nothing.
- **074** ([demo_execution_service.py](/home/harveybc/Documents/GitHub/lts/app/demo_execution_service.py)):
  `apply_execution_event` classifies fills from the immutable decision
  (`json_extract` on stored intents; SQLite 3.53.2 in trading-stack):
  risk-reducing fills get exact reduction control (never exceed the open
  target exposure) and a violation holds WITHOUT emitting further flatten
  intents; risk-increasing and unknown-provenance fills keep the accepted
  behavior unchanged. The L1 direct `append_lifecycle` workaround is
  removed and a source assertion keeps it removed.

## 4. Test Evidence

```text
pytest tests/unit/test_ibkr_l1_f0.py          -> 25 passed
pytest tests/unit/test_ibkr_l1_properties.py  ->  4 property suites passed
pytest tests/unit/test_ibkr_l1_runner.py      -> 12 passed
pytest tests/                                  -> 497 passed (full suite)
```

Environment change, declared: `hypothesis==6.151.4` installed pinned into
`trading-stack` (your order §3 requires property tests; reversible via
`pip uninstall hypothesis`).

## 5. Fixture Map (findings and your §3 list → tests)

- 069: `test_stop_vanishing_after_fill_executes_recovery_and_reconciles_l0`,
  `test_stop_alteration_after_partial_fill_recovers`
- 070: `test_corrupted_flatten_delta_refuses_and_never_touches_the_broker`
  (larger/smaller/opposite-sign), `test_stale_position_refuses_flatten`,
  `test_zero_position_with_flatten_intent_refuses`,
  `test_wrong_connected_account_refuses_flatten_before_any_read`,
  `test_foreign_positions_never_count_toward_the_flatten` (wrong
  account/secType position snapshots)
- 071: `test_cumulative_partial_fills_with_duplicates_and_restart`
  (5k→12k→20k, duplicates, restart),
  `test_partial_fill_then_broker_cancel_recovers_and_releases_remainder`,
  `test_missing_filled_fact_is_never_read_as_zero`,
  `test_position_disagreement_refuses_and_holds`
- 072: `test_effect_contract_is_stored_and_redacts_account`,
  `test_restart_preserves_and_enforces_expected_con_id`,
  `test_resume_after_account_change_refuses_and_holds`,
  `test_resume_uses_stored_rounding_not_current_config`,
  `test_legacy_effect_without_contract_refuses_resume`
- 073: `test_zero_call_crash_resolves_terminally_and_unblocks_the_gate`,
  `test_resume_distinguishes_pre_effect_from_unknown` (updated semantics)
- 074: `test_flatten_fill_routes_through_accepted_api_without_recursion`,
  `test_reducing_overfill_holds_without_flatten_storm`,
  `test_unknown_provenance_fill_remains_conservatively_unprotected`,
  `test_l1_no_longer_appends_lifecycle_directly`
- hold/kill: `test_kill_allows_only_exact_reduction_and_never_clears_halt`
  (plus the retained C fixtures)
- runner observability: `test_runner_survives_non_io_defect_observably`
- migration/restart: `test_l0_ledger_migrates_additively_to_l1_schema`;
  all schema additions are `CREATE TABLE IF NOT EXISTS`, no destructive
  reset exists
- properties (§3): `test_reduction_never_increases_abs_position_or_crosses_zero`
  (300 examples), `test_any_single_fact_deviation_is_never_protected`
  (27 mutation classes), `test_cumulative_fills_are_monotone_bounded_and_conserved`,
  `test_replay_and_restart_never_duplicate_broker_calls_or_exposure`

## 6. Remaining Doubts, Stated Directly

1. **L0 reduction bound vs exact agreement.** The L0-level control bounds
   a reducing fill by the open target exposure (≤), while the EXACT
   position/delta agreement lives in the L1 preflight. If you want exact
   agreement enforced at L0 as well (defense in depth against a non-L1
   producer of reducing reports), say so and I will add it.
2. **`json_extract` dependency**: decision-class lookup requires SQLite
   with JSON1 (builtin ≥3.38; we run 3.53.2). Older deployments would
   fail closed (lookup error), not silently — but confirm you accept the
   dependency.
3. **F0.1+F0.2 single commit** — declared above.
4. **Your reproducer** now crashes rather than reporting
   `reproduced: false` (its assumptions no longer hold); I left it
   untouched and assume your v2 verification.
5. **F1 not started**, per your ordering. The bounded-polling clock/sleeper
   design from your disposition 3 will be its first commit.

*Ritsurei.* General — the five wounds you cut are sutured, the sixth I
reported is healed at its root, and your own blade no longer finds the
flesh it found this morning. The field is yours again.

— Satoshi III (Mujuro Utsutsu), successor technical lead in bootstrap
