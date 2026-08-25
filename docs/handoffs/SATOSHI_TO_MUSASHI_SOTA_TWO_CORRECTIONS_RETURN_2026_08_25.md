# Satoshi to Musashi: two bounded corrections return (SOTA-F01/F02/F03)

Date: 2026-08-25
Order: `MUSASHI_TO_GENERAL_SATOSHI_FINAL_SOTA_TWO_CORRECTIONS_2026_08_25`
after `AUDIT_SATOSHI_FINAL_SOTA_CONTRACT_RETURN_2026_08_25.md`
(ACCEPT_WITH_TWO_BOUNDED_CORRECTIONS). CPU/code only; P1 untouched; no
screen materialized.

## C1 — canonical observation identity (SOTA-F01)

- Duplicate newline digest REMOVED as an identity source: the screen
  contract now imports and reuses
  `pipeline_plugins._observation_contract.feature_columns_sha256`
  (compact-JSON, single implementation). Unity fixture
  (`test_one_digest_across_producer_and_consumer`) proves producer and
  consumer hash identically and that the legacy digest stays distinct.
- `dd9e05d8...` preserved ONLY as
  `executed_feature_digest_legacy_newline`, labeled; canonical executed
  digest added to sealing evidence:
  `df2d981dc83367075643d20eb22d484e0543aef512ecdfc12ddb0f321ae3c682`.
- `executed_observation_identity` no longer infers state dims by
  arithmetic: flattened shape is emitted only under explicit config
  flags (`flattened_shape_basis` records the basis, else
  "unavailable"); contracts declaring `include_agent_state` without
  bound `agent_state_fields` are REFUSED.
- **Prospective contract artifact materialized**:
  `examples/config/phase_3_eth_sac_dynamics/systems/ethusdt_4h_l1_system_v2.json`
  — status PROPOSED_AWAITING_OWNER_RATIFICATION; declared 83 ordered
  list (canonical sha `c4697681c1323245...`), typical_price excluded,
  `include_price_window=false`, four bound agent-state fields
  (position, equity_norm, unrealized_pnl_norm, holding_duration_norm —
  the executed `live_stationary_v2` field order, sha
  `b5beeb97e2031b8b...`), window 32, flattened [2660].
- Doc 38 §23.5 updated: sealing carries BOTH digests.

## C2 — exact conservation and digest syntax (SOTA-F02/F03)

- `refresh_update_schedule(total, n)`: deterministic
  quotient/remainder, first `remainder` refreshes get +1. Regression:
  exact sums for (260000, 52/365/730) and odd cardinalities;
  deterministic placement asserted (713×120+712×245; 357×120+356×610).
  Doc 40 arithmetic corrected — the 259,880 shortfall is gone.
- Release guard: SHA-256 fields must be 64 LOWERCASE hex (7-char,
  uppercase and non-hex fixtures REFUSED), `code_commit` full 40 hex
  (truncated REFUSED), and the finalist must bind
  `ensemble_rule_schema` as `<name>@<version>` beside its digest.

## Validation

30 passed in `test_post_p1_screen_contract.py` (new: digest unity,
short/uppercase digest, truncated commit, missing ensemble schema,
missing state-field binding, no-arithmetic-inference, exact
conservation, deterministic placement); full suite 47 passed; registry
lint PASS {coverage heuristic_lint, files 9, sources 31}.

## Awaiting

Owner ratification of `ethusdt_4h_l1_system_v2.json`. P1 continues to
terminal under its executed 84-feature identity; no post-P1 screen is
materialized before your acceptance and the P1 seal.
