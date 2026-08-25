# Audit: Final SOTA Contract Return

Date: 2026-08-25
Audited series: `0176a7c0..bd08e31e`
Auditor: General Musashi
Focused reproduction: 38 tests passed; registry lint passed
Runtime mutation: none

## Verdict

`ACCEPT_WITH_TWO_BOUNDED_CORRECTIONS`.

F1-F4 are substantively corrected. The owner should ratify the prospective
observation contract proposed below. Two implementation inconsistencies must be
fixed before any post-P1 materialization. P1 remains untouched and finishes
under its honestly labeled executed identity.

## Owner Decision Recommendation: F1.5

Approve a new prospective contract identity:

- exact ordered 83-feature list from the declared system contract;
- exclude `typical_price`;
- `include_price_window = false`;
- `include_agent_state = true` with four explicitly named state fields;
- `window_size = 32`;
- flattened dimension `32*83+4 = 2660`.

Rationale: `typical_price` is a raw nonstationary level redundant with the raw
price family already excluded; the raw price window previously contributed to
actor collapse; neither should silently enter future screens. This is a new
identity, not a correction of historical P1 artifacts.

## Remaining Findings

### S2 — SOTA-F01: observation identity now has two digest algorithms

`tools/post_p1_screen_contract.feature_list_digest()` hashes feature names
joined by newlines. The canonical project helper
`pipeline_plugins._observation_contract.feature_columns_sha256()` hashes a
compact JSON array. They produce different hashes for the same ordered list.
The reported executed digest `dd9e05d8...` is the newline variant, while
existing observation contracts use the compact-JSON convention.

Correction: one canonical implementation and schema-qualified digest field.
Reuse the existing project helper or move it to a shared dependency; do not
duplicate it. Preserve `dd9e05d8...` only as a labeled legacy diagnostic hash,
and add the canonical digest to P1 sealing evidence.

Also derive flattened shape from the actual observation space or a fully
materialized contract. `executed_observation_identity()` currently adds four
state dimensions unconditionally even when the effective config does not carry
an explicit `include_agent_state` flag. Future contracts must bind the four
state-field names, order and digest, not infer them from arithmetic alone.

### S3 — SOTA-F02: equal-update arithmetic is short by 120 steps

Doc 40 states every adaptive arm receives exactly 260,000 updates, but:

- `712 * 365 = 259,880`;
- `356 * 730 = 259,880`.

Correction: materialize a deterministic quotient/remainder schedule whose sum
is exactly 260,000 (for example, add one update to the first 120 refreshes), and
test exact conservation for every cadence and scored-period cardinality.

### S3 — SOTA-F03: finalist digest syntax remains weak

The release guard accepts any digest string of seven characters. Require 64
lowercase hexadecimal characters for SHA-256 fields and a full 40-hex Git
commit. Bind the ensemble-rule schema/version as well as its digest.

## Accepted Corrections

- Exact 84-vs-83 reproduction and root-cause trace.
- Historical P1 preserved as executed 84-feature diagnostic evidence.
- Prospective mismatch refusal before model construction.
- Equal-total-compute causal cadence separated from operational
  cadence-plus-compute.
- Parsed temporal boundaries and fit/selection enforcement.
- Typed report-only allowlist and sole-finalist requirement.
- Honest heuristic-lint labeling and numeric-table coverage.
- Politis-White block selection bound to control returns only.

## Dispatch Status

No post-P1 screen is authorized by this verdict. After F01-F03 corrections and
P1 terminal aggregation, the rule-only baseline implementation may proceed
first under a separate acceptance check.

