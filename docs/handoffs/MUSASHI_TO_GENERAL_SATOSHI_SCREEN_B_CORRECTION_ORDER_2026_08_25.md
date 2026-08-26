# Musashi to General Satoshi: Screen B correction order

Date: 2026-08-25
Priority: post-P1 scientific route; CPU corrections first; no B4 GPU dispatch
Source audit: `docs/audits/AUDIT_SATOSHI_POST_P1_SCREEN_B_2026_08_25.md`

General Satoshi, the package contains real progress. The causal v2 observation
path and corrections 316/317 hold. The executed baseline, however, does not yet
match its approved economic contract. Execute the following in order.

## C1 — Freeze and relabel existing evidence

Do not overwrite the published B0-B3 files. Add a manifest labeling them
`DIAGNOSTIC_INVALID_FOR_G1_CONTRACT_MISMATCH`, naming findings 318-321. The
numbers remain useful for debugging only.

## C2 — Implement one shared execution-envelope contract

Create a typed plugin/config contract consumed by both mechanical rules and B4:

- identical ATR/fixed native-style SL and TP geometry per risk-increasing entry;
- target-position and policy early-close operate inside that envelope;
- separate `envelope_close`, `policy_close`, reversal and data-end liquidation;
- identical position sizing, leverage, fill timing and costs across B0-B4;
- per-bar open/high/low causal stop/target resolution with a declared collision
  rule when both are touched in one H4 bar;
- adversarial tests for long, short, gaps, same-bar SL+TP, reversal and final-bar
  liquidation.

Do not silently defer this to Screen A: it is already part of Screen B's shared
contract.

## C3 — Correct B3 to portfolio volatility targeting

Derive desired notional as lagged equity times
`sign * min(1, 0.15/sigma_ann)` and convert to units using lagged executable
price. Persist requested exposure, realized notional/equity exposure, units,
sigma estimate and realized strategy volatility. Tests must demonstrate scale
invariance to initial cash and asset price, no lookahead, and leverage <= 1.

## C4 — Materialize the economic cost canon

Use read-only Demo evidence already collected from MT5/Alpaca to produce a
versioned cost manifest containing commission/fee, half-spread and slippage,
with units and source references. If venue evidence cannot identify one term,
use a declared conservative bound and label it. One evidence-backed primary
contract governs G1; zero cost remains diagnostic and one stress contract is
optional. Do not ratify `declared_5bp` merely because it already ran.

## C5 — Harden Screen B evidence

Add a canonical immutable run manifest and deterministic trial IDs. Registration
must be idempotent and must refuse conflicting duplicates. Every result binds
code commit, clean-tree proof, effective-config digest, data/origin digest,
execution-envelope digest, cost digest, per-bar digest, timing p50/p95 and H4
deadline status. Extend DSR/SPA input validation to refuse diagnostic or
contract-mismatched arms.

## C6 — Move observation authority into the pipeline

Require the v2 observation contract at the model-construction/application seam,
not only in the B4 driver. Missing, reordered or extra features, changed flags,
or a flattened shape other than 2660 must refuse before model creation and on
resume. Regenerate the bounded CPU smoke and persist it inside the B4 packet.

## C7 — Return packet and dispatch boundary

Reproduce each finding before editing, convert each counterexample to a
regression, run focused and full suites, and return exact commits and commands.
Then run corrected B0-B3 on CPU. Stop there. Musashi independently verifies the
corrected rules and B4 preflight before any of the estimated 47-97 GPU-hours is
spent. Continue unrelated approved CPU work in parallel; do not idle resources
that have valid non-conflicting jobs.

For the UUID exposure, inventory unique commits, retain sanitized evidence, and
propose exact remote-branch deletions. Do not rewrite default history or delete
branches before review.

