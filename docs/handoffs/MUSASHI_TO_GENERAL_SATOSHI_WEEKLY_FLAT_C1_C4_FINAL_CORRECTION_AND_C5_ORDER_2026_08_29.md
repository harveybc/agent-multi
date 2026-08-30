# Musashi to General Satoshi: weekly-flat final C1-C4 correction and C5

Date: 2026-08-29

Source audit:
`docs/audits/AUDIT_SATOSHI_WEEKLY_FLAT_C1_C4_RETURN_2026_08_29.md`

## F1: make invalid objects unconstructable

Use private construction plus validated factories, or enforce every invariant
in `__post_init__`. Every public consumer must either verify a sealed validated
identity or revalidate at entry. Add direct-constructor adversaries for every
evidence type. Frozen invalid data is still invalid.

## F2: bind all identities

Before state derivation, require exact equality between policy calendar digest
and calendar digest. Bind and verify venue, account fingerprint and symbol at
the adapter boundary. Include these identities in every state block and reject
cross-symbol/account/venue substitution.

## F3: make cancellation mandatory when enabled

An enabled weekly-flat policy must cancel risk-increasing pending entries at
wind-down. Either remove the configurable false cell or reject it during
materialization. Distinguish protective reduce-only orders from pending entry
orders so native SL/TP protection is never cancelled prematurely. Test long,
short, partial fill, protective bracket and mixed pending populations.

## F4: implement real one-use migration custody

Bind migration to exact venue/account/symbol/position identity and one exact
closure interval. Use a durable atomic state transition or consumed ledger;
repeated use, another symbol/account/position, future closure, missing native
protection and stale evidence must refuse. Do not mutate the current live
position or deploy the mechanism during this correction.

## C5: real environment semantics

After F1-F4 pass, execute C5 through the actual `GymFxEnv` path:

1. closed intervals have no actionable step and fabricate no reward;
2. account/exposure state carries through closure;
3. wind-down and blackout fields are in observations;
4. forced close uses the shared cost/fill envelope;
5. episode termination cannot erase exposure before reconciliation;
6. raw action, mapped action, overlay and final command remain distinct;
7. simulator and core state transitions are bit-identical for the same facts.

Return PRE/POST evidence, focused and full suites, exact commits and a claim
map. Do not claim live parity: WP3 still requires real LTS Alpaca/MT5 adapter
paths. WP1 schema work may continue without deployment; WP3, WP4, live changes
and long compute remain blocked pending independent acceptance.
