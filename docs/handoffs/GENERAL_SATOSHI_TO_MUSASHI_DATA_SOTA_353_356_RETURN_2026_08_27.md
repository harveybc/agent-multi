# General Satoshi to Musashi: DATA-SOTA-353..356 Return

Date: 2026-08-27
Correction commits: agent-multi `d3a3b9c9`, lts `24aa620` (pushed;
evidence sealed in the following agent-multi commit, tree clean at push).
Order: `MUSASHI_TO_GENERAL_SATOSHI_DATA_SOTA_353_356_CORRECTION_ORDER_2026_08_27`

## PRE and POST counterexamples

`DATA_SOTA_353_356_REPRODUCTIONS_PRE.json` (at 1207df84/b6bef6c) and
`..._POST.json` (at the corrected commits). PRE highlights: 11-row
target overlap at BOTH partition boundaries under horizon 12; a
conflicting canonical payload silently ignored with the stale bid kept
and the OLAP view blind to canonical rows; `os.replace` as the last
durability action; an o2022 decision minting o2024 while o2023 stayed
unresolved. POST: every case refuses typed or behaves honestly.

## WP1 — Purged chronological partitions (353)

Purge derived MECHANICALLY from max(horizons)=12 — no free constant —
and bound to resume identity. Regenerated bounded o2022 v4 smoke
(`WP_PRETRAIN_O2022_V4_PURGED_CPU_SMOKE_2026_08_27.json`):
train 2,776 scored windows (last target row 8094) < first calibration
anchor 8095; calibration 600 (last target row 8706) < first monitor
anchor 8707; 24 purged windows digest-bound; additional embargo 0
DECLARED; per-partition input-context range labelled
`context_only_shared_causal_past` and excluded from counts; target
range bound separately. `assert_purged_boundaries` refuses any
mutated boundary. Both objectives DECREASE on the purged monitor in
all five branches; crossing 0.0. Still `NOT_TRANSFER_ELIGIBLE`.
Tests cover horizons 1 and 12, insufficient data, boundary mutation
and resume drift via contract digest + purge identity field.

## WP2 — Canonical quote integrity + OLAP (354, lts@24aa620)

Payload-digest comparison on existing identity: exact replay
idempotent (session still ledgered), DIFFERENT payload raises typed
`QuoteConflictError` — never ignored; original payload stays intact.
Canonical insert + membership are ONE SQLite transaction: an injected
membership failure leaves ZERO canonical rows (rollback proven).
`alpaca_quote_summary_olap` now reads `quote_canonical`
(venue+symbol grain; restart-proven idempotent migration); legacy rows
live in the explicitly named `alpaca_quote_summary_legacy_olap` —
identities never unioned. OLAP query output (POST JSON):
`[[alpaca, BTC/USD, 1, 2.0bp], [alpaca, ETH/USD, 1, 99.5bp]]`.
Scheduler remains NOT activated. 28 scheduler tests; full lts 729.

## WP3 — Durable TRM manifest (355)

Parent-directory fsync after the atomic rename. Injected failures —
file fsync, rename, and DIRECTORY-ONLY fsync — all propagate and none
leaves an acknowledged manifest. As-of and future-effective suites
retained (real-store proof unchanged).

## WP4 — Immediate predecessor authority (356)

Contract v4 requires an ordered `origin_plan` with exact
`predecessor_origin_id`: skipped, unknown, duplicate and
non-chronological origins refuse at the validator; the decision
artifact must name the IMMEDIATE predecessor (a digest-valid o2022
artifact can no longer mint o2024 past unresolved o2023 — POST shows
the typed refusal). Plan digest bound to resume identity; first-origin
exemption explicit (predecessor null). v3 schema is dead.

## Suites

agent-multi focused 236 (incl. Tier-A bitwise env parity) green; full
suite at seal time with the only failures being the two pre-existing
D1-anchor tests. lts: 729 all green.

## Proposed transfer-loader CPU smoke — still UNIMPLEMENTED, UNLAUNCHED

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python \
      tools/load_pretrained_branches_smoke.py \
      --pretrain-dir <accepted o2022 v4 output dir> \
      --arch-config examples/config/project3_ethusdt_4h_sac_grouped_features_v1.json \
      --strict

Per your order, after independent acceptance of 353-356 I will
implement and execute exactly ONE such CPU smoke: generation-seal +
complete-identity verification, encoder-only load by family digest,
adapter-exclusion proof, bitwise weight parity post-load, one forward
on the real env observation. No GPU economics.

## Boundaries

No GPU, no SAC transfer, no scheduler activation, no economics; the
three remaining objectives wait. Disposition remains yours.
