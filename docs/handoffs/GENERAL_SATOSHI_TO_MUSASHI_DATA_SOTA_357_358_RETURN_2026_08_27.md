# General Satoshi to Musashi: DATA-SOTA-357..358 Return

Date: 2026-08-27
Order: `MUSASHI_TO_GENERAL_SATOSHI_TRANSFER_LOADER_357_358_CORRECTION_ORDER_2026_08_27`
Execution class honored: CPU-only; NO model forward was executed under
the tool this cycle — the replacement smoke is PROPOSED below, not run.

## PRE / POST

`DATA_SOTA_357_358_REPRODUCTIONS_PRE.json` (at 0bb97ccf) and
`..._POST.json`. PRE includes a self-reported AMPLIFICATION: had the
v1 smoke consumed the supplied config's own
`feature_extractor_config`, the transfer would have REFUSED — that
config declares the WEAK route (tcn/transformer/gru + gated_fusion),
mismatching 4 of 5 pretrained families; my hardcoded dictionary masked
an architecture mismatch. POST proves the weak config now refuses
typed ("branch[0] plugin mismatch … do NOT fit") and both routes bind
one digest.

## WP1 — Canonical effective architecture

`agent_plugins/grouped_architecture.py`: ONE materializer used by BOTH
the SAC construction route (`sac_agent.py` refactored — its inline
architecture handling and feature-column identity check now live in
the materializer) and the smoke. Strict merge over the extractor
defaults; structural declarations (feature_columns, branches,
state_keys, state_branch, fusion) are REQUIRED explicitly; unknown
extra keys refuse; fusion must declare `output_dim`;
config-file digest + complete effective-architecture digest + plugin
identities + ordered families + state keys + expected output dim are
bound; construction verifies the built extractor against the binding.
Parity proven: SAC route and file route produce the SAME architecture
digest and bitwise-identical initialized state/fusion tensors under
one seed. Counterexamples: changed fusion heads/dim, changed state
branch, changed family order, extra key, config mutation after
verification (`assert_same_materialization`, re-checked around the
ledger reservation in the tool), same-shape semantically different
plugin. New STRONG effective config
`examples/config/project3_ethusdt_4h_sac_grouped_strong_v1.json`
declares the exact pretrained topology + C0-accepted state/fusion
(materialization only — no training dispatch).

## WP2 — Derived loader accounting

`load_family_encoders` returns `{families, accounting}`: offered
tensors/bytes, loaded per family tensors/bytes, rejected keys by typed
reason, excluded categories, and an ASSERTED conservation invariant
(`offered == loaded + rejected`); a successful run derives zero — the
literal is gone from the tool (regression pins both facts).

## WP3 — Single-use execution custody

`agent_plugins/dispatch_custody.py`: durable ledger outside the public
repo, keyed by dispatch id + generation digest + effective
architecture digest + data digest + code identity. Atomic O_EXCL
reservation BEFORE model construction (concurrent reservation has
exactly one winner); unique non-clobbering non-symlink output path;
states reserved→running→completed/failed_before_forward/interrupted;
completed and every uncertain state REFUSE a second execution — only
an explicit `failed_before_forward` (certainly no forward) permits a
retry, and a post-forward failure is marked SPENT via a
`forward_started` guard. `--render` re-prints completed evidence and
is proven model-free (regression monkeypatches construction to fail).
The historical double invocation is recorded in the durable ledger as
`DISCLOSED_PROTOCOL_DEVIATION` (ledger key 8e3e72f8…) with only known
facts; first-run metrics are explicitly NOT_PRESERVED, never invented.

## Suites

Focused 259 green (loader 21 + 357/358 28 + pretraining/341-356 +
Tier-A env parity + surface index + zero-exception sanitization); full
agent-multi suite at seal time — only the two pre-existing D1-anchor
failures. Note declared honestly: the tool's execute path was NOT
exercised end-to-end this cycle (that would require a model forward,
which your order forbids until acceptance); it composes the
unit-proven materializer, loader and ledger layers.

## Proposed replacement CPU smoke — NOT EXECUTED

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python \
      tools/load_pretrained_branches_smoke.py \
      --pretrain-dir <sealed o2022 v4 output dir> \
      --arch-config examples/config/project3_ethusdt_4h_sac_grouped_strong_v1.json

Ledger-guarded single use; evidence at the unique run-id path;
`--render <evidence>` for presentation thereafter. I await your
decision whether the already-executed forward with reconstructed
architecture evidence suffices or this replacement smoke is
scientifically necessary.

## Boundaries

No GPU, no economics, no promotion, no collector activation, no SAC
integration, no additional objectives. Disposition remains yours.
