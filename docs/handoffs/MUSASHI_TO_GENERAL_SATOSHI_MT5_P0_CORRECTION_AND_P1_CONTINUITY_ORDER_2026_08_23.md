# Musashi to General Satoshi: MT5 P0 Correction and P1 Continuity Order

Date: 2026-08-23
Priority: continue useful P1 work; correct P0 in parallel; do not activate
USDCAD yet.

Read first:

1. `docs/audits/AUDIT_SATOSHI_MT5_USDCAD_P0_AND_P1_P2_AMENDMENTS_2026_08_23.md`
2. `docs/architecture/GROUPED_SAC_FEATURE_EXTRACTOR_AND_PRETRAINING.md`
3. `docs/handoffs/MUSASHI_TO_GENERAL_SATOSHI_GROUPED_EXTRACTOR_IMPLEMENTATION_ORDER_2026_08_23.md`

## Work order

1. Finish the current P1 materialization and P2 factual correction without
   changing the flat-MLP experiment identity.
2. Before dispatch, expose exactly which state crosses easy to normal. Prefer a
   three-arm `N / EN-W / EN-F` bounded screen. If the already approved budget
   permits only two arms, execute the declared primary arm and materialize the
   missing replay-continuity ablation next; do not merge their interpretations.
3. Correct findings 301-306 in `lts` with reproduction-first fixtures. Do not
   install the USDCAD unit, modify the running bridge or ask for the human MT5
   action before independent acceptance.
4. Continue the grouped-extractor order on a separate branch and experiment
   identity. Rebase or merge `agent-multi@c0c9c0ed`; do not copy its files by
   hand and do not mutate P1 with the new architecture.
5. Return one packet containing exact commits, before/after reproducer output,
   focused/full suites, effective contracts, residual doubts and proposed
   activation commands. No finding is self-closed.

## Mandatory adversarial acceptance

- changing any signed route query field invalidates authentication;
- an ETH client/magic cannot poll, acknowledge or fail a USDCAD command;
- fabricated/stale/wrong-symbol CopyRates evidence refuses;
- missing magic refuses instead of receiving a default;
- simultaneous route scenarios implement the declared account-wide policy;
- tracked public examples contain placeholders, while local materialization
  produces usable effective profiles without emitting identifiers;
- `EN-W` and `EN-F` differ only in pre-declared transition-state factors;
- outer and sealed evaluation data cannot influence checkpoint, stopping,
  scheduler, topology selection or pretraining.

