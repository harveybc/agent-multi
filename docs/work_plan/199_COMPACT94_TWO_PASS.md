# Another ordered two-pass selection on the compact94 actor

Owner authorization:2026-11-04, immediately after exact compaction198.
This is another deletion sweep followed by one reinsertion sweep, the owner's
explicit operational definition; not repeated best-neighbor floating search.

## Frozen contract

- Start with94 inputs,30 learned groups and the preserved deterministic response
  in the original31-slot fusion. Never substitute the fresh94 trained policy.
- Use feature_selector `two_pass_greedy` v0.2.4, in existing column/block order.
  Sine/cosine and probability encoding blocks remain indivisible.
- Mask candidate columns to zero after fixed TRAIN scaling, no fitting.
  Retain the current removal if VALIDATION net return/week improves OR ties
  exactly; tolerance0. Reinsert a removed block only on strict improvement.
  Reinsertion considers only blocks removed in this round, not earlier rounds.
- Every comparison uses the current accepted subset. Both TRAIN and VALIDATION
  must maintain >2 orders/week. TRAIN remains fully evaluated and reported but
  cannot compensate a VALIDATION regression.
- Full TRAIN2020-2023 and VALIDATION2024, original eligible rows and timestamps,
 13 observations at stride2, same costs/calendar/risk/action semantics. TEST
  closed. No optimizer, replay updates, normalization change or state feedback.

## Tests and acceptance

1. Authenticate prior COMPLETE and every sealed artifact, bundle and source
   recipe; reject altered/unqualified evidence or wrong source checkpoint.
2. Load compact weights; exact saved probe and full native baseline replay
   before any candidate. Expected VALIDATION+10.4470635%,298 trades.
3. Exercise the actual selector with deterministic controls: nonworsening
   deletion, strict reinsertion, one round only, atomic blocks, minimum activity.
4. Authenticate cached evaluations by campaign/source/plugin/dataset/subset
   identity; rerun controls on resume and verify final source tensors unchanged.
5. Export final compact actor directly from the authenticated original108
   actor with the final subset. This preserves all earlier and newly emptied
   branch responses without treating the94-input graph as a fresh30-slot graph.
6. Compare the final candidate with the exported actor: predeclared1e-5
   absolute/relative logit tolerance, identical categorical actions and full
   native trades/equity; exact full reload. Never loosen gates after results.

## Automation and reporting

One bounded sequential primary-GPU worker with per-subset cache and exclusive
lock; progress and result receipts, automatic export and full account replay.
Partial cache survives restart. No intermediate decision or new training.
Record94 -> after deletion -> after reinsertion; selected/removed/reinserted
names; learned groups and constant slots; TRAIN/VALIDATION return, Sharpe,
drawdown, orders, wins, BUY/SELL and early-close diagnostics. Archive and publish
the result, not an independent-generalization claim: this VALIDATION has been
used repeatedly for development selection.

## Execution

Runtime `8f923c05`,30 focused tests pass. Authenticated baseline/probe and native
accounts replay exactly.94 columns form89 atomic blocks; maximum179 requests
including the baseline. First measured median1.09 seconds per uncached candidate.
Primary-GPU sequential worker has6GiB RAM cap, no swap, bounded automatic retry
and persisted per-subset results. Identity:
`950512db554647ec71b728812805f43275250711ddef5913f5c7717679608f76`.

Completed2026-11-04T21:10:22Z:94 ->93 ->93,90 unique evaluations,
`RAW_ELAPSED_HOURS` removed, no reinsertion and no exact ties.
Full final compact/native/reload parity and completed resume pass.
See [results200](200_COMPACT93_TWO_PASS_RESULTS.md).
