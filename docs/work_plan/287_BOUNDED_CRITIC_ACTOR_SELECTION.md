# Bounded Critic Actor Qualification And Selection

2026-11-11. Owner approves qualification/export of286 `critic_adamw` at the
fixed eight-update endpoint, seeds19/43, primary19 by declared order. No fitting,
checkpoint search, optimizer change or live promotion. Preserve285 and286.

## Contract And Test Design

| Requirement | Gate |
|---|---|
| Correct candidate | Require native-target-geometry schema, fixed endpoint8, seeds19/43 and declared critic_adamw arm |
| Frozen representation | Compare every actor tensor; only four head tensors may differ from285 |
| Saved performance | Full native action/trade/equity/close traces reproduce286; cold logits/state/accounts exact |
| Cumulative phase2 | New identity/cache;69 inputs,28 groups; all51 historical removals in120-input admitted universe |
| Selection rule | Ordered deletion accepts eligible improvement or exact tie; return accepts strict improvement only |
| Weight preservation | Donor expansion parity and packed cold replay; independent weight/decision audit |
| Automation | Bounded selection/report queue and finite authenticated collection after qualification succeeds |

Keep fixed TRAIN scaling, window13/stride2, full TRAIN2020-2023 and
VALIDATION2024; no TEST access. Score remains validation return/week with more
than2 orders/week in both periods. Coherent encodings stay in blocks; no phase1
invalid feature is restored. Archived donor readmission is a frozen screen,
not a claim about joint retraining. Both qualified source seeds stay available.

Expected candidate TRAIN12.708707%, VAL14.693262%; reference285 TRAIN12.392145%,
VAL14.693262%. Qualification must reproduce those exact accounts before the
primary candidate is evaluated for feature changes. No repeated training to
recreate weights. If the subset stays unchanged, do not refit it by default.

## Qualified Export And Launch

Both seeds qualified in30.86seconds. Full native action/trade/equity/close traces
match286; exact cold logits, state and accounts pass. Four head tensors differ
from285 while337 other actor tensors remain exact. Both change six TRAIN signals
and zero validation signals relative to285. This is not six separate profitable
trades:286 attributed the economic gain to one effective opening change.

Qualification identity:
`ab1d5704082715008c7a2b6d6a23d5a1fb30d0461fc095bcf67567c95491df47`.

| Seed | Exported model SHA256 |
|---|---|
|19|`f89df82adbdd99ade7a526d0ecc9331581984ecdb357afe28dc490c8b8724c8d`|
|43|`af7b55ca21a2fd3d5f70ec571b7eeb483bbdb1599f73bf848d5e57ed43379f20`|

30 focused tests and scoped lint pass. Runtime `45e0af8e` was exported
immutably; GPU readiness and full native baseline parity passed before the
RTX5090 started the fresh-cache pass. Selection, independent report and finite
authenticated collection are automated. No training or live changes.
Selection identity:
`ce6876c25356cc65c7d14ab27a6824c232c53658303a9d8a187e68f143056c13`.

## Completed Results

Selection completed in136.26seconds (142.90 including worker startup), followed
by3.12seconds for the independent report. Queue and collector both exited0 and
are inactive. Collection authenticated134 selection/report files, in addition
to68 qualification files. No optimizer step, feature-data change or live update.

| Item | Result |
|---|---|
| Initial / after deletion / after return |69 /69 /69|
| Retained groups |28|
| Evaluations |112: reference +65 deletion blocks +46 return blocks|
| Historical absent inputs tested |51|
| Accepted changes |0|
| Independent decision replay |exact|
| Packed cold logits / accounts |exact /exact|
| Retained actor tensors |313 exact; index buffers verified separately|

All65 deletion candidates reduce the score. Removing `cot.eur.net_share` also
fails the activity gate. Among return blocks45 reduce the score and
`RAW_ELAPSED_HOURS` ties exactly. Reinsertion requires strict improvement, so
the tie is not retained. No eligible deletion tie was skipped.

Source-versus-expanded donor graph logit error is at most2.1457672119140625e-6,
inside the frozen tolerance, with zero action disagreements over30234 valid
windows and exact native accounts. The later packed cold reload is bit-exact.
No old donor head overwrites the newly qualified head.

| Metric | TRAIN2020-2023 | VALIDATION2024 |
|---|---:|---:|
| Net return |12.708707%|14.693262%|
| Net USD, initial10000 |1270.87|1469.33|
| Sharpe |0.629028|2.179247|
| Maximum DD |7.930291%|4.044574%|
| Orders |743|206|
| Orders/week |3.5550|3.8148|
| Win rate |61.6420%|58.2524%|
| BUY orders / winners |230 /154|24 /15|
| SELL orders / winners |513 /304|182 /105|
| Agent close / TP / SL / forced |58 /402 /222 /61|6 /107 /67 /26|
| Helpful / harmful agent closes |29 /29|4 /2|

The feature pass preserves286's improvement; it does not add another gain.
Short-heavy validation and the harmful early-close count remain visible.
Frozen donor-assisted rejection does not establish irrelevance after a joint
refit or global optimality of the69-input subset.

## Evidence And Next Action

[Evidence manifest](generalization/cumulative_bounded_20261111/MANIFEST.json)
authenticates qualification, selection summary/trace, the full feature ledger
and weight audit. Standalone qualified bundles are under the private run root
`sac69-bounded-qualification-20261111/critic_adamw_{19,43}/bundle/`.
The selection packed artifact remains inference-only; use the qualified SAC
bundle, not that packed wrapper, as the starting point for a later refinement.

This closes the authorized cycle. Preserve primary19 and comparison43 alongside
285. Do not repeat this unchanged frozen pass or refit an unchanged subset by
default. The next design should contrast broader TRAIN anchor coverage against
the existing113 FIT anchors, from this new qualified policy, refreshing native
continuation targets and freezing coverage/sampling before any training. Keep
the unchanged model as control, MONITOR/validation out of target fitting, and
require account/activity checks before adopting a further update. No new training
or promotion is claimed by this receipt; live-producer work remains separate.
