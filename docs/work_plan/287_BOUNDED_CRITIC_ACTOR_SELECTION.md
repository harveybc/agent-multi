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
