# 412: bounded automatic TRAIN campaign

Exploratory, not a release gate. The 2023 market was inspected previously;
chronological fit is not independent confirmation. Do not use 2024, Dragon,
the live provider, a different encoder or cheaper fills. Unit is incremental
net account basis points vs waiting on the matched reference, not price MAE.

1. 411 ledger parity: reproduce all selected origins, timestamps, target cube
   and mean payoff. Compare top two scored actions and fixed BUY 6h against
   NOP at each selected origin. Separate both nonpositive, runner alone
   positive, winner alone positive, both positive. Record forecast margin
   and realized difference also by forecast-margin quartile.
2. Within 2023, fit on Jan-Jun labels only if their **weekly forced flat**
   predates July. Score Jul-Dec without refitting. Compare unchanged 409,
   abstention-only (ridge for actual winner payoff), reranking-only (same
   open ticks), both, fixed BUY 6h on original open ticks, and always wait.
   Inputs to both regressions are the eight 409 bids, plus winner indicator
   for abstention. Fixed normalized ridge penalty 0.1; intercept unpenalized.
3. Run native single-account replays with an empty account in H1, activity
   measured per H2 market session. Require net positive, above original
   account, >2 trades/session and positive hypothetical mean in every seed.
   No threshold sweep. Oracle best-of-eight vs NOP is a plumbing positive
   control with future labels, never a tradable rule or an account bound.
4. If no correction qualifies across all seeds, test a genuinely different
   estimator: mean signed payoff of nearest 2022 analogs in 208 frozen
   coordinates (standardize on 2022, Euclidean distance, k=ceil(sqrt(n))).
   Same H2 account and strict gates. Negative endpoint: reject these rules,
   no 2024 scoring or new feature ablation under a rejected router.
   Positive endpoint: designate for independent replication only.

Checks before interpreting: planted ordering, negative/no-op, tie exclusion,
same input with perturbed evaluation labels gives identical deployable
decisions, chronological separation, native-vs-ledger routing, sealed label
and native head hashes. Cached seed receipts bind input ledgers, dependencies
and code. An interrupted seed reruns, a completed one does not.

## Execution and outcome

Single restartable command: `bash tools/run_role412_campaign.sh`.
Result directory: `/var/tmp/role412-router-campaign/`. The main and analog
seed receipts include checksums and full metrics. DONE records the terminal
decision. These stages intentionally finish; this is not an always-on scheduler.

2,487 fit origins, 2,229 scored origins across 25 H2 market sessions per
seed. Net H2 return (%) / actual trades:

|Router|7|19|43|
|---|---:|---:|---:|
|Original| -20.83 / 476 | -23.12 / 541 | -16.27 / 502 |
|Gate only| -0.12 / 6 | 0 / 0 | -0.77 / 3 |
|Rerank only| -10.62 / 455 | -12.81 / 547 | -12.62 / 579 |
|Joint| +0.79 / 25 | -1.90 / 30 | +0.59 / 29 |
|Fixed BUY 6h| -18.45 / 557 | -23.06 / 592 | -19.08 / 581 |
|2022 analogs| -10.26 / 481 | -12.78 / 538 | -9.75 / 488 |
|NOP|0 / 0|0 / 0|0 / 0|

On the 2,555 / 2,795 / 2,681 original positive offers (full 2023), neither
top action beat waiting at 1,670 / 1,758 / 1,723 origins. In 1,103 /
1,202 / 1,201 cases a lower-ranked action did beat NOP; in 567 / 556 / 522
none of the eight did. The retrospective distinction is not a live input.
The actual mean top-minus-runner advantage was -0.17 / -0.14 / +0.29 bp.
The three oracle account controls were active and positive; this is not
evidence of attainable return. No deployable arm qualified. Do not promote
or use these routers to eliminate features. No confirmation year was scored.
