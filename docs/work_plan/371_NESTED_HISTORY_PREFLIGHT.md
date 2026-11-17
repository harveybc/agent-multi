# 371: nested history boundaries before expensive training

Owner-approved continuation of370. This preflight does not train, promote or
change running workers. It constructs6/12/24/48-month supports and identifies
fit-scope conflicts in the selected71 pipeline before GPU allocation.

## Test design

| Requirement | Test |
| --- | --- |
| Nested supports end2024-01-01 | Every smaller support is contained in larger support |
| Preserve13 observations and stride2 | Use existing slot timestamps; reject windows whose first slot precedes subset |
| No extra data from scaler | Explicit fresh-subset-scaler requirement, never copy old constants |
| No hidden pretrained network | Require fresh actor/critic/target initialization and fresh Adam/replay |
| Source scope honest | Fit ranges must be contained in subset; unknown producer scope cannot pass |
| Validation fixed | Same2024 row mask across arms; TEST unavailable |
| No fabricated eligibility | Unknown factor fit, changed upstream columns and warmup limitations remain explicit |

The initial supports are counts on the current store, NOT final regenerated
subsets. Refitting learned producers can change valid support; materialization
must remeasure counts and pin each arm's final data.

## Two different questions

Strict pipeline data quantity requires producer refits inside each interval,
cross-fitting activity with label-horizon purging, subset-only fixed scaling,
and fresh networks. Selected features/architecture remain common and previously
chosen; the conclusion remains conditional on them.

Frozen-upstream downstream learning instead measures how much RL exposure is
needed given already learned producers. It is useful but cannot establish that
the complete pipeline learned using only six months. Never label these two
experiments interchangeably.

History before each interval may provide explicitly declared causal warmup for
fixed filters; it must not supply fitted target labels or optimizer samples.
Here the scored network windows are strictly inside each interval. Long-memory
features such as COT rolling summaries need their warmup provenance retained.

## Executed CPU preflight

20 combined370/371 unit tests pass; scoped lint passes. Current-store measured
supports preserve all71 inputs,28 groups,13 observations and stride2:

| TRAIN months | Valid scored windows | Common validation windows |
| ---: | ---: | ---: |
| 6 | 2781 | 6208 |
| 12 | 5892 | 6208 |
| 24 | 11842 | 6208 |
| 48 | 24026 | 6208 |

Eight columns have learned upstream dependencies: two FX factor columns,
activity, one Kalman column and four GMM columns. The consumer manifest records
a2020-01-01..2023-07-01 fit scope for activity/Kalman/GMM. This scope is not
contained in any of the three shorter TRAIN suffixes. Factor fitting needs its
own source receipt; this preflight explicitly leaves unknown scope unqualified.
All71 columns are retained. No optimizer, data replacement or model promotion
occurred. Exact continuation: recover the factor fit receipt and implement
subset-fitted producer reconstruction with activity cross-fit/horizon purging,
then fixed subset scaling and fresh neural initialization. The support counts
must be remeasured on those rebuilt stores before dispatch.
