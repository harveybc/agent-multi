# Dynamic account Pareto: completed synthetic results

2026-10-29. Plan134 completed; runtime27min22s, population20,
generations300 for each of seeds7/19/43. No worker restart or failed unit.
Source implementation: agent-multi935ee95c; later context review b52db1aa
did not change the search. Identity:
`098e871909f0c587b1cbe855cc49b93723ed63a03c74845a52dd9009071a966d`.

These are synthetic model-output experiments, NOT financial VALIDATION,
TEST or learned RL results. Twelve independent-account weeks per panel start
at10000 synthetic USD each. Reports deliberately contain noisy future-path
information to design the controller. High win rates do not establish that a
learned reporter can supply this information.

## Completion and independent replay

17,018 distinct decoded policies evaluated. The empirical calibration archive
has632 nondominated policies, representing342 distinct objective vectors.
The five representatives were selected by the frozen policy-rank rule, not
confirmation performance. Duplicate vectors are retained; ranks therefore
need not be uniformly spaced along distinct objective vectors.

Post-run verifier authenticated every chained receipt and every population
fitness, checked all900 generations, recomputed dominance independently on
unique objective vectors, checked the representative rule and source hashes,
then replayed all288 native scenario accounts (12 policies x2 panels x12).
All stored detailed results matched exactly. Full actual-unit report decisions
also agree with the per-unit cache; maximum reported discrepancy is rounding
scale, about1e-15 in normalized advantage. This is self-verification, not
Satoshi's independent audit verdict.

All11 genes varied in the actual search: persistence1/2/3, size0.25-1,
and hundreds of distinct values for most continuous parameters. The full
observed spans and unique counts are in REPORT.json.

## Results

Return is the mean net percentage per synthetic week, not an annual return or
a compounded twelve-week account. DD is the worst mark-to-market percentage
drawdown among the twelve scenario accounts. Counts aggregate those accounts.

| Policy | Cal return% | Cal worst DD% | Confirm return% | Confirm worst DD% | Orders/week | Buy / sell | Win% | Early closes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Greedy all profiles | 2.1988 | 0.6190 | 2.1911 | 0.4034 | 3.33 | 20 /20 | 85.00 | 14/40 |
| Only3h | 3.0125 | 0.5945 | 2.9613 | 0.5545 | 5.67 | 34 /34 | 85.29 | 25/68 |
| Only6h | 2.8685 | 0.5768 | 2.9499 | 0.5378 | 4.58 | 27 /28 | 92.73 | 25/55 |
| P1 highest calibration return | 3.1169 | 0.4176 | 3.0115 | 0.5378 | 4.50 | 27 /27 | 98.15 | 20/54 |
| P2 | 2.9290 | 0.3396 | 2.7712 | 0.5378 | 4.08 | 24 /25 | 93.88 | 20/49 |
| P3 middle rank | 2.1927 | 0.2552 | 2.0751 | 0.4043 | 4.08 | 24 /25 | 93.88 | 20/49 |
| P4 | 1.2067 | 0.1435 | 1.2281 | 0.2371 | 4.67 | 27 /29 | 85.71 | 25/56 |
| P5 lowest calibration DD | 0.6877 | 0.0815 | 0.7040 | 0.1348 | 4.17 | 24 /26 | 96.00 | 23/50 |

All five representatives satisfy the activity rule in both panels. Equal
guarded allocation fails it in calibration;24h-only fails it in both panels.
NOP is the explicit zero-return/zero-risk ineligible control, not a winner.

| Policy | Confirm buy wins | Confirm sell wins | Mean native weekly Sharpe | Horizon orders3/6/12/24 | TP /SL /forced flat |
| --- | ---: | ---: | ---: | --- | --- |
| Greedy | 17/20 | 17/20 | 19.059 | 16/6/16/2 | 18/0/8 |
| Only3h | 29/34 | 29/34 | 19.249 | 68/0/0/0 | 36/0/7 |
| Only6h | 25/27 | 26/28 | 19.770 | 0/55/0/0 | 22/0/8 |
| P1 | 26/27 | 27/27 | 20.228 | 34/20/0/0 | 28/0/6 |
| P2 | 22/24 | 24/25 | 20.516 | 27/11/9/2 | 23/0/6 |
| P3 | 22/24 | 24/25 | 20.522 | 27/11/9/2 | 23/0/6 |
| P4 | 23/27 | 25/29 | 19.773 | 0/48/0/8 | 21/0/10 |
| P5 | 23/24 | 25/26 | 20.065 | 3/41/1/5 | 21/0/6 |

The Sharpe column is the arithmetic mean of12 short-sample native weekly
Sharpes (daily returns annualized by sqrt365), not a portfolio Sharpe or a
reliable long-run estimate. It is descriptive, never a selection objective.
Full per-world results, costs and ledgers are exported for recalculation.

## Findings to attack first

1. P1 improves on Only3h by0.0502 percentage points of mean confirmation
   return and0.0167 points of worst DD. No significance claim is made from
   this small paired synthetic panel. P1 uses only3h and6h in confirmation;
   this does not establish that longer specialists are useless elsewhere.
2. P2 is dominated by Only6h in confirmation: less return at identical worst
   DD. P3 is dominated by Greedy there. A calibration frontier need not remain
   a frontier out of calibration; no post-confirmation replacement is made.
3. P2 and P3 execute the same confirmation action sequence at different size.
   Part of the frontier is an exposure tradeoff, not improved pattern choice.
4. P1 closes early20/54 times (37.04%). There are no SL exits in these
   confirmation policies; many positions instead reach TP or forced flat.
   Thus this account panel alone cannot qualify difficult early-close-before-SL
   behavior. Prior131/133 counterfactual fixtures cover that component, but
   integrated stress with poorer reports is still an audit concern. No target
   close percentage is imposed to manufacture success.
5. At P3, disabling12h-buy improves the diagnostic scalar return-minus-DD by
   0.005858531. Therefore P3 is NOT an exact Nash equilibrium even in the tested
   on/off library. Disabling either24h specialist has zero calibration effect.
   This diagnostic is neither the Pareto objective nor a global certificate.
6. Geometries for SL/TP are fixed median profiles, not optimized genes in this
   run. The nominal horizons label these profiles, not separately trained
   policy models.12 calibration weeks and12 confirmation weeks are a bounded
   test panel, not exhaustive markets; mirrored scenarios are paired designs.
7. Smoke testing executed the fixed confirmation panel before this final run;
   the implementation and parameters were not retuned from those results.
   Confirmation is excluded from fitness, but is not advertised as never seen.

## Exact representative parameters

Gene order: weights3/6/12/24, trend coefficient, uncertainty penalty, duration
penalty, entry threshold, size multiplier, close margin/R, close persistence.

| ID prefix | Alias | Genes |
| --- | --- | --- |
| 08fca639649c | P1 | [0.922,0.943,0.379,0.582,0.076,0.100,0.816,0.022,1.000,0.000,1] |
| 7d161591d433 | P2 | [0.963,0.982,1.023,0.274,0.096,0.085,0.816,0.022,1.000,0.000,1] |
| 18c72e613b22 | P3 | [0.946,0.982,1.019,0.286,0.076,0.085,0.816,0.011,0.751,0.000,1] |
| 0622b2f8d354 | P4 | [0.993,1.168,0.688,0.878,0.120,0.102,0.326,0.000,0.440,0.003,1] |
| 89a4e072655d | P5 | [1.048,1.176,0.759,0.878,0.157,0.131,0.330,0.051,0.250,0.000,1] |

## Evidence and next step

[Evidence directory](../audits/evidence/dynamic_account_pareto_20261029/):
MANIFEST.json pins sources, rules and software; REPORT.json has metrics and
four2x2 payoff matrices; FRONTIER.json contains all632 policies;
CANDIDATES.jsonl.gz contains the17,018 original chained receipts;
SEED_*.json.gz contains every generation; REPLAYS.json.gz has complete native
ledgers and controller traces; VERIFICATION.json authenticates the export and
records the independent dominance calculation and exact288-account replay.

[Final audit request](../audits/SATOSHI_AUDIT_REQUEST_DYNAMIC_ACCOUNT_PARETO_2026_10_29.md).
Next: independent review of this bounded mechanism result, then matched
context ablations under136 and integrated adverse-report stress. No new
financial training, floating selection, deployment or global-optimum claim
follows automatically from these synthetic returns.
