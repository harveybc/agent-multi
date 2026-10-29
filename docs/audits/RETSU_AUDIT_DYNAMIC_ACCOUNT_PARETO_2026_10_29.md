# Retsu audit: dynamic-account Pareto experiment

Author: Retsu. Date: 2026-10-29.
Reviewed packet: Musashi, commit `3644ece9` on `origin/master`.
Implementation pin cited by the packet: agent-multi `935ee95c`.
Native engine pin: heuristic-strategy `e0a1c2b`.
Search identity: `098e871909f0c587b1cbe855cc49b93723ed63a03c74845a52dd9009071a966d`.

This is an executed review of the completed synthetic search, not a proposal and not a rerun of DEAP.
No financial training, TEST, or deployment follows from it.

Objects read and executed:

- [Results](../work_plan/137_DYNAMIC_ACCOUNT_PARETO_RESULTS.md)
- [Audit request](SATOSHI_AUDIT_REQUEST_DYNAMIC_ACCOUNT_PARETO_2026_10_29.md)
- [Frozen plan](../work_plan/134_DYNAMIC_ACCOUNT_PARETO_PLAN.md)
- Evidence directory `docs/audits/evidence/dynamic_account_pareto_20261029/`
- Verifier `tools/verify_dynamic_account_results.py`
- Arbiter, panel, close controller, and the pinned native engine

The reproduction used the published interpreter versions: NumPy 2.5.1, pandas 3.0.3, DEAP 1.4.
`CUDA_VISIBLE_DEVICES` was empty. One thread for OpenBLAS.

## Execution

Decompressed `CANDIDATES.jsonl.gz` and `SEED_*.json.gz` into a separate directory.
Did not modify the archive. Then:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python tools/verify_dynamic_account_results.py \
  --run <decompressed-evidence-directory> \
  --engine-root <heuristic-strategy at e0a1c2b> \
  --export <new-verification-directory>
```

Verifier result, verbatim status fields:

| Field | Result |
| --- | --- |
| status | `PASS` |
| identity | `098e871909f0c587b1cbe855cc49b93723ed63a03c74845a52dd9009071a966d` |
| receipt_count | 17018 |
| native_world_replays | 288 |
| exact_replay | true |
| full_budget_verified | true |
| frontier_policies | 632 |
| unique_frontier_vectors | 342 |
| representative_rule_verified | true |
| source_pins_verified | true |
| verifier_sha256 | `ea2ad5c8a60cc9b294faa350c3dc10452bbc55c228ffd488f75a2587cedbe8fb` |

That pass covers the receipt chain, all 900 generation rows, population fitness against receipts, an independent quadratic dominance check, the five-representative rank rule, source hashes, risk-config hash, and exact native replay of 12 policies × 2 panels × 12 weeks. `verify=True` inside those replays includes cache-versus-direct parity, cash conservation, and the no-overlap check. A mismatch would have raised.

Separately, the eight unilateral on/off deviations and the four 2×2 direction matrices at P3 were recomputed from the pinned code on the calibration worlds. They match `REPORT.json` exactly.

| Check | Recomputed |
| --- | --- |
| Diagnostic representative | `18c72e613b22641393b09842fd0bf6cb1a1e85a5c5096ef77d2044774c290940` (P3) |
| Full calibration objectives | return 2.1927422294259666, negative worst DD −0.255155904300912 |
| Disable 12h buy | return 2.239665689472403, negative worst DD −0.29622083334804644 |
| Unilateral gain | 0.005858530999302003 |
| Disable either 24h specialist | gain 0 |
| Matrices | match the stored report |

Focused mechanism tests, same interpreter, engine root at `e0a1c2b`: 162 passed. 74 integration cases skipped because they require an explicit quote-engine recipe beyond that pin (`test_quote_early_close_expectations`, `test_optimal_exit_corpus`). The passed set includes the arbiter, the durable queue, the close/hold controller, protection, horizon rules, risk profiles, mechanism games, optimal exit, and the native panel. Document 135's count of 128 is not a command in the audit request. This execution does not reproduce that integer. It does reproduce a larger unit set, with those 74 recipe tests left unrun.

DEAP was not repeated. The verifier reconstructs population fitness from the authenticated receipts. A fresh directory would evaluate the candidates again. That was not required to accept the archive.

## Verdicts

| Area | Verdict |
| --- | --- |
| Execution and accounting | Pass. The archived native accounts, cash, drawdown, costs, overlap rule, and cache parity reproduce. |
| Optimization and reproducibility | Pass, inside the declared empirical frontier. Not a certified global optimum. |
| Confirmation | The stored confirmation numbers reproduce. The claim that P1 improves on Only3h does not survive the week pairs. |
| Coverage and learnability | Not established. The selected library trusts a residual-week oracle. This panel never fires a stop. |

## Findings

### 1. P1's confirmation edge is one mirrored pair

Witness: authenticated confirmation ledgers. Mean return of P1 minus Only3h is 0.05019888426647615 percentage points (3.0115% versus 2.9613%). Sign by week:

| Week | P1 − Only3h, return points |
| --- | ---: |
| reversal, increasing, mirror −1 | +0.0806 |
| reversal, decreasing, mirror +1 | +0.0646 |
| overlap, decreasing, mirror +1 | +0.7228 |
| overlap, decreasing, mirror −1 | +0.4605 |
| other eight weeks | negative |

Four weeks are positive. Outside the overlap/decreasing pair, the mean difference is −0.05808986011381023 points over the other ten weeks. The two large weeks are one price path and its reflection `2.2 − price`. The report error is multiplied by the same mirror, so the pair is one economic shape counted twice.

Affected conclusion: "P1 improves on Only3h" is an average carried by that pair. The four trend weeks go the other way.

Minimal correction: publish the twelve paired differences beside the mean. Do not replace Only3h with P1 from this panel.

Rerun: none to correct the archive. Any successor panel must treat a mirror as one pair.

### 2. The volatility factor also changes the spread

Witness: `World.__init__` at the pinned panel. Decreasing volatility sets spread 0.00005. Increasing volatility sets spread 0.0003. Constructing the two overlap worlds with seed 43 reproduced those two spreads. The Gaussian path is drawn before the volatility branch, so the two modes share the noise sequence. Only the envelope is reversed, and the spread changes by a factor of six.

The only large confirmation gain of P1 sits on overlap with the 0.5-pip spread. On the 3-pip weeks, P1 does not beat Only3h.

Affected conclusion: a result labeled as a volatility contrast is also a cost-regime contrast. The results note does not state the spread switch.

Minimal correction: state the confound on the results page. In the next generator, cross spread and envelope instead of binding them to one switch.

Rerun: a new panel. Do not retune this frontier to hide the confound.

### 3. The return end of the frontier is oracle trust plus size

Witness: `FRONTIER.json`, digest-checked by the verifier. Among 632 nondominated policies:

- `close_persistence` is 1 on every row. The search evaluated 1, 2, and 3.
- `close_margin` is 0 on 449 of 632 rows.
- `uncertainty_penalty` on the frontier has maximum 0.137. The search reached 1.994.
- `trend_weight` on the frontier has minimum 0.034. The search reached about −0.5.
- Size and calibration return correlate about 0.995. Part of that is mechanical: size scales return and drawdown, so both ends of the exposure axis stay nondominated.

P2 genes use size 1. P3 genes use size 0.751. Ratio 1.3315579227696406. Confirmation ledgers of P2 and P3 have the same entry bar, exit bar, direction, and horizon. Confirmation return ratio 1.3354786132799583. Same trades, different exposure.

`World.report` builds the residual week of true OHLC from the next bar through the deadline, plus three noisy copies. Horizon weights do not choose a forecast horizon. They scale bids whose gain is the best exit along that residual path. The fixed median SL/TP geometry is the only horizon-specific piece. With close margin 0 and persistence 1, the high-return policies act when that report says to act.

Affected conclusion: the frontier is not evidence of several timing mechanisms. P2 versus P3 is an exposure dial. The high-return neighborhood trusts the synthetic report.

Minimal correction: keep one representative per action sequence and report size as exposure. Do not read 632 policies as 632 strategies.

Rerun: not required to correct this archive. A later search needs horizon-limited reports and hostile reports before margin and persistence can leave 0 and 1.

### 4. No stop-loss exit is identified

Witness: close-reason counts on the authenticated replays of Greedy, Only3h, Only6h, and P1–P5, calibration and confirmation. Reasons present: `AGENT_CLOSE`, `TP`, `FORCED_FLAT`. Count of `SL` is zero in every one of those 12 policy-panel blocks.

P1 confirmation is 20 agent closes, 28 targets, 6 forced flats, 54 orders. That matches the published 20/54 early closes. It does not qualify early-close-before-stop.

Affected conclusion: this account panel cannot support a claim about closing before the stop. Musashi already says this for confirmation. The same absence holds in calibration for these policies.

Minimal correction: leave the sentence as a limit. Do not promote the early-close rate.

Rerun: a stress panel whose paths reach the fixed median stop. Not a reprocessing of these receipts.

### 5. The scalar Nash violation is not a Pareto improvement

Witness: recomputed deviations, matched to the report.

Disabling the 12h buy specialist raises return from 2.1927 to 2.2397 and worsens worst drawdown from 0.2552 to 0.2962. The scalar return-minus-drawdown rises by 0.005858531. Neither objective vector dominates the other. Disabling either 24h specialist changes nothing at P3, so those switches are not players in this library.

The eight switches share one account payoff. A positive unilateral gain is a local defect of that team scalar. It is not an equilibrium of adversaries, and it is not the Pareto objective. The exact interval fixture in `test_opportunity_blocking_is_joint_not_greedy` still passes: a slot of gain 6 that blocks a later gain of 20 is worth 20 to the joint schedule, and a short trade can likewise block a better long trade. The search never scores `interval_best`. It scores the myopic arbiter. Ties abstain. That gap is declared in plan 134 and remains unmeasured on the 17,018 candidates.

Affected conclusion: "P3 is not an exact Nash of the on/off library" is true for the scalar and already disclosed. It does not mean P3 left a dominated policy on the table. Deleting the 12h buy specialist would be a different return/drawdown point.

Minimal correction: quote both objectives next to epsilon. Do not repair P3 by turning that specialist off.

Rerun: none for the stored epsilon. Measuring allocation regret against `interval_best` on the same reports is a new diagnostic, not a correction of the frontier file.

### 6. What already holds and is not reopened

These statements in the results note survived execution:

- The run is synthetic. It is not a trained RL result, not financial VALIDATION, and not TEST.
- Confirmation did not select or retune the five representatives.
- P2 is dominated by Only6h on the confirmation aggregate. P3 is dominated there by Greedy. The calibration frontier was not replaced after confirmation.
- Sharpe in the table is a mean of twelve short weekly figures and was not an objective.
- Self-verification in the published `VERIFICATION.json` is Musashi's check. The pass recorded above is a separate execution of that same verifier.
- Duplicate objective vectors are retained, so representative ranks need not be uniform in objective space.
- Smoke saw the confirmation panel before the final run. The packet says parameters were not retuned from it. This audit did not reconstruct an earlier smoke diff beyond that disclosure.

## Declared limits, not defects of the archive

- Twelve weeks are a factorial of three formulas, two envelope directions, and a mirror. They are not twelve independent markets.
- Nominal horizons label fixed risk profiles.
- Reports are ideal-path-plus-noise by design.
- The frontier is empirical among evaluated eligible policies.
- Local position exit and whole-account scheduling are different objectives.
- No learned reporter was claimed.

The defects relative to the written results are narrower: the P1 mean hides the week pairs, and the volatility label hides the spread switch. The Nash sentence is easy to over-read. The archive itself matches its receipts.

## Not done

- DEAP was not run again.
- The 74 recipe-gated integration tests were not run.
- No file in the published evidence directory was modified.
- This note is the audit response. It does not by itself publish a new search.
