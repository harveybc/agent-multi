# Independent audit: Musashi's dynamic-account Pareto experiment (agent-multi master @3644ece9)

Date 2026-10-29. Auditor working for Satoshi (expert-guided lane). Objects: agent-multi 3644ece9 (code identical
to implementation 935ee95c; b52db1aa and 3644ece9 change only docs, evidence, the verifier and TOOL_DECLARATIONS),
native engine heuristic-strategy e0a1c2b, search identity `098e8719…a966d`.
CPU only (`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`). Every heavy step ran under
`crispdm-run -q -W 1800 -m 2400M -t 7200s -n sat-da-<step>`, and no admission was refused. No tracked file was
modified, nothing was committed, and no process was signalled. Only synthetic 2020-01 bars were read (no rows dated
2025 or later).

Placeholders: `<am>` is the detached agent-multi worktree at 3644ece9. `<hs>` is a NEW detached heuristic-strategy
worktree at e0a1c2b962f4. No worktree named `hs-closehold-pin` existed. The only other checkout at e0a1c2b is a
branch worktree of another lane, so I left it alone. HEAD was e0a1c2b and the tree was clean before and after all
runs. `<scratch>` is the audit scratch directory; scripts are in `<scratch>/scripts/`, outputs in `<scratch>/*_out.json`.
Python 3.12.13, NumPy 2.5.1, pandas 3.0.3, DEAP 1.4 (the same versions as the MANIFEST).

## Verdicts

| Area | Verdict | Reason |
|---|---|---|
| Execution / accounting | **ACCEPT_WITH_CHANGES** | I recomputed all 1,173 trades of the 288 accounts with my own code and found exact agreement: cash max error 2.8e-14 USD, MTM drawdown 1.8e-14 pct-pts, Sharpe 6e-12, and 0 mismatches in units, margin, exits, gates or overlap. Five defects remain latent and none changes a committed number: F07, F08 and F10 are unfixed; the stale-report guard is vacuous in the panel (DA12); the cache key lacks the entry price (DA13). |
| Optimization / reproducibility | **ACCEPT_WITH_CHANGES** | The verifier export is byte-equal (10/10 files). I authenticated the receipt chain and the fitness of all 900 generations, and re-evaluated all 17,018 candidates from scratch in reverse order with 0 mismatches. An independent Kung skyline gives 632/342 and the same five representatives. Two changes: the representative inside a duplicate-vector group is picked by a hash tie-break (DA04), and one gene silently drives two computations (DA19). |
| Confirmation | **ACCEPT_WITH_CHANGES** | Every number in 137 reproduces. P2 is dominated by Only6h. P3 is dominated by Greedy and also by equal_guarded, which 137 does not state. P1 vs Only3h is +0.050 pts with CI [-0.076, +0.212], positive in 4 of 12 weeks. The 12 weeks are only 3 independent random draws per panel, and the report noise is shared by all worlds (DA16). |
| Coverage / learnability | **REVISE** | The reports are the true future path plus about 1 pip/sqrt(h) noise. Without early close, P1's confirmation return falls from 3.01 % to 0.91 % and DD rises from 0.54 % to 1.88 %. SL-first positions exist (14 of P1's 20 confirmation early closes pre-empt a native SL), but the reports foresee them, so no early close is ever tested under uncertainty. The prospective adverse stress leaves the foresight intact. No witness exists for protection pre-empting a discretionary exit (0 of 564 cases). |

## A. Focused tests

`<scratch>/A_tests.txt`, `<scratch>/A_tests_broad.txt`. `STRATEGY_ENGINE_ROOT=<hs>` was set, so the panel tests ran
instead of skipping.
- The four named files (`test_dynamic_account_arbiter/panel/queue.py`, `test_close_hold_controller.py`): **25 passed**.
- The broader family (the four above plus close_hold_reports, close_hold_trace_audit, early_close_metrics,
  absolute_exit_corpus, horizon_risk_rules, protection_reference, mechanism_games, optimal_exit, early_close_audit,
  early_close_grid_audit): **132 passed**.
- The "128 focused tests" in 135 are not listed anywhere, so that count cannot be reproduced (DA23).

## B. Verifier, digests, receipts

`<scratch>/B_verify.log`, `<scratch>/B_export/`, `<scratch>/B2_digests_out.json`, `<scratch>/B3_reeval_all_out.json`.
- I ran `tools/verify_dynamic_account_results.py --run <scratch>/run_in --engine-root <hs> --export <scratch>/B_export`.
  It printed PASS in 25.6 s with 17,018 receipts, 288 native replays, 632 policies and 342 vectors. **All ten exported files
  are byte-equal to the committed evidence**, including VERIFICATION.json (sha d66dca80…), REPORT 5760fb13…,
  REPLAYS c19a5998…, FRONTIER 116102193… and MANIFEST 966b0292….
- I wrote my own check (`B2_digests.py`), which imports nothing from agent-multi. Results:
  - The MANIFEST identity recomputes.
  - 12 agent-multi and 29 heuristic-strategy source pins match, the hs file set is identical, and the CONFIG sha matches.
  - The CANDIDATES, REPLAYS and FRONTIER digests match REPORT.
  - Every VERIFICATION artifact sha matches, including the uncompressed shas, and the verifier's own sha matches.
  - The receipt chain has 17,018 links, 0 bad, and the correct tail.
  - All genes are on the 0.001 grid and inside bounds.
  - All 900 generations × 20 fitness values equal their receipts. 16,920 candidates are eligible and 98 are not.
- **Full re-evaluation** (`B3_reeval_all.py`): I re-evaluated all 17,018 receipts on fresh calibration worlds, in REVERSE
  receipt order so the cache state differs from the search. Result: **0 mismatches** (1,643 s; see `B3_reeval_all_out.json`).
  The verifier itself checks only the chain and the population fitness; it never re-evaluates a candidate, so this run
  goes further.

## C. Frontier (independent algorithm)

`<scratch>/scripts/C_frontier.py`, `<scratch>/C_frontier_out.json`. I used a Kung divide-and-conquer maxima algorithm
that splits by x value so equal-x points stay together, and cross-checked it with a brute-force O(n²) pass on distinct
vectors.
- Results: **632 policies, 342 distinct vectors**, the same key set, and the same (-ret, -negDD, key) order as
  FRONTIER.json.
- The frontier has no equal-x and no equal-y ties. The duplicate multiplicities include groups of 64, 44, 10 and 8, plus
  241 singletons.
- Eligibility: 0 ineligible candidates would sit on the unconstrained frontier, so the activity rule never removed a
  nondominated vector.
- Representatives: positions 0/158/316/473/631, giving **P1 08fca639649c, P2 7d161591d433, P3 18c72e613b22,
  P4 0622b2f8d354, P5 89a4e072655d**. Their genes equal the 137 table.
- The positions are evenly spaced in policies but not in distinct vectors: the distinct-vector ranks are 0/37/126/218/341.
  P1 is one of 64 genomes with the identical calibration vector. Those genomes differ in w12 (26 values), w24 (23),
  trend (16) and margin (3). The hash tie-break picked which one goes to confirmation (DA04).

## D. Accounting (own code, from the REPLAYS ledgers)

`<scratch>/scripts/D_accounting.py`, `<scratch>/D_accounting_out.json`. World supplied only the exogenous bars,
calendar and cost constants. I recomputed everything else.
- **What was recomputed for each of the 1,173 trades:**
  - Quote side: entry at the ask for a buy and the bid for a sell; exits on the exit side.
  - SL/TP: my own Decimal rounding (buy floor, sell ceil) from the median references plus a 1-pip buffer.
  - Sizing and margin: q = c·min(1, ρB/L1), with L1 including market and stop slippage, round-trip commission and the
    positive-only funding reserve to the forced flat. Units = floor(q·floor(B·0.1·50/E)). Margin = units·E/50 ≤ q·0.1·B.
  - Exits: native first passage (open gap first, SL before TP intrabar, `h ≥ tp+through`), agent close at the open,
    forced flat at the close of bar 107.
  - Cash: spread not double-counted, stop and market slippage, commission per side, rollovers in (entry, exit],
    triple on the third.
- **Results:** units mismatch 0, margin over q·0.1·B 0, risk over budget 0, exit bar or reason mismatch 0,
  **max |net error| 2.84e-14 USD**.
- **MTM drawdown:** equity at every bar close with an exit-side mark, entry slippage and commission, and accrued
  financing. Max return error 0.0 and **max DD error 1.8e-14 pct-pts**. The closed-trade-only DD is far smaller
  (P1 confirmation 0.002 % vs MTM 0.538 %), so the reported DD really is mark-to-market.
- **Overlap and gates:** 0 overlaps and 0 same-tick reopens. Entry fills all satisfy
  Mon 10:00 ≤ t < Fri 10:00 (session_open+12h ≤ t < session_close−12h), and every position is flat by the close of
  bar 107 (Fri 10:00). There are 0 entry-bar voluntary closes. 806 of 1,173 trades re-enter at the first permitted bar
  (DA25).
- **Two last-hour entries.** P4 and P5 in calibration each fill a buy (6h profile) at bar 107, Fri 09:00 UTC, in the
  reversal / increasing / mirror −1 world. Both are forced flat at Fri 10:00, for nets of +1.82 and +1.03 USD.
  Against BUSINESS_CONTRACT.json `entry_time_rule` (`session_open + 12h <= executable_entry_time < session_close − 12h`)
  the executable fill time 09:00 is before 10:00, so these are **not gate violations**. They are guaranteed one-hour
  forced-flat trades. The engine permits them on purpose: heuristic-strategy `app/short_horizon/expert_guided/absolute.py:74`
  (`if self.v[k] >= self.ff[si]`) replaces the legacy `env.py:175` rule (`v[k]+1h >= ff`). No fix or rerun is needed.
  An optional variant would exclude entries whose only possible exit is the forced flat (DA06).
- **Weekly return and win %.** Every return, DD, order count, buy/sell split, win %, buy/sell wins, TP/SL/forced-flat
  split and early-close count in both 137 tables is reproduced exactly. Win % counts net > 0 over all exit reasons.
  For P1 in confirmation: TP 28/28 wins, early closes 19 wins / 1 loss, forced flat 6/6 wins, so 53/54 = 98.15 %.
  The counts are internally consistent (TP + SL + FF + early = orders for every policy). Across all 288 accounts there
  is **exactly 1 SL exit** (equal_guarded, calibration).
- **Sharpe.** I reproduce the native value to 6e-12. Each world has 6 calendar-day returns (Sunday partial through
  Friday), ddof=1, multiplied by sqrt(365). That treats a single 5-trading-day sample as if 365 i.i.d. daily returns
  continued for a year, so the "≈20" values mainly reflect very high daily mean / low daily SD under foreseen paths and
  n = 6. The cross-week alternative (mean / SD of the 12 weekly returns × sqrt(52)) gives P1 conf 17.8 and Only3h 20.3,
  which ranks the two the other way round from the 137 column (20.23 vs 19.25) (DA07).

## E. Questions 3, 4 and 11: causality, cache, entry bar

`<scratch>/scripts/ownval.py`, `E_replicate.py`, `E_adversarial.py`, `Q4_counts.py`, and the matching `*_out.json`.
- **Independent replication of every decision.** I wrote my own first-passage valuation, common-action advantage,
  persistence/urgency logic and arbiter score; none of them imports value_paths, protection_exit or choose. Over the
  288 accounts:
  - 3,938 flat gate ticks: **0 entry mismatches** (1,173 entries with the same side and horizon).
  - 23,710 occupied ticks: **0 close mismatches** (564 closes).
  - My actual-unit advantage vs Musashi's per-unit cached advantage: max difference **1.3e-15**. Per-unit vs actual-unit:
    1.3e-15, because values and risk scale linearly in units.
- **Stale and future reports.** `decide()` at the environment clock refuses a report from b−1 and one from b+1
  (STALE_OR_FUTURE_REPORT). But the panel passes `int(r.issued_at)` as the decision time on both routes
  (`app/dynamic_account_panel.py:180` and `:184`), so the guard compares the report with itself. Witness: a World whose
  `report(b)` returns `report(b−1)` runs to completion under `verify=True`, and the confirmation trend-up world's return
  changes from 1.795 % to 1.881 %. The cache adapter cannot check staleness at all, because it never receives a report
  time. The committed results are unaffected: every report used has `issued_at == v[b]+1h`,
  `deadline == forced flat` and `H == end−b−1`, all asserted on the 23,710 ticks (DA12).
- **Changed entry, profile, side and funding.**
  - Changing the profile, side or funding changes the valuation, as it should.
  - A cache **miss** with a changed entry raises CACHE_GEOMETRY.
  - A cache **hit** with a changed entry is **served the stale valuation silently**: the key `(h, side, entry_bar, b)` at
    `:157` omits the entry price, and the geometry check at `:160` runs only on a miss. This is latent; the entry quote is
    deterministic per key (DA13).
- **Causal plane.** `allow_synthetic=False` refuses both the cache adapter (SYNTHETIC_CACHE_FORBIDDEN) and a
  synthetic report (ORACLE_PLANE_FORBIDDEN). The separation is a constructor flag, not a capability boundary (DA14).
- **Scenario names and labels.**
  - Mutating `spec.pattern` or `spec.volatility` after construction leaves the trace unchanged.
  - Mutating `spec.mirror` changes the trace: the report error is multiplied by the scenario label at `:95`. The bias
    direction therefore depends on the scenario label; this is part of the declared fixture, but it is a label input.
  - The report noise realization is keyed only by (seed, b), so it is identical in all 12 worlds of a panel (DA15, DA16).
  - No agent state is read (`AbsoluteBarEnv.state` raises). past_efficiency reads closes up to bar b.
  - Every horizon uses the same close/hold rule; the horizon selects only the geometry.
- **Entry-bar close:** none. Candidate valuation drops column 0 (`:128`), and the ledgers show 0 closes on the entry bar.
- **Question 4: pre-emption, gaps, ambiguous bars.**
  - All 564 close decisions executed as AGENT_CLOSE at the next open; **0 were pre-empted by native open protection**.
  - 192 of the 288 world-runs carry one deterministic −12 pip gap (reversal/overlap). 161 trades span a gap
    (exits: TP 89, early 55, FF 16, SL 1).
  - At most one same-bar SL/TP ambiguity is possible, since there is a single SL exit.
  - So the account panel has no witness for "first passage pre-empts a discretionary exit" (DA24).

### Prior findings F07, F08, F09/F15, F10

| Prior | Fixed in 935ee95c? | Effect on THIS experiment (measured) |
|---|---|---|
| F07 TP-through `(fav−tp) >= through` vs native `h >= tp+through` | **No** (`app/protection_reference.py:22` unchanged; close_hold diff vs 8bf75c81 is refactor only) | **0 affected.** 24,224 native bars checked, 0 disagreements, 0 within 1e-7 of the boundary (prices are not tick-quantized). My native-expression valuation agrees with every cached valuation (1.3e-15) and reproduces every decision on all 27,648 decision ticks. |
| F08 forced-flat close-time spread | **No** | **0 affected.** Each world uses one constant spread (asserted at every open and at the close time). It becomes live as soon as an hour-dependent schedule is used. |
| F10 truncated self-consistent report accepted | **No** (witness: 3-bar report, HOLD, accepted) | **0 affected.** The panel binds the deadline to the forced flat; asserted on 23,710 ticks. |
| F09/F15 no intrabar / deadline / near-boundary cases | **Partly**: the account panel has intrabar TP (P1 conf 21), at-open TP (7) and forced flat (6) | Intrabar SL appears once in the whole corpus, near-boundary TP never, protection pre-emption never. The panel contains SL-destined positions, but the reports foresee them (DA15). |

## F. Gene reach (question 7)

`<scratch>/scripts/FH.py`, `<scratch>/FH_out.json`. For each gene I perturbed P1 and P3 by ±0.05 (persistence +1)
and replayed calibration. The columns are:
- **Entry-score reach:** bid scores changed, out of 2×ticks per horizon (P1 134 flat ticks, P3 123).
- **Choices changed:** entry choices changed at the baseline states.
- **Close reach:** advantage / qualifies / action changed, out of P1 1,018 or P3 1,029 occupied ticks.
- **Δobjective:** whether the calibration account vector changed.

| Gene | P1 entry-score reach | P1 choices changed | P1 close reach | P1 Δobjective | P3 choices / close / Δobj |
|---|---|---|---|---|---|
| w3 | 268 (3h only) | 7 | 0 | yes (3.098/−0.437) | 10 / 0 / yes |
| w6 | 268 (6h only) | 11 | 0 | yes | 16 / 0 / yes |
| w12 | 268 (12h only) | **0** | 0 | **no** | 3 / 0 / yes |
| w24 | 268 (24h only) | **0** | 0 | **no** | 0 / 0 / **no** |
| trend | 1,072 (all, ∝ d·eff·h/24) | 1 | 0 | yes | 1 / 0 / yes |
| uncertainty | 1,038 (all bids with unc>0) | 0 | **adv 1,006, qualifies 1, action 1** | yes | 0 / 1,016-1-1 / yes |
| time | 1,072 | 4 | 0 | yes | 2 / 0 / yes |
| threshold | 1,072 | 6 | 0 | yes | 1 / 0 / yes |
| size | 0 (units only: 45,428→43,156; trace equal) | 0 | 0 | yes (2.959/−0.397) | 0 / 0 / yes |
| margin | 0 | 0 | qualifies 8, action 8 | yes | 0 / 7-7 / yes |
| persistence | 0 | 0 | action 29 | yes (2.467/−0.675) | 0 / 26 / yes |

Every gene reaches its intended computation. At P1, w12 and w24 change only the scores of bids that never win, so
they are inert at account level (P1 never trades 12/24h in calibration). **The uncertainty penalty is a single gene
driving two computations**: the entry score and the close controller's z (`app/dynamic_account_panel.py:171`,
`CloseConfig(genes[5],…)`). 134/135 list it as one gene without saying so (DA19).

**P2 and P3:** traces (bar, action, horizon) and ledger events (entry, exit, reason, side, horizon) are **identical in
both panels**. q ratio 0.75099–0.75102 equals the size ratio 0.751, and the units ratio is 0.743–0.751 because of the
floor. Their other gene differences (w12, w24, trend, threshold) are inert at action level, so the P2–P3 difference
is pure size. Return ratio: 2.075/2.771 = 0.749.

## G. Controls and robustness (questions 6, 8, 9) and the rotation question (item 5)

`FH_out.json` (bootstrap), `G_cluster_out.json`, `G_stress_out.json`, `R_reentry_out.json`, `Q8_gapfore_out.json`.
I declared the protocol **before running**: `<scratch>/G_PROTOCOL_PROSPECTIVE.md`, sha 9177a23d….
- **P1 − Only3h per week.**
  - Confirmation diffs: −0.135, −0.154, −0.107, −0.079, −0.010, +0.081, +0.065, −0.096, −0.034, −0.111, +0.723, +0.461.
    Mean **+0.050**, 12-week paired bootstrap (100k, seed 20261029) **95 % CI [−0.076, +0.212]**, positive in **4 of 12**
    weeks.
  - The mean comes from the overlap / decreasing-volatility mirror pair. With a 6-mirror-pair cluster bootstrap the
    CI is [−0.094, +0.275], with 2 of 6 pairs positive.
  - Calibration: mean +0.104, CI [−0.027, +0.250], 8 of 12 weeks positive.
  - Worst DD: P1 0.538 vs Only3h 0.555 in confirmation. In 9 of 12 weeks P1's DD ≤ Only3h's.
- **Dominance:**
  - P2 (2.7712 / 0.5378) is dominated by Only6h (2.9499 / 0.5378): VERIFIED.
  - P3 (2.0751 / 0.4043) is dominated by Greedy (2.1911 / 0.4034): VERIFIED, and **also by equal_guarded**
    (2.1518 / 0.4034), which is eligible in confirmation at exactly 3.0 orders/week. 137 does not say this.
  - P1 weakly dominates Only6h.
  - **Duplicate groups in confirmation (G4):** P1's 64 calibration-identical genomes give 2.9904–3.0196 at DD 0.5378
    (5 vectors); P2's 44 give 2.7669–2.7776 (6); P3's 10 give exactly one value. Every P2-group genome is dominated by
    Only6h, so the tie-break never flips a conclusion.
- **Report-noise model** (`panel.py:89-101`): paths = true future OHLC bars b+1…forced flat, plus
  (bias ± noise) × mirror. Bias = cumsum N(+5e-6, 6e-5) and noise = cumsum N(0, 1e-4), giving three antithetic samples
  with weights 1/3. The weighted mean is the true path plus bias. Cumulative dispersion is about 2.4 pips at 6 h and
  4.9 pips at 24 h, against SL distances of 8.4–17.2 pips.
- **Scenario support** (`panel.py:47`): the rng is keyed by (seed, pattern) only. Within a pattern the 4 worlds share
  turn, phase, gap bar and innovations: increment correlation between the volatility variants is 0.72–0.86, and mirror
  pairs are exact reflections. The report noise realization is identical across all 12 worlds. Gap bars are 58/73 in
  calibration and 83/88 in confirmation; trend worlds have no gap. **So each panel has 3 independent random draws, each
  shown under 2 envelopes × 2 mirrors.** The spread is also tied to the volatility label (3 pips increasing, 0.5 pip
  decreasing).
- **SL-first situations exist; the reports foresee them.** Passive counterfactual of each early close (true bars,
  native rules):
  - **P1 confirmation (20 early closes):** passive outcome SL 14, forced flat 5, TP 1. 19 helpful, 1 destructive.
  - P1 calibration: SL 20, FF 13, TP 2, with 4 destructive.
  - Only3h confirmation: SL 15, FF 9, TP 1, with 6 destructive.
  - In 2,713 of 23,710 occupied ticks (11 %) every report sample shows the SL coming.

  The panel therefore exercises early-close-before-SL, but always with foresight. The "0 SL exits" is a result of
  clairvoyance and is no evidence of robustness. Item 9's caution is correct; its stated reason ("many positions instead
  reach TP or forced flat") understates how often an SL was avoided.
- **Prospective stress results.** G1 is the requested arm: adverse bias 0.5 × stop distance, on entry bids (bars 1…)
  and on occupied reports (all bars). "Dest." counts destructive early closes.

| Arm (confirmation) | P1 ret % | P1 DD % | P1 orders | SL | early | dest. | Only3h ret % | DD % | orders | SL | early | dest. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline (= committed) | 3.012 | 0.538 | 54 | 0 | 20 | 1 | 2.961 | 0.555 | 68 | 0 | 25 | 6 |
| **G1 adverse 0.5·D** | 2.961 | 0.465 | 51 | 0 | 18 | 8 | 3.112 | 0.479 | 64 | 0 | 23 | 8 |
| G1b adverse, occupied only | 2.991 | 0.733 | 85 | 0 | 49 | 21 | 2.867 | 0.622 | 89 | 0 | 45 | 21 |
| G2 favourable 0.5·D | 2.884 | 0.662 | 61 | 2 | 26 | 3 | 2.679 | 0.593 | 58 | 3 | 14 | 1 |
| G3a never early close | **0.906** | **1.881** | 39 | 13 | 0 | – | **1.307** | **2.070** | 46 | 12 | 0 | – |
| G3b no re-entry 1 bar after early close | 2.962 | 0.538 | 52 | 0 | 25 | 5 | 2.942 | 0.555 | 65 | 0 | 28 | 10 |
| G3c no re-entry within 6 bars | 2.717 | 0.538 | 51 | 0 | 25 | 3 | 2.756 | 0.555 | 56 | 0 | 26 | 6 |

The calibration rows are in `G_stress_out.json`. They tell the same story: G3a P1 1.199/1.219 vs 3.117/0.418.
All arms stay activity-eligible.

Reading the stress table:
1. The requested adverse stress produces no SL and moves P1 by only −0.05 pts. Only3h even improves. A level shift does
   not remove the true future *shape* from the report, so the adverse arm does not test early close under
   uncertainty. Destructive closes still rise from 1 to 8.
2. An optimistic reporter (G2) is the arm that produces SL exits (2 and 3).
3. Early close is the return engine: removing it costs 2.1 pts (P1) and 1.65 pts (Only3h) and triples the DD.

**Rotation (request item 5).**
- Ledgers: 806 of 1,173 trades open at the first permitted bar after the previous exit.
- P1 in confirmation: 36 such re-entries. 15 follow an early close, all on the opposite side. 21 follow a TP (5 opposite,
  16 same side). None follows a forced flat, because the week ends.
- The trades opened one bar after an early close carry 38.8 % of P1's confirmation net (1.17 of 3.01 pts). For
  Only3h it is 40.5 % (1.20 pts). The full per-policy distribution by side, previous reason and gap is in
  `R_reentry_out.json`.
- But the full-trajectory ablation G3b removes those immediate re-entries and costs only **0.05 pts (P1) / 0.02 pts
  (Only3h)**. A 6-bar cooldown costs 0.29 / 0.20 pts.
- Re-entries per policy and panel. A re-entry is any trade after the first of its week; "gap" = entry bar minus
  the previous exit bar, and 1 is the earliest permitted. There are no re-entries after a forced flat, because the
  week ends. The last two columns give the share of net and the return points carried by trades opened one bar
  after an early close ():

| Policy / panel | re-entries | gap1 after early (opp/same) | gap1 after TP (opp/same) | gap1 after SL | gap 2-3 | gap 4-10 | gap >10 | net share of gap-1-after-early trades | pts |
|---|---|---|---|---|---|---|---|---|---|
| P4|calibration | 54 | 32/2 | 4/11 | 0 | 5 | 0 | 0 | 60.5 % | 0.730 |
| P4|confirmation | 44 | 15/4 | 2/19 | 0 | 4 | 0 | 0 | 40.4 % | 0.497 |
| P1|calibration | 56 | 25/0 | 8/10 | 0 | 10 | 3 | 0 | 43.8 % | 1.365 |
| P1|confirmation | 42 | 15/0 | 5/16 | 0 | 4 | 2 | 0 | 38.8 % | 1.169 |
| P3|calibration | 52 | 24/1 | 7/9 | 0 | 8 | 3 | 0 | 46.6 % | 1.022 |
| P3|confirmation | 37 | 12/2 | 5/13 | 0 | 4 | 1 | 0 | 35.0 % | 0.726 |
| P2|calibration | 52 | 24/1 | 7/9 | 0 | 8 | 3 | 0 | 46.7 % | 1.367 |
| P2|confirmation | 37 | 12/2 | 5/13 | 0 | 4 | 1 | 0 | 35.0 % | 0.971 |
| P5|calibration | 55 | 31/3 | 4/11 | 0 | 6 | 0 | 0 | 59.7 % | 0.410 |
| P5|confirmation | 38 | 13/4 | 2/19 | 0 | 0 | 0 | 0 | 40.2 % | 0.283 |
| equal_guarded|calibration | 20 | 7/0 | 7/3 | 1 | 2 | 0 | 0 | 17.1 % | 0.257 |
| equal_guarded|confirmation | 24 | 6/0 | 4/14 | 0 | 0 | 0 | 0 | 21.5 % | 0.462 |
| greedy|calibration | 31 | 18/2 | 6/3 | 0 | 2 | 0 | 0 | 40.5 % | 0.890 |
| greedy|confirmation | 28 | 5/2 | 5/12 | 0 | 4 | 0 | 0 | 18.8 % | 0.411 |
| only_12|calibration | 38 | 18/7 | 10/1 | 0 | 2 | 0 | 0 | 53.7 % | 1.101 |
| only_12|confirmation | 28 | 6/2 | 4/12 | 0 | 4 | 0 | 0 | 23.5 % | 0.472 |
| only_24|calibration | 22 | 20/0 | 0/0 | 0 | 2 | 0 | 0 | 52.7 % | 0.779 |
| only_24|confirmation | 23 | 7/0 | 3/7 | 0 | 6 | 0 | 0 | 30.5 % | 0.538 |
| only_3|calibration | 65 | 33/5 | 9/16 | 0 | 2 | 0 | 0 | 53.4 % | 1.607 |
| only_3|confirmation | 56 | 18/1 | 6/25 | 0 | 6 | 0 | 0 | 40.5 % | 1.198 |
| only_6|calibration | 64 | 39/5 | 4/13 | 0 | 3 | 0 | 0 | 66.1 % | 1.896 |
| only_6|confirmation | 43 | 15/2 | 4/18 | 0 | 4 | 0 | 0 | 44.5 % | 1.312 |

- Re-entries per policy and panel. A re-entry is any trade after the first of its week; "gap" = entry bar minus
  the previous exit bar, and 1 is the earliest permitted. There are no re-entries after a forced flat, because the
  week ends. The last two columns give the share of net and the return points carried by trades opened one bar
  after an early close (`<scratch>/R_table.md`):

| Policy / panel | re-entries | gap1 after early (opp/same) | gap1 after TP (opp/same) | gap1 after SL | gap 2-3 | gap 4-10 | gap >10 | net share of gap-1-after-early trades | pts |
|---|---|---|---|---|---|---|---|---|---|
| P4|calibration | 54 | 32/2 | 4/11 | 0 | 5 | 0 | 0 | 60.5 % | 0.730 |
| P4|confirmation | 44 | 15/4 | 2/19 | 0 | 4 | 0 | 0 | 40.4 % | 0.497 |
| P1|calibration | 56 | 25/0 | 8/10 | 0 | 10 | 3 | 0 | 43.8 % | 1.365 |
| P1|confirmation | 42 | 15/0 | 5/16 | 0 | 4 | 2 | 0 | 38.8 % | 1.169 |
| P3|calibration | 52 | 24/1 | 7/9 | 0 | 8 | 3 | 0 | 46.6 % | 1.022 |
| P3|confirmation | 37 | 12/2 | 5/13 | 0 | 4 | 1 | 0 | 35.0 % | 0.726 |
| P2|calibration | 52 | 24/1 | 7/9 | 0 | 8 | 3 | 0 | 46.7 % | 1.367 |
| P2|confirmation | 37 | 12/2 | 5/13 | 0 | 4 | 1 | 0 | 35.0 % | 0.971 |
| P5|calibration | 55 | 31/3 | 4/11 | 0 | 6 | 0 | 0 | 59.7 % | 0.410 |
| P5|confirmation | 38 | 13/4 | 2/19 | 0 | 0 | 0 | 0 | 40.2 % | 0.283 |
| equal_guarded|calibration | 20 | 7/0 | 7/3 | 1 | 2 | 0 | 0 | 17.1 % | 0.257 |
| equal_guarded|confirmation | 24 | 6/0 | 4/14 | 0 | 0 | 0 | 0 | 21.5 % | 0.462 |
| greedy|calibration | 31 | 18/2 | 6/3 | 0 | 2 | 0 | 0 | 40.5 % | 0.890 |
| greedy|confirmation | 28 | 5/2 | 5/12 | 0 | 4 | 0 | 0 | 18.8 % | 0.411 |
| only_12|calibration | 38 | 18/7 | 10/1 | 0 | 2 | 0 | 0 | 53.7 % | 1.101 |
| only_12|confirmation | 28 | 6/2 | 4/12 | 0 | 4 | 0 | 0 | 23.5 % | 0.472 |
| only_24|calibration | 22 | 20/0 | 0/0 | 0 | 2 | 0 | 0 | 52.7 % | 0.779 |
| only_24|confirmation | 23 | 7/0 | 3/7 | 0 | 6 | 0 | 0 | 30.5 % | 0.538 |
| only_3|calibration | 65 | 33/5 | 9/16 | 0 | 2 | 0 | 0 | 53.4 % | 1.607 |
| only_3|confirmation | 56 | 18/1 | 6/25 | 0 | 6 | 0 | 0 | 40.5 % | 1.198 |
| only_6|calibration | 64 | 39/5 | 4/13 | 0 | 3 | 0 | 0 | 66.1 % | 1.896 |
| only_6|confirmation | 43 | 15/2 | 4/18 | 0 | 4 | 0 | 0 | 44.5 % | 1.312 |

- So freeing the slot adds little. The synthetic return comes from closing before foreseen stops and gaps, and the
  slot is simply re-used one bar later. The objective adds no fictitious missed-opportunity cash: the returns equal my
  own equity recomputation (D).

## H. Question 10: payoff matrices at P3

`FH_out.json` → `H`. The diagnostic representative is P3 (18c72e613b22). I recomputed the 4 × (2×2) matrices and the
8 unilateral on/off deviations natively on calibration, and all match REPORT exactly (max difference 0.0).
- **ε = 0.005858531** (disabling 12h-buy; objectives 2.2397/−0.2962). The activity-feasible ε is the same.
- Disabling 24h-buy or 24h-sell has an exact 0 effect.
- The largest loss comes from disabling 3h-sell (−0.302).

So P3 is not an on/off Nash point in this restricted library, as Musashi states. This is not a global certificate.

## Findings

Severity: BLOCKER / MAJOR / MINOR / INFO. In the Verdict column, VERIFIED, REFUTED and UNVERIFIED are mutually
exclusive.

| ID | Sev | Claim | Verdict | Reproducible witness | Affected conclusion | Minimal correction | Rerun scope |
|---|---|---|---|---|---|---|---|
| DA01 | INFO | Export, digests, receipt chain and population fitness authentic | VERIFIED | B_export (10/10 byte-equal), B2_digests_out.json | none | – | none |
| DA02 | INFO | Receipts are true evaluations | VERIFIED | B3_reeval_all_out.json (17,018 re-evaluated, reverse order, 0 mismatches) | none | Add a sampled re-evaluation to the verifier (it checks only the chain). | verifier only |
| DA03 | INFO | 632 policies / 342 vectors / five representatives; eligibility never removes a nondominated vector | VERIFIED | C_frontier.py (Kung D&C + brute force) | none | – | none |
| DA04 | MINOR | Representatives are evenly spaced along the frontier | REFUTED[spaced in policies, not vectors: distinct ranks 0/37/126/218/341; P1/P2/P3 chosen from 64/44/10-genome tie groups by SHA key; confirmation range of P1's group 2.990–3.020, P2's 2.767–2.778] | C_frontier_out.json, G_stress_out.json G4 | 137's "five representatives"; no dominance conclusion changes | `tools/optimize_dynamic_account.py:200-201`: rank distinct vectors, and confirm and report every genome of the representative's group (min/max), or the lexicographically smallest genome | FRONTIER.json, REPLAYS (representatives), REPORT/SUMMARY, 137 tables |
| DA05 | INFO | Native cash, sizing, margin q·0.1·B at leverage 50, quote side, slippage, commission, funding timing, integer units, MTM DD, gates, no overlap, no entry-bar close | VERIFIED | D_accounting_out.json (1,173 trades; 2.8e-14 USD, DD 1.8e-14, 0 mismatches) | none | – | none |
| DA06 | INFO | Two bar-107 entries (Fri 09:00 fill) break the 12 h gate | REFUTED[fill 09:00 < session_close−12h = 10:00 and flat at 10:00, so it satisfies `entry_time_rule`; P4/P5 calibration, +1.82/+1.03 USD] | R_reentry_out.json `entries_on_bar107`, D gates 0 | none | Optional variant: refuse entries whose only exit is the forced flat (absolute.py:74) | none |
| DA07 | MINOR | Mean native weekly Sharpe is a meaningful descriptor | REFUTED[6 daily returns per week × sqrt(365); cross-week Sharpe·sqrt(52) reverses P1 vs Only3h (17.8 vs 20.3)] | D_accounting_out.json `weekly_sharpe_sqrt52` | 137 Sharpe column | Report cross-week mean/SD and Sortino with n; label sqrt(365)-on-6-days as not annualizable | REPORT/SUMMARY, 137 |
| DA08 | INFO | Return / win % / TP / SL / FF / early tables | VERIFIED | D_accounting_out.json per_policy | none | – | none |
| DA09 | MINOR | F07 fixed | REFUTED[`app/protection_reference.py:22` unchanged; 0 affected bars (24,224) in this experiment] | D F07_boundary, E max_errors | none here | Use `h >= tp+through` / `l <= tp−through` verbatim | none for 137; future panels with tick-quantized prices |
| DA10 | MINOR | F08 fixed | REFUTED[no terminal close-time spread; 0 affected (constant spread)] | D spread assertion | none here | Carry the close-time spread in PathReport | any hour-dependent-spread panel |
| DA11 | MINOR | F10 fixed | REFUTED[truncated self-consistent report accepted, `close_hold_controller.py:74-78`; 0 affected (deadline = forced flat on 23,710 ticks)] | E_adversarial_out.json `F10_truncated_report_accepted` | none here | Bind the deadline to the position calendar in `validate` | none |
| DA12 | MINOR | Stale/future reports cannot pass the panel route | REFUTED[`dynamic_account_panel.py:180,:184` pass `int(r.issued_at)`; a b−1 report runs under verify=True (1.795→1.881 %)] | E_adversarial_out.json `panel_with_stale_reports_verify_true` | question 3 "cannot bypass"; committed numbers unaffected | decision_time = `int(e.v[b])+HOUR` in both calls; `assert r.issued_at == decision_time` in `report`/`close_values`/`candidate_bids`; give the cache adapter `issued_at` | rerun verifier (expect byte-equal) |
| DA13 | MINOR | Per-unit cache keyed exactly | REFUTED[key `:157` omits entry price; a hit with a changed entry is served stale (check `:160` only on a miss); 0 affected (1.3e-15)] | E_adversarial_out.json `cache_hit_changed_entry_served_stale` | none here | Add entry price (and units-invariance assertion) to the key, or check geometry on every hit | rerun verifier |
| DA14 | INFO | A causal consumer cannot use the synthetic cache | VERIFIED[refused when allow_synthetic=False]; the separation is a constructor flag, not a capability | E_adversarial_out.json | – | Separate the class/module for the synthetic plane | none |
| DA15 | MAJOR | The panel cannot test early-close-before-SL because positions reach TP / forced flat | REFUTED[SL-destined positions are frequent: P1 conf 14/20 early closes pre-empt a native SL; reports = true path + ~1 pip/sqrt(h), bias sign from `spec['mirror']` (`:95`), noise shared by all worlds (`:92`)] | E_replicate_out.json `early_close_counterfactual`, G_cluster_out.json | 137 finding 4; the reason it gives for the "0 SL" | State that the reports foresee the SL-first paths; add per-world independent noise, a label-independent bias sign, and a quality ladder (noise / AR(1) / omitted SL mode) | new panel version; full search optional |
| DA16 | MAJOR | 12 independent weeks per panel | REFUTED[rng keyed (seed, pattern) at `:47`: 3 independent draws per panel; mirror pairs exact; report noise identical across worlds] | G_cluster_out.json | significance language, bootstrap n, "independent-account weeks" | Key the rng by (seed, pattern, volatility, mirror-pair); draw report noise per world; add ≥ 10 draws per family | new panels and confirmation; search if calibration changes |
| DA17 | INFO | P2 dominated by Only6h; P3 by Greedy | VERIFIED; P3 is also dominated by equal_guarded (2.1518/0.4034), not stated | FH_out.json DOM | 137 finding 2 incomplete | Add the sentence | 137 text only |
| DA18 | INFO | P2/P3 identical actions, difference pure size | VERIFIED (traces + ledger events identical in both panels; q ratio 0.7510) | FH_out.json P2P3 | – | – | none |
| DA19 | MINOR | 11 independent genes each reach their computation | VERIFIED for reach; the uncertainty penalty is SHARED by entry score and close z (`:171`), undeclared; w12/w24 inert at P1 | FH_out.json F table | gene-reach claim, gene interpretation | Declare the coupling or split into `entry_uncertainty` and `close_uncertainty` | search if split |
| DA20 | INFO | Matrices, 8 deviations, ε = 0.005858531 | VERIFIED (0.0 difference) | FH_out.json H | – | – | none |
| DA21 | MAJOR | Results show a coordinating allocation policy | UNVERIFIED[return is driven by foreseen early closes: no-early-close P1 0.906/1.881 vs 3.012/0.538; rotation adds only 0.05 pts] | G_stress_out.json G3a/G3b/G3c, R_reentry_out.json | any reading of P1–P5 as allocation skill; gene priors | Report G3-type ablations with every frontier result | report only |
| DA22 | MAJOR | Robustness to adverse reports | UNVERIFIED[declared 0.5·D adverse level shift keeps the future shape: P1 −0.05 pts, 0 SL, destructive 1→8; optimistic arm yields SL 2/3] | G_PROTOCOL_PROSPECTIVE.md, G_stress_out.json | question 8/9 robustness | Shape-destroying stresses: AR(1) report errors, lagged reports, omitted SL mode, quality ladder with break-even | stress panel |
| DA23 | MINOR | 128 focused tests | UNVERIFIED[set not listed; named 4 files 25 passed, broad 14 files 132 passed] | A_tests*.txt | 135 | List the test node IDs | none |
| DA24 | MINOR | Question 4: protection pre-emption, gaps and ambiguous bars exercised at account level | REFUTED[0 of 564 closes pre-empted; ≤ 1 SL; no near-boundary TP] | Q4_counts_out.json | question 4 | Account fixtures with a gap through SL on a CLOSE tick, ambiguous SL/TP bars, near tp+through | new fixtures |
| DA25 | INFO | No same-tick close/reopen | VERIFIED; 806/1,173 re-entries at the first permitted bar (P1 conf: 15 opposite after early close) | R_reentry_out.json | – | – | none |
| DA26 | INFO | Sizing symmetric by side | REFUTED[the positive-only funding reserve makes long q < 1 early in the week (e.g. 0.783) while shorts get q = 1; it follows the contract, but buy/sell P&L are not size-comparable] | D ledgers (q) | buy/sell comparisons | Report side-normalized net per R | report only |

No BLOCKER: no defect changes a committed number.

## Answers to the 11 questions

1. **Digests, receipts, fitness, frontier.** All recomputed: DA01–DA03, plus a full re-evaluation of 17,018
   candidates. My Kung frontier gives 632/342, the same key set and order, and the same representatives. Duplicate
   vectors are retained, and 0 ineligible vectors would be nondominated. The hash tie-break within duplicate groups is
   DA04.
2. **Cash path.** Exact (DA05): DD is mark-to-market, financing falls in (entry, exit] with exit at the open or bar end,
   spread is quote-side and not double-charged, stop/market/TP slippage are correct, commission is charged per side,
   units are integer, margin = q·0.1·B at leverage 50, there is no daily cap, the 12 h gates hold and nothing overlaps.
3. **Cache vs actual units.** Equal to 1.3e-15 on all 23,710 occupied ticks, and all decisions are reproduced by my
   independent code. Changed profile, side or funding change the valuation. A changed entry on a cache hit is served
   stale (DA13, latent). The full route refuses stale and future reports only if it is given the environment clock; the
   panel gives it the report's own clock (DA12). The cache cannot be reached when `allow_synthetic=False`.
4. **Occupancy and pre-emption.** expected_duration uses first passage, and the native order (protection at the open
   before CLOSE) is respected. There is no entry-bar voluntary close. Pre-emption never occurs (0 of 564), gaps are
   foreseen, and ambiguous bars occur at most once (DA24).
5. **Opportunity costs.** Alternatives are complete native trajectories with common exogenous randomness, and no
   unrealized counterfactual enters cash. Locally optimal closes vs global allocation: freeing the slot is worth only
   0.02–0.05 pts in confirmation, while early close itself is worth 1.6–2.1 pts (DA21). Without early close P1 keeps only 0.91 %/week (DD 1.88 %), so the allocation
   layer by itself carries a minority of the synthetic return under these reports.
6. **Controls.** P2 is dominated by Only6h and P3 by Greedy and equal_guarded. P1 − Only3h = +0.050, CI [−0.076, +0.212],
   4 of 12 weeks positive; not significant and driven by one mirror pair (DA16, DA17). P4 and P5 are not dominated in
   confirmation.
7. **Size-only trade-offs.** P2 and P3 are identical in actions and differ only in exposure (DA18). All 11 genes reach
   their computation. Uncertainty is coupled into two computations, and w12/w24 are inert at P1 (DA19).
8. **Report errors and support.** Reports are the ideal path plus a small antithetic error, with the bias sign set by the
   mirror label and noise shared across worlds. Each panel has 3 independent draws (DA15, DA16). The declared limits
   hold; there is no learnability evidence.
9. **Early close.** Of P1's 20 confirmation early closes, 14 pre-empt a real SL, 5 a worse forced flat and 1 a TP. All of
   them are foreseen. With the 0.5·D adverse shift, destructive closes rise to 8 but no SL appears. The optimistic
   reporter produces SL exits. The 131/133 fixed-position fixtures still lack intrabar, near-boundary and
   omitted-SL-mode witnesses. The integrated adverse witnesses missing at account level are: AR(1) or lagged reports, an
   omitted SL mode, an optimistic reporter, gap-through-SL on a CLOSE tick, and ambiguous bars (DA22, DA24).
10. **Payoff matrices.** The four matrices and eight deviations reproduce exactly, with ε = 0.005858531 and the same
    feasible ε (DA20). This is not a global certificate.
11. **Locks and hidden inputs.** No regime lock exists: the close rule ignores the opener's horizon and native protection
    always acts. No scenario name enters a decision except the mirror sign inside the report fixture (DA15). No agent
    state or learned-model input is read. SL/TP geometry is fixed (median profiles, buffer 1 pip, ρ = 0.005).

## I. Satoshi-lane relevance

**(1) How Musashi's arbiter and panel differ from the C3Z full_f4 strategy**

| Aspect | Musashi (agent-multi 935ee95c) | Satoshi C3Z full_f4 |
|---|---|---|
| Inputs | One shared PathReport per tick: S = 3 joint OHLC future paths to the forced flat (truth + bias ± antithetic noise), plus the signed 6-bar past efficiency. Bids are derived inside the harness by valuing each (h, side) order on those paths. | Market-only heads per tick: the multi-resolution ZigZag hierarchy per level, first-passage / best-continuation grids BC[side, cell], a quality ladder with AR(1) noise (φ = 0.587) and a Kalman filter, truth by replay. |
| Flat menu | 8 absolute (h ∈ 3/6/12/24, side) orders + NOP. Unique positive maximum of `w_h·gain/R − λ·unc/R + κ·d·eff·h/24 − τ·dur/24 − θ`; ties abstain; no reversal. | Absolute (h, side) orders (coordinator ruling) with horizon_rule finest/coarsest, EV side comparison, NOP veto, no-open-at-high-risk gates (W15/W17), optional reversal variant. |
| Sizing | q = c·min(1, ρB/L1), ρ = 0.005, c = size gene ∈ [0.25, 1]. L1 includes market+stop slippage, round-trip commission and a **positive-only funding reserve to the forced flat**, so long sizes are smaller early in the week. | Same contract formula, with c and ρ fixed in the DEAP. Whether L1 carries the funding reserve must be checked; if it does not, the two lanes size longs differently. |
| Closer | Position-conditioned common-action valuation over sampled joint paths: `adv = min_j(mean − z·sd)/R > max(margin, 1e-12)`, persistence 1–3, urgent 0.25 R. Ties HOLD. z is shared with the entry score. | Market-only closer head + Bayes comparator, EPS_CLOSE band (close unless hold beats close by more than EPS), Kalman filter upstream. Ties CLOSE. |
| Search | NSGA-II, 2 objectives (mean weekly return, worst MTM DD), activity ≥ 3/week, 20 × 300 × 3 seeds, 12 calibration weeks. | Sharpe-only DEAP (pop 16 × 30) constrained by sealed witnesses on 48 worlds + hard pairs, 21 strategy genes, risk fixed. |
| Worlds | One 120-bar week, constant spread per world (tied to the volatility label), one deterministic gap, 3 independent draws per panel. | 869 base cells × 5 variants of 3-year hourly real-calendar worlds, hour-dependent spreads, sealed witnesses W0–W17 and E0–E8 (O*, regret, forced-entry episodes). |

**(2) Feasibility of cross-running**

- **Musashi's P1 through Satoshi's behaviour matrix: feasible, with an adapter.**
  - (a) **Multi-week calendar.** World assumes one segment (`cal.segments[0]`, `self.end` from `ff[0]`, rolls of segment 0).
    Generalize it per segment: report deadline = that segment's forced flat, `end` and rollovers per segment.
  - (b) **Report adapter from the matrix world.** At the ideal rung, the true OHLC from b+1 to that segment's flat
    (exactly `World.report`). At the noisy rungs, our AR(1) and conditional-mean error model with per-world draws, not
    the shared antithetic noise.
  - (c) **Spreads.** Hour-dependent spreads at bar opens. This makes **F08 live**: a terminal close-time spread is
    required, so fix F08 first.
  - (d) **Inputs and order adapter.** Funding arrays from our calendar; `propose()` geometry from PROFILES.json (already
    shared).
  - (e) **Engine parity.** Run on heuristic-strategy `AbsoluteBarEnv` e0a1c2b, or prove parity with our full_f4 engine
    (our S2 560-case parity is the template).
  - (f) **Record exporter.** Map Musashi's ledger and trace to our record schema (net σ/trade mean and median, hold bars,
    horizon mix, refusals by reason, levels_clipped, exit reasons) so W0–W17 read it.
  - (g) **Forced-entry E0–E8 episodes.** `World.run` cannot start with an open position. Add an entry point that seeds a
    PositionView, the controller state and a native position (the 131 fixed-position corpus already does this for the
    controller alone).
  - (h) **Fixes before running.** DA12 (environment clock) and DA13 (cache key), so stale reports cannot enter
    silently over multi-year runs.

  Cost estimate: about 0.008 s per world-week, so roughly 1–2 CPU-hours for 4,345 worlds × 156 weeks.
- **Satoshi's champion through Musashi's 12 + 12 panels: feasible but degenerate unless history is added.**
  - The 120-bar week gives the coarse ZigZag levels (e.g. rev800) almost no pivots, so C3Z's coarse gates degenerate.
    Prepend a warm-up history generated with the same seed. Otherwise, report C3Z under the panel's foresight reports
    via a ZigZag-from-truth head.
  - Execute on `AbsoluteBarEnv`: map C3Z's (h, side) order to `AbsoluteOrder` via `propose()`, and its closer to
    HOLD/CLOSE.
  - Score on Musashi's objectives (mean weekly return, worst MTM DD, activity), not our Sharpe-only fitness, which is
    not meaningful on 6-day weeks (DA07).
  - Use the 6-pair cluster bootstrap (DA16).

**(3) What a shared strategy should take from Musashi's components**

- **Adopt as-is:**
  - The native AbsoluteBarEnv execution and accounting path (independently exact to 2.8e-14).
  - `propose()` / `levels()` geometry and sizing (already the shared contract).
  - `objectives()` MTM drawdown.
  - The chained Receipts, durable search and verifier pattern (byte-exact export, 0 of 17,018 mismatches).
  - The Pareto frontier routine (verified).
  - Full-trajectory counterfactuals with common exogenous randomness, with no fictitious missed-profit cash.
  - The on/off payoff-matrix ε as a diagnostic.
- **Adopt with adaptation:**
  - `value_paths` / `CloseController`: after F07/F08/F10, with the decision time from the environment clock (DA12) and a
    signed tie band so our EPS_CLOSE CLOSE-favouring policy is expressible. Keep z separate from the entry score (DA19).
  - The arbiter score `choose()`: as an allocation layer over our (h, side) menu, fed by a joint-path adapter from our
    first-passage / ZigZag heads, which is where the modelling risk sits.
  - The per-unit cache, with the entry price in the key (DA13).
  - The representative rule, on distinct vectors with group ranges (DA04).
  - The 3-pattern panel as an extra confirmation family, but with per-world independent noise and draws, a
    label-independent bias sign, ≥ 10 draws per family, and optimistic / AR(1) / omitted-SL-mode reporters (DA15, DA16, DA22).
- **Do not adopt:**
  - The foresight report fixture as a proxy for learned outputs.
  - The P1–P5 genes as priors for our DEAP: margin ≈ 0 and persistence 1 encode clairvoyance, and P2/P3 differ only in size.
  - The per-week daily Sharpe × sqrt(365) as fitness or descriptor.
  - The 12-week panel as sufficient confirmation, or "0 SL exits" as early-close evidence.
