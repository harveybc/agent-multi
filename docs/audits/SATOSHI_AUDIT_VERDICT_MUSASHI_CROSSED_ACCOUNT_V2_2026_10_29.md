# Independent audit: Musashi's crossed-account v2 (agent-multi master @5ef458ec)

Date 2026-10-29. Independent auditor for Satoshi (expert-guided lane).

Object:
- agent-multi `5ef458ec`, with agent code `e889a2d4`;
- native engine heuristic-strategy `e0a1c2b962f4`;
- run identity `4415bd3fdab307bf0869051a37f8a69035d1a727a6b642cb9afd880aa0725455`.

The audit request is `docs/audits/SATOSHI_AUDIT_REQUEST_CROSSED_ACCOUNT_V2_2026_10_29.md`.

Placeholders used in this report:
- `<am>` is a new detached worktree of agent-multi at 5ef458ec.
- `<hs>` is the reused detached heuristic-strategy worktree at e0a1c2b.
- `<scratch>` is this audit's scratch directory:
  - `code/` holds every script I wrote;
  - `logs/` holds every command output;
  - `out/` holds the JSON results;
  - `run/` holds the read-only copy of the public evidence, prepared as the request's reproduction section specifies.

## Verdicts

| Area | Verdict | Basis |
|---|---|---|
| Execution / accounting | **ACCEPT_WITH_CHANGES** | I re-executed **all 27,360 confirmation accounts** on fresh native worlds: 27,000 archived plus the 360 post hoc ones. **0 mismatches.** My independent ledger code found **0 errors** in cash, costs, financing, integer units, the q·0.1·B margin, the gates and mark-to-market DD. The changes are tests only: the panel never exercises the terminal-spread repair (RR-N2 is still non-discriminating), and the RR-N1/N3/N4 hardening is still open but latent. No rerun is needed. |
| Search / reproducibility | **ACCEPT** | Every digest in ANALYSIS.json reproduces **byte-equal**, as does POST_HOC_P1_ONLY6. The 1,800-account P1 replay is exact. All three DEAP histories re-execute exactly from the receipts with 0 new evaluations. A brute-force nondominance check returns the same 199 policies and 180 vectors, and the representative rule holds. |
| Confirmation / statistics | **ACCEPT_WITH_CHANGES** | Every summary and every paired cluster difference reproduces with my own code. Changes required in the interpretation: <br>– 9 of the 35 P1-vs-control contrasts do not survive a Holm adjustment. <br>– Report errors are shared across the spread and mirror pairs. <br>– Two thirds of P1's correlated return comes from the overlap family. <br>– At size=1, P3 and P4 are dominated by P2. <br>– 41 % of P1's early closes are destructive against holding. |
| Strategy/output interface for learned-model selection | **REVISE** | <br>– The fitness rung keeps the true future path shape (DA15 persists). <br>– The only causal rung (omitted reversion) loses for every active policy. <br>– The 3-path dispersion is uncalibrated on every non-ideal rung: band coverage is 35–55 % under correlated errors and 1–9 % under omitted reversion. <br>– A causal error detector exists and is strong (median track-record z of 47.6 against 1.3), but the policy cannot read it. <br>– No horizon-specific positive control exists (97.9 % of entries use 6 h). <br>– No learned adapter exists. <br>These gaps need NEW experiments; fixing this run will not close them. |

## Setup, constraints and environment

- **Interpreter:** the `trading-stack` environment, matching the manifest exactly (Python 3.12.13, NumPy 2.5.1, pandas 3.0.3, DEAP 1.4).
- **Environment variables:** `CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`. `PYTHONDONTWRITEBYTECODE=1` keeps the worktrees free of bytecode.
- **Job control:**
  - Every heavy step ran under `crispdm-run -q -W 3600 -m 2400M -t 14400s -n sat-cx-<step>`, one job at a time, and every job was admitted.
  - The largest footprint was Musashi's analyzer with `--replay-p1` and four workers. Its main process peaked at 0.96 GB and the run took 77 s. It fit under the 2400M cap, so **no subset fallback was needed**.
  - My own jobs peaked at 0.3 GB or less.
- **Worktree state:**
  - `<am>`: `git status` was empty before and after.
  - `<hs>`: HEAD was e0a1c2b962f41039bb59a76614b70e326f2cf0c1 before and after, with an empty `git status`. No file is newer than the start of this audit; the ignored `__pycache__` dates from an earlier audit.
- **Things I did not do:** no process was killed, nothing was committed or pushed, and no Musashi service, queue, worker or timer was touched. Only synthetic 2020-01 bars were read.

## A. Tests (`logs/A_tests.log`, `out/focused_suite.xml`)

- The 16 preflight files from the queue's `TESTS` list plus `test_crossed_account_analysis.py`: **166 passed, 0 skipped, 0 failed** (2.90 s).
- The two files named in the request (`test_crossed_account_analysis.py`, `test_crossed_account_queue.py`): **7 passed.**

## B. Producer analyzer reproduction (`logs/B_*`, `out/analysis_replay/`, `out/analysis_noreplay/`)

1. I copied the evidence into `<scratch>/run` and decompressed only `CANDIDATES.jsonl` and `SEED_{7,19,43}.json`. I removed the analysis outputs and made the directory read-only.
2. All 199 entries in ANALYSIS.json `files` match: compressed sha256 against the evidence, and original sha256 against the decompressed files (`logs/hash_check.txt`).
3. `tools/analyze_crossed_account_results.py --run <scratch>/run --engine-root <hs> --export <scratch>/out/analysis_replay --replay-p1` printed `EXACT_P1_REPLAY` for all five rungs (1,800 accounts).
   - **My ANALYSIS.json is byte-equal to the committed one.**
   - So is `POST_HOC_P1_ONLY6.json.gz`.
   - All 200 exported files are byte-equal to the evidence directory.
   - The analyzer source sha256 is b93060de…, which matches `analysis_source_sha256`.
4. Without `--replay-p1`, the output differs only in `files`, `fresh_native_p1_replays_exact` and `post_hoc_p1_only6`, as expected.

## C. Second implementation (`code/C_stats.py`, `code/C_search.py`, `code/C_deap_replay.py`)

### C1. Summaries

I recomputed all 75 final jobs from the REPLAY ledgers: return, worst DD, orders, buys/sells, wins by direction, early closes and eligibility. The result has **0 mismatches** against the FINAL_*.json files. Ledger cash divided by capital equals `return_pct` for every policy. Full table: `logs/C_summary_table.txt`.

Spot checks against the request's numbers:

| Claim | Recomputed |
|---|---|
| Correlated P1 | 3.32469 % / 1.77887 %; 3,156 orders (3,090 at 6 h, 65 at 3 h, 1 at 12 h); 2,103 AGENT_CLOSE / 858 TP / 132 SL / 63 FF; 721 negative early closes |
| Omitted P1 | −1.81228 % / 15.12537 %; 1,700 of 3,313 orders reach SL; 100 % of entries at 6 h |
| Ideal Only3 | 5.12503 % / 1.00663 %, beating P1 (3.94665 % / 1.25487 %) on both objectives |
| Correlated Only24 | worst DD 1.20191 %, lower than P1's |

The ideal-rung counts are also confirmed: P2–P5 and all three size=1 variants fail the activity threshold there.

### C2. Eligibility, nondominance, ties and the representative rule (`out/C_search.json`)

- I re-verified the receipt chain with my own canonical-JSON digest: 17,028 receipts, 0 breaks, and the tail equals COMPLETE.
- The manifest identity recomputes.
- Eligibility: `eligible == (mean_orders ≥ 3)` and `orders/360 == mean_orders` hold for every receipt. 16,481 receipts are eligible and 547 are not.
- All genes are in bounds and rounded to 3 decimals.
- **Brute-force O(n²) dominance**, independent of the producer's sorted sweep. A point counts as dominated only when another point is at least as good on both objectives and strictly better on one, so exact ties are retained. Result: **199 nondominated keys, identical to FRONTIER.json, with 180 distinct vectors.**
- Duplicate groups among the 180 vectors: 170 of size 1, 8 of size 2, 1 of size 3 and **1 of size 10**.
- The representative ranks are `linspace(0,179,5)` = [0, 44.75, 89.5, 134.25, 179], which gives [0, 45, 90, 134, 179] under both half-even and half-up rounding.
- Choosing the lexicographically smallest genome within each group reproduces the five frozen representatives.
- **P1 is the minimum of the 10-genome duplicate group.** Its members differ only in genes 1, 2, 3, 4 and 6 (for example, weight_24 ranges from 0.291 to 0.351). Those differences are inert on calibration: 24 h is never selected. The P1 genome is therefore not identified by the data beyond the active genes. This is a disclosure item, not a defect.
- **Frontier genes:**
  - close_persistence is 1 in all 199 frontier policies;
  - close_margin has a median of 0.002;
  - the size gene equals 1 in only 20 of the 199;
  - corr(size gene, calibration return) over the frontier is **0.959**.
  As in v1, the frontier is largely an exposure dial (see F).
- **DEAP re-execution** with the producer's `search()` against a frozen receipt set, where any new evaluation would raise: the history equals SEED_7, SEED_19 and SEED_43 exactly, with 0 new evaluations. The original seeds had 5,687, 5,678 and 5,663 new evaluations.
- P2 and P3 never appear in any saved post-selection population (they were evaluated offspring). Seed 43 first saw 84 of the 199 frontier members, against 11 for seed 7 and 25 for seed 19, so the three searches do not converge on one front.

### C3. Paired differences, cluster bootstrap and multiplicity (`out/C_paired.json`, `logs/C_paired_table.txt`)

**Cluster definition, verified in the generator.** The price RNG is seeded by `SeedSequence([seed, pattern_index, 7401])`; envelope, spread and mirror are not part of the seed. Each (seed, pattern) is therefore one independent draw, and its 12 cells are deterministic transforms of it. I reproduced the generator independently (360/360 OHLC arrays equal per bank). Every cluster has 12 paired cells.

My implementation:
- cluster means of the per-account return differences;
- 20,000 percentile bootstrap resamples of the 30 cluster means (`default_rng(777)`; the producer uses 20261029 and 10,000 resamples);
- a t-interval;
- a 100,000-draw sign-flip randomization p-value;
- Holm adjustment over the 35 P1-vs-control contrasts (7 controls × 5 rungs);
- a bootstrap of the **worst-DD difference** (resample clusters, recompute the max-account DD of both policies).

Selected rows. Returns are in percentage points; positive means P1 is better on return. For the worst-DD column, negative means P1's worst DD is lower.

| Contrast | Mean Δ | +clusters | Boot 95 % | Sign-flip p | Holm p | Worst-DD Δ [boot 95 %] |
|---|---:|---:|---|---:|---:|---|
| corr P1 − Only6 | +0.6576 | 24/30 | [0.392, 0.914] | 8e-5 | 0.0011 | −2.339 [−2.78, −1.12] |
| corr P1 − Greedy | +0.9393 | 30/30 | [0.766, 1.130] | <1e-5 | <1e-4 | −0.625 [−1.06, −0.18] |
| corr P1 − Only3 | +0.7798 | 24/30 | [0.498, 1.052] | 2e-5 | 0.0003 | −2.507 |
| corr P1 − Only24 | +1.2491 | 30/30 | [0.873, 1.690] | <1e-5 | <1e-4 | **+0.577 [+0.14, +0.67]** (P1 worse) |
| corr P1 − P1-only6 (post hoc) | +0.0052 | 15/30 | [−0.0096, 0.0205] | 0.52 | – | 0.000 |
| ideal P1 − Only3 | −1.1784 | 0/30 | [−1.485, −0.921] | <1e-5 | <1e-4 | +0.248 |
| ideal P1 − Only6 | −0.7300 | 2/30 | [−0.960, −0.542] | <1e-5 | <1e-4 | −0.021 |
| delayed P1 − Greedy | +0.1152 | 18/30 | [−0.191, 0.408] | 0.47 | 0.95 | −0.063 |
| omitted P1 − NOP | −1.8123 | 13/30 | [−3.338, −0.516] | 0.016 | 0.13 | +15.125 |

- Every point estimate and producer interval in 142 reproduces within bootstrap Monte-Carlo error. For example, P1 − Only6 is +0.65759 [0.38868, 0.91819] (producer) against my [0.392, 0.914]; P1 − P1-only6 is +0.00523 [−0.00943, 0.02070] against my [−0.0096, 0.0205].
- **All seven correlated-rung control contrasts survive Holm.**
- **9 of the 35 do not survive (adjusted p > 0.05):**
  - ideal and delayed P1 − Greedy;
  - delayed P1 − Only6 and P1 − Only12;
  - five omitted-rung contrasts.
- **P1 is worse than a control on mean return in 6 of the 35:** ideal Only3 and Only6; omitted NOP, equal_guarded, Only12 and Only24.
- On worst DD, P1 does not beat Only24 on the correlated rung: the cluster bootstrap excludes 0 in the wrong direction.
- The post hoc only-6 ablation shows that **no multihorizon arbitration benefit is detectable** (p = 0.52; 15/30 clusters positive).

## D. Accounting and contract from archived ledgers (`code/D_replay.py`, `out/D_replay_<rung>.json`)

For every confirmation cell (1,800 cells over 5 rungs), I built one fresh `CrossedWorld` and ran all 15 policies on that same instance, warm, as the queue's persistent shards do. For the correlated rung I also ran the post hoc P1-only6. That makes **27,360 native accounts with `verify=True`.**

Two cross-checks:
- **Exact-dict check:** each full account dict (trace, ledger, costs, Sharpe and cache error included) was compared with the archive using `!=`. Result: **0 mismatches.**
- **Cold check:** for every 10th cell, I built a fresh World per policy (**2,736 cold accounts**). Result: **0 mismatches.**

Independent accounting, re-derived from my own reimplementation of the generated bars and the declared profile constants. The engine's trade records supplied only the TP/SL levels.

| Check (all 27,360 accounts) | Errors |
|---|---:|
| Bars used by the native env equal my generator | 0 |
| Entry quote = open + d·s(t)/2, using my own spread schedule (weekly-open hour; 2× in hour 10–11) | 0 |
| Integer units = floor(q·floor(0.1·B·50/p)), with B = capital + previous nets | 0 |
| Margin = units·p/50 ≤ q·0.1·B, with 0.1 applied once | 0 |
| Exit price by reason (open quote for AGENT_CLOSE; close quote at the flat-time spread for FF; level or gap-open for TP/SL) | 0 |
| First passage: my own bid/ask protect logic places the TP/SL bar and reason exactly, and no earlier hit for AGENT_CLOSE/FF | 0 |
| Spread, slippage and commission per trade | 0 |
| Financing: rollovers at open + 24·k; triple on k = 3; long pays 7/lot, short receives 2.5/lot; window v[entry] < r ≤ exit time | 0 |
| Net = d·u·(exit − entry) − slippage − commission − financing; ledger equals native | 0 |
| Gates: v[entry] − open ≥ 12 h, entry < forced flat, FF exits at bar 107, exits ≤ 107; one position at a time (entry > previous exit) | 0 |
| MTM DD and return from my own equity mark (close quote at close-time spread, accrued financing, spread credit) | 0 |

- **No daily cap:** the maximum number of entries in one business day is 9 on correlated, 14 on omitted, 7 on ideal, 10 on delayed and 9 on missing. Leverage 50 and rel_volume 0.1 are also enforced by the engine's `ABSOLUTE_BUSINESS_CONTRACT`.
- **Report clocks, deadlines and spreads (139 repairs):** 184,896 reports were checked, every available report at every decision bar of every cell:
  - issued_at = first_fill = v[b] + 1 h, and observed_through ≤ issued_at;
  - deadline = ff = v[108];
  - support = 107 − b bars;
  - per-bar spreads equal my schedule;
  - terminal spread = spread at the flat instant.

  Result: **0 violations.** However, **0 of 184,896 reports discriminate** between the terminal spread and the last bar-open spread. The flat instant falls at hour 12 mod 24, while the doubled-spread hour is [10, 11), so the F08 terminal-spread repair is never exercised by this panel (CX-05).
- **Preemption at account level (RR-DA24b):** 450 of the 93,926 CLOSE requests in the archive were preempted by native protections at the open (428 SL, 22 TP), including 11 of P1's 2,114 correlated requests. All of them are native-exact in my re-execution. The behaviour therefore exists and is correct in the data, but no unit test targets it.
- **Cache parity:** the maximum `cache_max_error` over the 27,000 archived accounts is 1.78e-15 risk units.

### Open re-review items: can they affect v2's numbers?

| Item | Status in v2 code | Effect on v2 numbers | Witness |
|---|---|---|---|
| RR-N1 (report/bid caches keyed without execution config) | **Unchanged** since 2318b0e3. `reports` is keyed by b and `bids` by (b, report_identity). | **0 affected accounts.** Each World instance has one immutable spec (quality, spread), and its config is asserted unchanged after all runs. Spread and quality changes always create a new World. Warm-vs-archive: 0/27,360 differ; cold-vs-archive: 0/2,736; warm-vs-cold report identities: 0 mismatches. | `out/D_replay_*.json` |
| RR-N2 (non-discriminating deadline-spread fixture) | **Unchanged** (`test_external_audit_repairs.py` identical to 2318b0e3) | No effect on numbers, but no v2 report discriminates either (above). | D report check |
| RR-N3 (decide() can be fed report-sourced deadlines) | Unchanged API. The panel uses `e.ff[e.pos.seg]` for closes and `env.ff[0]` for bids; the panel has a single segment. | None | D report check (deadline = ff for all 184,896) |
| RR-N4 (adapter values not bound to report identity) | Unchanged | None. All 27,000 archived accounts ran with `verify=True` direct-route parity, max 1.78e-15. | `out/F2_archive_counts.json` |
| RR-N8 (migration note) | No new note; the code is unchanged | – | – |
| RR-DA24b (account-level preemption witness) | Still no dedicated test | None. 450 preemptions occur in the data and are exact. | F2 |

Scope limit: I did not re-execute the 12,600 calibration-control accounts or the 17,028 × 360 DEAP accounts. They run on the same per-spec World structure, so the same RR-N1 argument applies. The queue itself re-checked the five representatives' calibration receipts.

## E. Close quality: fixed-position native counterfactuals (`code/E_close.py`, `out/E_close_<rung>.json`, `out/E_close_rows_<rung>.json`)

**Method.**
1. Every archived P1 account was replayed from its archived trace through the native engine. The trace replay reproduces every ledger exactly (0 mismatches), and my position clone reproduces every executed close exactly.
2. At every executed AGENT_CLOSE, the open position was cloned and evaluated in two ways:
   - **(a) hold:** held passively to its native outcome (TP, SL or forced flat);
   - **(b) close later:** closed at every later executable open up to that outcome. Native open-gap protections apply first, so a gapped SL is the executable value.
3. O* = max of (a) and all (b).

**Classification.**
- Δ = executed − hold.
- **Helpful** if Δ > 0.02σ·units, **destructive** if Δ < −0.02σ·units, **neutral** otherwise, with σ = 0.001 as the panel's sigma.
- Regret = (O* − executed)/R, where R is the engine's modeled cash risk at entry.

I evaluated **all** early closes, not only a sample. The deterministic stratified 300-row sample asked for (150 evenly spaced from the negative-net closes, 150 from the others, ordered by world and entry bar) is reported alongside.

| Rung | Closes | Helpful | Destructive | Neutral | Σ Δ vs hold (pp of mean account return) | Regret/R mean, median, p90 | Executed = O* |
|---|---:|---:|---:|---:|---:|---|---:|
| Correlated, all | 2,103 | 1,220 (58.0 %) | **867 (41.2 %)** | 16 | **+0.678** | 0.586, 0.262, 1.70 | 637 (30 %) |
| Correlated, 721 negative-net | 721 | 441 (61.2 %) | 274 (38.0 %) | 6 | **−0.181** | 0.796, 0.472, 2.18 | 207 |
| Correlated, stratified 300 | 300 | 182 | 118 | 0 | +0.090 | 0.649, 0.298, 1.86 | 83 |
| Omitted, all | 1,277 | 787 (61.6 %) | 489 (38.3 %) | 1 | **−0.519** | 0.802, 0.304, 2.26 | 463 |
| Omitted, 958 negative-net | 958 | 655 | 303 | 0 | −0.454 | 0.818, 0.134, 2.42 | 416 |
| Omitted, stratified 300 | 300 | 159 | 141 | 0 | −0.117 | 0.811, 0.584, 1.93 | 85 |
| Ideal (positive control) | 353 | 351 | **0** | 2 | +0.885 | **0, 0, 0** | 353 (100 %) |

By hold outcome, on the correlated rung:
- the **997 closes of SL-destined positions are all helpful** (+1.783 pp);
- the **599 closes of TP-destined positions are 594 destructive** (−1.072 pp; regret 1.28 R);
- the 507 FF-destined closes are mixed (−0.033 pp).

By family, the net close value on the correlated rung is +0.634 pp for overlap, +0.046 pp for reversal and −0.002 pp for trend.

On the omitted rung, TP-destined closes cost −0.924 pp and SL-destined closes save +0.566 pp. The net is destructive, with a negative Σ in every family.

**Reading.**
- The ideal rung is a clean positive control for the counterfactual tool: every close is optimal (regret 0).
- On the selection rung, early closing adds about +0.68 pp of the 3.32 % mean return relative to holding each closed position. That figure is fixed-position only and ignores the freed capacity for re-entry, which favours closing.
- Early closing also destroys TP-bound value in 41 % of cases.
- "Negative realized close" says nothing about quality by itself: 61 % of those closes are helpful, yet in aggregate they are net-destructive (−0.18 pp).
- Under the confidently wrong causal reporter, closing is net-destructive (−0.52 pp).
- Absence of destructive closes, and optimal exit timing, are both **refuted** for the non-ideal rungs.

## F. Exposure (attack point 5) (`out/D_replay_*.json` `size_traces`, `logs/F_exposure_table.txt`)

**Traces are identical.** On all five rungs, P3, P4 and P5 have **360/360 identical full action traces** against their size=1 versions. Their trade sequences are identical as well: entry bar, exit bar, direction, reason and horizon.

**All of the change comes from q.** The per-trade unit ratio equals the size gene to within compounding:

| Policy | Unit ratio (5 rungs) | Return ratio | Worst-DD ratio | Size gene |
|---|---|---|---|---:|
| P3 | 0.796–0.799 | 0.795–0.803 | 0.799–0.805 | 0.798 |
| P4 | 0.446–0.449 | 0.443–0.454 | 0.450–0.458 | 0.448 |
| P5 | 0.249–0.251 | 0.247–0.254 | 0.251–0.257 | 0.250 |

So **100 % of the return and DD difference between Pk and Pk_size1 is caused by q.** None of it comes from the action sequence.

**At equal size, the low-risk representatives are dominated:**
- On correlated, P2 (2.693 / 1.67053) dominates P3_size1 (2.521 / 1.67053) and P4_size1 (2.544 / 1.67053).
- P5_size1 (2.206 / 1.67049) escapes only by a 0.00004-point DD margin.
- The pattern is the same on missing and on ideal.

`rel_volume = 0.1` is a constant applied once. q = size·min(1, ρB/L1) with ρ = 0.005. At size 1, the mean q is about 0.69–0.75, so the risk budget usually binds.

## G. Coverage and report semantics (`code/G_coverage.py`, `out/G_coverage.json`)

**Generator enumeration**, from my reimplementation, which is bit-equal to `market()` on 720/720 cells:

| Feature | Coverage |
|---|---|
| Families | trend (constant drift plus a sinusoid of amplitude 2–7 pips and period parameter 3–9); reversal (two cuts in bars 25–95, counter-drift of 0.6–1.5× strength); overlap (12/36/96-bar sinusoids, 3–15 pips each) |
| Trend strength | 0.25–1.0 pip/bar |
| Innovations and widths | i.i.d. N(0, 1.8 pip) innovations; per-bar width 0.3–1.6 pip |
| Volatility envelopes | constant, increasing (0.3→1.8) and **decreasing (1.8→0.3)** |
| Gaps | 3 signed per world in bars 15–103, 3.1–18.0 pips (calibration up-share 0.48, confirmation 0.54) |
| Spreads | 0.5 / 3 pips, doubled in hour 10–11 since open (mod 24), weekly-open hour at the base spread |
| Mirrors | exact (2.2 − p, H/L swapped) |
| Missing reports | i.i.d. Bernoulli(0.2) per bar |

Innovation correlations across the 30 draws: mean −0.002, max |r| 0.28 over 120 samples, consistent with independence.

**Banks are distinct:**
- calibration seeds are 50100–50109, confirmation 60100–60109, and smoke 70100 and 80100 (the first two cells of each bank, seed + 20000);
- there is no seed overlap between banks;
- the 364 distinct OHLC hashes (180 + 180 + 4) have no cross-bank collisions.

**Prices are identical across cost and quality transformations.** I hashed the native env's O/H/L/C and timestamps for all 1,800 confirmation worlds (5 qualities × 2 spreads per group). Each of the 180 (seed, pattern, envelope, mirror) groups has exactly **one** hash.

**Report errors are not independent per world.** They are shared as follows:
- **Spread pair:** errors are identical, 12/12 checked on each of the correlated, delayed and omitted rungs. The report RNG key (seed, pattern, envelope, b, stream) excludes spread and mirror, and Musashi's own test asserts this.
- **Mirror pair:** the report is the exact mirror of the original, 12/12 (`paths' = 2.2 − paths` with H/L swapped). The mirror pair is therefore a pure direction-symmetry check, not a second error draw.
- **Envelopes and seeds:** errors differ, 0/12 identical.
- **Missing rung:** every available report equals the correlated report exactly (275/275), and the availability mask is shared across the spread and mirror cells. The missing rung is a pure availability mask on the correlated rung, which is a good paired design.

Net effect: there are 90 independent report-error realizations per bank (30 draws × 3 envelopes), not 360. The 30-cluster bootstrap stays valid because it is coarser.

**Foresight and causality witness.** I shifted every bar from b+2 onward by 30 pips and re-issued the report at b:

| Rung | Reports that changed |
|---|---|
| ideal | 18/18 |
| correlated | 18/18 |
| delayed | 18/18 |
| omitted reversion | **0/18** |

The **omitted-reversion rung is the only causal reporter.** It is a median-drift extrapolation of the last 8 closes with a near-zero band. It is not "the future with the reversal removed".

The correlated rung (the DEAP fitness rung) is the true future OHLC path plus a smooth additive AR error applied equally to O/H/L/C. Gap timing, reversal timing and intrabar shape are preserved. Correlation between predicted and realized close change is **0.62, 0.77, 0.83 and 0.79 at 1, 3, 6 and 12 h**, with sign agreement of 67–78 %. The delayed rung is still future-aware at long horizons (0.73 / 0.82 at 6 / 12 h).

**The 3-path band is uncalibrated** everywhere except ideal. Realized-close coverage at 1, 3, 6 and 12 h:

| Rung | Coverage |
|---|---|
| correlated | 35, 48, 53 and 55 % |
| delayed | 8–10 % |
| omitted reversion | 9, 5, 1.7 and 1.2 % |

The median |error|/sd on the omitted rung is 7 to 51.

## H. Adverse-report failure (`code/H_adverse.py`, `out/H_adverse.json`)

**Anatomy of omitted-rung P1:**
- −1.81 % mean return and 15.13 % worst DD;
- 204 of 360 accounts lose;
- 1,700 SL, 278 TP, 1,277 AGENT_CLOSE and 58 FF exits;
- 100 % of entries at 6 h.

**The loss is concentrated in the overlap family:** −6.01 % mean, 114/120 accounts losing, 1,456/2,155 SL, and the worst DD of 15.13 %. A momentum extrapolator is maximally wrong on oscillations. In contrast, reversal is +0.35 % (worst DD 3.1 %) and trend is +0.23 % (worst DD 2.7 %). The loss is worse at 3 pips (−2.26 %) than at 0.5 pips (−1.37 %).

The correlated rung is mirrored: P1's gain is concentrated in the same overlap family (+6.62 % mean, 0/120 losing accounts, about two thirds of P1's correlated return) against +1.63 % for trend and +1.72 % for reversal.

**A causal error-detection signal is available in principle, and it is strong.** At each P1 entry decision b, I used only reports issued at b−1..b−6 and closes up to b. The track-record statistic z is the mean |realized c[b] − predicted c[b]| divided by those reports' own sample sd.

| Rung | Median z | p90 z | AUC (z → trade loss) | AUC (z → SL) |
|---|---:|---:|---:|---:|
| correlated | 1.32 | 2.23 | 0.51 | 0.53 |
| delayed | 21.4 | 50.8 | 0.77 | 0.81 |
| omitted reversion | **47.6** | 106.6 | **0.75** | 0.71 |

A one-line causal gate on z (anything around 5–10 separates the medians) would suppress nearly all omitted-rung and delayed-rung trading while leaving correlated trading intact. **The policy cannot see this signal.** Its only causal market input is `past_efficiency`, a 6-bar efficiency ratio with P1 trend weight −0.079. Its uncertainty term is the 3-sample sd of the report itself, which is narrowest precisely when this reporter is most wrong.

**Calibration claims.**

Justified:
- A confidently wrong, narrow-band reporter makes every active policy and control lose, with NOP best.
- Losses scale with selectivity and size.
- Failure is family-specific: catastrophic on oscillations, roughly break-even on trend and reversal.

Not justified:
- that the 3-path dispersion measures uncertainty (its coverage is 35–55 % even under correlated errors);
- that the uncertainty penalty gene provides protection;
- any calibration statement about a learned model;
- the magnitude of the failure for real reporters (the stress model is declared, not fitted);
- that market inputs cannot support detection (H shows a simple causal statistic does).

## Mapping of prior MAJOR findings

| ID | Addressed in v2? | How / residual |
|---|---|---|
| DA15 foresight | **Partially** | The label-dependent *systematic* bias is gone: the error is mean-zero AR, and the mirror sign gives equivariance only. There are separate ladder rungs, including a causal one. But the **fitness rung (correlated) still contains the true future path shape**, with pred/real correlation up to 0.83 and the SL-first paths visible. Per-world noise independence holds only across envelopes and seeds, not across the spread or mirror pairs. |
| DA16 only 3 draws | **Yes** | 30 independent draws per bank, cluster bootstrap over 30, and new calibration and confirmation banks disjoint from the smoke banks. Residual: 90 independent error streams, not 360 (G). |
| DA21 early-close attribution | **Not by the producer**; supplied here | v2 reports no ablation without early close. My fixed-position counterfactual (E) shows the source of close value: +1.78 pp saved on SL-bound positions, −1.07 pp lost on TP-bound ones, net +0.68 pp of 3.32, almost all from the overlap family. |
| DA22 adverse stress preserved shape | **Mostly** | The delayed, omitted (causal, shape-destroying) and missing rungs exist. There is no noise or break-even ladder: one φ = 0.6, one error scale. The omitted rung conflates "wrong shape" with "narrow band", so the two cannot be separated. |

## Findings

Severity is one of BLOCKER, MAJOR, MINOR or INFO. Verdict words are exclusive.

| ID | Sev. | Claim (as made by Musashi unless stated) | Verdict | Executable witness | Affected conclusion | Minimal repair | Exact rerun scope | New experiment or fix? |
|---|---|---|---|---|---|---|---|---|
| CX-01 | INFO | 166-case focused suite passes with zero skips; the two named files pass | VERIFIED | `code/run_tests.sh` → `logs/A_tests.log` (166 passed; 7 passed) | – | – | – | – |
| CX-02 | INFO | Analyzer authenticates receipts, the 18,000 fitness entries, the frontier, the representatives, controls, finals and archives; 1,800 fresh P1 replays are exact | VERIFIED | `code/run_analyzer_replay.sh`; ANALYSIS.json and POST_HOC byte-equal; 200/200 files equal | – | – | – | – |
| CX-03 | INFO | Search is reproducible; frontier is 199 policies / 180 vectors; tie rule holds | VERIFIED | `code/C_deap_replay.py` (3/3 histories equal, 0 new evaluations); `code/C_search.py` (brute force 199/180; representatives equal) | – | – | – | – |
| CX-04 | INFO | Native accounting, 1:50, one position, no daily cap, 12 h gates, integer units, all costs, MTM DD are correct (attack point 7) | VERIFIED | `code/D_replay.py` (27,360 warm + 2,736 cold accounts exact; 0 errors in 11 independent checks; max 14 entries/day) | – | – | – | – |
| CX-05 | MINOR | "External clock/deadline exact, terminal spread explicit" (CP04) exercises the 139 terminal-spread repair | REFUTED[0 of 184,896 v2 reports have a terminal spread different from the last bar-open spread: the flat is at hour 12 and the doubled band at [10, 11); the RR-N2 fixture is unchanged since 2318b0e3] | `out/D_replay_*.json` `terminal_discriminating_reports = 0` | Coverage of the F08 repair only; no number changes | Put a schedule boundary at the flat hour in one panel profile and in the unit fixture; assert before/after discrimination | Unit test, plus one optional diagnostic panel | Fix (test) |
| CX-06 | INFO | Cache repairs mean no stale value reaches v2 (RR-N1, RR-N3, RR-N4) | VERIFIED for v2 numbers: 0/27,360 affected. The latent API defects remain unrepaired (code unchanged since 2318b0e3). | D warm/cold/report-identity witnesses; F2 max cache error 1.78e-15 | None for v2 | As in the re-review (config digest in the cache keys; bind values to `report_identity`) | None | Fix (hardening) |
| CX-07 | MINOR | Account-level preemption of a requested CLOSE is witnessed (RR-DA24b) | UNVERIFIED[no dedicated test; but 450 of 93,926 CLOSE requests in the archive are preempted (428 SL, 22 TP), all native-exact on re-execution] | `code/F2_archive_counts.py` | Attack point 7: the model cannot act before an unobservable gap, and the data confirm the engine handles it | Promote one archived preemption (for example P1 correlated) to a regression test | Unit test | Fix |
| CX-08 | MINOR | Separate worlds have different report errors; noise is independent per world (CP03, design 140) | REFUTED[the spread pair has identical errors and the mirror pair has exactly mirrored reports (12/12 each); the missing rung equals correlated wherever available (275/275); 90 independent error realizations per bank, not 360] | `out/G_coverage.json` `report_sharing` | Wording of independence; the 30-cluster bootstrap is unaffected | State the sharing in 140/142; for future panels, key report noise by spread too, or declare the pair as a cost-only contrast | None | Fix (doc); NEW panel if independence is wanted |
| CX-09 | MAJOR | Correlated/delayed/missing reports are future-aware fixtures (disclosed); calibration on the correlated rung measures strategy response, not causal forecasting | VERIFIED, with consequence: the fitness rung keeps the true path shape (pred/real r = 0.62–0.83; 18/18 reports change under a future shift). DA15 persists on the selection rung. | `out/G_coverage.json` `causality`, `foresight_calibration` | Any reading of P1's genes as suited to a causal reporter | Calibrate on causal or shape-destroying reporters with a quality ladder; keep correlated as a diagnostic | New search | NEW experiment |
| CX-10 | MAJOR | "Omitted reversion" models a reporter that misses reversals | REFUTED[it is a purely causal 8-bar median-drift extrapolator with a near-zero band (0/18 reports change under a future shift); band coverage 1–9 %; it is the ONLY causal rung, and every active policy loses on it] | G `causality`; H tables | Attack point 3: v2 has **no** positive result with any causal reporter | Rename it and split the stress into (wrong shape) × (band calibration) factors; add a calibrated causal baseline reporter | Five-rung confirmation of the frozen representatives on the new rungs | NEW experiment |
| CX-11 | MAJOR | 3-path dispersion carries uncertainty used by the entry and close penalty | REFUTED[band coverage is 35–55 % (correlated), 8–10 % (delayed), 1–9 % (omitted); median \|err\|/sd 1.0–1.8 (correlated) up to 51 (omitted)] | G `foresight_calibration` | Attack point 3; any abstention or uncertainty claim | Calibrated sampler (≥ 32 paths, coverage tests) and a causal track-record channel | – | NEW experiment |
| CX-12 | MAJOR | (auditor) A causal error-detection signal exists in principle | VERIFIED: the track-record z is 47.6 (omitted) and 21.4 (delayed) against 1.32 (correlated); AUC(z → loss) 0.75 and 0.77; the policy cannot read it | `code/H_adverse.py` → `out/H_adverse.json` | Attack point 3; the readiness verdict | Expose a causal report-track-record input (or conformal recalibration) to entry and close; test abstention | New search plus confirmation | NEW experiment |
| CX-13 | MINOR | P1 improves return over controls on the correlated rung, conditionally | VERIFIED (7/7 control contrasts survive Holm). Not universal: 9/35 contrasts fail Holm, and P1 is worse in 6/35 (ideal Only3/Only6; omitted NOP, guarded, Only12, Only24). Worst DD: P1 > Only24 [+0.14, +0.67]. | `code/C_stats.py` → `out/C_paired.json` | Attack point 1 (already declared conditional) | Publish the Holm column and the worst-DD bootstrap next to the descriptive intervals | None (analysis only) | Fix (report) |
| CX-14 | MINOR | P1's advantage reflects coefficient tuning, not multihorizon arbitration | VERIFIED: 97.9 % of entries at 6 h (100 % on omitted); P1 − P1-only6 = +0.0052 pp, p = 0.52, 15/30 clusters; 10-genome duplicate group differing in inert genes | `out/C_paired.json`, `out/C_search.json` | Attack point 2 | No multihorizon claim without a horizon-specific positive control (I-PC4) | – | NEW experiment |
| CX-15 | MINOR | (auditor) P1's correlated gain is concentrated in one family | VERIFIED: overlap +6.62 % (120 accounts, 0 losing), trend +1.63 %, reversal +1.72 %; the close value is +0.634 of +0.678 pp from overlap; the same family drives the omitted-rung loss (−6.01 %) | `out/H_adverse.json`, `out/E_close_*.json` | Interpretation of the conditional advantage | Report family-stratified tables and the family-cluster bootstrap | None | Fix (report) |
| CX-16 | MAJOR | Early closes are beneficial / timing is optimal (attack point 4; not claimed, but untested) | REFUTED[correlated: 867/2,103 (41.2 %) destructive against hold; 594/599 TP-bound closes destructive (−1.07 pp); regret mean 0.59 R; omitted: net −0.52 pp; ideal control: 0 destructive, regret 0] | `code/E_close.py` → `out/E_close_*.json` | Any close-optimality reading; the readiness verdict | Add the O*/regret witness to the acceptance list; report helpful/destructive splits by hold outcome per rung | E rerun for every frozen representative (cheap, ~3 s per rung) | Fix (analysis) plus NEW closer experiment |
| CX-17 | MINOR | Lower-risk points are exposure, not better information processing (attack point 5; disclosed) | VERIFIED: 360/360 identical traces on 5 rungs; return and DD ratios equal the size gene; at size 1, P2 dominates P3 and P4 (P5 escapes by 4e-5 DD points); frontier corr(size, return) = 0.959 | `out/D_replay_*.json` `size_traces`; `logs/F_exposure_table.txt` | Attack point 5; the Pareto reading | Separate size from signal genes in the search; report frontiers at fixed size | New search if the frontier is to mean "policies" | NEW experiment |
| CX-18 | INFO | Price paths identical across cost/quality; mirrors exact; banks disjoint; coverage as declared (attack point 6) | VERIFIED: 180/180 groups have one hash over 10 worlds; my generator equals `market()` 720/720; 0 bank collisions; the trend/decreasing-volatility/overlap/gaps/spreads/missing enumeration is as declared | `out/G_coverage.json` | – | – | – | – |
| CX-19 | INFO | Optimization claims are limited (no global optimum, no Nash; shared entry/close uncertainty gene) (attack point 8) | VERIFIED: the three seeds' fronts barely overlap (frontier first-seen by seed 11/25/84); persistence = 1 on all 199 | `out/C_search.json` | – | – | – | – |
| CX-20 | MINOR | DA21 correction (ablations without early close reported with frontier results) adopted | UNVERIFIED[not produced in v2; CX-16 supplies a fixed-position substitute, not a full-trajectory ablation] | – | Attribution of P1's return | Add the G3a/G3b ablations to the analyzer | Analysis plus 5 × 360 accounts per representative | Fix |

## Answers to the nine attack points

1. **Conditional improvement.** VERIFIED as stated. All seven correlated-rung control contrasts are positive and survive Holm when the 30 draw clusters are used: P1 − Only6 = +0.658 pp [0.392, 0.914], 24/30 clusters positive. Across all 35 control contrasts, 9 fail Holm and 6 favour the control. On worst DD, P1 beats Only6 and Greedy but loses to Only24 (+0.58 pt [0.14, 0.67]).

2. **Concentration.** 97.9 % of P1's correlated entries are 6 h (100 % on omitted). The same-genes only6 ablation gives +0.0052 pp (p = 0.52) with identical worst DD. The benefit comes from coefficient tuning: threshold 0.968, uncertainty 0.334 and time weight 0.658 against BASE. No multihorizon arbitration benefit is demonstrated. The P1 genome is one of 10 calibration-identical genomes that differ in inert genes.

3. **Adverse reports.** See H:
   - this is the only causal rung;
   - the loss is concentrated in the overlap family;
   - the band is narrow and uncalibrated;
   - a strong causal track-record detector exists (median z 47.6 against 1.3; AUC 0.75) that the policy cannot see.

   Justified claims: vulnerability to confident wrong signals, and NOP dominance. Unjustified: any calibrated-uncertainty or abstention claim.

4. **Close quality.** Counterfactuals now exist (E). On the correlated rung, 58 % of closes are helpful and 41 % destructive, with net +0.68 pp against holding. SL-bound closes save +1.78 pp; TP-bound closes lose −1.07 pp. Mean regret against O* is 0.59 R, and executed equals O* in only 30 % of closes. The 721 negative closes are 61 % helpful by count but net −0.18 pp. On the omitted rung the net is −0.52 pp. The ideal rung is a perfect positive control. Optimal exit timing is refuted.

5. **Exposure.** Identical traces 360/360 on every rung. All of the return and DD change is q. At equal size, P3 and P4 are dominated by P2. `rel_volume = 0.1` is applied once, separately from q.

6. **Coverage.** Verified as declared, with three qualifications:
   - report errors are shared across the spread and mirror pairs (90 independent streams);
   - the missing rung is the correlated rung masked;
   - the fitness rung keeps the true future shape.

   No learned-model accuracy and no real-market generalization follow from this run.

7. **Contract and engine.** Every element verified on 27,360 accounts, with 0 errors. 450 CLOSE requests were preempted by native protection at a gapped open. The model never acts before an unobservable gap, and the engine handles the gap correctly. The 139 clock and deadline repairs are exercised and hold in 184,896 reports. The terminal-spread repair is not exercised (CX-05).

8. **Optimization.** Eligibility, nondominance with ties, duplicate ranks and the lexicographic rule all reproduce by brute force. The three seeds explore different fronts. No global, Nash or opportunity-cost certificate is claimed, correctly.

9. **Output feasibility.** See section I.

## I. Output feasibility and relevance to Satoshi's lane

**Current interface.** A `PathReport` carries:
- S = 3 joint OHLC mid paths from b+1 to the forced flat;
- per-bar spreads;
- weights;
- terminal spreads;
- an external clock and deadline binding.

The policy values fixed-geometry (h, side) orders, where h selects a TRAIN risk profile, through first passage on those paths. The close controller compares closing now with the best common waiting column. Horizon labels are risk profiles, not experts.

### Minimum adapter before model-specific floating selection

1. **Causal producer.** Use `provenance='causal'` (the synthetic test-plane flag off) and set `observed_through ≤ v[b]+1h`. The model identity and hash must be part of the report identity. One report per decision bar, bound to `ff[seg]` from the calendar (multi-segment ready, not `ff[0]`).
2. **Marginal-to-joint sampler.** It turns per-horizon heads (first-passage curves P_TP(h), P_SL(h), a conditional mean path, an H/L range model) into **S ≥ 32** joint OHLC paths through the deadline. Its RNG must be keyed per world *and* per cost cell. The terminal spread comes from the cost schedule at the flat instant.
3. **Calibration layer.** Conformal or isotonic recalibration of the path band, with coverage tests at 1/3/6/12/24 h. Accept only if coverage is in [0.8, 0.95] for nominal 0.9. On v2's rungs it is 1–55 %.
4. **Causal track-record channel.** A z-type realized-error statistic of the model's own past reports, as in H, fed to entry and close as a gate or penalty. Separate entry and close uncertainty genes, ending the shared-gene restriction.
5. **Size and signal separation.** Size genes are fixed or reported at fixed size in any comparison between models.

### Positive controls (all sealed before running)

| ID | Control | Acceptance |
|---|---|---|
| PC1 | Ideal truth routed through the full adapter with S = 1 | Reproduces the ideal-rung archive exactly |
| PC2 | Synthetic heads with a calibrated quality ladder (AUC/MAE steps, serial errors, *calibrated* bands) | Monotone return and Sharpe against quality, with a measured break-even quality; oracle ceiling and NOP floor on every plot |
| PC3 | Calibrated but uninformative reporter (climatology) | About 0 trades or about 0 return |
| PC3b | Confidently wrong causal reporter (the v2 omitted rung) | The track-record gate abstains, with return ≥ −0.1 % and DD ≤ 2 % |
| PC4 | Horizon-specific control: informative only at 12 h (noise elsewhere) | Allocation moves to 12 h, demonstrating that arbitration works; v2 is at 98 % 6 h |
| PC5 | Close counterfactual (E) | Destructive share and regret against O* below declared ceilings per rung; ideal gives 0 destructive |
| PC6 | Mirror and side symmetry, independent sampling seeds, price-identity hashes across cost cells | Pass |

### Comparison with Satoshi's lane

The facts about Satoshi's lane in this subsection come from the brief. I did not re-measure them.

- **Shared foundation.** C3Z full_f4 uses an absolute (h, side) menu built from Musashi's PROFILES, plus risk-budget sizing with a funding reserve to the forced flat. Both lanes therefore share geometry, sizing and the business contract, and the order-level semantics are already interchangeable.
- **Closers differ.** Satoshi's closer is a **market-only head with a Kalman filter and the ties-HOLD rule**. Musashi's closer reads the report paths. In v2 those paths are future-aware on 4 of 5 rungs, and that closer is net-destructive under the one causal rung (−0.52 pp). A market-only closer is structurally immune to report miscalibration.
- **What remains unproven in Satoshi's lane.**
  - Its closer still needs the E-type counterfactual against O* on comparable worlds.
  - Its 869-cell behaviour matrix with sealed witnesses passes 97.4 % with perfect heads. That shows plumbing and behaviour coverage: 3-year real-calendar worlds, hour-dependent spreads, forced-entry episodes, O* and regret witnesses. It has no imperfect-reporter quality ladder of v2's kind, and no 30-draw cluster inference.
- **Entry selectivity.** Satoshi's measured finding: ZigZag-hierarchy entry vetoes lose Sharpe at every quality while raising σ/trade by up to 38 %. This has the same structure as v2's P2–P5, where higher threshold and lower size give higher win rates and lower totals. In both lanes, per-trade quality rises while total value falls. Neither lane should present higher selectivity, or lower q, as better information processing without fixed-exposure, fixed-count comparisons.

### What each lane should take from the other

| From | To | Take |
|---|---|---|
| Musashi | Satoshi | <br>– Native-exact account replays with cash-conservation, native-parity and cache-parity assertions, sealed receipts and byte-reproducible analyzers. <br>– Independent price draws with explicit paired transforms (envelope, spread, mirror) and a cluster bootstrap at the draw level. <br>– The report-quality ladder (ideal / correlated / delayed / causal-wrong / missing), with **missing implemented as a mask on the same errors**. <br>– Same-genes horizon ablations (only-h) to separate coefficient tuning from arbitration. <br>– Size=1 diagnostics with trace-equality checks. |
| Satoshi | Musashi | <br>– O*/regret close witnesses (E0–E8) and forced-entry both-side episodes, sealed before any search; the E table here is the minimum. <br>– A market-only (causal) closer, or at least the causal track-record gate. <br>– Noise sweeps with break-even quality instead of a single φ = 0.6. <br>– Multi-week, real-calendar worlds whose spread schedules hit the flat hour (CX-05). <br>– σ/trade and count reporting next to totals. <br>– The EPS_CLOSE tie variant reported side by side with ties-HOLD. |

**Readiness:** **REVISE.** The strategy and output interface is not ready for model-specific floating selection until the adapter (points 1–5) exists and PC1–PC6 pass. All of these need new experiments; none can be repaired retroactively in this run.

## Commands and artifacts

All of the following ran in this order, each under `crispdm-run -q -W 3600 -m 2400M -t 14400s -n sat-cx-<step>`, sourcing `code/env.sh`.

| Step name | Command | Output |
|---|---|---|
| (setup) | `git worktree add --detach <am> 5ef458ec`; copy the evidence to `run/`, decompress CANDIDATES and SEED_*, `chmod a-w` | `logs/hash_check.txt` |
| tests | `code/run_tests.sh` | `logs/A_tests.log`, `out/focused_suite.xml` |
| analyze | `code/run_analyzer_noreplay.sh` | `logs/B_analyzer_noreplay.log`, `out/analysis_noreplay/` |
| replay-p1 | `code/run_analyzer_replay.sh` | `logs/B_analyzer_replay.log`, `out/analysis_replay/`, `logs/B_compare.txt` |
| stats | `code/run_py.sh code/C_stats.py` | `logs/C_stats.log`, `out/C_summaries.json`, `out/C_paired.json`, `logs/C_paired_table.txt`, `logs/C_summary_table.txt` |
| search | `code/run_py.sh code/C_search.py` | `logs/C_search.log`, `out/C_search.json` |
| deap | `code/run_py.sh code/C_deap_replay.py` | `logs/C_deap_replay.log`, `out/C_deap_replay.json`, `out/deap_replay/` |
| probe, dprobe | `code/D_probe.py`; `code/run_py_lim.sh 3 code/D_replay.py correlated` | timing and pilot only |
| d-correlated, d-omitted_reversal, d-ideal, d-delayed, d-missing | `code/run_py.sh code/D_replay.py <rung>` | `logs/D_replay_<rung>.log`, `out/D_replay_<rung>.json` |
| e-correlated, e-omitted_reversal, e-ideal | `code/run_py.sh code/E_close.py <rung>` | `logs/E_close_<rung>.log`, `out/E_close_<rung>.json`, `out/E_close_rows_<rung>.json` |
| G_coverage | `code/run_py.sh code/G_coverage.py` | `logs/G_coverage.log`, `out/G_coverage.json` |
| H_adverse | `code/run_py.sh code/H_adverse.py` | `logs/H_adverse.log`, `out/H_adverse.json` |
| F2_archive_counts | `code/run_py.sh code/F2_archive_counts.py` | `logs/F2_archive_counts.log`, `out/F2_archive_counts.json` |

The first G_coverage attempt failed on a read-only array in my own perturbation witness (`logs/chain2.txt`, exit 1). I fixed the script to copy the arrays and re-ran it under the same name and cap; only the fixed run is reported.

The worktree state checks are in `logs/worktree_state_midrun.txt`. The final check was empty `git status` for both trees, with `<hs>` at e0a1c2b.
