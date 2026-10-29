# Independent audit: Musashi shared ZigZag / risk-rules packet (agent-multi master @82862801)

Date: 2026-10-29. Auditor: independent (for Satoshi, technical lead). Object: handoff
`docs/handoffs/MUSASHI_TO_SATOSHI_SHARED_ZIGZAG_RISK_2026_10_28.md`, work-plan docs 113-128,
`docs/work_plan/mechanism_synthesis/BUSINESS_CONTRACT.json`, evidence under
`docs/audits/evidence/{shared_risk_rules_20261028,zigzag_risk_profiles_20261028,zigzag_k3_p20_g300_20261028,mechanism_design_20261028}`
and the code/tests listed in the brief.

Working copies: detached worktree of agent-multi at 82862801 (`<am-audit-worktree>`), detached worktree of
heuristic-strategy at b936d01 (`<hs-engine-pin-worktree>`), scratch dir `<scratch>` (this directory). Both worktrees
were clean (`git status --short` empty) after all runs; nothing was committed, pushed or modified in tracked files.
CPU only (`CUDA_VISIBLE_DEVICES=""`), heavy python under `crispdm-run -m 2400M`, Python = conda env `trading-stack` 3.x.
No data rows were read: the TRAIN tape, `SPEC.private.json` and `h*_OPPORTUNITIES.npz` are not in the repository.

Verdict words are exclusive per claim: VERIFIED / REFUTED[counter-evidence] / UNVERIFIED[reason].

## Findings table

| ID | Sev. | Claim (source) | Verdict | Evidence (`<scratch>/…`) |
|---|---|---|---|---|
| A-01 | INFO | Focused suites pass (doc 128 "70 focused tests"; doc 113 M0 tests) | VERIFIED: 105 passed / 0 failed in 1.03 s across the 6 named files; per file 22+18+23+7 = 70 (risk/zigzag) + 34+1 (mechanism) | `A_pytest.log`, `A_counts.txt` |
| B-01 | INFO | Engine pin `05179b24…` is `app/short_horizon/execution.py` (tool hashes only that file) | VERIFIED: `tools/verify_horizon_risk_rules.py` lines 165-170 hash `engine_root/app/short_horizon/execution.py`; blob introduced by heuristic-strategy commit 95abcaf2 (2026-10-19); present in 11 existing worktrees; also at b936d01 (branch `satoshi/expert-guided-rl-20261019`) | `B_verify_stdout.log` |
| B-02 | INFO | Replay reproduces CONFIG / CASES / REPORT byte-for-byte (doc 128) | VERIFIED: my run produced CONFIG `99d7d4dc…`, uncompressed CASES `d5c156e1…`, REPORT `bda1ab6c…`; `cmp` against committed files = identical (CONFIG_IDENTICAL, REPORT_IDENTICAL, CASES_IDENTICAL) | `B_digests.txt`, `B_verify_out/`, `B_verify_stdout.log` |
| B-03 | MINOR | The verify tool pins the engine by `execution.py` only; `zigzag.py` is imported from the same root but not hashed | VERIFIED (observation): any checkout with `execution.py`=05179b24 passes `--engine-sha` regardless of its `zigzag.py` (two variants coexist: `fc24c6f2…` pre-2026-10-26 and `c06592f6…`). Harmless for this tool (it does not call zigzag) but the pin is narrower than "engine root" suggests | `B_verify_stdout.log`, worktree hash listing in this report §B |
| B-04 | INFO | Independent recomputation of the 3,456 ledger rows | VERIFIED: 3,456 records, 3,456 ORDER / 0 NOP; `units == floor(q*floor(10000*0.1*50/1.1))`, `margin == units*1.1/50`, `loss_per_unit == dist(entry,SL_rounded)+0.00014` hold on every row; max margin 999.988; max utilization 0.9999996; 0 budget violations; 0 rounding-direction violations; q ranges equal doc 128 table to 6 decimals | `B_cases_independent_check.log` |
| C-01 | INFO | PROFILES.json digest `aa2cb1f8…` (handoff, doc 126) | VERIFIED: sha256 of committed file = `aa2cb1f8f524db4091f8d3cd3b86b7b0f31af6f2ff71bb5a974f49e9d33d9d96` | `C_spotcheck.log` |
| C-02 | INFO | Doc 126 directional table and support table equal PROFILES.json `positive_calendar` | VERIFIED: all 8 rows (n, TP mean/std/median, MAE mean/std/median, duration median/p90) and all 4 support rows (complete, censored, excluded-by-calendar, zero-MAE) match exactly (brief asked for 3; all 8 checked) | `C_spotcheck.log` |
| C-03 | INFO | "134,916 directional paths reconstructed, max error 0" | VERIFIED as self-reported in PROFILES.json (39,240+38,216+32,382+25,078 = 134,916; max_abs_error 0.0 each). Not re-executed: the NPZ arrays and `SPEC.private.json` are outside the repo | `C_profiles_audit_fields.log` |
| C-04 | INFO | Hash chain PROFILES → run RESULTS → public RESULTS | VERIFIED: PROFILES `source_sha256[RESULTS.json]` = `dd956798…` = public `RESULTS.json.audit.results_sha256`; public k3 `RESULTS.json` itself hashes `d12b57cb…` (different file, by design); `audit.source_sha256` = current `tools/audit_zigzag_opportunity_results.py` (`ba332540…`) | `C_profiles_audit_fields.log`, `F_k3_results_meta.log` |
| D-01 | INFO | Formulas dSL, dTP, SL, TP, q, units, margin, loss (handoff / doc 127 RR03-RR06) | VERIFIED from `app/horizon_risk_rules.py`: see §D below; rel_volume enters once (inside `Sizing.units_max`), leverage once (`units_max`; `required_margin` divides it back out so margin = q·0.1·balance) | `D_adversarial.log` |
| D-02 | INFO | "Round SL away from entry, TP toward entry" | VERIFIED: buy → both ROUND_FLOOR (SL 8.400→8.430 pips, TP 20.000→19.970); sell → both ROUND_CEILING (SL 8.400→8.470, TP 20.000→19.930); 0 violations over 3,456 ledger rows | `D_adversarial.log`, `B_cases_independent_check.log` |
| D-03 | MINOR | "Modeled stop loss never exceeded the supplied risk budget" | VERIFIED as an algebraic identity under 6 assumptions (§D): units=floor(q·umax) ≤ q·umax and q ≤ R/(umax·lpu) ⇒ loss ≤ R. It is a cap on the *model*, not on realized loss: 100-pip adverse gap on the baseline order → 460.90 USD vs budget 50 (native `_protect` fills at the gap open); doc 128 already states this | `D_adversarial.log` |
| D-04 | INFO | Fixture (i): tiny balance, integer units round to 0 | VERIFIED behaviour: balance 0.2 → `units_max`=0 → NOP `BELOW_MINIMUM_UNITS`; balance 1.0 → 4 units ORDER (margin 0.088); balance 100 / rho 0.001 → q 0.2248, units 102, loss 0.09996 ≤ 0.1. No upward rounding anywhere | `D_adversarial.log` |
| D-05 | INFO | Fixture (ii): spread larger than buffer | VERIFIED behaviour: buffer does NOT cover spread (by design, spread lives in the quotes). SL = 8.4 pips below ask; spread 2 and 5 pips → ORDER (liquidation-to-SL distance 6.4 / 3.4 pips ≥ 0.5); spread 8.0 pips → NOP (0.4 < min_distance 0.5); MAE=0 + 1-pip buffer: spread 0.5 → ORDER, spread ≥0.9 → NOP; buffer 0 + MAE 0 → NOP. Loss model unchanged by spread (ask−SL) | `D_adversarial.log` |
| D-06 | MINOR | Fixture (iii): negative / NaN / inf sigma | VERIFIED behaviour, with a caveat: `propose` RAISES `ValueError('INVALID_LEVEL_INPUT')` (not a NOP status) for sigma −1e-4, NaN, inf; `-0.0` is accepted as 0. Negative `b_sl`/`b_tp`, NaN rho, rho>1, a_sl=0 raise `INVALID_RISK_GENE`. A consumer must catch exceptions or a bad sigma estimator aborts the decision step instead of abstaining | `D_adversarial.log` |
| D-07 | INFO | sigma genes reach the formula | VERIFIED: sigma 30 pips with b_tp=1 erodes TP to 10 pips below entry → NOP `LEVELS_NOT_EXECUTABLE`; b_sl=1 widens SL to 38.4 pips → q drops 1.0 → 0.2764, loss 49.997 ≤ 50 | `D_adversarial.log` |
| E-01 | INFO | `zigzag.py` sha `c06592f6…` (doc 124) | VERIFIED: blob introduced by heuristic-strategy commit 28ad98e8 (2026-10-26, symmetric-first-pivot change); present at b936d01 (`satoshi/expert-guided-rl-20261019`); 13 other worktrees carry the older `fc24c6f2…` (diff = `select_pivots` first-pivot rule + `select_pivots_legacy`; `candidate_maps` lines 56-90 identical) | this report §E |
| E-02 | INFO | Deviation is inactive for any deviation ≥ 0 in this implementation (doc 124 lines 60-89 reasoning) | VERIFIED: line 77 `lowmap[i] = val if low[i]==val else 0` emits a candidate only when `low[i]==val`, where `low[i]-val = 0 ≤ dev` for every dev ≥ 0, so line 71 `if low[i]-val > dev` can only reject bars that line 77 discards anyway (mirror lines 83/89 for highs). Probe: synthetic 200-bar path, (depth,backstep) ∈ {(3,1),(4,2),(7,6),(12,11)}, deviation 0 / 5 / 50 / 1e6 → identical candidate maps and identical pivot lists (67/43/24/12 pivots). Brute force 3,000 random short paths dev 0 vs 1e9: 0 differences | `E_deviation_probe.log`, `E_deviation_probe.py` |
| E-03 | INFO | The deviation comparison line is live (not dead code) | VERIFIED positive control: deviation −1 (out of contract) empties every candidate map (0 pivots) on the 200-bar path and differs in 500/500 random paths, so the branch executes; doc 124 reported no positive control (`outside_contract_positive_control: null`) — this probe supplies one | `E_deviation_probe.log` |
| E-04 | INFO | Doc 124 lines 93-134: selection phase has no deviation parameter / no reversal test | VERIFIED by reading `select_pivots` (lines 99-134): signature `(lowmap, highmap)` only; replacement rule compares `lowmap[i] < last_low` / `highmap[i] > last_high`, no amplitude threshold | pinned source |
| E-05 | MINOR | Residual channel: skipped backstep cleanup when deviation rejects | VERIFIED inactive empirically (E-02 brute force, doc 124's 10,935 synthetic cases, 1,276 real groups) — but this is empirical, not a proof; doc 124 says the same. No change requested | `E_deviation_probe.log` |
| F-01 | INFO | BUSINESS_CONTRACT.json constants | VERIFIED listing: leverage 50; rel_volume 0.1; max_simultaneous_positions 1; max_entries_per_day null; entry_wait_after_session_open_hours 12; forced_flat_before_session_close_hours 12; early_close enabled, daily_count_limit null, independent_of_opener_timeframe true; mandatory controls stop_loss / take_profit / relative_margin_fraction; status OWNER_DEFINED_DESIGN_CONTRACT_NOT_DEPLOYED; sha `73f32e43…` (matches M0 report pin). Handoff, docs 113/127/128 and `propose()` (`rel_volume != .1 or leverage != 50` → BUSINESS_SIZING_CONTRACT) are consistent with it | `F_constants_grep.txt`, `G_m0_run1.json` |
| F-02 | MAJOR | The contract equals the owner's decision | UNVERIFIED[two lane records disagree]: Musashi's contract (commit ba2af54a 09:47 -0500 "remove daily entry cap per owner business clarification") says no daily cap, entries allowed until close−12h, leverage 50. Satoshi's record of the same day's owner decision (predictor memory `business-contract-calendar-entries-20261028`) says max 4 entries/day, NO entries in the 24 h before the forced flat (= 36 h before close), leverage still open (100 vs 50). The 24h pre-flat no-entry window is absent from BUSINESS_CONTRACT.json and from every doc 113-128. Only the owner can settle which is final; both lanes must load the same file (Satoshi F-11) | `F_constants_grep.txt`; contract file |
| F-03 | MINOR | Population: "population 30" (docs 114-118) vs "population 20" (docs 119-128, handoff) | VERIFIED inconsistency, explained by chronology: 114/115 = 30×40 baseline; 116/117 = 30×300 pre-correction extension; 118 line 32 "Population 30, generations 300 … remain the requested budget" was overtaken by the owner-requested 20×30 comparison (119) and the 20×300 K=3 run (123, RESULTS.json `population: 20, generations: 300`). Doc 118 was never amended; the handoff's "compared with corrected 20x30 runs, all 360 prefix records matched" refers to 119 → 123 | `F_constants_grep.txt`, `F_k3_results_rows.log` |
| F-04 | INFO | K=3 chosen; horizons 3/6/12/24 | VERIFIED: RESULTS.json `future_pivots: 3` on all rows and grid control; `tools/zigzag_risk_profiles.py` refuses `future_pivots != 3 or horizons != [3,6,12,24]`; doc 114 explicitly replaced an earlier "1/2/6/12/24 ladder" with 3/6/12/24; no other horizon set appears in 113-128 or the handoff | `F_k3_results_rows.log`, `F_constants_grep.txt` |
| F-05 | INFO | Handoff champion table (3/5/1, 4/5/2, 7/25/6, 12/25/11; fitness 0.271176912 / 0.237675675 / 0.152872027 / 0.098117667) | VERIFIED against RESULTS.json `results.rows` and PROFILES `rows[*].params`; 12h +1.215882 % vs 30 gens matches `comparison.rows[2].relative_improvement_pct` | `F_k3_results_rows.log`, `C_spotcheck.log` |
| F-06 | INFO | Bounds depth 2..96, backstep 1..min(24,depth−1), deviation 1..120 (0.1) | VERIFIED consistent with observed diversity (depth 2..96 unique 95; deviation 1..119.9 unique 1,010; backstep 1..24) and the 2,004-pair grid count | `F_k3_results_meta.log` |
| F-07 | INFO | Tape facts: 24,895 rows, 19,879 origins, 209 chunks, 2020-01-02 .. 2023-12-29 | VERIFIED as recorded in RESULTS.json `results.data`; not re-derivable here (no data) | `F_k3_results_meta.log` |
| F-08 | INFO | Horizon eligibility "median spacing in [h/√2, h√2], ≥100 spans" | VERIFIED in `app/zigzag_opportunities.py` line 158; observed medians 3/5/9/17 h all inside their bands | grep in this report |
| G-01 | INFO | `tools/mechanism_m0.py` runs, is deterministic and equals committed `M0_EXACT_REPORT.json` | VERIFIED: two runs byte-identical; JSON equals committed report; `source_sha256` pins match current `app/mechanism_games.py` (`6b6b47db…`), `tools/mechanism_m0.py` (`ae11d9bf…`), BUSINESS_CONTRACT.json (`73f32e43…`) | `G_m0_run1.json`, `G_m0_run2.json`, `H_file_digests.txt` |
| G-02 | INFO | What the M0 fixtures prove | VERIFIED scope: (a) 3×3 coordination game has exactly 3 pure Nash (0,0),(1,1),(2,2) with values 0/2/3 and Pareto front {(2,2)}; (b) 20 players × 27 policies: unilateral certificate at the all-zero profile costs 521 distinct profiles (1+20·26), finds epsilon = 1 (last player's action 26) — linear, not 27^20; (c) 5 identical closers: all difference rewards 0, group Shapley 2 each from 32 coalitions; (d) 3×2 table: moving (1,1)→(2,1) lowers account 4→3 while player-1 Shapley rises 2→13/2. All in exact Fractions. They prove nothing about markets, learning or closers (training_runs = market_rollouts = 0) | `G_m0_run1.json`; doc 113 §3-5 |
| G-03 | INFO | Doc 113 §8 disposition of Satoshi's F-01..F-21 | VERIFIED text comparison: F01-F04, F06-F10, F12, F13, F15-F19 accepted (F06 amended: Shapley is diagnostic, not a learning reward — same as Satoshi's own note); F05 accepted with an explicit separating fixture (two longs, same quote 100, path 98→104, A SL99/TP106 → CLOSE, B SL97/TP104 → HOLD) but curves not mandated; F11 declared resolved by owner clarification (see F-02 above: contested); F14 amended (neither mirror-loss nor n·risk bound is universal under gaps); F20 and F21 partially rejected (synthetic-inflation-never-improves-Φ and 80 % matching-scale quota rejected as universal). M0 does not implement the F05 separating fixture, filter/persistence policies (F07) or cross-scale matrix (F08): doc 113 §9 defers them to the next boundary | `G_satoshi_verdict.md`; doc 113 |
| G-04 | INFO | Referenced commits exist | VERIFIED in agent-multi: ba2af54a (contract), 3326f898 (Satoshi verdict), 660511a0 (guided-lane audit), `docs/audits/MUSASHI_VERDICT_SATOSHI_GUIDED_MECHANISM_2026_10_28.md` present | git cat-file in this report |
| H-01 | MAJOR | "Compatibility finding: pinned BarEnv._levels clips to multiplier bounds and effectiveness_c·sigma_rule; check_order enforces that cap" (handoff) | VERIFIED in `satoshi/expert-guided-rl-20261019` @b936d01: `expert_guided/env.py` lines 201-216 clip `tp_mult`/`sl_mult` to `tp_bounds=(0.25,3.0)` / `sl_bounds=(0.5,3.0)` and, under `level_policy == "clip"`, clip distances into `[min_distance, effectiveness_c·sigma_rule[b+1]]`; `execution.py` `check_order` line 219 raises when `max(dist_tp, dist_sl) > effectiveness_c·sigma` (`effectiveness_c = 1.5`). Absolute pip levels from `horizon_risk_rules.propose` (e.g. 24h p75 TP ≈ 60+ pips) cannot be routed through this API unchanged | `<hs-engine-pin-worktree>` sources |
| U-01 | INFO | ZigZag DEAP run used `zigzag.py` = `c06592f6…` and engine root = a specific checkout | UNVERIFIED[engine root and code pins live in `SPEC.private.json`, not in the repo; public RESULTS.json carries no zigzag sha] | `F_k3_results_meta.log` |
| U-02 | INFO | 17,280 native protection witnesses "passed" | VERIFIED count only (3,456×5 fixtures, REPORT `protection_witnesses: 17280`, and my replay re-executed them without AssertionError); the fixtures are single-bar synthetic quotes around the order's own levels, not market paths | `B_verify_stdout.log` |
| U-03 | INFO | "Verified 3,600 generation records, 6,144 candidate paths, 2,048 scalar paths" (doc 123) and "13,232 distinct genomes / 6,688 real-engine calls" (doc 124) | UNVERIFIED[requires the private run root and tape; only RESULTS.json self-report available] | `F_k3_results_meta.log` |

Summary counts: 0 BLOCKER, 2 MAJOR (F-02 contract divergence between lanes — unverified which is the owner's final; H-01 consumer cap, which Musashi already flagged and which blocks absolute-level routing), 5 MINOR, rest INFO. Nothing in the packet was REFUTED.

## A. Tests

Command (from `<am-audit-worktree>`):
`crispdm-run -q -W 1800 -m 2400M -t 3600s -n sat-audit-A -- <trading-stack python> -m pytest -q -p no:cacheprovider tests/unit/test_zigzag_opportunities.py tests/unit/test_zigzag_risk_profiles.py tests/unit/test_horizon_risk_rules.py tests/unit/test_zigzag_extension.py tests/unit/test_mechanism_games.py tests/unit/test_mechanism_m0_runner.py`
Result: `105 passed in 1.03s`. Per file: opportunities 22, risk_profiles 18, horizon_risk_rules 23, extension 7, mechanism_games 34, m0_runner 1.
Note: `test_horizon_risk_rules.py` uses a `SizingFixture`, not the native Sizing; the native class is exercised only by `tools/verify_horizon_risk_rules.py` (task B) and by my D script.

## B. Replay of the shared-risk-rules verification

- Pin resolution: `--engine-sha` hashes `<engine-root>/app/short_horizon/execution.py`. Blob `05179b24…` = heuristic-strategy commit 95abcaf2 (2026-10-19). Worktree hashes observed (exec / zigzag): 11 worktrees `05179b24 / fc24c6f2`; one worktree (`satoshi/expert-guided-rl-20261019` @b936d01) `05179b24 / c06592f6`; `codex/quote-execution-20261027` and `codex/master-strategy-integration-20261028` carry a newer `execution.py` (`a0b374d6`), `codex/execution-cost-modes-20261026` carries `2939d8a1`. I created my own detached worktree at b936d01 (`<hs-engine-pin-worktree>`) so both pinned files match and no shared checkout received `__pycache__` writes (`PYTHONDONTWRITEBYTECODE=1`; worktree status clean afterwards).
- Command: `crispdm-run … -n sat-audit-B -- <python> tools/verify_horizon_risk_rules.py --profiles docs/audits/evidence/zigzag_risk_profiles_20261028/PROFILES.json --profiles-sha aa2cb1f8… --engine-root <hs-engine-pin-worktree> --engine-sha 05179b24… --config docs/audits/evidence/shared_risk_rules_20261028/CONFIG.json --output <scratch>/B_verify_out`
- Result: exit 0; CONFIG `99d7d4dc92f6437c84b8b36de4c29a1ba474489a7c07444899699c00203c3c0e`, CASES `d5c156e1b6115c32c0ed9f913e16b8e23de2c9d06832c472077fb9b73d1a9030`, REPORT `bda1ab6c9ab26a668ba4b12f9568cd8416b609172a724d63b66f032012caadda` — byte-identical to the published files (committed CASES.json.gz decompresses to the same bytes).

## C. PROFILES.json

Digest verified. All eight doc-126 rows and the four support rows match the JSON exactly (`C_spotcheck.log`). CONFIG.json references are the `positive_calendar` median and p75 of PROFILES (e.g. h3 buy TP 20.0 / 35.9 pips, MAE 7.4 / 15.1 pips). TP/MAE correlation range −0.0205..+0.0695 as stated in doc 126.

## D. Formula audit of `app/horizon_risk_rules.py` (sha `89afca8d…`)

Derived from the code (Decimal arithmetic for levels, floats elsewhere):

```
dSL  = a_sl*MAE_ref + b_sl*sigma + buffer_pips*0.0001          (raw_sl_distance)
dTP  = a_tp*TP_ref  - b_tp*sigma                                 (raw_tp_distance)
SL   = round(entry - dir*dSL, tick)   TP = round(entry + dir*dTP, tick)
       rounding = FLOOR for buy (dir=+1), CEILING for sell       -> SL away, TP toward entry
NOP (LEVELS_NOT_EXECUTABLE) if any of: SL<=0, TP<=0, dTP<=0, dSL<=0, TP/SL on wrong side of entry OR of
       liquidation quote, or min(|TP-p|,|p-SL|) < min_distance for p in {entry, liquidation}
umax = floor(balance*rel_volume*leverage/entry)                  (native Sizing.units_max; rel_volume & leverage used HERE only)
NOP (BELOW_MINIMUM_UNITS) if umax < min_units
lpu  = dir*(entry - SL_rounded) + loss_cost_per_unit + funding_reserve_per_unit
R    = rho*balance
L1   = umax*lpu
q    = quality*min(1, R/L1)         (size_from_risk; q==0 -> NOP ZERO_BUDGET_OR_QUALITY)
units= floor(q*umax)                 (native Sizing.order_units, risk_accepts_minimum=False; None -> NOP BELOW_MINIMUM_UNITS)
margin = units*entry/leverage  (= q*rel_volume*balance up to floor)   estimated_loss = units*lpu
raise NATIVE_SIZING_CAP_BREACH if margin > 0.1*balance or estimated_loss > quality*R
```

rel_volume is applied exactly once (in `umax`); leverage exactly once (in `umax`, then divided back out by `required_margin`, so margin never carries leverage). Spread is not debited in `lpu` (entry is the ask for a buy, SL is a bid-level, so ask−SL already contains it). Stop slippage, market slippage and both commissions enter `loss_cost_per_unit` (grid: 0.00002+0.00005+2·3.5/100000 = 0.00014 per EUR).

"Modeled stop loss never exceeded the budget" rests on: (1) loss linear in units (no ticket fees, no lot steps beyond integer units); (2) stop fills at the SL price plus a fixed per-unit slippage — no gap (the 100-pip gap fixture realizes 460.90 USD against a 50 USD budget; doc 128's own fixture 90.64 vs 10); (3) commissions fixed per unit; (4) funding beyond `funding_reserve_per_unit` ignored (grid uses 0); (5) entry at the known fill quote (no next-open revalidation); (6) EURUSD with USD account, so price distance × units is already USD (no conversion). Within those, `floor` only lowers `units`, so the inequality is exact; the 0.9999996 maximum utilization is the floor residue.

Adversarial fixtures (native Sizing from the pinned engine, `D_adversarial.log`):
(i) balance 0.2 USD → umax 0 → NOP BELOW_MINIMUM_UNITS; 0.5 → 2 units; 1.0 → 4 units, margin 0.088; balance 100 with rho 0.001 → q 0.2248, 102 units, loss 0.09996 ≤ 0.1. No rounding up.
(ii) spread > buffer: SL 8.4 pips below the ask; spread 2 / 5 pips → ORDER; 8.0 / 8.3 / 8.4 / 9 / 20 pips → NOP (liquidation-to-SL < 0.5-pip min_distance or wrong side). MAE 0 with 1-pip buffer: spread 0.5 → ORDER, 0.9..1.5 → NOP; buffer 0 + MAE 0 + zero spread → NOP. The buffer is a cushion against the MAE reference, not against spread; the liquidation-side check is what rejects wide spreads.
(iii) sigma −1e-4 / NaN / inf → `ValueError('INVALID_LEVEL_INPUT')` (exception, not NOP); −0.0 accepted. Negative b_sl/b_tp, NaN or >1 rho, a_sl = 0 → `INVALID_RISK_GENE`. sigma 30 pips with b_tp = 1 → TP below entry → NOP; with b_sl = 1 → q 0.2764, loss 49.997 ≤ 50.

## E. Deviation inactivity (heuristic-strategy `app/short_horizon/zigzag.py` @ `c06592f6…`)

Line-level: `candidate_maps` (lines 56-90). `dev = deviation_pips*PIP` (60). For lows: `val = lmin[i]` (window minimum including bar i); if unchanged → 0 (67-68); else the deviation test `if low[i]-val > dev: val = 0` (71-72) else backstep cleanup (74-76); finally `lowmap[i] = val if low[i]==val else 0.0` (77). Since the window includes bar i, `low[i] ≥ val`, and a candidate is emitted only when `low[i] == val`, i.e. when `low[i]-val = 0 ≤ dev` for every dev ≥ 0. The deviation test can therefore only zero a `val` that line 77 zeroes anyway. Highs mirror this (78-89). The only side channel is the skipped backstep cleanup when the test fires; neither doc 124's 10,935 synthetic cases nor my 3,000 random paths found a case where that changes the maps. `select_pivots` (99-134) takes only the two maps and has no amplitude test. Doc 124's reasoning is VERIFIED. My positive control (deviation −1, out of contract) empties the maps on every path, proving the branch is executed.

Probe (`E_deviation_probe.py`, 200-bar synthetic path, seed 20261028): for (3,1), (4,2), (7,6), (12,11) the pivot index lists are identical for deviation 0, 5, 50, 1e6 (67 / 43 / 24 / 12 pivots; first indices for (3,1): 0,1,3,4,8,12,17,25,29,30,35,41).

## F. Business contract and cross-document constants

Constants in BUSINESS_CONTRACT.json: listed in F-01. Consistent with handoff, docs 113/125/126/127/128 and the sizing-contract check in `propose()`. Inconsistencies found: (a) population 30 (docs 114-118) vs 20 (119-128, handoff): doc 118 line 32 is stale, the executed K=3 run is 20×300 (F-03); (b) owner-contract divergence between lanes on daily cap (null vs 4), pre-flat no-entry window (none vs 24 h) and leverage (50 vs open) (F-02, MAJOR, unverified which is final); (c) nothing else: K=3, horizons 3/6/12/24, seeds 7/19/43, leverage 50, rel_volume 0.1, 12h/12h appear identically everywhere they occur.

## G. Mechanism M0

`python tools/mechanism_m0.py` (stdout only, no CLI flags, no training) runs in well under a second, twice byte-identical, equal to the committed `M0_EXACT_REPORT.json`. What it proves and what doc 113 concedes: see G-02 / G-03. The M0 code is a generic exact finite-game toolkit (potential value oracle, unilateral epsilon certificate, strict best response, group Shapley, Pareto dominance), independent of markets.

## H. Extraction for Satoshi's expert-guided (supervised ZigZag-label) lane

REUSABLE AS-IS (paths relative to agent-multi @82862801; sha256):
- `docs/work_plan/mechanism_synthesis/BUSINESS_CONTRACT.json` (`73f32e43…`) — as the single loaded contract, ONLY after the owner settles F-02 (daily cap, 24 h pre-flat window, leverage).
- `docs/audits/evidence/zigzag_risk_profiles_20261028/PROFILES.json` (`aa2cb1f8…`) — TRAIN 2020-2023 directional TP/MAE/drawdown/duration quantiles per horizon and side, with year strata; descriptive references only.
- `docs/audits/evidence/shared_risk_rules_20261028/CONFIG.json` (`99d7d4dc…`) — consumable references + baseline genes + search bounds + synthetic execution assumptions; `REPORT.json` (`bda1ab6c…`), `CASES.json.gz` (gz `8cdc263c…`, uncompressed `d5c156e1…`).
- `app/horizon_risk_rules.py` (`89afca8d…`) — fill-time SL/TP/q/units construction; pure, injected Sizing; EURUSD/USD only.
- `app/zigzag_risk_profiles.py` (`19c329fc…`) — `describe`, `verify_paths`, `profile`, `size_from_risk`.
- `app/protection_reference.py` (`6f22267…`) — independent OHLC barrier-order reference (`protection_exit`), useful as a second opinion against the native `_protect`.
- `tools/verify_horizon_risk_rules.py` (`7f0bfdaf…`) — replayable grid + native-primitive witnesses; pins `execution.py` 05179b24.
- `app/mechanism_games.py` (`6b6b47db…`) + `tools/mechanism_m0.py` (`ae11d9bf…`) — exact finite-game certificates / Shapley / Pareto arithmetic (Satoshi's own reuse table already classes these SHARED_UNCHANGED).
- ZigZag champion genes per nominal horizon (3/5/1, 4/5/2, 7/25/6, 12/25/11) with the declared convention deviation = 5 (inactive); and the verified fact that deviation is a dead gene for any deviation ≥ 0 in `zigzag.py` c06592f6 and fc24c6f2 alike (`candidate_maps` is identical in both), so Satoshi's DEAP/label pipelines can drop that gene.
- Unit tests `tests/unit/test_horizon_risk_rules.py`, `test_zigzag_risk_profiles.py`, `test_mechanism_games.py` (as regression fixtures for the shared modules).

REUSABLE WITH ADAPTATION:
- `propose()` levels into Satoshi's `BarEnv`: NOT routable through `BarEnv._levels` / `check_order` as-is (H-01). Adaptation: an absolute-level entry path with `level_policy` that bypasses the `(0.25..3)·sigma` multiplier clip and the `effectiveness_c·sigma_rule` cap, with next-open revalidation and `check_order`'s wrong-side / min-distance checks retained. Note the lane's current `CAP_P1_SIGMA = 2.09` audit constant will no longer describe the cap.
- Exception semantics: `propose()`/`levels()`/`coefficients()` raise on non-finite sigma or invalid genes (D-06). In a supervised lane where sigma comes from a model/estimator, wrap in a NOP-on-exception adapter or validate upstream.
- Reference quantiles: PROFILES are hindsight-oracle, overlapping, calendar-filtered populations (every complete origin has a positive opportunity). For expert-guided labels use them only as SL/TP *priors* per horizon; the lane's own label population (ZigZag-level labels with maturity/censoring) must produce its own MAE/TP quantiles if labels differ from "next-3-events best exit".
- `size_from_risk` / q: carries `quality ∈ [0,1]` as an explicit input; in the supervised lane `quality` would be a calibrated head confidence — calibration (ECE / proper scoring) must be established first (Satoshi F-20); c = 1 control until then.
- Sigma estimator and window: not selected by the packet (grid uses synthetic 5/15/30 pips). The lane must pin its own causal sigma (its `decision_sigma` / `sigma_rule`) and treat b_sl/b_tp as genes over that definition.
- Cost model: synthetic (`ASSUMED_SYNTHETIC_NOT_BROKER_VERIFIED`: tick 1e-5, min_distance 0.5 pip, slippage 0.2/0.5 pip, 3.5 USD/lot/side, funding 0). Replace with the lane's declared cost convention; the oracle's `fill_cost`/`financing` tape convention is different from this adapter (handoff says so).
- Verification tool: `--engine-sha` pins only `execution.py`; if the lane changes `execution.py` (quote-execution and cost-mode branches already do: `a0b374d6`, `2939d8a1`), re-pin and re-run; the witnesses depend on `_protect` and `Sizing` semantics.
- Doc 113 §6 separating fixture (two longs, same market, A CLOSE / B HOLD) — reusable as the lane's blocking test for position-dependent closers, once written in the native engine with quotes/costs.

NOT APPLICABLE to the expert-guided lane:
- The ZigZag oracle *objective* (per-origin best score over next-K events, `net/(max(dd,cost,1pip)·hours)`), its DEAP search receipts, the K=2/3/4/5 contrast and the 2,004-pair grid: these pick indicator genes for Musashi's demonstration targets; the lane's labels (ZigZag-level labels, confirmation lag, maturity) are a different target definition. Do not re-run the searches (handoff asks this) but do not adopt the fitness numbers as label quality either.
- `tools/zigzag_risk_profiles.py`, `tools/audit_zigzag_opportunity_results.py`, `tools/zigzag_opportunity_deap.py`, `tools/extend_zigzag_opportunity.py`: bound to `SPEC.private.json`, the private run root and the NPZ records; cannot run from the public repo.
- M0 game interpretation (20 RL players × 27 policies, difference rewards, PPO credit): RL-only per Satoshi's own table; the arithmetic module is shared, the player model is not.
- The native protection witnesses (17,280) as evidence of early-close behaviour: they test `_protect` on single synthetic bars only; the lane's early-close policy needs its own oracle-validated witnesses (Satoshi F-12).
- "Every complete origin has a positive hindsight opportunity" as any statement about abstention: the oracle population cannot qualify NOP (doc 126 says so).

## Claims I could NOT verify and why

1. The TRAIN tape identity (24,895 rows / 19,879 origins / 209 chunks), the 134,916-path reconstruction and the 3,600-record / 6,144-path / 2,048-path audits: the tape, `SPEC.private.json` and `h*_OPPORTUNITIES.npz` are not in the repository; only self-reported hashes exist (C-03, F-07, U-03).
2. Which `zigzag.py` blob and engine root the DEAP run actually loaded (U-01): pinned inside the private spec.
3. The owner's final business constants (F-02): two first-hand records from the same day disagree; no owner-signed artifact in either repo.
4. Doc 124's claim of 1,276 real-data deviation groups and 6,688 intercepted engine calls (U-03): only the aggregate counts are in RESULTS.json.
5. Broker realism of the synthetic execution constants: declared ASSUMED by the packet; no broker evidence to check against.

## Files in `<scratch>`

`A_pytest.log`, `A_counts.txt`, `B_verify_stdout.log`, `B_verify_out/{CONFIG,CASES,REPORT}.json`, `B_digests.txt`,
`B_cases_independent_check.log`, `C_spotcheck.py`, `C_spotcheck.log`, `C_profiles_audit_fields.log`,
`D_adversarial.py`, `D_adversarial.log`, `E_deviation_probe.py`, `E_deviation_probe.log`, `F_constants_grep.txt`,
`F_k3_results_meta.log`, `F_k3_results_rows.log`, `G_m0_run1.json`, `G_m0_run2.json`, `G_satoshi_verdict.md`,
`H_file_digests.txt`, `COMMANDS.md`.
