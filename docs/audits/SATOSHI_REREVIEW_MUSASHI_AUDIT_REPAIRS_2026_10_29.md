# Targeted re-review: Musashi's audit repairs (agent-multi master @2318b0e3)

Date 2026-10-29. Independent auditor for Satoshi (expert-guided lane). Object: agent-multi `2318b0e3`
("Repair audited close valuation and cache contracts; preserve 288 account replays"), predecessor `3644ece9`,
native engine heuristic-strategy `e0a1c2b962f4`.

## Overall verdict per repaired finding

- **F07 / DA09 (TP-through boundary): FIXED.** The controller now equals native cash at the exact boundary, long and short. The 27.50 USD / 17.39 USD gaps are now 1.8e-15 / 0.0, and a 60,000-case pure-function sweep has 0 mismatches (11,497 before).
- **F08 / DA10 (forced-flat spread): FIXED.** Our F5 witness drops from 2.84 USD to 8.9e-16 USD. Musashi's own native "deadline_spread" fixture does not discriminate: it has no spread jump and also passes on the predecessor (RR-N2).
- **F10 / DA11 (truncated report): FIXED.** A truncated self-consistent report, a report with the right deadline but early-stopping bars, and a mismatched expected_deadline are all refused when the deadline comes from the calendar. This holds on the direct, cached and native-world routes.
- **DA12 (stale/future reports, engine clock): FIXED.** A b−1 or b+1 report is refused on the direct route, the cached-adapter route, the close-cache route and the entry-bid route, with warm caches and with verify set to False or True. The run that went from 1.795 % to 1.881 % now raises STALE_OR_FUTURE_REPORT.
- **DA13 (cache identity): PARTIALLY_FIXED.** The close-value cache now binds entry, profile, side, funding, slippage and report content. But two other caches are keyed by bar only and ignore the execution config: a spread-schedule change reuses the stale report, and a cost change reuses stale entry bids (RR-N1).
- **DA24 (boundary fixtures): PARTIALLY_FIXED.** Gap-SL preemption of CLOSE, the ambiguous bar and the exact TP boundary hold on every common action against native, in both directions. The deadline-spread fixture does not discriminate (RR-N2). The account panel still has no protection-preempts-CLOSE witness; that is deferred to the plan-138 panel.

## Setup and constraints

Placeholders: `<am>` is a new detached worktree of agent-multi at 2318b0e3. `<am-pre>` is the existing detached
worktree at 3644ece9 from our dynamic-account audit; it was clean and only read. `<hs>` is the reused clean detached
heuristic-strategy worktree. Its HEAD was e0a1c2b962f41039bb59a76614b70e326f2cf0c1 and its status was clean before
and after; the only extra content is ignored `__pycache__` from the earlier audit (timestamp 02:03, before this
review). `<ch-branch>` and `<da-branch>` are new detached read-only worktrees of our two audit branches (f971b152 and
c39a0a07). `<scratch>` is this review's scratch directory; scripts are in `<scratch>/scripts/` and the unmodified
originals in `<scratch>/scripts_orig/`.

CPU only (`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`). Every heavy step ran under
`crispdm-run -q -W 1800 -m 2400M -t 7200s -n sat-rr-<step>` and was admitted every time; peak RSS was about 0.2 GB.
Nothing was killed, committed or pushed, and no tracked file was modified (`git status` of `<am>` is empty). Only
synthetic 2020-01 bars were read.

**Interpreter.** The evidence interpreter is the `trading-stack` environment: Python 3.12.13, NumPy 2.5.1,
pandas 3.0.3 (`<scratch>/versions.txt`). My first pass used the base interpreter by mistake (Python 3.12.7,
NumPy 1.26.4, pandas 2.2.2). I archived those outputs in `<scratch>/base_interpreter_run/` and re-ran everything
under trading-stack. All C-step JSON outputs are identical across the two interpreters. The one exception is RR-N5.

## A. Tests (`<scratch>/A_tests.sh`, `A_tests.txt`, `A_tests_earlier.txt`, `tests.xml`)

- Receipt command, 14 files (the exact list in report 139): **148 passed, 0 skipped, 0 failed**. The JUnit node-ID
  set equals the 148 `focused_tests.nodes` in REPAIR_VERIFICATION.json, and all 14 test-file sha256 match the receipt.
- Our earlier sets, on 2318b0e3:

  | Set | Before | Now |
  |---|---|---|
  | Close/hold named 5 files | 34 passed | 35 passed (+1 is `test_synthetic_value_cache_matches_report_and_cannot_bypass_causal_api`, added at 3644ece9) |
  | Close/hold broad family | 86 passed, 72 skipped | 87 passed, 72 skipped (the recipe-gated integration file, deliberately not supplied) |
  | Dynamic-account named 4 files | 25 passed | 37 passed (+12 new panel tests) |
  | Dynamic-account broad 14 files | 132 passed | 144 passed |

## B. Verifiers (`<scratch>/B_verify.sh`, `B_verify.log`, `B_repair/`, `B_oldverifier.log`)

- `tools/verify_external_audit_repairs.py --engine-root <hs> --test-report <scratch>/tests.xml --output <scratch>/B_repair`
  printed `{"status": "PASS", "accounts": 288}`.
- For all 12 policies × 2 panels × 12 worlds, `evaluate(..., verify=True, details=True)` compares the full account
  dict with the archived replay using exact `!=`: orders, traces, cash, MTM DD, costs, Sharpe, counts and cache parity.
- **My REPAIR_VERIFICATION.json is byte-equal to the committed one.** That file includes the predecessor identity
  098e8719…, the source hashes, the verifier sha 3226924b… and the 12 paired differences.
- The 10 predecessor archive files have the same sha256 before and after the run, and
  `git diff 3644ece9 2318b0e3 -- docs/audits/evidence/dynamic_account_pareto_20261029/` is empty. **The archive is untouched.**
- The verifier allows exactly three changed pinned sources (`protection_reference`, `close_hold_controller`,
  `dynamic_account_panel`) and refuses any other scope. The native engine pins are unchanged.
- The old account verifier `tools/verify_dynamic_account_results.py` on patched code raises `ValueError: SOURCE_PINS`,
  as Musashi states, so that safeguard was not weakened.
- The same verifier also passed 288/288 under NumPy 1.26.4.

## C. Original counterexamples re-run

Call-site adaptations, all marked `MIGRATION` in the scripts:

1. `fixtures_lib.py` and `D_native_fixtures.py`, function `evaluate`:
   - Before: `decide(p, r, r.issued_at, fo, fe)`.
   - After: `decide(p, r, int(e.v[e.next_decision])+HOUR, fo, fe, expected_deadline=int(e.ff[e.pos.seg]))`. The clock
     comes from the engine and the deadline from the position's calendar segment.
   - `terminal_spreads` needed no change in these scripts, because the reports come from the app's `build_report`.
     That function now fills them with `profile.spread_at(forced_flat, market_open)`.
2. `E_adversarial.py` became `C_F10_DA12_DA13.py`:
   - `decide` gains `expected_deadline=int(e.ff[0])`.
   - `decide_synthetic_values` gains `report=` and `expected_deadline=`.
   - The truncated report keeps `terminal_spreads`.
   - The duck-typed env objects passed to `World.close_values` gain `cfg` and `ff`, and the position gains `seg`,
     because `close_values` now reads them. This dependency is **not in the migration list**. Run unadapted, the
     original script fails with `AttributeError: 'E' object has no attribute 'cfg'`, not with a contract error
     (`C_E_adversarial_unadapted_after_err.txt`).
3. Bug fix in my own new harness. The first run of the DA13 block reused one World across mutations, so funding
   doublings carried over. I fixed this with a fresh World per case and re-ran; only the fixed run is reported.

Before numbers are the ORIGINAL scripts on `<am-pre>` (`C_D_before_out.json`, `C_D2_before_out.json`,
`C_E_adversarial_before_out.json`, `C_F07_sweep_before_out.json`, `C_DA24_disc_before_out.json`). After numbers are
the adapted scripts on `<am>`.

### F07: exact TP-through boundary (`C_D*_out.json`, `C_F07_DA24_out.json`)

Values are the maximum absolute difference between the controller and native, in USD, over all common actions.

| Fixture | H | Before | After | Native passive |
|---|---:|---:|---:|---|
| F6 long, bid high = TP + through | 94 | **27.50 (1.066 R)**; controller passive −3.64 vs native 20.79 | **1.8e-15** | TP |
| F6s short, ask low = TP − through | 94 | **17.39 (0.674 R)**; 9.32 vs 20.80 | **0.0** | TP |
| F6 long, 1e-6 inside | 94 | 1.8e-15 | 1.8e-15 | FORCED_FLAT |
| F6x short, 1e-6 inside (new) | 94 | – | 1.8e-15 | FORCED_FLAT |
| F6x long, 1e-6 beyond (new) | 94 | – | 1.8e-15 | TP |
| Pure-function sweep, `protection_exit` vs native `_protect` | 60,000 cases (exact edge, ±1 ulp, d = ±1, 7 through values) | **11,497 mismatches** | **0** | – |

Erratum in our own earlier evidence: the case labelled "F6s_short_TP_through_below" was in fact 1e-6 *beyond* the
boundary, not inside (native TP, and the controller agreed). The genuinely inside short case F6x is added above.

### F08: forced-flat spread (F5)

- F5, with the spread schedule boundary at the flat instant: the max difference goes from **2.841 USD (0.040 R)** to
  **8.9e-16 USD**. Controller and native both give −4.2043, FORCED_FLAT.
- F5c, default schedule: 2.2e-16 before and after.
- Our independent `audit_lib` still differs by 2.84 USD on F5, and by 27.50 / 17.39 USD on F6. It encodes the
  pre-repair semantics on purpose (bar-open spread at the deadline, `(fav−tp) >= through`), so those differences are
  expected. Native cash is the ground truth here.

### F10 / DA11: truncated reports (`C_F10_DA12_DA13_out.json`, blocks `F10_unit` and `F10_native`)

| Case (deadline from the calendar) | Before | After |
|---|---|---|
| Unit, full report to calendar deadline 13 | – | ACCEPTED, HOLD, adv −2.4 R |
| Unit, truncated self-consistent report (deadline 12; the B2 pattern) | B2: ACCEPTED, flips HOLD to CLOSE (+0.5 R) | **REFUSED EXTERNAL_DEADLINE_MISMATCH** |
| Unit, deadline field = calendar but only 2 of 3 bars | – | **REFUSED REPORT_TIME_SUPPORT** |
| Unit, expected_deadline 12 or 14 vs full report | – | **REFUSED EXTERNAL_DEADLINE_MISMATCH** (both) |
| Unit, step 2 report vs calendar 13 | – | **REFUSED EXTERNAL_DEADLINE_MISMATCH** |
| Unit, cached adapter with truncated report | – | **REFUSED EXTERNAL_DEADLINE_MISMATCH** |
| Native world (E_adversarial), 3-bar truncated report | ACCEPTED, HOLD | **REFUSED EXTERNAL_DEADLINE_MISMATCH** |
| Native world, deadline = forced flat but 3 bars | – | **REFUSED REPORT_TIME_SUPPORT** |
| Native world, full report, expected_deadline = flat − 1 h | – | **REFUSED EXTERNAL_DEADLINE_MISMATCH** |
| Misuse: truncated report with `expected_deadline=r.deadline` | – | ACCEPTED, CLOSE (residual; RR-N3) |
| `value_paths` called directly on the truncated report | – | ACCEPTED (residual; RR-N3) |

### DA12: stale/future reports and engine clock (block `DA12`)

| Case | Before (3644ece9) | After |
|---|---|---|
| Direct route, b−1 report at the engine clock | REFUSED | REFUSED STALE_OR_FUTURE_REPORT |
| Direct route, b+1 report | REFUSED | REFUSED |
| Cached adapter, b−1 values at the engine clock | **ACCEPTED, HOLD** | **REFUSED** (b−1 and b+1) |
| Warm `close_values` cache with an injected b−1 / b+1 report | served | **REFUSED** |
| Warm `candidate_bids` cache with an injected b−1 / b+1 report | served | **REFUSED** |
| Whole panel, P1 confirmation trend/increasing/+1, true reports | 1.7950627842 % | **1.7950627842 %** (unchanged) |
| Whole panel, `report(b)` returns b−1, verify=True | **ACCEPTED, 1.8810 %** | **REFUSED**, also with verify=False |
| Whole panel, b+1 reports | – | REFUSED (verify False and True) |
| Stale only on the close route via `close_values` | – | REFUSED (both) |
| Stale only on the value passed to the adapter/direct call | – | REFUSED (both) |
| Stale only on the entry-bid route | – | REFUSED (both) |
| Misuse: `decide(rs, decision_time=rs.issued_at, expected_deadline=rs.deadline)` | ACCEPTED | ACCEPTED (residual; RR-N3) |
| Cached adapter: current report plus values computed from the b−1 report | – | ACCEPTED (residual; RR-N4) |

Entry bids are validated against `v[b]+1h` and `ff[0]` from the World's engine before the cache lookup.

### DA13: cache identity (block `DA13`; each case uses a fresh World, warmed, then mutated)

| Mutation after a warm hit | Before | After |
|---|---|---|
| Entry +10 pips, old SL/TP kept (the E_adversarial witness) | **served stale** | REFUSED CACHE_GEOMETRY |
| Entry +1e-7 with consistent geometry | – | recomputed, equals a cold recomputation |
| Profile (horizon 3 → 6) | – | recomputed, equals cold |
| Side | – | recomputed, equals cold |
| Funding (rollover weights ×2) | – | recomputed, equals cold |
| Market slippage in the World config | – | recomputed, equals cold |
| Report paths +1 pip | – | recomputed, equals cold |
| Env config ≠ World config | – | REFUSED CACHE_EXECUTION_CONFIG |
| **Spread schedule in the World config (0.3 → 6 pips)** | – | **served the warm baseline exactly; differs from cold** (RR-N1) |
| **Entry-bid cache, market slippage changed** | – | **served the warm bids exactly; differs from cold** (RR-N1) |
| **Entry-bid cache, spread schedule changed** | – | **served the warm bids exactly; differs from cold** (RR-N1) |

### DA24: Musashi's eight fixtures, checked by me against native on every action

Files: `C_F07_DA24_out.json` (full menu) and `C_DA24_disc_*_out.json` (his test assertion, verbatim, on both trees).
"Before" and "After" are his test assertion run on 3644ece9 and on 2318b0e3.

| Fixture | Direction | Native reason | Controller vs native, all 96 actions | Before | After |
|---|---|---|---:|---|---|
| gap_close (requested CLOSE preempted) | long / short | SL on all 96 actions | 0.0 / 0.0 | pass | pass |
| ambiguous bar | long / short | SL (stop wins); CLOSE now AGENT_CLOSE | 7.1e-15 / 0.0 | pass | pass |
| through (quote extreme − boundary = 0.0, a true exact edge) | long / short | TP | 1.4e-14 / 0.0 | **fail: 87.54 / 84.09 USD** | pass |
| deadline_spread (his schedule `(0,10),(10,24)`) | long / short | FORCED_FLAT | 7.1e-15 / 3.6e-15 | **pass: last bar-open spread = flat spread = 0.0004, no jump** | pass |
| deadline_spread with boundary at hour 12 (my variant, same construction) | long / short | FORCED_FLAT | – | **fail: 6.23 / 7.96 USD** | pass (7.1e-15 / 2.7e-15) |

The engine's `spread_at` uses hours since market open mod 24. The flat instant is hour 108 → 12 and the last bar
opens at hour 107 → 11, so both are in his `[10,24)` band. The gap and ambiguous fixtures are coverage witnesses that
also pass on the predecessor; they were coverage gaps (F09/DA24), not defects. **The account panel itself still has
no witness of protection preempting a CLOSE:** none of the new tests runs `World.run` through such an event.

## D. Regression search

1. **Can expected_deadline still be sourced from the report?**
   - Not in `app/` or `tools/`:
     - `close_hold_verification` uses `e.ff[e.pos.seg]`.
     - The panel close route uses `e.ff[e.pos.seg]`; the bid route uses `self.env.ff[0]`, which is the single-segment
       assumption, not report-sourced.
     - `prefix_report` gets its deadline from the caller.
   - The unit-test helper `decide()` in `tests/unit/test_close_hold_controller.py` passes `r.issued_at` and
     `expected_deadline=r.deadline`, so those tests never exercise the external binding (RR-N3).
   - The API cannot detect a caller that copies the report's own fields (misuse rows above). `value_paths` remains
     public without clock or deadline arguments.
2. **Does the cache key include everything that affects a valuation?** For the close-value cache, yes. Its key
   `(PositionView, report_identity, fo, fe)` covers every input of `value_paths`, and I confirmed recomputation for
   entry, profile, side, funding, slippage and report content. **The report cache `self.reports[b]` and the bid cache
   `(b, report_identity)` do not include the execution config.** A spread-schedule change therefore reaches the close
   valuation through a stale report, and cost or spread changes reach entry bids stale (RR-N1). This is latent:
   in-repo panels never mutate the config after construction, and all 288 accounts are exact.
3. **Do valuations still equal native cash within 1e-12 USD?**
   - The 16,320-action check from our close/hold audit (194 corpus positions, every 9th, first tick, every common
     action) gives max **2.842e-14 USD** on the patched code, identical to before.
   - I also re-ran all 8,720 close/hold rollouts and 25 prefix witnesses of plan 133 on patched code. Under the
     evidence interpreter, **CASES bytes are equal to the committed CASES.json.gz** and all 25 witnesses are equal
     (254 s; `D_closehold_8720_out.json`).
   - Under NumPy 1.26.4, 2,450 records differ, all in `trace[].expected_close_net`, max 2.84e-14. There are 0 action
     changes and 0 metric changes. The cause is the dot-product rounding of `w@values[:,0]`: the patched and
     pre-repair `value_paths` arrays are bitwise equal (`base_interpreter_run/D_closehold_diag_out.json`). This is RR-N5, an environment
     note, not a patch defect.
4. **Did any of the 288 accounts change only because a fixture was relaxed?** No:
   - The verifier compares full account dicts with exact `!=` against the immutable archive.
   - The diff changes no tolerance: CACHE_DIRECT_PARITY 1e-10, NATIVE_ACCOUNT_PARITY 1e-7 and CASH_CONSERVATION 1e-7
     are all untouched.
   - The changes to the old tests only add the new mandatory arguments. No assertion was weakened or removed.
   - The equality is expected for three reasons. Each panel world has one constant spread, so terminal = bar-open
     spread. Panel prices are not tick-quantized, so the boundary is never hit (we had already measured 0 affected
     bars). And all reports already satisfied clock = v[b]+1h and deadline = forced flat.

## E. Interpretation changes (report 139)

| Required change | Done? | Where |
|---|---|---|
| All 12 paired P1/Only3h differences | **Yes.** All 12 equal my values to 6 decimals; mean +0.050199, 4/12 positive, ex-pair −0.058090 | 139 table; REPAIR_VERIFICATION `paired_p1_only3_confirmation` (byte-equal on rerun) |
| Withdraw P1 superiority | **Yes** ("P1 is not promoted over Only3h"; handoff "withdraws any inference of robust P1 superiority") | 139, handoff |
| 3-draw structure and shared noise | **Yes** ("Three independent price draws per panel, not twelve… Noise was shared across worlds and its bias sign followed the mirror") | 139 §Scientific 1 |
| Duplicate/exposure frontier, uncertainty-gene coupling | **Yes** (hash tie-break, P2/P3 same actions, "Entry and close share one uncertainty-penalty gene") | 139 §4; 138 |
| 14 preempted stops | **Yes** ("14 of P1's 20 confirmation early closes preempt a native SL") | 139 §2, handoff |

Not quoted: the bootstrap CIs ([−0.076, +0.212] for weeks, [−0.094, +0.275] for the cluster bootstrap). The point
estimate and sign counts are published and the inference is withdrawn, so this is acceptable.

Our MAJOR dynamic-account findings:

| ID | Status | Commitment |
|---|---|---|
| DA15 foresight (SL-first paths foreseen; label-dependent bias; shared noise) | **Addressed in interpretation; correction deferred to the new panel** | Plan 138 "Next experimental version": independent price draws, separate report-quality levels, delayed/missing/biased reports, correlated path-shape errors, missed SL modes; expectations frozen before results. A label-independent bias sign is implied ("confounds to remove") but not stated as a requirement. |
| DA16 draws (3 independent per panel) | **Addressed in interpretation; deferred** | ≥10 independent draws per family; spread crossed independently of volatility; mirrors as pairs; old confirmation not reused as unseen. The cluster/pair bootstrap is not adopted explicitly. |
| DA21 early-close attribution (no-early-close 3.01/0.54 → 0.91/1.88) | **Addressed in interpretation** ("mainly a demonstration of foresight-assisted closing, not proof of a robust allocation mechanism"; "early-close benefit dominates rotation benefit") | Our correction, "report G3-type ablations with every frontier result", is **not adopted as an explicit rule**. Plan 138 only commits to frozen fixed controls on identical paths and exposure before any new search. Deferred without an explicit commitment. |
| DA22 adverse stress preserved shape | **Addressed in interpretation** ("The original adverse-bias test still preserved future path shape") **and deferred with commitment** | Plan 138 lists shape-destroying stresses (delayed/missing/biased, correlated path-shape errors, missed SL modes) and report-quality levels. A break-even sweep is not mentioned. |

Other items:
- DA04 / DA19: disclosed. "Next versions must separate exposure from action diversity and declare tie rules"; splitting the gene is not committed.
- DA07: disclosed ("not validated as a reliable annual performance estimate").
- DA17: added ("P3 is also dominated by equal_guarded").
- DA23: fixed (148 node IDs enumerated).
- DA26 (side-asymmetric sizing from the positive-only funding reserve) and DA14 (constructor flag, not a capability boundary): **not mentioned (ignored; both INFO)**.
- Close/hold F18 (signed tie band): explicitly kept as a future variant, not repaired.
- Close/hold F15 (near-boundary TP controls in the matrix): not named in plan 138.

## F. Retsu vs our audits

**Retsu has, we lack:**
1. *Spread–volatility cost confound as a finding.* Volatility switches spread 0.5 vs 3 pips and shares the Gaussian
   sequence, and P1's only large gains sit on the 0.5-pip weeks. We recorded the spread tie inside DA16 but did not
   draw the cost-regime inference.
2. *Frontier gene statistics:*
   - `close_persistence` = 1 on all 632 rows;
   - `close_margin` = 0 on 449/632;
   - uncertainty max 0.137 on the frontier vs 1.994 searched; trend min 0.034;
   - size–return correlation about 0.995.
   Hence "the return end is oracle trust plus size". We stated only qualitatively that margin ≈ 0 and persistence 1
   encode clairvoyance.
3. *The scalar Nash violation is not a Pareto improvement* (disabling 12h-buy raises return but worsens DD), and the
   search never scores the joint `interval_best` schedule, so allocation regret is unmeasured. We verified ε (DA20)
   without this reading.
4. *Confirmation not untouched:* the smoke run saw the confirmation panel before the final run.

**We have, Retsu lacks:**
1. Every code defect: F07/F08/F10 carried into the panel (DA09–DA11), DA12 vacuous stale guard (1.795 → 1.881 %),
   DA13 stale cache hit.
2. The full 17,018-candidate re-evaluation in reverse order (DA02) and an independent Kung frontier (DA03).
3. Independent accounting of all 1,173 trades (DA05) and the last-hour bar-107 entries (DA06).
4. The representative hash tie-break inside duplicate groups, with confirmation ranges (DA04).
5. Sharpe sqrt(365) on 6 days and the cross-week ranking reversal (DA07).
6. SL-first foresight: 14/20 preempted stops, label-dependent bias, noise shared by all 12 worlds (DA15).
7. P3 also dominated by equal_guarded (DA17).
8. The uncertainty gene drives entry and close (DA19).
9. Ablations: no-early-close (DA21), rotation G3b/G3c, adverse stress preserving shape and optimistic SL arm (DA22).
10. No preemption witness at account level (DA24), and side-asymmetric sizing (DA26).
11. Paired and cluster bootstrap CIs.
12. The whole close/hold component audit: F09 at-open-only matrix, F15 TP controls far from the boundary, F16 guarded
    default, F18 tie band.

**Overlap:** P1's edge is carried by one mirrored pair (Retsu 1 = our G/DA16); P2/P3 are an exposure dial (Retsu 3 =
DA18); no stop exits in the named policies (Retsu 4 ⊂ our DA15/D count of 1 SL in 288); the test count of 128 is not
reproducible (Retsu ≈ DA23).

## Findings

Severity: BLOCKER / MAJOR / MINOR / INFO. The verdict words VERIFIED / REFUTED[…] / UNVERIFIED[…] are exclusive.

| ID | Severity | Claim | Verdict | Witness | Correction |
|---|---|---|---|---|---|
| RR-A1 | INFO | 148 enumerated focused tests pass with zero skips | VERIFIED | `<scratch>/A_tests.txt`, `tests.xml` (node set = receipt; 14 file hashes match) | – |
| RR-B1 | INFO | 288/288 predecessor accounts equal the patched executions exactly; receipt reproducible | VERIFIED | `<scratch>/B_verify.log`; `B_repair/REPAIR_VERIFICATION.json` byte-equal to committed | – |
| RR-B2 | INFO | Original archive bytes untouched; old verifier still rejects patched sources | VERIFIED | sha256 before/after equal, empty git diff vs 3644ece9; `B_oldverifier.log` SOURCE_PINS | – |
| RR-F07 | MINOR (was) | Reference protection equals native arithmetic at the exact TP-through boundary, both directions (AR01) | VERIFIED | F6 27.50 → 1.8e-15 USD, F6s 17.39 → 0.0; sweep 11,497 → 0 of 60,000; `C_D*_out.json`, `C_F07_*_out.json` | – |
| RR-F08 | MINOR (was) | Deadline liquidation uses the deadline spread (AR02, controller side) | VERIFIED | F5 2.841 → 8.9e-16 USD; boundary-12 variant 6.23 / 7.96 → ≤7.1e-15 USD | – |
| RR-F10 | MAJOR (was) | External calendar owns the deadline; truncated, short-bar and mismatched reports are rejected (AR03) | VERIFIED | `C_F10_DA12_DA13_out.json` F10_unit / F10_native: all REFUSED; full report HOLD −2.4 R accepted | – |
| RR-DA12 | MINOR (was) | Stale and future reports are refused on direct, cached, close-cache and entry-bid routes, warm caches, verify on or off (AR04) | VERIFIED | DA12 block: 1.881 % stale run now REFUSED; true run 1.7950627842 % unchanged | – |
| RR-DA13 | MINOR (was) | Close cache never serves another position/report: entry, profile, side, funding, slippage, report (AR05, close cache) | VERIFIED | DA13 block: geometry REFUSED; 6 mutations recompute and equal cold recomputation | – |
| RR-N1 | MINOR | "Cache binds position/report/funding/costs" (139) and "changed reports, costs or funding cannot reuse wrong key" (TRACEABILITY AR05) hold for every cache | REFUTED[`World.reports` is keyed by bar and `World.bids` by (bar, report_identity), without execution config; after a spread-schedule change `close_values` serves the warm baseline exactly (differs from cold); after a slippage or spread change `candidate_bids` serves the warm bids exactly (differs from cold)] | DA13 block `hit_changed_spread_schedule`, `bids_hit_changed_*` | Key both caches by an execution-config digest (profile, sizing, rolls, risk_config), or refuse when `self.env.cfg` differs from the config captured at construction; no rerun needed for the archive |
| RR-N2 | MINOR | The native deadline-spread fixtures "with spread jump in both directions" witness AR02 (TRACEABILITY AR02; 139 table) | REFUTED[his schedule `(0,10),(10,24)` puts last bar open (hour 11) and flat (hour 12) in the same band, both 0.0004; the test passes on 3644ece9 (err 7.1e-15 / 2.7e-15)] | `C_DA24_disc_before_out.json` (deadline_spread pass before; boundary12 fails before by 6.23 / 7.96 USD) | Move the schedule boundary to the flat hour, e.g. `((0,12,.00005),(12,24,.0004))`, and assert the before/after discrimination |
| RR-DA24 | MINOR | Native boundary coverage: gap-SL preempts CLOSE, ambiguous bar, exact TP boundary, terminal spread, both directions | VERIFIED for gap, ambiguous and through on all 96 actions (≤1.4e-14 USD; through is a true exact edge, quote − boundary = 0.0) | `C_F07_DA24_out.json` DA24 rows | Terminal spread: see RR-N2 |
| RR-DA24b | MINOR | The account panel exercises protection preempting a discretionary CLOSE (original DA24 scope) | REFUTED[the new fixtures use the fixed-position `create_case`, never `World.run`; the account panel still has 0 preemptions of 564 closes] | test_dynamic_account_panel.py source; our Q4_counts | Add one World-level witness (gap through SL on a CLOSE tick) in the plan-138 panel |
| RR-N3 | INFO | The independent clock/deadline cannot be sourced from the report under test | UNVERIFIED[enforced in all app/tools callers, but not enforceable by the API: `decide(r.issued_at, expected_deadline=r.deadline)` accepts the truncated report and flips to CLOSE; public `value_paths` has no context check; the unit-test helper `decide()` sources both from the report] | F10_unit `MISUSE_*`, `value_paths_direct_truncated`; DA12 `MISUSE_direct_stale_at_report_clock` | Have the test helper take an explicit calendar; make `value_paths` private or add the context arguments; consider a calendar object the report cannot construct |
| RR-N4 | INFO | The cached adapter binds values to the report | UNVERIFIED[it checks step, shape and weights only; values computed from the b−1 report with the current report are ACCEPTED] | DA12 `cache_adapter_current_report_with_stale_values` | Test-plane only; pass `report_identity` with the values or keep the adapter private to `World` |
| RR-N5 | INFO | Close/hold CASES bytes are reproducible under patched code | VERIFIED under NumPy 2.5.1 (8,720/8,720 byte-equal, 25 witnesses equal); under NumPy 1.26.4, 2,450 records differ ≤2.84e-14 in `expected_close_net` only (dot rounding; valuations bitwise equal; 0 action/metric changes) | `D_closehold_8720_out.json`; `base_interpreter_run/D_closehold_8720_out.json`; `base_interpreter_run/D_closehold_diag_out.json` | Record the interpreter/NumPy version in the manifests; compare `expected_close_net` with tolerance across environments |
| RR-N6 | INFO | 16,320-action valuation = native cash within 1e-12 USD after the patch | VERIFIED (max 2.842e-14 USD, unchanged) | `C_D2_after_out.json` | – |
| RR-N7 | INFO | No account changed only because a fixture or tolerance was relaxed | VERIFIED (exact dict equality vs the immutable archive; no tolerance line changed; old-test edits only add mandatory arguments) | `git diff 3644ece9 2318b0e3 -- app tests tools` | – |
| RR-N8 | INFO | The migration list is complete | REFUTED[`World.close_values` now also requires `e.cfg`, `e.ff` and `e.pos.seg`; a duck-typed caller fails with AttributeError, not a contract error] | `C_E_adversarial_unadapted_after_err.txt` | Add this to the migration note, or validate the env object explicitly |
| RR-E1 | INFO | Report 139 publishes the 12 pairs, withdraws P1 superiority, states 3 draws and shared noise, discloses the duplicate/exposure frontier and gene coupling, names the 14 preempted stops | VERIFIED | 139 text; REPAIR_VERIFICATION paired table equals my numbers | – |
| RR-E2 | MINOR | DA21 correction (report no-early-close ablations with every frontier result) adopted | UNVERIFIED[139 accepts the interpretation; plan 138 commits only to frozen fixed controls before a new search; no ablation-reporting rule] | 138 "Next experimental version" | Add G3a/G3b-type ablations to the plan-138 acceptance list |
| RR-E3 | INFO | DA26 (side-asymmetric sizing) and DA14 (capability boundary) dispositioned | UNVERIFIED[not mentioned in 138/139 or the handoff] | – | Add a one-line disposition |
| RR-T1 | INFO | Shared component expresses our CLOSE-favouring tie band | REFUTED[`CloseConfig` still rejects `margin_risk < 0`; 139 keeps it as a future variant] | close_hold_controller.py `CloseConfig.__post_init__` | Use the named variant below |

## Migration notes for Satoshi's lane

**Report adapter, mandatory fields.**
- `PathReport(issued_at, observed_through, first_fill, step, deadline, paths[S,H,4], spreads[S,H], weights[S], provenance, terminal_spreads[S])`.
- `terminal_spreads[s]` is the spread **at the forced-flat instant** (close time of the last bar). Take it from our
  cost schedule as `profile.spread_at(forced_flat_ns, market_open_ns)`, not from the last bar-open spread. Our
  hour-dependent schedules make F08 live (a 19:00 UTC close puts the flat on a schedule boundary).
- The contract requires `paths[s,-1,CLOSE] − terminal_spreads[s]/2 > 0`.

**Decision call.**
- `decide(p, r, decision_time, opening_funding, ending_funding, *, expected_deadline)`.
- `decision_time` comes from our engine clock: the next-open fill time of the decision bar (`v[b]+1h` on the
  contiguous hourly grid, `v[b+1]` in general).
- It must equal `r.issued_at == r.first_fill`, with `r.observed_through ≤ decision_time`.
- `expected_deadline` comes from **our calendar's forced flat for the position's segment** (`ff[seg]`), never
  `r.deadline`. The report must cover exactly `(expected_deadline − first_fill)/step` bars.
- For multi-week runs, bind per segment. Musashi's `World` hard-codes `ff[0]`.
- If we reuse `decide_synthetic_values`, also pass `report=` and `expected_deadline=`, and compute the values from
  that same report (RR-N4).
- If we reuse `prefix_report`, pass `terminal_spread=`.
- Callers that pass a duck-typed env to `World.close_values` must supply `cfg`, `ff` and `pos.seg` (RR-N8).
- If we reuse `World`, build a new World per execution config, never mutate one (RR-N1).

**Signed tie policy as a named variant `SATOSHI_TIE_CLOSE_EPS_v1`.** Keep Musashi's controller untouched and wrap its output:
- Configure `CloseConfig(uncertainty_penalty=0, margin_risk=0, persistence=1, urgent_risk=None)`, so that
  `advantage_risk = (E[C] − max_j E[W_j]) / R`.
- Action = CLOSE iff `advantage_risk ≥ −EPS_R`, with `EPS_R = 0.02·σ·units / R` (σ in price units). This is our rule
  "hold only if hold beats close by more than EPS".
- Do not use the controller's own `action`, streak or persistence; the Kalman filter stays upstream in the adapter.
- Report every result under both `MUSASHI_TIE_HOLD` (default: CLOSE iff `advantage_risk > max(margin, 1e-12)`) and
  `SATOSHI_TIE_CLOSE_EPS_v1`, side by side. The two disagree exactly in `−EPS_R ≤ advantage_risk ≤ 1e-12`.
- Never change the other lane's default.

**Marginal-to-joint adapter (Musashi's request).** Our first-passage curves give marginal hit probabilities by
horizon. The controller needs joint OHLC paths through the deadline. The adapter must:
- sample paths (our conditional-mean path + AR(1) innovations + an H/L range model, with independent draws per
  world);
- pass first-passage consistency tests: sampled P_TP(h), P_SL(h) and survival-conditional open distributions must
  match the head within Monte-Carlo error;
- carry the modelling risk itself; the controller does not.

The adapter is not yet demonstrated; it is open work for our lane.

## Commands

All commands ran from `<scratch>` with `. env.sh` (trading-stack interpreter, CPU-only variables), each under
`crispdm-run -q -W 1800 -m 2400M -t 7200s -n sat-rr-<step> -- bash <script>`:

| Step name | Script | Outputs |
|---|---|---|
| versions | `versions.sh` | `versions.txt` |
| A-tests | `A_tests.sh` | `A_tests.txt`, `tests.xml` |
| A-earlier | `A_tests_earlier.sh` | `A_tests_earlier.txt` |
| B-verify | `B_verify.sh` | `B_verify.log`, `B_repair/` |
| B-oldverifier | `B_oldverifier.sh` | `B_oldverifier.log` |
| C-native-fixtures | `C_D.sh` | `C_D_{before,after}_out.json`, `C_D2_{before,after}_out.json` |
| C-adversarial | `C_adv.sh` | `C_E_adversarial_before_out.json`, `C_F10_DA12_DA13_out.json`, `C_F07_sweep_before_out.json`, `C_F07_DA24_out.json` |
| C-da24-discrimination | `C_da24disc.sh` | `C_DA24_disc_{before,after}_out.json` |
| D-closehold-8720 | `D_8720.sh` | `D_closehold_8720_out.json` |
| D-closehold-diag | `D_diag.sh` (base-interpreter pass only) | `base_interpreter_run/D_closehold_diag_out.json` |

The first-pass outputs from the base interpreter are kept in `base_interpreter_run/` for RR-N5.
