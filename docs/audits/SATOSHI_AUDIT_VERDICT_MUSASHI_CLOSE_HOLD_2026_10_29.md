# Independent audit: Musashi's position-conditioned close/hold controller (agent-multi @8bf75c81)

Date 2026-10-29. Auditor working for Satoshi (expert-guided lane). Scope: the COMPONENT verdict requested in
`docs/audits/SATOSHI_AUDIT_REQUEST_CLOSE_HOLD_COMPONENT_2026_10_29.md` (plans 129-133), not a strategy verdict.
CPU only (`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`), every heavy step under
`crispdm-run -q -W 1800 -m 2400M -t 5400s` (peak RSS observed 0.2 GB). No tracked file modified, nothing committed,
no training/DEAP/feature selection, no financial rows read (all inputs are the synthetic 2020-01 matrix).

Placeholders: `<am-worktree>` = detached agent-multi worktree at 8bf75c81; `<hs-checkout>` = existing clean
heuristic-strategy worktree at exactly e0a1c2b962f4 (reused; HEAD re-checked before and after the runs);
`<scratch>` = the audit scratch directory. Python 3.12.13 / NumPy 2.5.1 / pandas 3.0.3 (same as the evidence).

## Verdicts

| Area | Verdict | One-line reason |
|---|---|---|
| (a) execution / metric correctness | **ACCEPT_WITH_CHANGES** | Byte-exact reproduction; valuation equals native cash to 2.8e-14 USD on 16,320 actions; two latent valuation mismatches (limit-through boundary, forced-flat close-time spread) and a matrix that never exercises intrabar or deadline exits. |
| (b) decision rule under supplied reports | **ACCEPT_WITH_CHANGES** | Formula, common-action semantics, ties-HOLD, persistence/urgency all verified independently; the controller accepts a truncated report (earlier self-consistent deadline) and can flip HOLD to CLOSE; tie policy not configurable to our CLOSE-favouring band. |
| (c) signal sufficiency | **NOT_EVALUATED** | Interface is sufficient for the open-loop finite-action valuation by construction; no producer exists, the prefix producer fails, and the matrix is too easy to stress it (0 destructive closes even at an adverse bias of 0.5 x stop distance). |
| (d) full-mechanism eligibility | **NOT_EVALUATED** | Prerequisites absent by declaration: learned/causal report adapter, entry arbitration, account-level trajectory, joint optimization. |

## A. Focused tests

`<scratch>/A_tests.txt`, `<scratch>/A_tests_broad.txt`.

- Named files (`test_close_hold_controller.py`, `test_close_hold_reports.py`, `test_close_hold_trace_audit.py`,
  `test_early_close_metrics.py`, `test_absolute_exit_corpus.py`): **34 passed**.
- Broader early-close family (+ `test_early_close_audit`, `test_early_close_grid_audit`, `test_horizon_risk_rules`,
  `test_optimal_exit`, `test_protection_reference`, `tests/integration/test_quote_early_close_expectations.py`):
  **86 passed, 72 skipped** (the integration file skips without `QUOTE_ENGINE_ROOT`/`PINNED_RECIPE_PATH`, a pinned
  financial recipe, deliberately not supplied).
- The "134 focused tests" of REPLAY.json/133 is not enumerated anywhere in the delivery; I could not reproduce
  that count (finding F03).

## B. Reproduction

`<scratch>/B_run.log`, `<scratch>/B_out/`, `<scratch>/B131_run.log`, `<scratch>/B131_out/`, `<scratch>/pin_check.txt`.

- Engine pin check: all 41 source hashes in MANIFEST (`agent_multi` 12, `heuristic_strategy` 29) match the
  checkouts (0 mismatches). Engine HEAD e0a1c2b962f41039bb59a76614b70e326f2cf0c1, clean.
- `tools/verify_close_hold_controller.py --engine-root <hs-checkout> --output <scratch>/B_out`: COMPLETE in
  253.5 s, 8,720 rollouts, 25 prefix witnesses. MANIFEST.json, REPORT.json and CASES.json.gz are **byte-equal**
  to the committed evidence. Identity `e490dbeb63e09b493e65cd85bc07ae6dd58f69bbb9338a4b66c98a6e48fbbc94`,
  cases `8e0d495ea658ccf493dd3d4630b18512a543a79cb7dc211bc0a5677145879d35`.
- Plan 131 `tools/verify_absolute_early_close.py` (cheap, ~12 s): also **byte-equal** (identity `7f6c0f46…`,
  cases `79035aeb…`, max cash error 2.842e-14 USD).
- Reproduced aggregate table (all 1,744 positions):

| Arm | Early closes | Destructive | Missed optimal | Mean regret/R | Capture |
|---|---:|---:|---:|---:|---:|
| ideal_immediate | 1152 | 0 | 0 | 2.6e-17 | 1.000000 |
| ideal_guarded | 1152 | 0 | 552 | 0.063218 | 0.941054 |
| noisy_immediate | 1152 | 0 | 0 | ~0 | 1.000000 |
| noisy_guarded | 1152 | 0 | 582 | 0.071463 | 0.933279 |
| prefix_guarded | 0 | 0 | 1152 | 1.071048 | 0.000000 |

## C. Formula check (request item 2)

Code (`app/close_hold_controller.py` l.135-146): `delta = values[:,[0]] - values[:,1:]`; `mean = w@delta`;
`sd = sqrt(w@(delta-mean)^2)` (weighted population sd); `advantage = min(mean - z*sd)/p.risk`;
`qualifies = advantage > max(margin, 1e-12)`; `urgent = qualifies and advantage >= urgent_risk`;
`close = qualifies and (urgent or streak >= persistence)`; streak continues only for the same position identity
and `decision_time == last_tick + step`. This is exactly `min_j(mu_j - z*sd_j)/initial_risk` over the
samples of `D_sj = C_s - W_sj`, with W covering every later open close plus the passive exit.

Independent check (`<scratch>/audit_lib.py`, my own first-passage valuation, no import of
`protection_reference` or controller math; `<scratch>/C_formula_check.py`): per-sample values are not stored in
CASES, so the reports were rebuilt by replaying the native env along the stored actions and recomputed from
scratch. Three cases (matrix_0000 long gain, matrix_0288 short loss, matrix_1440 long tp 24h) x three arms
(noisy_guarded, ideal_guarded, noisy_immediate), 20 ticks: max |advantage error| **4.4e-16** (tolerance 1e-12),
max |expected close net error| 1.4e-14 USD, persistence/urgency/action reimplementation **0 mismatches**, native
exits identical. Extension to all arms and positions: `<scratch>/C_formula_check_ALL_out.json` (see C-ALL below).
500 random reports (S 1-5, H 1-7, random weights, z in [0,2], both sides): identity error 0.0
(`<scratch>/E_common_action_out.json`, E4).

Ties HOLD (strict `>` with a 1e-12 R floor): verified; e.g. native fixture F1 (next-open gap through SL) gives
advantage exactly 0.0 and HOLD. Persistence reset on gap/position change and urgency bypass: unit test
`test_persistence_urgency_gap_reset_and_serialization` (t 13 -> 15 gap resets, new identity resets, urgent 0.5
bypasses persistence 5) passes, and my independent replay matched every stored streak.

C-ALL (beyond the 3 requested cases): every one of the 1,744 positions x 5 arms (8,720 rollouts, 62,834 decision
ticks including the prefix arm) recomputed independently: max |advantage error| **8.9e-16**, max expected-net error
2.8e-14 USD, **0** qualifies/streak/urgent/action mismatches, every native exit reason identical (4 min 23 s,
0.17 GB peak).

## D. Native priority, quotes, fees, financing (request item 3)

From heuristic-strategy e0a1c2b `app/short_horizon/expert_guided/env.py` (`step`, `_risk_hit`, `_close`,
`_financing_events`, `quote_price`, `_fill_cost`) and `app/short_horizon/execution.py` (`_protect`):

1. At the next open k (decision after bar b, k = b+1): `_risk_hit(k, open_only=True)` runs first, on the EXIT-side
   quote (long: bid = mid - spread/2; short: ask = mid + spread/2; spread at the bar-open time). A gap through SL
   fills at the open quote with stop slippage; a gap through TP fills at the open quote (better than TP) with zero
   slippage. Inclusive comparisons (`o <= sl`, `o >= tp`). An agent CLOSE cannot pre-empt this.
2. Otherwise an agent CLOSE fills at the open quote with market slippage (exit_ns = open time).
3. Then intrabar protection on bar k: SL (adverse extreme touches SL, fill at SL, stop slippage) is checked before
   TP (favourable extreme beyond TP by `limit_through`, fill at TP, zero slippage) so the stop wins an ambiguous
   bar; intrabar quotes and costs use the bar-open spread, financing runs to exit_ns = open + 1 h.
4. Margin closeout (stop = max(SL, margin level)), then forced flat at the CLOSE of the last bar before the flat
   time; that close is quoted with the spread at exit_ns = open + 1 h (the close time), market slippage.
5. Cash: entry at the entry-side open quote; spread is not double-charged (gross is credited back the spread
   attribution); net = side*units*(exit quote - entry quote) - units*(entry market slip + exit slip +
   2*commission/lot) - financing. Financing: rollover instants market_open + 24k inside the segment, entry_time <
   t <= exit_ns, side-specific per-lot rate, triple on index 3.

`value_paths` replicates 1-3 and 5. Own fixtures (`<scratch>/D_native_fixtures.py`, `<scratch>/D2_native_more.py`,
outputs `*_out.json`), each comparing the controller's value of EVERY common action (close at each later open, and
passive) with the native engine actually executing that action on a deep copy, default RealisticProfile
(hour-dependent spreads, slippage, commission, swaps):

| Fixture | H | max abs controller-native (USD) | Close-now native reason | Immediate decision |
|---|---:|---:|---|---|
| F1 long, next open gaps through SL | 94 | 1.4e-14 | SL (protection pre-empts CLOSE) | HOLD, adv 0.0 |
| F2 short, next open gaps through TP | 94 | 7.1e-15 | TP at open (54.09, better than TP) | HOLD, adv 0.0 |
| F3 long, CLOSE at open vs same-bar SL touch | 94 | 3.6e-15 | AGENT_CLOSE -14.54 vs passive SL -25.79 | CLOSE, adv 0.436 |
| F3b short, ambiguous bar (SL and TP) | 94 | 7.1e-15 | passive SL (stop wins) | CLOSE, adv 0.524 |
| F4 long, Monday rollover between now and waits | 95 | 3.6e-15 | 84 funded actions | HOLD |
| F4b short, Wednesday triple rollover, flat to deadline | 47 | 6.7e-16 | passive FORCED_FLAT | HOLD |
| F5c forced flat, default schedule | 7 | 2.2e-16 | FORCED_FLAT | HOLD |
| F7 two positions same market, tight stop | 94 | 3.6e-15 | AGENT_CLOSE vs SL | CLOSE, adv 1.229 |
| F7 two positions same market, wide stop | 94 | 1.4e-14 | passive TP 54.88 | HOLD, adv -0.818 |
| Corpus sample, every 9th position (194), first tick | - | 2.84e-14 over 16,320 actions | - | - |
| **F5 forced flat, spread changes at the flat instant** | 7 | **2.84 (0.040 R)** | native -4.2043 vs controller -1.3636 | - |
| **F6 long, bid high exactly TP + limit_through** | 94 | **27.50 (1.066 R)** | native TP, controller no TP | - |
| **F6s short, ask low exactly TP - limit_through** | 94 | **17.39 (0.674 R)** | native TP, controller no TP | - |
| F6/F6s 1e-6 inside the boundary | 94 | <=1.8e-15 | agree | - |

Mismatch 1 (F6/F6s): native `_protect` tests `h >= tp + through` (short `l <= tp - through`); the controller (and
the 131 reference `app/protection_reference.py`) test `direction*(favourable - tp) >= through`. These are not
equivalent in floating point; at the exact boundary, which tick-quantized quotes reach, the native engine fills TP
and the controller values a non-TP path. Fix: use the native expression verbatim.

Mismatch 2 (F5): the native forced-flat close uses the spread at the close time (open + 1 h); the report contract
has one spread per bar (bar open), so a schedule boundary at the flat instant is unrepresentable. Zero with the
default schedule on 22:00 closes (F5c); reachable on early session closes (e.g. a 19:00 UTC close puts the flat at
the 9 h schedule boundary). Fix: carry a terminal close-time spread or let the caller supply it.

Coverage: in the committed 1,744-position matrix **every passive exit is an at-open gap** (gain 576 SL, loss 576
SL, tp 576 TP, gap 16 SL; bar width 0.02 x stop distance). No intrabar SL/TP, no limit_through, no forced-flat passive
outcome was ever executed natively in 131/133 evidence; those paths were covered only by zero-cost unit tests. My
fixtures above close that gap for correctness, but the frozen matrix itself does not support the request's
"before intrabar barriers" claim.

## E. Common action vs sample-specific foresight (request item 4)

`<scratch>/E_common_action.py`, `<scratch>/E_common_action_out.json`. Code: each column of `values[:,1:]` is one
closing time common to all samples; the minimum is taken over column statistics, never `E[max]`.
Long, entry 100, SL 99, TP 110, zero costs, R = 1:

- E1, both samples touch SL (at different bars), weighted-mean path minimum low 99.05 (never touches): per-path
  values C=0, W1=-0.7, W2=-1, passive -1, so the controller **CLOSEs** (adv 0.70); the same controller fed the mean
  path sees +0.05/+0.25/+0.25 and **HOLDs** (adv -0.25).
- E2, only one sample touches SL, mean path minimum low 99.05: controller **HOLDs** (E W = +0.2/+0.3/+0.3,
  adv -0.30); mean-path controller **CLOSEs** (adv +0.05). The nonlinearity flips the decision in both directions.
- E3, sample-specific foresight: E over samples of max_j W_sj = 1.0 versus max_j E W_j = -0.5; the controller
  CLOSEs (adv 0.5) where a foresight rule would HOLD.
- E5 two positions, same report (both sides): tight stop CLOSE, wide stop HOLD; native counterpart F7 above.

## F. Prefix diagnosis (request item 7)

`<scratch>/F_prefix_trace.json` (tool rerun, JSON-equal to PREFIX_TRACE_AUDIT.json), `<scratch>/F_independent.py`,
`<scratch>/F_independent_out.json`. Confirmed: 864 first positive requests, 0 executable early requests, 880 with no
positive request, in both ablations. Independent recount over all 16,528 prefix ticks: 864 qualifying ticks in
total (one per affected position); **every one** is at the decision whose fill bar equals the passive gap-exit bar;
no positive request at ANY tick is executable. 864 = 576 gain + 288 loss; 880 = 288 loss + 576 tp + 16 gap. The
cause is the 8-bar median-drift extrapolator lagging a 5-bar collapse that crosses SL by a gap, as stated. This
says nothing about selected features or SAC.

## G. Noise stress (request item 6)

Antithetic noise preserves the weighted mean path: max |mean - truth| = 2.2e-16 over 3,428 ticks.
Controls were declared before execution in `<scratch>/G_PROTOCOL_PROSPECTIVE.md` (G1, then addendum G2 declared
after G1 and before G2 ran). Driver `<scratch>/G_noise_controls.py` imports the harness without editing it.
Support: every 9th position (194: gain 64, loss 64, tp 64, gap 2). The unbiased baselines match the committed
CASES for those positions exactly. Outputs `<scratch>/G_noise_controls_out.json`, `<scratch>/G2_noise_controls_out.json`.

| Report family | Config | Early | Destructive | Missed optimal | Capture |
|---|---|---:|---:|---:|---:|
| IDEAL / NOISY baselines | immediate | 128 / 128 | 0 / 0 | 0 / 0 | 1.0000 / 1.0000 |
| IDEAL / NOISY baselines | guarded | 128 / 128 | 0 / 0 | 62 / 65 | 0.9398 / 0.9330 |
| BIAS_FAV_NOISY k0.5 / k1 (1e-4 sqrt h) | immediate | 128 / 128 | 0 / 0 | 2 / 6 | 0.9999 / 0.9995 |
| BIAS_FAV_NOISY k0.5 / k1 | guarded | 128 / 128 | 0 / 0 | 66 / 70 | 0.9305 / 0.9156 |
| BIAS_FAV_IDEAL k0.5 / k1 | immediate | 128 / 128 | 0 / 0 | 2 / 6 | 0.9999 / 0.9995 |
| BIAS_ADV_NOISY k0.5 / k1 | immediate | 128 / 128 | 0 / 0 | 1 / 7 | 1.0000 / 0.9989 |
| BIAS_SL_FAV_NOISY k0.25 / k0.5 (x stop distance) | immediate | 128 / 128 | 0 / 0 | 6 / 8 | 0.9995 / 0.9984 |
| BIAS_SL_FAV_NOISY k0.25 / k0.5 | guarded | 128 / 120 | 0 / 0 | 68 / 71 | 0.9231 / 0.8812 |
| BIAS_SL_ADV_NOISY k0.25 / k0.5 | immediate | 128 / 128 | 0 / 0 | 6 / 46 | 0.9971 / 0.9431 |
| BIAS_SL_ADV_NOISY k0.25 / k0.5 | guarded | 128 / 128 | 0 / 0 | 56 / 45 | 0.9506 / 0.9642 |
| MISSING_DROP (SL-ending samples removed) | both | 0 | 0 | 128 | 0.0000 (regret 1.0696 R) |
| MISSING_UNDERWEIGHT (x0.25) | both | = NOISY | 0 | 0 / 65 | 1.0000 / 0.9330 |

Reading: (1) a favourable bias costs late exits and, under guarded, 8 positions never close (capture 0.881); (2) even a
pessimistic bias of half the stop distance produces **zero destructive closes**: the 64 TP controls are far from the
decision boundary (TP is about 2.3 x stop distance and is reached by an opening gap), so "0 destructive closes" in 133
is not robustness evidence; (3) omitting the adverse mode is catastrophic: in 2,660 of 3,428 ticks every sample ends
at SL, the reporter has nothing left, the controller never closes, and the result equals never-close;
(4) MISSING_UNDERWEIGHT is degenerate on this matrix (renormalization restores the weights when every sample is SL);
I keep it reported as declared. All of this is a finite synthetic retrospective stress, not learned-report evidence.

## Findings

| ID | Severity | Claim | Verdict | Evidence |
|---|---|---|---|---|
| F01 | INFO | Two fresh executions reproduce manifest/report/cases bytes; identity e490dbeb…, cases 8e0d495e… | VERIFIED | `<scratch>/B_out/`, B_run.log |
| F02 | INFO | Plan 131 runner reproduces identity 7f6c0f46…, cases 79035aeb… | VERIFIED | `<scratch>/B131_out/` |
| F03 | MINOR | "134 focused tests pass" | UNVERIFIED[set not enumerated; named files 34 passed; broad family 86 passed, 72 skipped] | A_tests*.txt |
| F04 | INFO | advantage = min_j(mu_j - z sd_j)/R, weighted mean and population sd over D_sj = C_s - W_sj | VERIFIED | C_formula_check_out.json (4.4e-16), C_formula_check_ALL_out.json (62,834 ticks, 8.9e-16), E4 (0.0) |
| F05 | INFO | Ties HOLD; margin + 1e-12 epsilon; persistence resets on gap/position change; urgency bypass | VERIFIED | code l.140-146, unit test, F1 adv 0.0, 0 replay mismatches |
| F06 | INFO | Native order: open protection > agent CLOSE at open > intrabar SL > intrabar TP; exit quote bid (long) / ask (short); fees and rollover timing as in D | VERIFIED | D/D2 fixtures, 16,320 corpus actions max 2.84e-14 USD |
| F07 | MINOR | Controller valuation replicates the native first-passage rule exactly | REFUTED[limit_through boundary: native fills TP, controller does not; 27.50 USD = 1.07 R long, 17.39 USD = 0.67 R short; same expression in protection_reference] | D_native_fixtures_out.json F6, D2 F6s |
| F08 | MINOR | Report spreads suffice to value the forced-flat exit | REFUTED[native quotes the close-time spread; 2.84 USD = 0.040 R with a schedule boundary at the flat instant; 0 with default schedule] | D_native_fixtures_out.json F5/F5c |
| F09 | MAJOR | The 1,744-position native matrix exercises priority before intrabar barriers (and deadline exits) | REFUTED[1744/1744 passive exits are at-open gaps; 0 intrabar, 0 limit_through, 0 forced-flat] | refs count in this report §D; F_independent_out.json |
| F10 | MAJOR | "Short/incomplete support … refused" / "shorter reports are rejected" (132/133) at the controller | REFUTED[a truncated report with an earlier self-consistent deadline is accepted and flips HOLD to CLOSE; only the harness binds deadline = forced flat] | `<scratch>/B2_truncated_report.txt` |
| F11 | INFO | Each wait action is one common closing time; no E[max]; per-path barriers before expectation | VERIFIED | E1/E2/E3, code l.134-139 |
| F12 | INFO | Two positions under the same market need different decisions | VERIFIED | E5 both sides; native F7 tight CLOSE 1.229 R vs wide HOLD -0.818 R |
| F13 | INFO | Prefix trace: 864 late first requests, 880 none, 0 executable, also without guards | VERIFIED | F_prefix_trace.json, F_independent_out.json (all 864 at the gap-exit fill bar) |
| F14 | INFO | Antithetic noise preserves the mean path | VERIFIED | G stats, 2.2e-16 |
| F15 | MAJOR | 0 destructive closes under supplied reports indicates robustness of the rule | REFUTED[adverse bias 0.5 x stop distance still 0 destructive on 64 TP controls; TP controls are far from the boundary; omitted SL mode gives never-close, regret 1.07 R] | G/G2 outputs |
| F16 | MINOR | Guarded configuration is a reasonable default | REFUTED[552/582 missed optimal exits, 5.9-6.7% capture lost; under adverse bias guarded misses fall to 31-56, so its delay dominates] (Musashi also does not promote it) | REPORT.json, G2 |
| F17 | INFO | Open-loop finite-action rule is a stated limitation, not optimal stopping (max_j E W_j <= adaptive value <= E max_j W_sj) | VERIFIED | E3 (-0.5 vs 1.0) |
| F18 | MINOR | Config can express a CLOSE-favouring tie band | REFUTED[CloseConfig rejects margin_risk < 0 and qualifies uses max(margin, 1e-12)] | close_hold_controller.py l.51, l.142 |
| F19 | INFO | Margin closeout can be ignored in valuation under the business sizing | UNVERIFIED[no fixture built; native stop = max(SL, margin level) is not modeled by the controller] | env.py `_risk_hit` |
| F20 | INFO | Reports require a uniform gap-free hourly grid to the deadline | VERIFIED | validate l.74-78; AbsoluteBarEnv gap refusal |

Required changes for (a)/(b) acceptance without conditions: F07 (use native comparison verbatim, also in
protection_reference), F08 (terminal close-time spread), F10 (bind the deadline to the position/calendar and
reject mismatches), and a new matrix version (do not alter the frozen records) with intrabar-touch, through-boundary,
forced-flat and near-boundary TP controls (F09, F15).

## H. Relevance to Satoshi's lane

Our lane: market-only closer head emitting first-passage curves evaluated at the position's remaining
distances; AR(1) conditional-mean noise (phi = 0.587); Kalman filter over the head history; Bayes closer
comparator; EPS_CLOSE = 0.02 sigma tie band favouring CLOSE (hold only if hold beats close by more than EPS).

1. **Can it replace our closer as a shared component?** Only the decision/valuation layer, and only if a report
   adapter supplies what it needs: per tick a `PathReport` with `paths [S,H,4]` joint mid OHLC samples through the
   forced-flat deadline on a gap-free hourly grid, `spreads [S,H]` (bar-open), `weights [S]`, timestamps, plus a
   `PositionView` (entry quote, side, units, SL, TP, fixed R, entry/market/stop slippage, round-trip commission per
   unit, limit_through) and per-unit cumulative funding at each open and each bar end. Our head's first-passage
   curves are marginal hit probabilities by horizon; they determine the passive value
   (P_TP, P_SL, P_neither, E[deadline]) but NOT the value of closing at a later open given survival, which needs the
   joint law of (open_j, no barrier before j). So it is not a drop-in replacement: an adapter must sample joint paths
   consistent with our head (e.g. our conditional mean path + AR(1) innovations + a range model for H/L), and that
   adapter, not the controller, carries the modeling risk. Our Kalman filter belongs upstream in the adapter; with
   it in place set persistence = 1, or the two smoothers stack delay (guarded already loses about 6% capture).
2. **What our ideal-output simulator must emit** for this controller: for each decision tick and each quality rung,
   S joint OHLC paths (S = 1 for the ideal rung; S >= 3 for noise rungs) starting at the next open, covering exactly
   (deadline - next open)/1 h bars, with intrabar highs/lows (first passage depends on them), bar-open spreads,
   weights summing to 1, `issued_at = first_fill = next open`, `observed_through <= issued_at`,
   provenance `synthetic_ideal`/`synthetic_noisy`; plus the known funding schedule arrays. Our AR(1) phi = 0.587 noise
   applied as one correlated error path per sample is NOT mean-preserving per tick unless generated antithetically;
   we should run both (mean-preserving and biased), and add the omitted-mode control of §G, which on this matrix
   collapses the controller to never-close. Also emit near-boundary TP and intrabar-touch cases, which the frozen
   matrix lacks (F09, F15).
3. **Conflicts with our tie band.** Musashi closes only when advantage > max(margin, 1e-12 R): ties and near-ties HOLD,
   the z*sd penalty and persistence push further toward HOLD. Ours closes unless HOLD beats CLOSE by more than
   EPS = 0.02 sigma: near-ties CLOSE (consistent with the owner's mandatory early-close rule). In his notation ours is
   margin = -EPS (converted to R units: EPS_R = 0.02 sigma x units / R) with z = 0 and persistence 1, which
   `CloseConfig` rejects (F18). The two rules disagree exactly in the band -EPS_R < advantage <= 0. With ideal reports
   this band is mostly the exact-zero gap ties, where native protection pre-empts any CLOSE anyway (F1, F2), so outcomes
   coincide there; with noisy reports they differ systematically. Proposal for a shared component: an explicit signed
   tie policy (`margin_risk` allowed negative, or `tie: HOLD|CLOSE` with a band), and both lanes report both settings
   instead of either lane changing the other's contract.
4. **Reuse map.**
   - As-is: `value_paths` native-equivalent valuation (after the F07/F08 fixes it is exact on everything I could build);
     the common-action/min_j aggregation; `PositionView`/report validation; `early_close_metrics` (plan 130 definitions:
     regret/R, capture as ratio of sums, null for zero opportunity); the 131 fixed-position counterfactual corpus,
     oracle and native AbsoluteBarEnv execution path; the prefix-invariance witness pattern.
   - With adaptation: tie policy (negative margin), deadline binding (F10), report adapter from our first-passage
     head to joint paths, funding arrays from our calendar, persistence = 1 when our Kalman filter is upstream, z = 0
     for our Bayes comparator arm, a new matrix version with intrabar/deadline/near-boundary controls.
   - Not applicable: `prefix_report` (a deliberately weak extrapolator, not our head); the antithetic 1e-4 noise as a
     proxy for our AR(1) error model; the z*sd dispersion penalty as a substitute for our Bayes comparator (it is an
     uncalibrated heuristic); the guarded engineering configuration.
