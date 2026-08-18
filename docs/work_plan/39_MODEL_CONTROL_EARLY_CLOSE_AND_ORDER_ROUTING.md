# 39. Model Control, Early Close and Order Routing

Status: active execution and research contract, v1.0.0, 2026-08-18
Owner: project owner
Scope: ETH reference stack first; Paper/Demo only until separately authorized

## 1. Decision

An open position does not suspend inference. The policy observes its current
position, equity, unrealized PnL and holding duration on every due closed bar.
It can target long, short, flat or preserve exposure. Every entry has native
stop-loss and take-profit protection at submission; model-driven early close is
an additional control and never a substitute for that bracket.

The corrected scalar contract is `target_exposure_hysteresis_v2`:

| Raw SAC action | Flat route | Open/pending route |
| --- | --- | --- |
| `a >= entry_threshold` | enter/target long | hold long; close short first |
| `a <= -entry_threshold` | enter/target short | hold short; close long first |
| `abs(a) <= exit_threshold` | hold flat | explicitly close/cancel |
| between thresholds | hold | preserve current exposure |

Normal uses entry `0.10`, exit `0.02`. Easy uses `0/0`; every non-zero action
targets direction and exact zero targets flat. Opposite targets always
`close_then_wait`: a later bar may enter the other side after direct venue
evidence proves flat. Same-bar reversal is forbidden.

All pre-2026-08-18 P1LR performance artifacts were trained without this
explicit close-to-flat action and cannot select the new controller.

## 2. Exact Current Input and Policy

- asset/timeframe: `ETHUSD@4h`;
- source: 18,085 model-ready rows, 2017-09-28 through 2025-12-31;
- fit: 11,509 rows through 2022;
- train monitor: 2,190 scored 2022 rows;
- inner validation: 2,190 scored 2023 rows;
- outer validation: 2,196 scored 2024 rows;
- sealed release test: 2,190 rows in 2025, inaccessible during selection;
- 83 ordered engineered features, 32-bar policy window;
- rolling z-score over 256 rows, clip at +/-10;
- four binary-pass-through fields remain pinned as currently materialized
  pending their separate feature-contract review;
- four live-observable state values; no episode-length or future state;
- exact flattened observation: `32 * 83 + 4 = 2,660`;
- live raw-bar history: at least 800 closed H4 bars, sufficient for the longest
  nested causal warm-up plus the scaling window;
- actor: MLP `2660 -> 256 ReLU -> 256 ReLU -> mu/log_std(1)`;
- twin critics: each `2661 -> 256 ReLU -> 256 ReLU -> Q(1)` plus targets;
- complete SB3 policy state: 3,737,606 parameters including target critics.

The live feature implementation was compared on 2026-08-18 against all 18,085
rows of the model-ready file: all 83 columns matched within declared float
tolerances. That parity becomes a regression gate; a finite vector with
incomplete warm-up is refused rather than zero-filled.

## 3. Current L1 Calibration Parameters

The active P1LR experiment is a causal calibration, not a DOIN optimization.
Its only factors are:

| Factor | Levels |
| --- | --- |
| phase-1 dynamics | normal-realistic; easy chronological continuation |
| one LR used in both phases | `1e-4`; `3e-5` |

Held fixed:

- one zero-update genesis per seed, seeds 101/202/303/404;
- SAC MLP `[256,256]`, batch 256, buffer 40,000;
- `learning_starts=1000`, `train_freq=1`, `gradient_steps=1`;
- `gamma=0.99`, `tau=0.005`, fixed `ent_coef=0.2`;
- `target_update_interval=1`, `target_entropy=auto`, `use_sde=false`;
- 20,000 transitions per pass-equivalent;
- global ceiling 2,000 pass-equivalents: at most 1,000 phase-1 checkpoints and
  at least 1,000 reserved normal phase-2 checkpoints;
- patience 60 after floor 40, minimum improvement `1e-4`, best restoration;
- no activity terminator and no positive-profit gate;
- market parent entries only, pending TTL zero;
- commission `0.0002`, spread `0.0001`, leverage 1, relative volume 0.05;
- ATR protection with current fixed `k_sl=2`, `k_tp=3` and mandatory SL/TP.

These held values are not called optimal. They are isolated so the current
comparison can answer its declared question. Tunable values enter later typed
DOIN domains with measured bounds.

## 4. Entry Order Families

Direction and order family are different decisions:

- **market:** highest fill certainty, explicit spread/slippage exposure;
- **limit:** price improvement is possible, but non-fill and adverse selection
  after a fill must be measured;
- **stop:** confirms a breakout but can pay gap/slippage;
- **stop-limit:** caps execution price but combines trigger uncertainty with
  non-fill risk after triggering;
- **trailing-stop and venue-specific variants:** deferred until basic families
  establish material incremental value.

Buy/sell is direction. Limit/stop/stop-limit is entry mechanism; they do not
replace directional policy, SL, TP or early-close authority. Venue capability
is discovered directly because not every symbol/account accepts every family.

Entry stop orders and protective stop-loss orders are different objects. An
entry stop may open risk after a breakout; the mandatory stop-loss reduces risk
after a fill. They must have distinct intent classes, identifiers and metrics.

The initial responsibility split is:

| Decision | Initial owner | Reason |
| --- | --- | --- |
| long/short/flat target | directional SAC | this is the learned alpha and lifecycle decision |
| explicit early close | the same SAC under O0 | preserves one coherent exposure policy for the first causal comparison |
| position size and account caps | deterministic LTS risk layer | account-specific authority is not delegated to a model |
| mandatory SL/TP geometry | deterministic risk contract, later bounded DOIN genes | protection cannot disappear because a router is uncertain |
| market/limit/stop entry | deterministic O1 router, then optional O2 execution model | execution quality is separable from directional alpha |
| pending cancel/replace/expiry | deterministic execution state machine | restart and partial-fill behavior must be exact |
| emergency/risk-reducing close | market by default | fill certainty dominates price improvement while risk is open |

The router never changes long to short, changes the requested size, removes SL
or TP, or reverses on the same decision bar. It may only choose an executable
entry mechanism and its bounded offset/TTL for an already-authorized target.

### 4.1 Current implementation inventory

The capability baseline is intentionally asymmetric:

- `gym-fx` already implements and tests native market, limit and stop entry,
  deterministic pending expiration, optional market fallback, cancellation on
  a new flat target and protected SL/TP children;
- the historical `adaptive` simulator route is a heuristic implementation, not
  accepted evidence that adaptive routing improves return/risk;
- Alpaca Paper currently submits a market parent with native bracket children;
- IBKR Paper currently enforces a market parent, LMT take-profit and STP
  stop-loss in both plan and broker translation; and
- the MT5 Demo EA currently sends market DEAL entries and market closes.

Therefore O0 is executable across venues now. O1 first reuses the existing
simulator primitives under a new evidence contract; live limit/stop adapters
are added only after paired replay justifies them and each venue's atomic
protection behavior is independently tested.

## 5. Why Separate Models Are Not the Default

Separate entry, exit and order-type models do not guarantee improvement. They
increase compute, produce more selection opportunities and can issue
incoherent decisions unless trained under one account/exposure contract.
Vanilla Stable-Baselines3 SAC has a continuous Box action and does not natively
solve a hybrid categorical-order-family plus continuous-price-offset action.

Use this staged comparison:

1. **O0 control:** one target-exposure SAC, market entries, model early close.
2. **O1 deterministic router:** freeze the same signals and risk; route among
   market/limit/stop with causal spread, volatility, urgency and breakout facts.
3. **O2 learned router:** a small contextual-bandit or supervised execution
   head chooses family/offset/TTL from counterfactual execution labels while
   the directional SAC remains frozen.
4. **O3 shared representation, separate heads:** jointly trained entry target,
   exit hazard/value and execution-family heads only if O2 leaves measured
   headroom and gradient-interference tests pass.
5. **O4 separate specialists:** separate entry/exit/order models only if the
   shared-head arm loses robustly and the incremental GPU cost is justified.

For early close, compare in the same order: unified target-exposure actor;
unified actor plus deterministic exit baseline; shared encoder with distinct
entry/exit heads; fully separate models last. Every exit candidate is evaluated
counterfactually against holding to native SL/TP using the same entry set.

Do not train separate long and short routers initially. Side is an input and
price offsets are represented in side-normalized ticks, so one router can learn
the symmetric mechanics while retaining side-specific interactions. A separate
model by side, venue or family is admitted only after an interaction test shows
that conditioning is inadequate. This avoids multiplying trials and false
discoveries before a business effect exists.

O2 is not another full SAC by default. It is a small execution-value model that
estimates, for every currently supported family and bounded offset/TTL:

```text
expected directional value after costs
- non-fill opportunity cost
- adverse-selection cost
- latency/slippage penalty
- protection/rejection penalty
```

It can be trained as a supervised counterfactual utility model when replay can
label all alternatives, or as a contextual bandit when only the selected action
has credible feedback. The frozen directional SAC supplies target, confidence
and urgency; causal lower-timeframe execution state supplies spread, volatility,
distance, liquidity and session features. Promotion uses paired replay and then
Paper/Demo shadowing against O0/O1, never training return alone.

A monolithic hybrid RL policy remains a research arm, not the baseline. It
requires a library or policy implementation that correctly supports categorical
family plus continuous offset/TTL actions, action masks for venue capability,
pending-order state and partial fills. Encoding categories as arbitrary points
inside a scalar SAC Box action would impose a false geometry and is rejected.

## 6. Data and Simulator Requirements

An H4 OHLCV bar cannot reliably determine queue position, bid/ask path, whether
a limit and stop were both touched first, or realistic intrabar cancellation.
Order-routing promotion therefore requires:

- H4 decision cutoff with causally aligned H1/15m or tick execution replay;
- bid/ask or defensible spread model, latency and slippage;
- parent/child acknowledgement, partial fills, cancel/replace and expiration;
- pending-order TTL and model-close cancellation;
- symbol-specific minimum distance, size/step and supported-order facts;
- native SL/TP attached to every filled risk-increasing parent; and
- identical deterministic replay in simulation and Paper/Demo event journals.

Historical bar-only routing remains a feasibility screen, not business proof.

## 7. Metrics and Decision Rules

Hold direction, size, SL/TP, signal timestamps and evaluation partitions fixed
while comparing entry mechanisms. Report:

- net and gross weekly return, RAP, drawdown and expected shortfall;
- submitted, filled, partial, cancelled, expired and rejected counts;
- fill ratio and time-to-fill;
- effective spread, slippage and price improvement;
- opportunity cost of non-fill;
- adverse selection after fill at fixed horizons;
- early-close count, beneficial/harmful close rate and PnL avoided/forgone;
- native protection completeness and time unprotected (must remain zero);
- turnover, holding time and compute/GPU cost.

No family is promoted solely for better entry price. It must improve downstream
normal-realistic return/risk without material safety, activity or coverage loss.

The minimum attribution packet compares the same timestamped directional
intents under O0 and the candidate router. It reports both conditional results
given a fill and unconditional account results. This prevents an apparently
excellent limit strategy from winning merely by declining difficult trades.

No architecture guarantees improvement. O2/O3/O4 are rejected when their
paired out-of-sample benefit is smaller than uncertainty, when gains disappear
after opportunity cost, or when operational/compute cost exceeds the measured
benefit. A rejected complex arm is useful evidence and does not block O0.

## 8. DOIN Domains and Gene Boundaries

After the P1LR calibration, use sequential domains rather than one giant genome:

| Domain/stage | Candidate genes | Fixed/excluded here |
| --- | --- | --- |
| FS/representation | source/family/feature masks, context/window, scaling, encoder family/depth/width/latent | policy/execution fixed |
| SAC dynamics | actor/critic depth/width/activation, LR, batch/buffer, gamma, tau, entropy, update schedule, reward scale | order family market |
| control/risk | entry/exit thresholds, ATR period, `k_sl`, `k_tp`, sizing/risk caps | selected observation/topology |
| execution O1/O2 | family, router thresholds, offsets, TTL, cancel policy | directional artifact, size and protection geometry fixed for attribution |
| cadence | bar-aligned retraining schedule, lookback, warm/reset/refit, replay recency | mature input/action interface |
| restricted integration | only genes/ranges surviving isolated domains | rejected families remain closed |

The historical full-genome defaults and old champions are evidence, not best
values for this corrected contract. Until a corrected domain completes, the
only valid "best so far" values are the current P1LR fixed controls and its two
LR strata; no nonexistent DOIN champion may be fabricated.

## 9. Execution Order

1. Finish and verify corrected train/live observation and action parity.
2. Run the four-cell mechanics smoke from clean 2,660-input genesis artifacts.
3. If at least one non-constant trained arm survives, run the four-seed P1LR
   decision; otherwise return typed collapse and repair the measured cause.
4. Publish the selected artifact and exact executable observation manifest.
5. Attach it to ETH MT5 Demo; retain linear controllers only as labeled shadows
   on non-ETH infrastructure seats.
6. Collect at least one week of action, close, bracket and live/sim divergence
   evidence while component domains continue.
7. Run O0 versus O1 with lower-timeframe execution replay.
8. Admit O2 and later specialization only when preceding evidence shows
   material headroom relative to their compute and complexity.

Paper/Demo may lose virtual money; low exposure limits damage. Real-capital
authority remains outside this document.

## 10. First Live-Control Findings (2026-08-18)

The first deployment of explicit model close found three business-control
defects that simulation-only tests had not exposed:

1. Alpaca reports a short redundantly as negative `qty` and `side=short`.
   Applying both signs made the runner read a short as long and request an
   incorrect close. Quantity is now normalized exactly once.
2. An equity market close requested outside the regular session cannot be
   assumed immediately executable. The runner now defers the model close and
   preserves the existing native protection while the market is closed.
3. "Close first" did not initially mean "wait for another bar": MT5 could
   close on one daemon tick and enter the opposite side on the next tick of the
   same H4 bar. A durable due-bar close fact now consumes the entire bar on all
   three venues and blocks duplicate close, reversal and new risk.

The third defect is direct evidence against immediately placing order-family
selection inside the directional SAC. Position state, market session, pending
state, fill state and bar identity must be coherent before any router can add
value. O1 therefore begins as a deterministic execution policy over a frozen
directional artifact. A learned router or separate specialist is admitted only
after the paired O0/O1 journal proves headroom.

Current Paper/Demo seats remain deliberately labeled: MT5 runs the ETH linear
infrastructure canary until a corrected SAC artifact exists; Alpaca SPY and
IBKR USD.CAD are cross-venue infrastructure canaries and are not substitutes
for the ETH champion. Every risk-increasing position still requires native SL
and TP. The MT5 SAC route is implemented but remains fail-closed until the EA
publishes sufficient bars and a corrected artifact/manifest is available.
