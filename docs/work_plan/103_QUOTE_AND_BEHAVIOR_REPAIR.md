# Quote and behavior repair: engineering qualification, not financial release

Date: 2026-10-27. Implements the first steps of plan 102 with new code identities.
Historic engine and experiment artifacts are not modified.

## Test design before implementation

Q01 long enters ask, short enters bid; quantity uses that quote.
Q02 broker protection and liquidation mark use the opposite quote.
Q03 a spread-only stop crossing must execute; midpoint-only TP must not execute.
Q04 entry and liquidation minimum distances; gap refusal; stop priority; margin.
Q05 spread charged once, financing/event cash/daily equity independently reconcile.
Q06 old midpoint mode retained for replay; quote mode explicit in every receipt.
Q07 native policy wrapper consumes new entry-price method, not the old midpoint.

B01 independently coded forward-barrier reference: gap then stop then target,
with literal examples; it cannot call the engine's step, protection or accounting.
B02 early-close diagnostic compares continuation to liquidation after fees, not
just terminal movement. Known stop-first case closes; known target-first holds.
B03 risk-adjusted entry diagnostic uses net expected gain / monetary loss at SL,
with configurable minimum ratio; compares both sides together. Equal scores NOP.
B04 literal mirror cases swap side and barriers; no first-valid-BUY behavior.

The reference is DIAGNOSTIC_ONLY and can read synthetic future paths. It must
never become a live input, financial feature or claim that a trained network can
forecast that path. A default ratio of 1 is an engineering test condition, not a
new universal business rule; probability-aware entry utility remains a design
choice for the new optimization protocol. Negative controls are required.

Execution correctness, oracle behavior and learning qualification are separate.
The C4 failure remains active. No financial selector or new DEAP campaign may
reuse a success label from the prior rejected strategy.

## Engineering grid, frozen before execution

`configs/validation/quote_repair_grid.json` declares 1,312,200 combinations:
five drift values before and after (including flat and two strengths each way),
three volatility scales before and after, three transition positions/durations,
optional superposed 8/72-bar components, three seeded innovation processes
(Gaussian, variance-normalized Student-t5, clustered variance), none/gap/jump,
three market seeds, exact mirrored paths, 4/24/72-bar reference horizons and
normal/tripled spreads. The clock includes a New York DST transition and omits
unobserved closure bars. The transition resets every 240 bars, creating repeated
regimes. Prefix invariance and exact random-path reflection are tested.

This is the Cartesian product of DECLARED bins, not all possible markets. It
uses the symmetric, unoptimized control `(0.5,0.5,0.2)` for each side. It is NOT
new evidence for any rejected DEAP finalist. The reference computes prospective
exit on separate code; the native engine executes full 240-bar episodes and
independent cash/size/cost checks. Each case retains decisions, trades and metrics
in compressed immutable batches, including NOP and early-close decisions.

The synthetic forecast path is fixed by the scenario seed. Its desired command
depends on the actual position and proposed protections. Unlike the old fixed
terminal-direction tape, it can therefore test SL-before-TP and continuing an
existing order. This deliberate oracle dependency is NOT hidden or a trained
policy, and must not be used to claim a gene-independent categorical tape.

The runner resumes completed batches by identity/hash, stops on any counterexample,
requires a 20 GiB disk reserve, and emits an engineering-only closure. Sample or
limited runs always say SMOKE, never full coverage. Full execution has six CPU
workers and no GPU/financial data. No following optimization is enabled until
behavior expectations and the separate learning qualification are reviewed.

## Verified implementation

- Quote engine: heuristic-strategy `842320a`, branch `codex/quote-execution-20261027`.
  57 tests passed, including the legacy environment suite and new quote witnesses.
- Consumer: native wrapper sizes at the quote, independent ledger checks handle
  quote P&L without double spread, seeded path and reference tests plus native
  full-episode integration and immutable-close/resume corruption checks. 41 focused tests passed, zero skipped with explicit
  pinned recipe/engine supplied.
- A diverse deterministic sample of 128 full episodes passed with zero failures
  before adding the final broker-protection-priority reference guard. The final
  producer must run its own smoke before the full grid; the earlier sample is
  not relabeled as evidence for later code.
- The reference gain/risk threshold is explicit in JSON (`minimum_gain_risk=1`),
  not an unrecorded business change or a secretly optimized acceptance threshold.
- No C4 retraining, final strategy selection or financial fitting is claimed.

## Execution receipt, 2026-10-27 09:00 UTC

Both code revisions are published: engine `842320a`, consumer `972fb7aa`.
The service executes exported snapshots, not either editable checkout. Its own
128-case deterministic sample completed with zero failures at 08:55:32 UTC;
identity `8df3849b29832afefc40ec4c60453277b1853d77c051ff32d655f33c12f0db93`.

The full grid is running automatically: 12,064 / 1,312,200 completed at 09:00 UTC,
zero recorded failures. Six CPU workers, 6 GiB soft / 8 GiB hard RAM limits,
2.8 GiB measured use. Compressed batches retain decisions, transactions and
metrics. The service is enabled for restart, with bounded process retries;
completed batches are authenticated before reuse. An assertion failure never
becomes a passing cell or silently changes the expected behavior.

Measured remaining estimate at this observation: 27,839 seconds, about 7 h 44 m.
This is an engineering-grid ETA only, not an ETA for repaired learning,
reoptimization, superiority confirmation or independent final approval.

### What this execution does not do

- The fixed symmetric six-parameter reference is not a DEAP winner.
- The full grid uses ideal diagnostic outputs. NOP and inverted controls have
  focused native episode tests; they have not been run across every grid cell.
- Per-case independent cash, size, cost and routing checks are not a proof of
  profitable learning. The original C4 failure remains explicitly active.
- The current actual learning consumer is categorical PPO with a grouped TCN,
  not continuous SAC. Its failed BUY-dominant behavior has not been repaired by
  changing the execution engine or supplying an oracle to synthetic episodes.
- No financial feature selection, TEST reading or new optimization starts from
  this closure. Plan 102 still requires the learned positive control, new paired
  searches and fresh confirmation before a final strategy audit request.

### Handoff to Satoshi

The rejection is accepted, including the failed planted control and missing
final-report disclosure. No replacement final audit is requested yet. These
published changes supply quote-side execution witnesses, seeded full-episode
coverage and an independent protection-path reference for review. The next final
submission must additionally include qualified learning, the finalist/control
comparisons, sensitivity and negative controls, fresh confirmation, and all
failures. The prior Sharpe-only recommendation remains withdrawn.
