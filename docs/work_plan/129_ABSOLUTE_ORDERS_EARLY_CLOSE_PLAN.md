# Absolute-order integration and synthetic early-close qualification

Continue after 128. Implement in heuristic-strategy's integration worktree,
not Satoshi's active checkout. Keep agent-multi responsible for orchestration,
risk proposals and synthetic oracle evaluation; reuse the actual native
quote-side protections, costs, funding and ledger rather than a new simulator.

## Acceptance contract, before implementation

- AO01: opt-in AbsoluteBarEnv accepts an immutable open-time context containing
  only known quotes, cash, direction and timestamps. A decision-time closure
  captures references, sigma and coefficients; the callback resolves the
  submitted template at the next open. No future HIGH/LOW/CLOSE in context.
- AO02: absolute SL/TP are not clipped by legacy sigma/effectiveness limits.
  Revalidate both executable quote sides, ticks, minimum distances, integer
  size, margin and supplied modeled loss budget at the actual fill.
- AO03: leverage50, rel_volume0.1, no daily cap, one position; no implicit
  reversal. While occupied, only HOLD/CLOSE are actionable; entry requests
  do not close the existing order. q belongs to the position, not a mutable
  global config. Risk/cost helpers remain common with plan127.
- AO04: enforce 12h opening wait, entry strictly before forced-flat, and flat
  at the known close 12h before session end. Last eligible H1 entry may close
  at that deadline. Never borrow a quote across a missing open-session bar.
- AO05: existing broker protection at the next open precedes agent CLOSE;
  otherwise CLOSE executes there before intrabar SL/TP. Closing needs an
  existing position but no opener-horizon permission. Stop wins ambiguous
  intrabar touches. Preserve native financing and bid/ask accounting.
- AO06: sequential one-pass decisions, unknown actions, incomplete replay,
  invalid plan and prohibited model-state feedback are explicit errors.
- AO07: legacy engine default replay must remain identical. Baseline digest
  for two 120-bar synthetic runs with 18 trades each (mid_cash_v2/bid_ask_v3):
  `51e9aa9226c49238d20198da7df37abb6f70c65c1d7a021655f1c0e9a772c219`.
- EC01: synthetic close optimum is best executable net liquidation from now
  until passive protection/deadline, not a ZigZag opening pivot. Compare
  close-now, best later close and passive exit on the same position.
- EC02: use an independent scalar quote/cost calculation for the oracle and
  match actual native trade cash. Future oracle data is synthetic test truth,
  never an RL observation or an assertion of learnability.
- EC03: predefined mirrored gain-preservation and loss-mitigation paths
  have known future best close times; a HOLD-to-TP control must not be
  destructively closed. Include gaps, spread, funding, calendar, no-order,
  repeated opens and two different positions under the same market.

## Architecture and tests

Extract the legacy entry-preparation block into a default method with identical
behavior. The additive absolute subclass overrides preparation/gating, keeps
the native step/protection/ledger, and records per-position q. Optional daily
cap None must not break state/summary code. Unit fixtures precede code.

Cross-repo CPU runner consumes the sealed shared CONFIG and provides risk
callbacks through `app.horizon_risk_rules.propose`. Run legacy tests and
byte-replay, absolute API tests, then a finite mirrored multiscale synthetic
close matrix. Every scenario declares its expected behavior before replay;
counterexamples are fixes, not grounds to alter its intended behavior.

No financial bars, learned models, neural training or DEAP in this integration
qualification. Its results are test coverage and oracle regret/cash checks,
not strategy profitability. After this component is qualified, complete the
causal market-report/uncertainty and entry arbitration interfaces before
joint account-level coefficient optimization and floating selection.
