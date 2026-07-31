# Ten-Invariant Test Mapping

Date: 2026-07-31
Owner: Musashi
Purpose: input packet for `AT-QUAL-024`
Source contract: `docs/work_plan/09_TESTING_SECURITY_AND_OPERATIONS.md:17-28`

## Findings

The audit's narrowed finding 010 is correct. The repositories contain strong
causality, replay, accounting and constraint tests, but the ten declared
metamorphic invariants are not all encoded exactly. Three have direct coverage,
two have partial adjacent coverage and five are named gaps.

| # | Declared invariant | State | Existing evidence | Exact missing test |
| --- | --- | --- | --- | --- |
| 1 | Zero exposure creates no trading P&L before account fees | gap | Nautilus reconciliation and flat replay exist in `gym-fx/tests/test_nautilus_bakeoff.py` | Explicit zero-target replay with non-zero price movement and account-fee separation |
| 2 | Linear notional scaling produces linear P&L/costs while unconstrained | gap | Cost and sizing unit tests exist, but no metamorphic scale pair was located | Replay identical path at `k` and `2k` below all constraints; assert P&L and variable costs scale by two |
| 3 | Asset/cell order permutation cannot alter results | gap | Portfolio allocation determinism exists in `tests/unit/test_project3_portfolio_supervisor.py:326` | Permute input asset/cell order and compare canonical weights, intents and ledger facts |
| 4 | Unavailable assets cannot fill | covered | `gym-fx/tests/test_nautilus_bakeoff.py::test_unavailable_market_rejects_downstream_intent_without_fill`; the full Nautilus path receives a downstream intent, emits an explicit `MARKET_UNAVAILABLE` rejection, leaves balance unchanged and records zero fills | None |
| 5 | Future input mutation cannot alter earlier decisions | covered | `gym-fx/tests/test_feature_window_preprocessor.py:113`; earlier fill facts in `gym-fx/tests/test_nautilus_bakeoff.py:124`; portfolio test exclusion in `tests/unit/test_project3_portfolio_supervisor.py:247` | None for the declared property; retain all three layers |
| 6 | One-cell portfolio matches single-asset behavior | gap | Single-cell and portfolio paths are separately tested | Feed one identical intent through both paths and compare fills, costs, equity and attribution |
| 7 | Same hashes/seed produce identical deterministic replay | covered | Nautilus replay `gym-fx/tests/test_nautilus_bakeoff.py:35`; portfolio DB replay `tests/unit/test_project3_portfolio_supervisor.py:409`; shared population tests in `tests/unit/test_default_optimizer_shared.py` | Add cross-host packet only when hardware-determinism scope is declared |
| 8 | Tighter hard risk limits cannot increase permitted exposure | partial | Max-weight cap is tested in `tests/unit/test_project3_portfolio_supervisor.py:303`; same-asset cap in line 461 | Metamorphic pair over multiple monotonically tighter caps, including leverage and drawdown governor |
| 9 | Invalid/stale signal cannot create a larger position | covered | `gym-fx/tests/test_nautilus_bakeoff.py::test_stale_signal_cannot_increase_existing_position`; the full intent path rejects before order creation, preserves existing exposure and persists `STALE_OR_INVALID_SIGNAL` | None |
| 10 | Net instrument target equals sum of virtual cell targets | covered | `lts/tests/unit/test_portfolio_invariants.py`; exact long/short cell attribution, permutation invariance and duplicate-cell rejection exercise `DefaultPortfolio.net_instrument_targets` | None |

## AT-QUAL-024 Execution Order

Implement gaps in this order because each protects a larger downstream
surface:

1. future mutation remains mandatory in CI;
2. unavailable asset cannot fill (completed 2026-07-31);
3. stale/invalid signal cannot increase exposure (completed 2026-07-31);
4. net target equals cell sum (completed 2026-07-31);
5. one-cell parity;
6. risk-limit monotonicity;
7. input-order permutation;
8. zero-exposure accounting;
9. linear notional scaling.

The deterministic replay invariant is already directly covered but should be
included in the Tier A CI gate.

## Minimal Tier A CI Position

Finding 009 is accepted. CI should begin as a small repository-local workflow,
not a monolithic cross-repository environment:

- Python syntax/import and config/schema validation;
- each repository's fast unit/contract tests;
- the ten-invariant suite once materialized;
- DOIN consensus-focused tests;
- publication scaffold validation in `agent-multi`;
- no broker credentials, GPU requirement or paid-data access;
- pinned Python and dependency-lock identity recorded in the job;
- full GPU, broker, fleet and 24-hour gates remain scheduled evidence jobs,
  not pull-request checks.

Repository-local Tier A workflows now exist for `agent-multi`, `doin-core`,
`doin-node`, `doin-plugins`, and `lts`. Each installs a generated hash-pinned
lock with `pip --require-hashes`; `doin-node` checks out exact commits of
`doin-core` and `doin-plugins`, while `doin-plugins` checks out exact commits
of `doin-core`, `doin-node` and the adapted `agent-multi` optimizer. All five
local clean-environment reproductions and all five clean GitHub runner jobs
passed. Finding 009 therefore meets its stated closure condition, subject to
independent audit verification.
