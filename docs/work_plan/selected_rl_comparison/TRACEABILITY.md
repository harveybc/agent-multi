# Selected Comparison Traceability

| ID | Requirement | Synthetic evidence |
|---|---|---|
| SEL-1 | Scoped admission, not all-inventory completion | scope refusal and scoped status tests |
| SEL-2 | Exact ordered names, bytes, current selection/population pins | identity, stale selection and tamper tests |
| SEL-3 | TRAIN2020-23, VALID2024, completed-bar availability | period, support, future availability negatives |
| SEL-4 | Same selected information and causal scaling in both SAC arms | task config equality; gym prefix perturbation test |
| SEL-5 | Existing independent trainable actor/critic Conv1D/fusion/core | real model construction and forward test, no learning |
| SEL-6 | Same finite SAC budgets, controls, policy, costs and SAC-first design ordering | reference-design preservation and exact arms-order assertion; shared queue delegation tests |
| SEL-7 | No private input/fit/service needed for verification | all fixtures synthetic under tmp_path; dispatch doubles |
| SEL-8 | Shared queue resume/status/publication and verified physical copy | publication refusal/idempotence and backup tests |
| SEL-9 | Final selection OLAP/two-copy closure required before source reads or dispatch | 21 close negatives, fresh/resume rechecks, deployment-receipt positive and source-close tamper test |
| SEL-10 | Mandatory SL/TP and user balance allocation in both arms | v2 budget binding/negative tests; gym balance-budget unit and real Backtrader protection tests |

Data admission and causal transformation reproduction belong to the upstream
owner. These tests verify consumption and transport, not source truth or
financial eligibility. No synthetic result may replace an admitted campaign.

Verification on 2026-10-19: all 31 new unit cases and the one no-learning
integration passed. The combined focused regression selection passed 91 tests
in 5.61 seconds in the existing trading-stack environment, CPU-only. Exact
paths and checks are recorded in `PROJECT_METHOD_STATE.json`. Real source
admission and scientific campaign acceptance remain unclaimed.

Publication-gate follow-up: 21 refusal cases first failed because the gate was
absent. After implementation, the same focused regression command passed 116
tests in 9.01 seconds (original 91 plus 25 cases). Closure identity is now
mandatory before source reads and rechecked before dispatch/resume. This
verification used only synthetic receipts; no mirrors or services were read.

Balance-budget amendment: the seven new request tests plus existing selected,
no-learning integration, input, queue and collector regressions passed127/127
on CPU with the matching gym checkout. Gym's budget/protection/session suites
passed108/108. Thirty-one budget/protection cases were observed red before the
gym change; request-negative cases were observed red before the v2 adapter.
This establishes engineering behavior, not a profitable trading policy.

Leverage amendment: request v3 binds broker leverage and both sizing leverage
fields together. Eight test-first request cases cover50/100 and malformed
values. The focused CPU regression set passed135/135. Real gym entries at
10/50/100 verify both protection children and effective broker leverage.
