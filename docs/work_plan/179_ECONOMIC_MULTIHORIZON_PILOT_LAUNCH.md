# Economic Multihorizon Pilot Launch

Launched2026-10-31 at13:27:25UTC from immutable code `cff9af34`, implementing
[plan178](178_ECONOMIC_MULTIHORIZON_ARBITRATION.md).
Identity: `6c4686e3cbb0c8ffed95ef5629882225d6c651e5350edf617c3ab576df6cf151`.

## Verified Execution

-33 focused CPU tests pass; scoped Ruff correctness checks pass.
- Real-data smoke:328 finite labels covering all76 role/action calibrators.
- Persistent sequential service: three seeds7/19/43, one CPU quota,4GiB cap.
- Source, original model/data receipts, configuration and dependency identity
  are checked before execution. Authenticated sessions and seed results resume;
  completed closures are verified without rewriting their artifacts.
- Permanent contract/integrity failures exit78 without automatic retry. The
  service has a restart limit for process failures; no silent fallback models.
- First seed finished178 TRAIN session batches and fitted76 calibrators:
  26,896 fit rows,4,656 chronological diagnostic rows; minimum216 fit rows
  and32 diagnostic rows per calibrator. These counts include correlated
  counterfactuals, not independent samples.
- First seed advanced automatically to native TRAIN account replay. At that
  check the service had zero restarts and used about1GiB RAM.

These are execution checks, not evidence of economic improvement. The complete
account comparison and every executed close's counterfactuals still run in the
same queue. No original SAC weights, data selections or input scalers changed.

## Automatic Outputs

Each seed writes its target batches, fixed calibrators and calibration errors,
then reference and joint accounts for TRAIN and INNER_A. Metrics include net
profit, return, Sharpe, drawdown, orders, activity, buy/sell counts and win rates,
native costs, close reasons, horizon usage and estimated-versus-realized
continuation gaps. Joint accounts additionally write fixed-position close
regret and whole-account CLOSE-versus-one-tick-HOLD counterfactuals.

The final stage writes `RESULTS.md`, `AUDIT_REQUEST.md`, `SUMMARY.json` and a
hashed `CLOSE.json`, including negative results. No intermediate approval is
required. There is no automatic promotion, DEAP sweep, new SAC fit or floating
feature selection following this pilot.2024 and TEST remain unopened.

## Reproduction

With the pinned source, original campaign and authenticated frozen collection:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m tools.run_economic_curve_arbiter \
  --campaign <original-campaign> --collection <frozen-collection> \
  --output <new-pilot-root> \
  --config examples/config/real_curve_economics_v1.json
```

Use an exclusive process lock and the documented service limits. Adding
`--smoke` creates a distinct identity and performs one-session target generation,
not full strategy validation. No access to2024/TEST or rollout future is allowed
in runtime action selection.
