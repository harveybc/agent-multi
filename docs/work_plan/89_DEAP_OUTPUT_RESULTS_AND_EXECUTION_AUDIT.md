# DEAP output comparison: results and execution audit

Closed2026-10-26 at09:13:46 America/Bogota; audit completed09:17:59.
Producer `e35c2af4`, nine searches, no process restarts, exit0. Peak cgroup
memory1.10GB below the2GiB limit. The completed service was disabled after audit;
no GPU campaign or financial feature-selection artifact was changed.

## Measured results

3611 unique genomes,21666 calibration trajectories and66 confirmation
trajectories. Every search stopped by patience, after40 to71 generations, not
the100-generation maximum. Population20, patience10, relative threshold1e-8.
All3 seeds for each objective satisfy the declared confirmation eligibility.

Means over the three seeds and the two ideal-output confirmation markets D/E:

| Objective | Sharpe | Net profit USD | Net return % | Mean maximum DD % | Orders/business day |
|---|---:|---:|---:|---:|---:|
| Sharpe | 8.85595 | 885.33 | 8.85330 | .17418 | .87302 |
| Sharpe + net return | 8.94266 | 1094.65 | 10.94648 | .20701 | .87302 |
| Sharpe + net return + activity | 8.94266 | 1094.65 | 10.94648 | .20701 | .87302 |

Each synthetic market starts with USD10000, with the same costs and business
rules. D has40 orders (32 buys/8 sells,22.5% profitable); E has30 (6 buys/24
sells,100% profitable) in each final seed. These are simulated ideal outputs
with deliberate future knowledge, not a trained policy's validation results.
Per-side win rates, exit reasons, fees and every final genome are in the
[mechanical summary](../audits/evidence/deap_output_calibration_20261026/SUMMARY.json).

Adding return changed the chosen exposure and delivered higher synthetic profit
with slightly higher drawdown. This is an observed tradeoff, not universal
superiority or proof of real-market profitability. No winner is selected on D/E.

### Activity contrast has no discriminating variation here

Across all3611 evaluated genomes, calibration activity was exactly
.8541666666666667 orders/business day. Therefore adding activity was a constant
offset: F2 and F3 produced identical genomes and search paths for each seed.
The diagnostic command generator and daily gate fixed entry counts over this
sampled domain. This experiment cannot establish whether rewarding activity
helps a learned policy or a strategy with different entry decisions. Do not
call the activity term generally useless, nor claim it was optimized separately.
The theoretical q=0 boundary is tested, but genetic samples did not produce an
activity contrast. No extra entry parameter was invented to hide this limitation.

## Defect found and repaired

The v1 checkpoint identity bound seed and global run settings but not the
objective name. Directory separation prevented accidental mixing in the normal
driver, but exchanging two state files with the same seed could bypass identity
validation. This was an integrity weakness, not evidence that an exchange occurred.

The v2 runner now binds the objective explicitly. A new regression rejects
cross-objective adoption. Another verifies exact continuation after a failure
inside a generation. No producer file or original checkpoint was modified.

Rather than repeat numerical experiments, the execution audit:

1. Verified every producer/source pin and all3611 cache receipts.
2. Reconciled433320 calibration trades, net profit and costs; checked direction,
   actual q, protections and the fixed activity denominator.
3. Recomputed every retained population's scores under its own objective,
   including eligibility, patience, elite history and termination reason.
4. Recreated all nine DEAP searches from seed using authenticated cached
   evaluations. Population, scores, RNG, history and terminal state matched
   exactly, including the legacy checkpoints.
5. Executed the66 final confirmation simulations again: all outputs and trade
   records matched exactly. Recomputed the comparison table and verified3645
   sealed artifact hashes.

Result: PASS. The weakness did not contaminate these results; there is no
numerical basis to discard or retrain these studies. Future searches use v2;
replaying v1 uses the frozen producer plus the explicit objective audit, not
silent conversion of receipts. The final focused suite passed56 tests.

## Evidence and reproduction

Study CLOSE sha256:
`a4709827e15d03539a3459d9980870d75b0dd8e53ddb1d552881a9fda0f58d78`.
Execution AUDIT sha256:
`28a34c36eb93e515789964208107cf0a7c4d57fca47b091698338c5dabbc9f1d`.

The auditor writes only to a separate output directory:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m tools.audit_deap_output_calibration \
  --run "$DEAP_OUTPUT" --runtime "$FROZEN_PRODUCER" \
  --campaign "$CAMPAIGN_MIRROR" --engine-root "$PINNED_ENGINE" \
  --output "$ISOLATED_AUDIT" --require-complete
```

The private index supplies these paths. This completes the synthetic-output
comparison, not the cross-auditor's decision on financial strategy admission.
The partial-observability and learned-policy behavior findings in report84 are
not removed by these ideal-output results. No financial selection is released
by this report, and no business/model configuration is silently replaced.
