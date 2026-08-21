# Audit: WP4 GPU Smoke Runtime Finding

Date: 2026-08-20 America/Bogota
Audited commit: `526cebec`
Runtime: Omega RTX 4070, seed 101, 20,000 steps/epoch
Disposition: GPU smoke stopped after four epochs; correction required

## Accepted corrections

- Strict CLI rejects unknown arguments.
- CUDA preflight proves the requested/effective RTX 4070 and UUID.
- Global sensitivity merge independently regenerates 586 rows and reproduces
  quantiles and candidate scores.
- Focused suite: 48 passed.
- The GPU run is real: actor/critic parameters move and actions/trades vary.

## Runtime evidence

Four epochs and 80,000 steps completed. Every epoch had real activity in
train, train-tail and validation, yet every epoch reported
`trade_gate=FAIL`, `checkpoint ineligible`, and incremented the L1
`no-activity` counter.

Examples:

- epoch 1: train 18, train-tail 1, validation 7 trades;
- epoch 3: train 34, train-tail 1, validation 7 trades;
- epoch 4: train 40, train-tail 1, validation 3 trades.

Continuing to 50 epochs would only burn GPU while the evidence authority
misclassifies observed activity, so the auditor stopped the bounded run.

## Root cause

The executing summary and the return-trace evidence disagree:

- train-tail summary: 1 trade; final trace field `trades`: 0;
- validation summary: 3 trades; final trace field `trades`: 2.

`_activity_authority.verify_evidence()` assumes the final value of the CSV
column named by `fact_key="trades"` is the cumulative episode count. The
return-trace producer does not provide that invariant. The verifier therefore
derives a different count and emits `EVIDENCE_FACT_MISMATCH_*`, making every
active checkpoint ineligible.

## Immediate correction order

1. Define one unambiguous cumulative field, e.g. `closed_trades_cumulative`, in
   the return-trace schema. Its final value must equal the environment summary's
   `trades_total` exactly.
2. Bind the evidence descriptor's `fact_key` to that cumulative field. Do not
   reinterpret an event/step field as cumulative.
3. Add adversarial tests for zero, one, several and a close on the final bar;
   summary count, final cumulative trace count and authority-derived count must
   be identical.
4. Preserve backwards compatibility by refusing legacy traces lacking the
   cumulative field; never silently use the old ambiguous `trades` column.
5. Add the activity-authority reason codes and both passed/derived counts to the
   epoch history and WP4 report so future failures are directly observable.
6. Re-run the CPU smoke and prove an active epoch passes the gate and selects a
   checkpoint. Then return a corrected GPU preflight. Do not require another
   owner phrase and do not run the GPU smoke before independent reproduction.

The interrupted GPU artifacts remain outside Git under
`~/.local/share/agent-multi/wp4_gpu_smoke_20260820_seed101/`.
