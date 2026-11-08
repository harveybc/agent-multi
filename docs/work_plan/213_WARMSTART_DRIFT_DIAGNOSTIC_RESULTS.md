# Warm-Start Drift Diagnostic Results

Completed2026-11-08 at13:09UTC. Runtime `8cb1a0bc`; identity
`45883acada99460e1d1801fb24040f06256743084c9ebbe3cc457198fc5ef02c`.
Six arms completed successfully in405.58 seconds using the verified cached replay.
The terminal manifest's19 files were independently rehashed successfully. Ten
focused tests passed, including the CUDA state-comparison regression. The first
attempt's failure and missed ETA are recorded in [the plan](212_WARMSTART_DRIFT_DIAGNOSIS.md).

## Paired Results

Every learning arm received256 updates on the same128-row minibatch schedule,
drawn from4096 initial TRAIN transitions under the preserved stochastic policy.
Accounts below use the full4-year TRAIN and1-year VALIDATION periods, unchanged
inputs/window/scaler/costs/strategy. All runs start from the same eligible epoch0.

| Arm | TRAIN return % | VAL return % | VAL Sharpe | VAL orders | VAL wins % | VAL max DD % |
|---|---:|---:|---:|---:|---:|---:|
| No update | 11.5145 | 13.8085 | 2.0971 | 206 | 57.77 | 3.6249 |
| Actor only | 11.5729 | 13.6654 | 2.0158 | 210 | 57.62 | 3.3971 |
| Critics only | 11.5145 | 13.8085 | 2.0971 | 206 | 57.77 | 3.6249 |
| Actor and critics | 9.8219 | 12.2785 | 1.9019 | 192 | 57.81 | 4.0106 |
| Both, adaptive alpha | 9.8219 | 12.2785 | 1.9019 | 192 | 57.81 | 4.0106 |
| Both, learning rate /10 | 11.3881 | 13.2395 | 1.9791 | 207 | 57.49 | 3.6249 |

Critic-only has identical trades by construction: the unchanged actor, not Q,
selects deployed signals. It is an isolation control, not a new trading winner.
No diagnostic model replaces the preserved checkpoint.

## Mechanism Evidence

Q/distribution probes use512 fixed valid windows per period, distinct from the
full-account evaluations above. On the VALIDATION probe:

| Arm | Q argmax NOP % | Actor argmax NOP % | Mean absolute TD error on common TRAIN batch | Alpha |
|---|---:|---:|---:|---:|
| No update | 62.11 | 80.66 | 0.069247 | 0.00531291 |
| Actor only | 62.11 | 80.27 | 0.071133 | 0.00531291 |
| Critics only | 90.43 | 80.66 | 0.067958 | 0.00531291 |
| Both | 90.43 | 82.81 | 0.067939 | 0.00531291 |
| Both, adaptive alpha | 90.43 | 82.81 | 0.067940 | 0.00535695 |
| Both, learning rate /10 | 74.61 | 80.66 | 0.068988 | 0.00531291 |

The isolated critic update changes its preference toward NOP while slightly
reducing the sampled TD residual. Coupling actor updates to those critics degrades
the full accounts more than actor-only in this short paired run. This supports
investigating critic targets/calibration and actor-critic coupling first; it does
not establish whether those Q estimates are wrong or responding correctly to the
training objective and replay distribution.

Adaptive alpha increased only0.83% over256 updates and produced the same trades
as fixed alpha. This does not rule out its effect over thousands of updates.
Reducing LR mitigated but did not eliminate the return loss. Greedy actions can
remain identical on the512-row probe while changing elsewhere in the full account;
this is why full trading evaluation is retained.

## Six-Epoch Snapshot

The authenticated completed epoch6 snapshot reproduced its saved accounts on the
diagnostic GPU. All online encoder values and the frozen temperature matched
epoch0 exactly. On the fixed VALIDATION probe, actor NOP share rose80.66% ->98.24%,
Q NOP share62.11% ->91.80%, and mean entropy0.68423 ->0.23271. Native validation
orders fell206 ->35 and return13.80849% ->-3.31670%. The initial account remains
reproducible: no bias replacement for removed zero-valued channels is indicated.

## Scope and Next Action

This is a single-seed operator diagnostic using an initial TRAIN replay segment,
not a completed full-data learning comparison or a feature-selection verdict.
Next investigate Q-target calibration, stochastic training versus greedy account
evaluation, and replay calendar composition; then confirm the resulting
mechanism-specific intervention across full trajectories and seeds. Do not widen
the architecture or discard inputs on this evidence. The primary refinement and
independent replication continue under their original contracts.

Raw evidence: `CONFIG.json`, `INITIAL.json`, `SNAPSHOT.json`, all six arm JSONs,
`BATCH_SCHEDULE.json`, replay file/tensor hashes, `REPORT.md`, and `DONE.json` in
the private diagnostic artifact root associated with the identity above.
