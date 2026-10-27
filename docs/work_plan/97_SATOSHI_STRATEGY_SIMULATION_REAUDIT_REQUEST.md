# Independent re-audit request: ideal outputs and strategy scenarios

For Satoshi. Owner requested a new rigorous, pragmatic audit. Read
[report 96](96_STRATEGY_SIMULATION_REAUDIT_REPORT.md) first. Evidence code pins
are `e3468172` (full matrix), `87252b92` (JSON), `e3498b48` (C4), and engine
`cd6489f`. The documentation commit carrying this request is not the producer.
Prior verdict `2fd4eeda` remains in force unless a specific correction is proven.

## Questions to answer

1. Are ideal outputs specified independently enough to prove correct strategy
   behavior, or are tests reproducing the simulator's own mistakes?
2. Does each tested scenario reach the claimed market/agent state and execute
   its expected action? Does aggregate PASS hide an absent state or interaction?
3. What EXACT additions are required for the owner's trend-strength,
   volatility and multihorizon requirement? Do not confuse oscillation period
   with decision horizon or transition speed.
4. Are named JSON coefficients consumed correctly, bounded and identifiable
   before DEAP tuning? A usable interface is not proof of optimized parameters.
5. Is the learner/output distinction preserved throughout the evidence and
   decisions? C4 failed and must not be certified by synthetic execution tests.

## Attack checklist

| ID | Attack | Required evidence |
|---|---|---|
| R01 | Reconstruct matrix independently | all 5832 unique cells; actual axis values and absent dimensions |
| R02 | Inject known ideal outputs | BUY/SELL/NOP and both closes through native policy, decoder and engine; both directions positive but SELL greater must select SELL |
| R03 | Detect circular oracles | independent monotonic/event/ledger checks versus same-engine counterfactuals; demonstrate any shared-error blind spot |
| R04 | Temporal horizon and transitions | show four-bar-only and fixed drift magnitude; define additional strengths, regime durations, transition speeds and horizons before executing replacements |
| R05 | State reachability | independently recompute 880 protection-closed cases and coverage by side/regime; absence must not masquerade as a passed comparison |
| R06 | Counterfactual/impossible outputs | contradictory heads, exact/near ties, NaN/Inf, fractional command, wrong-side close, illegal entry, invalid risk parameters; typed behavior and no unintended order |
| R07 | Entry, hold and early exit | next-open timing, both sides, close before adverse move, hold when appropriate, session restriction and protective priority; no forced trades to pass activity |
| R08 | Accounting and size | independent signed gross/net, fees, financing, fill-time sizing, q=0/1, broker minimum and cap; no risk increase to meet minimum |
| R09 | Stop/target/event ordering | adverse and favorable gaps, both barriers in one bar, forced flat, margin and stale/missing observations; list unexecuted combinations explicitly |
| R10 | JSON reaches actual execution | asymmetric long/short coefficients, endpoint behavior, unknown business override rejected; exact effective config and hashes |
| R11 | Parameter effect and optimization | perturb one coefficient, hold support/outputs fixed, observe intended effect; retain rejection if SL or fitness components are ineffective |
| R12 | Resume and artifact integrity | alter copied spec/recipe/cell/summary/hash, missing/duplicate cell and closed run; distinguish semantic from byte integrity checks |
| R13 | Learner localization | replay planted checkpoint and swapped rows; inspect mask, tie behavior, input dependence and all-seed C4 result; no financial selection from this probe |
| R14 | Decision scope | separate execution, coverage, parameter optimization and learning verdicts; no blanket acceptance of a selected strategy |

Use mutants as negative controls: BUY-first routing; swapped SELL decoder;
wrong short-P&L sign; same-bar fill; ignored close; removed costs; disabled
calendar; q above cap; ignored SL; a removed scenario falsely counted as PASS.
Run them only on disposable copies. If a mutant survives, identify the missing
test and supply the smallest counterexample. Do not change the expected answer
to agree with the mutation or a failing strategy.

## Coverage extension design to review

Create the scenario catalogue BEFORE implementing additional tests. For each
scenario declare: initial agent state, synthetic price/regime path, calendar,
costs and liquidity assumptions, ideal head outputs, expected command, expected
fill/SL/TP/volume/close, and independent numeric or inequality oracle.

At minimum distinguish:

- Down/flat/up before and after, multiple weak/medium/strong magnitudes, abrupt
  versus gradual reversal, and duration relative to the decision horizon.
- Low/medium/high innovation amplitude before and after, volatility bursts,
  contraction, and combinations with trend transitions; realized volatility
  must be measured rather than assumed equal to generator amplitude.
- Short and long comparison horizons independently of oscillation period;
  horizons that cross the calendar's forced-flat boundary remain constrained.
- Flat, long and short; position age, favorable/adverse floating return,
  proximity to SL/TP, daily cap and balance/margin conditions.
- Ideal open/hold/close/NOP outputs; deliberately contradictory or invalid
  outputs and feature-data defects. Invalid prices are rejection fixtures,
  not plausible synthetic markets admitted into training.

Specify finite bounds and a coverage obligation for each axis/interaction;
identify full Cartesian coverage versus justified interaction designs. No
claim of exhaustive continuous-space coverage follows from a finite grid.
Do not start a new DEAP search or alter the active policy during this audit.

## Reproduction and isolation

Private index resolves the roles below without publishing private paths:
`systematic_matrix`, `systematic_matrix_runtime`, `engine_repo`, `c4_archive`,
`direction_probe`, `direction_probe_script`, `json_audit_runtime`,
`json_audit_smoke`, `json_test_receipt`. Verify closure hashes against the
[public manifest](../audits/evidence/strategy_reaudit_20261026/MANIFEST.json).

Read source artifacts only. Replay into a NEW directory from the correct
export; do not use one version's code to claim reproduction of another.
Full grid is CPU-only, measured 8.82 min, one compute thread. Keep at most
3 GiB RAM and no GPU. No fits, TEST reads, service changes, credential access,
source mutation or historical artifact deletion are required or requested.

For a JSON replay, from the `87252b92` export and private resolved variables:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$PYTHON" -m tools.systematic_strategy_matrix \
  --recipe "$RECIPE" --source-spec "$SPEC" --engine-root "$ENGINE" \
  --strategy-config configs/validation/directional_strategy_midpoint.json \
  --output "$NEW_AUDIT_OUTPUT" --limit 12
```

This is a smoke; it does not reproduce the 5832-cell pre-JSON execution. Use
the original export and omit the new flag for that full replay. The raw C4
closure already includes all fitted artifacts; replay does not require refits.

## Return

Publish a Markdown verdict and machine-readable R01-R14 table with independent
commands/evidence, severity, counterexample, affected conclusions and minimum
correction. Separate implemented fixes from proposed coverage expansion.
Return four explicit dispositions: execution contract; scenario coverage;
parameter optimization; learned policy. Refuted or unproven items are not
administrative obstacles: give executable witnesses and precise remedies.
No recommendation to abandon the market follows from these engineering tests.
