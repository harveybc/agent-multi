# Actor Gradient, Optimizer Memory and Observation Ambiguity

Owner approved2026-11-09. Continue diagnosis in parallel with forward delivery.
Keep selected69 and the selected70 control immutable. Diagnose the documented
70-feature epoch40 deterioration using its authenticated replay; do not silently
change the selected69 inference or trading contract.

## Fixed Design

1. CPU gradient decomposition on eight128-row batches spread across full TRAIN.
   Compare epoch0, epoch40, and the4096-update realized-return anchor. Separate
   actor value and entropy terms and verify their gradients sum to the actual
   TorchRL actor gradient. Report Q action differences, per-head gradient norms,
   cosine similarity and local directional change in mean NOP probability.
   This derivative is not an AdamW update or a trading-performance prediction.
2. Primary GPU sequential paired critic anchor: retained AdamW state versus
   cleared AdamW state. Both restore exactly the same epoch40 weights, preserve
   all hyperparameters and use4096 updates, batch128, the same covering schedule,
   realized TRAIN soft returns and frozen actor/encoders/temperature. Record
   0/1/64/256/1024/2048/4096; authenticate each finished arm before continuing.
   Re-run the retained-state control in the same runtime. No need for a third
   expensive training arm or to occupy slower devices with duplicate work.
3. CPU observational ambiguity panel: align action-seed7/19/43 TRAIN trajectories
   by timestamp for epoch0 and40. Compare rows with identical market timestamp
   and action but different position directions; report immediate reward,
   controller-command and return differences alongside equal-position controls.
   Require identical input logits for each timestamp. Future action draws,
   account balances and positions differ, so this does NOT isolate a causal
   position effect or establish irreducible error. No state enters the network.

## Requirements to Tests

| ID | Check |
|---|---|
| G01 | Value+entropy actor loss and gradients match native TorchRL, including per-layer norms |
| G02 | Adding an action-independent Q offset preserves the actor gradient |
| G03 | Clearing optimizer state leaves weights, parameter-group settings and LR unchanged; states become empty |
| G04 | Identical observation/action matching refuses mismatched timestamps and separates position strata |
| G05 | Frozen actor, temperature and encoders exact at all checkpoints; TRAIN-only fitting |
| G06 | Equal replay/target/schedule identities and budgets for the two optimizer arms |
| G07 | Failures recorded; sequential driver stops on failed verification; completed units are authenticated on restart |
| G08 | Endpoints report calibration against constant predictors, not a trading-policy improvement |
| G09 | Resolve GPU by physical identifier and require the expected device name before loading data |
| G10 | Component export automatically writes a hashed weight profile without modifying tensors |

Read code and prior213-225 receipts first. Unit tests precede execution; native
gradient parity and authenticated full-TRAIN replay precede result admission.
Existing epoch40 labels are matched to its stochastic policy, not greedy
evaluation. Same-date alternate action draws are not independent market years.

## Next Decision

Interpret all three diagnostics together. A better critic RMSE is not enough to
replace a trading model. No new live/paper orders or long joint refinement are
launched by this contrast. Weight/gradient inspection becomes a phase2 checkpoint
gate after structural export and before/after refinement, not every feature
ablation. Preserve all failures and negative outcomes.

## Execution Corrections

- The first launch referenced the recovery location of epoch6, but that
  snapshot had already been archived separately. It failed before training.
  The corrected launch uses the authenticated archive; the driver now checks
  all snapshot files before starting a subprocess.
- CUDA ordinal1 resolved to the secondary RTX5070Ti, not the RTX5090 shown
  at ordinal1 by nvidia-smi. The partial run was stopped and kept separately.
  The paired run restarted from the original checkpoint on the RTX5090 using
  its private physical identifier. Both child configs now record the actual
  GPU name and reject an unexpected device. Previous index-only hardware
  attributions in224/225 are unverified; their numerical results are unchanged.
- Cross-device probability reproduction initially failed the strict check:
  maximum differences6.59e-6 at epoch0 and3.21e-5 at epoch40, zero greedy action
  differences. CPU epoch40 probabilities do NOT pass the1e-5 tolerance. The
  observational comparison uses authenticated original GPU surfaces and matches
  archived chosen probabilities exactly. The CPU discrepancy remains explicit
  in the report; no tolerance was increased to declare probability parity.

GPU runtime:7dbaa182, queue artifact label sac70-optimizer-memory-v3-20261109.
Both arms completed4096 updates with frozen actor checks passing. CPU gradient
and position analysis also completed. Results and the next controlled
distinction are in[227](227_ACTOR_LEARNING_MECHANISM_RESULTS.md).
