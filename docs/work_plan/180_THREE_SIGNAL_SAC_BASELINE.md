# Three-Signal SAC Baseline

Owner approval2026-10-31: implement the simplest categorical BUY/SELL/NOP
controller with phase-1 inputs, substantial training and early stopping.
This replaces neither historical evidence nor the audited execution engine.

## Contract and Discovery

One market-only actor emits exactly three logits:0 NOP,1 BUY,2 SELL.
Flat: BUY/SELL opens that direction when the native calendar permits it.
Occupied: opposite signal closes only; same direction and NOP both HOLD.
No simultaneous close/reopen. No agent-state input, learned price levels,
economic calibrator, rolling normalization or oracle input. All three signal
symbols remain available to the actor, including when an opening cannot
execute. The external business layer enforces legality. The duplicated HOLD
semantics are deliberate and logged; they are not silently masked away.

Keep phase-1 recipe:120 columns,31 groups,13 observations at stride2,
fixed TRAIN scalers and small double-TCN encoder. Use one6h risk profile as
an engineering control, not a winner selected from prior results. Native
reference medians plus the already-declared probe volatility coefficients
(`b_sl=0.25`, `b_tp=0.1`) determine distances; all other risk coefficients
are unchanged. Volume uses the existing risk budget and maximum margin
`q * rel_volume * current_balance`, with rel_volume0.1 and leverage50.
SL/TP/volume are fixed after fill. They are not optimized in this baseline.

## Requirements and Tests Defined Before Implementation

BS01: all nine signal/position combinations implement the table exactly;
opposite closes without reversing; NOP never closes an occupied position.
BS02: learned modules receive only market tensors; emitted dimension3;
SAC twins and categorical actor share the same action semantics.
BS03: native mirrored entries and early exits conserve cash, fees and risk;
future perturbation cannot affect the current signal or decision-time sizing.
BS04: terminal liquidation is included in the final economic reward; skipped
observation bars accumulate discounted reward and matching bootstrap duration.
BS05: sequential synthetic training uses the native execution engine and net
marked-equity reward, not terminal +/-1 action labels or imitation losses.
Held-out generator seeds/mirrors and exact deterministic signal control must
exercise both directions, early close and waiting. A flat-market control
must show that trading pays costs while NOP preserves capital.
BS06: validation never enters replay, gradient updates or scaler fitting;
fixed chronological monitor blocks cannot be selected for favorable outcomes.
BS07: early stopping has a large hard budget, minimum learning duration,
patience counted in actual evaluations, best checkpoint retention and reload
parity. Inactive checkpoints cannot become the admissible baseline.
BS08: independent fresh financial weights after synthetic qualification;
authenticated source/data/config/checkpoints, bounded workers, automatic
next seed, explicit failed outcomes and final result/audit documents.

## Learning and Stopping

Three independent seeds7/19/43. TorchRL discrete SAC, unchanged small grouped
encoder, AdamW and fixed recipe weight decay. Reward is100 times each native
equity increment divided by initial capital, including all native costs.
No reward for trade count, winning labels or artificial closing frequency.
The identical reward and signal decoder are used in training and evaluation.

Financial budget:1,048,576 decisions, minimum131,072 before patience starts;
4,096 random-action warmup, one update every4 decisions, batch128,
65,536 replay transitions. Evaluate every32,768 decisions. Patience10
evaluations, min_delta0.00001 return percentage points per market week.
Record complete historical passes separately from gradient updates and
evaluation count; RL decisions are not called Keras epochs.

Monitor: half the mean TRAIN-block return/week plus half the mean internal
validation-block return/week. Each block is eight calendar weeks and starts
flat at the same initial capital. TRAIN blocks:2020-07-05..08-30,
2021-07-04..08-29,2022-07-03..08-28. Internal validation blocks:
2023-07-02..08-27,08-27..10-22,10-22..12-17. Dates are fixed before results.
Monitor activity must average strictly more than2 orders/week in both halves
for an admissible best checkpoint. Also retain the best unrestricted checkpoint
to expose inactivity, never reward forced trading. If no admissible checkpoint
exists after ten checks beyond the minimum budget, stop as
`no_eligible_checkpoint`, not a substituted success.

Weights/replay fit2020..2023H1. Repeated monitor selection uses2023H2; these
are DEVELOPMENT results, not an untouched validation estimate. Upstream
phase-1 feature/risk selection already saw2023H2.2024 and TEST stay closed.
After stopping, reload best admissible weights and evaluate full TRAIN and
INNER_A, saving logits, Q, trades, equity, all costs, side-specific win rates,
close reasons, holding times and early-close diagnostics.

## Synthetic Qualification and Automation

An observed exogenous regime drives the next synthetic price increment.
It is causal by construction; randomized regime lengths include positive,
negative and flat stretches. Native calendar, protections and accounting
remain active. Train sequentially through episodes, with no supervised action
reward. Every fifth training episode is entirely flat, so flat-market NOP
has explicit learning support. Three frozen synthetic validation worlds use
independent seeds and mirrors. They are monitored repeatedly, not claimed
as an untouched confirmation set.
Required: positive net profit, both directions and both early-close directions,
plus at least half the deterministic signal controller's aggregate net profit.
The deterministic controller must itself pass before any learning is launched.
Maximum131,072 qualification decisions; checks every8,192 after16,384.
Failure is reported and the remaining independent seeds still execute; no
financial run is admitted from failed synthetic qualification.

Execution: primary accelerator handles seeds7 then43, secondary handles19;
CPU coordinator collects and authenticates results. Queue and collection are
persistent and bounded. No intermediate owner approval, automatic DEAP sweep,
phase-2 feature selection or strategy promotion follows this baseline.

Prelaunch review fixed a probability/position variable collision in the final
early-close diagnostic. Both directions now have a regression executing that
diagnostic. Requested-command entropy merges HOLD aliases but is not claimed
to be entropy after broker gates or protection preemption. Market-only actor
and critics intentionally remain partially observable with respect to position.
