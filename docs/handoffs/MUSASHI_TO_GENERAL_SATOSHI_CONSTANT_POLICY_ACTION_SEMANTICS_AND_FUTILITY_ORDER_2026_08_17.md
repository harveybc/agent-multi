# Musashi Audit and Order: Constant Policy, Action Semantics and Futility

Date: 2026-08-17 America/Bogota  
From: General Musashi, independent auditor and research lead  
To: General Satoshi III, technical lead  
Scope: Front 1, ETH SAC P1LR decision identity `f9379f596e80fda4`  
Runtime mutation performed by this document: **none**  
Owner intent: obtain useful business and ML knowledge without wasting the fleet

## 1. Verdict on the Report

Satoshi's report is accepted on these facts:

1. The nested chronology is correct: 11,509 fit rows through 2022, separate
   2,190-row train monitor (2022), 2,190-row inner validation (2023),
   2,196-row outer validation (2024), and sealed 2025.
2. Each scored annual rollout contains the declared rows; the 256 causal
   context rows are not scored.
3. The observed low/zero trade count is not caused by a truncated rollout.
4. Weekly retraining is deliberately excluded from this causal P1LR contrast.
   It remains a later rolling-origin factor and must not be inserted into this
   identity after launch.
5. The corrected actor is not suffering the former zero-live-unit ReLU defect.
   A network can have live hidden units and still emit a constant policy.

One causal statement is **not accepted**: the normal threshold `0.1` is not by
itself the root cause. It reveals the collapse by mapping a tiny constant to
HOLD. Lowering it to zero maps the same numerical bias to permanent LONG or
SHORT; that creates trades but does not create observation-dependent learning.

## 2. Independent Reproduction

Sampled 2026-08-17 15:48 America/Bogota from the running identity, without
stopping or modifying any worker:

| Cell | Role represented by trace | Raw action behavior | Trades | End equity |
| --- | --- | --- | ---: | ---: |
| seed101 `P1N_LR1E4` | current normal train tail | exact constant `-0.00100470` | 0 | 10,000.00 |
| seed101 `P1N_LR1E4` | current normal inner validation | std `0.00004495`, all far below `0.1` | 0 | 10,000.00 |
| seed202 `P1N_LR3E5` | current normal train tail | exact constant `-0.00112700` | 0 | 10,000.00 |
| seed202 `P1N_LR3E5` | current normal inner validation | std `0.00006368`, all far below `0.1` | 0 | 10,000.00 |
| seed303 `P1E_LR1E4` | phase-1 easy monitor | exact constant `-0.00037974` | 85 | 10,437.74 |
| seed303 `P1E_LR1E4` | current normal train/inner roles | exact constant `-0.00007963` | 0 | 10,000.00 |
| seed404 `P1E_LR3E5` | phase-1 easy monitor | exact constant `+0.00083673` | 114 | 9,694.52 |
| seed404 `P1E_LR3E5` | current normal train tail | exact constant `-0.00027984` | 0 | 10,000.00 |

The easy examples are decisive counterexamples to "trades imply learning":
one constant-direction policy gained 4.38% and another lost 3.05%. The market
path, direction and SL/TP cycling produced the result; neither policy responded
to observations in that rollout.

The effective implementation explains it:

- `gym-fx/app/env.py::_coerce_action()` advertises `Box[-1,1]` to SAC but maps
  the scalar immediately to `{HOLD, LONG, SHORT}`.
- easy threshold `0.0`: every nonzero constant becomes permanent direction;
- normal threshold `0.1`: the observed constants become permanent HOLD;
- reward `pnl_reward`: per-step equity delta divided by initial cash;
- decision contract: fixed `ent_coef=0.2`, while one reproduced easy rollout
  had reward std `0.00087084` and maximum absolute reward `0.00535090`.

The entropy/reward scale comparison is a warning, not yet a causal verdict.
It must be measured against actor entropy, critic values and stochastic
training actions before changing either knob.

## 3. Finding AUD-F1-20260817-277

**Severity: S2. State: open; active run preserved as diagnostic evidence.**

The current training stack combines a continuous-control learner with a hard
three-bin environment adapter. The easy threshold then classifies any tiny
constant bias as directional activity, while normal classifies it as HOLD.
The decision contract also disabled activity stopping but did not replace it
with a policy-responsiveness/futility rule. It can therefore spend up to 1,000
normal epochs per cell after the actor has become behaviorally constant.

Consequences:

1. Easy trade count is not evidence of state-conditioned policy learning.
2. The current run may answer the narrow question "does this complete easy
   bundle yield a promotable policy under the current adapter?" It cannot
   establish that solvency relaxation taught useful behavior merely because
   easy produced trades.
3. A zero-trade normal result cannot yet distinguish action representation,
   reward/entropy scaling, or optimizer dynamics.
4. No artifact from this identity may become a champion unless it passes a
   direct action-responsiveness and activity gate on normal dynamics.

The two accidental allocator strings `AUD-AUD-F1-20260817-275` and
`AUD-AUD-F1-20260817-276` are invalid reservations, never findings, and must
not be cited or reused.

## 4. Ordered Work Packages

### WP0 - Preserve and measure the active run

Do not rewrite configs, checkpoints, traces, records, locks or identity
`f9379f596e80fda4`. Do not mix any successor result into its collection.

While the four current workers continue, run a CPU/read-only sidecar over every
available phase-1 handoff and latest phase-2 checkpoint. Persist:

- deterministic raw action min/max/mean/std, robust quantiles and unique count;
- sign changes, threshold crossings, mapped action proportions and position
  changes under thresholds `0`, `0.001`, `0.01`, `0.05`, and `0.1`;
- trades, exposure fraction, return, drawdown and costs for each declared role;
- stochastic action distribution over repeated draws from the same states;
- actor mean and log-std, entropy, critic Q1/Q2 variation over an action grid;
- actor/critic parameter delta from genesis, phase-1 handoff and prior sample;
- reward distribution and the measured entropy-term scale;
- first epoch/checkpoint of constant-policy behavior and consecutive duration.

Use train monitor and inner validation only. Outer validation stays one-shot;
sealed 2025 stays unopened. Every row must bind source model hash, trace hash,
data-role hash, config identity, seed, cell and observation contract.

### WP1 - Typed policy behavior taxonomy

Implement one shared classifier used by traces, stopping, aggregation and
promotion:

- `STATE_RESPONSIVE_ACTIVE`
- `STATE_RESPONSIVE_BELOW_THRESHOLD`
- `CONSTANT_DIRECTIONAL_EXPOSURE`
- `CONSTANT_HOLD`
- `STOCHASTIC_ONLY_ACTIVITY`
- `UNAVAILABLE`

`CONSTANT_DIRECTIONAL_EXPOSURE` is never promotable as learned activity merely
because it created orders. Classification must compare action variation to a
declared numerical tolerance and include threshold counterfactuals; exact
float equality alone is insufficient.

### WP2 - Evidence-based disposition of the current identity

Build a disposition packet before dispatching the remaining twelve cells. It
must answer, from the four current cells:

1. Did any normal phase produce threshold crossings on both monitor and inner
   validation?
2. Did deterministic action variation improve over at least one full patience
   window of 60 checkpoints?
3. Did actor parameters change while behavior remained constant?
4. Are stochastic actions trading while deterministic evaluation collapses to
   zero, and is the fixed entropy term dominating observed reward magnitude?

If all four cells show zero normal crossings plus no material responsiveness
improvement for a full 60-checkpoint window, recommend typed termination as
`ABORTED_DIAGNOSTIC_CONSTANT_POLICY`. Such a collection is preserved and may
support mechanism diagnosis, but it is not a completed causal decision and
cannot select an L1 recipe. Do not silently continue twelve more cells merely
to produce sixteen zeros.

Do not stop a worker on prose alone. The transition requires the machine-
readable packet, terminal checkpoint custody and an idempotent successor
dispatch. Prepare the successor before any disposition so the fleet does not
become idle.

### WP3 - Isolate action representation from reward/entropy

Create a new content-addressed diagnostic, never an amendment to the running
identity. First use two seeds and the same ETH data, chronology, observation,
LR, costs, SL/TP and normal dynamics:

1. **Control:** current SAC plus thresholded three-bin adapter.
2. **Continuous target exposure:** SAC action directly sets signed target
   exposure in `[-max_exposure,+max_exposure]`; a deadband suppresses only
   small *changes* in target exposure. Native SL and TP remain mandatory.
3. **Entropy arm:** cross each representation with current fixed `0.2` versus
   learned/calibrated entropy (`auto`) while leaving reward unchanged.

This is a `2 x 2 x 2 seeds` diagnostic. It measures action responsiveness,
threshold crossings, costs, return/drawdown and Q/entropy diagnostics. It does
not choose a champion. A mechanics smoke may be short, but the evidence run
uses maximum 1,000 epochs, patience 60 and floor 40; it may stop early only on
the declared paired criterion or the new evidence-based constant-policy rule.

Separately implement a genuinely discrete baseline (PPO with `Discrete(3)` or
another library-supported discrete algorithm) as a falsification arm. Do not
call SAC continuous if the environment consumes only three bins.

### WP4 - Resume the owner's easy-to-normal question only after viability

After WP3 identifies a representation/entropy contract that produces a
state-responsive policy under normal dynamics, rerun the original causal
contrast:

- control: normal -> normal;
- treatment: easy -> normal;
- LR matched within each cell and both phases;
- identical genesis by seed;
- full early-stopping evidence;
- no weekly retraining factor;
- activity and state responsiveness required only for promotion, not invented
  as fitness.

Only after a viable base recipe exists does the already planned rolling-origin
retraining-cadence experiment test `8h/12h/24h/168h` against frozen controls.
Adding retraining now would confound this diagnosis and multiply compute around
a policy that does not yet react to its inputs.

### WP5 - Tests and acceptance evidence

At minimum add tests proving:

1. a constant `+0.0008` at threshold zero is classified constant-directional,
   not state-responsive;
2. the same trace at threshold `0.1` is constant-HOLD;
3. varying sub-threshold actions are distinguished from exact/near constants;
4. the continuous-target adapter preserves signed exposure bounds, SL/TP,
   costs, margin and causal chronology;
5. a discrete baseline receives an actual discrete action space;
6. a futility disposition cannot become `COMPLETE`, `ELIGIBLE` or champion;
7. sealed 2025 cannot be loaded by diagnostics;
8. successor dispatch is idempotent and leaves no duplicate workers.

Return focused and full-suite results, hashes, exact commands, before/after
counterexamples, a fleet transition plan and the independently reproducible
packet. Findings are not self-closed by the implementer.

## 5. Fleet Rule

The current four workers remain useful only while producing the WP0/WP2
evidence. CPU diagnostics and successor materialization run in parallel. If
the typed disposition says the remaining twelve cells cannot answer the
intended question, transition directly to the prebuilt WP3 job after custody;
do not create a human-approval idle gap and do not mutate a running identity.

