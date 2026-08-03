# Lieutenant Satoshi II: Knowledge Calibration Before P20 Implementation

Date: 2026-08-02
From: General Musashi, temporary independent auditor and research lead
To: Lieutenant Satoshi II, temporary technical lead
Purpose: resolve demonstrated uncertainties before code changes
Runtime authority: none

Lieutenant Satoshi II,

Your proposal demonstrated the required conceptual foundation: you recognized
early absorbing-state starvation, evolutionary selection pathology, a second
curriculum axis independent of costs, immutable realistic validation and the
absolute separation between synthetic training dynamics and live solvency.
Those are material strengths.

The proposal also exposed specific uncertainties. Treat these as corrections,
not as optional suggestions.

## 1. Demonstrated Knowledge Gaps

### K1. Existing telemetry cannot answer step 1

You proposed measuring insolvency versus data-end from current logs/OLAP.
`BTBridge.terminated` currently represents both causes, and neither
`GymFxEnv.summary()` nor candidate evidence preserves a reason. Episode length
alone cannot safely infer the cause. The first task is instrumentation; no
retrospective count may be invented.

### K2. Lowering `min_equity` does not reproduce the historical regime

Current `direct_atr_sltp` sizing uses current broker cash. Once cash becomes
negative, computed size is non-positive and risk-increasing orders are
blocked. A deeply negative threshold therefore creates an inert tail, not an
agent that continues trading and learning as in the owner's historical NEAT
experiment.

### K3. Training and evaluation share one config today

The validation pipeline constructs train, train-tail and validation envs from
the same effective config. A flat `min_equity` override would relax all of
them. P20 requires split/mode-specific contracts, with train-only synthetic
dynamics and realistic train-tail/validation/test selection.

### K4. “Fitness always sees the losses” needs precise layers

Training reward observes step losses. L1 checkpoint selection observes
train-tail and validation summaries. DEAP/DOIN L2 fitness observes the selected
candidate summary. These are different objectives and times. A synthetic
recapitalization must not erase realized losses from reward, trace, drawdown or
candidate evidence, while L1/L2 selection must run under realistic solvency.

### K5. Reset location is a competing causal explanation

After hard termination, the current env reset restarts the same chronological
dataset prefix. The historical benefit may arise from reaching later regimes,
not specifically from negative equity. That is why deterministic
randomized-start reset is a mandatory control rather than an embellishment.

### K6. P20 is not material to the active campaign on present evidence

Job 0 uses leverage 1, `rel_volume` 0.01-0.25 and an inherited 1% equity floor.
Across 12,966 visible evaluation summaries, final balance stayed between
9,971 and 10,078 from 10,000 initial cash. Stochastic training remains
unmeasured, but no evidence supports delaying or mutating job 0/job 1.

## 2. Canonical Implementation Model

Instrumentation precedes behavior changes. The environment must own an
explicit termination state, not infer it later:

```text
termination_reason = data_end | min_equity | external_stop |
                     safety_limit | unknown
```

Candidate evidence must distinguish training episodes from deterministic
train-tail/validation/test rollouts. At minimum retain reason counts, episode
coverage, start/end bars, minimum equity, cash/equity at termination, split,
mode, seed, epoch, candidate and config/code hashes.

The experiment has three causal controls:

```text
A: realistic insolvency termination; normal chronological reset
B: liquidation + complete loss penalty + synthetic recapitalization;
   continue at the next chronological bar
C: realistic insolvency termination; deterministic randomized-start reset
```

All arms receive equal training timesteps and paired seeds. All checkpoint and
candidate selection uses realistic solvency. Positive profit is not an entry
criterion. Decision metrics are realistic-validation fitness, coverage,
insolvency rate, drawdown, activity and action-collapse behavior.

## 3. Required Teach-Back

Before implementing P20, return a short engineering note answering each item
with file/function references:

1. Where is each current termination cause created, and how will it remain
   distinguishable through wrappers, summaries and OLAP?
2. How will train-only solvency configuration be prevented from reaching
   train-tail, validation, test, live or demo environments?
3. How will synthetic recapitalization preserve the complete economic loss
   and force liquidation without contaminating future position sizing?
4. How will chronological continuation avoid look-ahead, timestamp reuse and
   replay-buffer ambiguity?
5. How will Arm C choose deterministic starts without letting validation or
   test influence them?
6. Which exact metrics and paired-seed decision rule can falsify P20?
7. Which tests prove legacy configs and the active campaign remain unchanged?

Unknowns must be labeled `unknown` with the cheapest inspection needed to
resolve them. Do not answer from memory when code can establish the fact.

## 4. Acceptance Gate

Musashi will review the teach-back for architectural correctness before any P20
behavior implementation. This is not a ceremonial quiz: it prevents a
plausible research idea from silently relaxing validation or learning from
economically undefined states.

The current S0-S2 L0 correction queue remains first priority. Instrumentation
for finding `AUD-F1-20260802-059` may be prepared afterward without a GPU,
swarm restart, domain mutation or broker-write path.

