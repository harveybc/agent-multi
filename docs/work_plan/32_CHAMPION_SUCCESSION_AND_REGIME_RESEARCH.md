# 32. Champion Succession and Regime-Specialist Research

Status timestamp: 2026-08-03 America/Bogota
Version: 1.1.0
Authors: Satoshi III (initial proposal); General Musashi (independent review
and D1-D5 disposition)
Authority: owner directives of 2026-08-03 require continuous selected-model
Paper/Demo operation, evidence-gated champion succession, and a staged
regime-specialist research track.

This document governs model replacement and regime research. It does not
authorize Live capital, increase a risk envelope, alter an active DOIN chain,
or place an LLM/Hermes process in the order path. Deterministic services make
and execute every runtime decision. The owner may halt, reject, or narrow an
operation, but no human or agent may bypass account identity, native SL/TP,
capability, reconciliation, exposure, or loss-limit controls.

## Part I - Champion Succession

### S1. Exact succession seat

A champion never replaces a global abstract "best model". A succession is
bound to this exact key:

```text
{asset, timeframe, venue, route, policy_role}
```

Changing any key component is a new activation, not a routine succession.
Every record carries the incumbent and challenger artifact/config/data/code
hashes, training stage, campaign lineage, seeds, metric contract and intended
seat key.

### S2. Two-layer evidence gate

A new simulation champion does not automatically take a Paper/Demo seat. All
of these conditions must hold:

1. **Independent offline promotion panel.** Challenger and incumbent are
   evaluated on the same frozen, causally constructed 52-week promotion panel
   that was not used for DOIN candidate fitness, hyperparameter choice, early
   stopping, or model selection. This is separate from the protected test,
   which remains excluded from promotion.
2. **Paired weekly comparison.** Persist challenger-minus-incumbent weekly RAP
   differences for at least 26 common eligible weeks. Compute a one-sided 95%
   simultaneous lower confidence bound using a paired moving-block bootstrap,
   default block length four weeks and 10,000 deterministic resamples.
3. **Multiplicity family.** The simultaneous bound uses a max-statistic over
   every promotion-eligible challenger considered against the incumbent since
   its tenure began. The family membership, count, hashes and seed are frozen
   before opening the promotion panel. If complete weekly traces are missing,
   no multiplicity claim and no promotion are allowed.
4. **Seed stability.** Across three frozen training seeds, median paired
   improvement is positive and no seed violates activity or safety gates.
5. **Operational shadow.** The challenger runs with zero orders for at least
   seven calendar days and at least 90% of expected bar/decision coverage.
   This proves continuity and runtime compatibility, not profitable
   superiority.
6. **Safety and parity.** Feature parity, model/artifact identity, order
   protection, route capability, reconciliation, stale-input and risk gates
   all pass. Any missing fact is a refusal, not a zero or success.
7. **Seat availability.** The current model remains active while the
   challenger shadows. Promotion waits for the route to be flat, then uses
   the existing verified drain, manifest replacement and session reseed from
   actual post-close broker cash/equity.

The paired bootstrap controls temporal dependence and the declared comparison
family. It is not called a Deflated Sharpe Ratio and does not claim to solve
all adaptivity in evolutionary search. DSR and PSR remain secondary
Sharpe-specific diagnostics only.

### S3. Tenure and synchronized replay

Every successful handover freezes an append-only tenure record:

- realized metrics and direct venue facts over the full reign;
- synchronized causal replay of the outgoing policy over the exact persisted
  live inputs and modeled costs;
- realized-minus-replay divergence with data, config and code hashes;
- the outgoing champion as a zero-order shadow for at least seven days and
  90% expected coverage beside its successor; and
- the successor decision packet, gate results, notices and rollback result.

The replay is called counterfactual only when a stated intervention and its
identification assumptions exist. Ordinary same-input simulation is
"synchronized replay".

### S4. Daily operations and weekly decisions

Daily observations trace mechanics and divergence. They do not establish
alpha from a tiny sample:

- realized versus modeled spread, slippage, fees and financing;
- action agreement on persisted live inputs;
- exposure overlap, fill latency/quality and calibration drift;
- expected/observed bar and decision coverage; and
- protection, account, route and reconciliation integrity.

Native-protection loss, wrong account/route, unexplained exposure, duplicate
effects or reconciliation failure have zero tolerance and trigger the
existing deterministic hold/recovery contract. Cost/drift thresholds are
versioned only after at least 30 eligible fills or four weeks of baseline
evidence; before then they are reported as unavailable rather than invented.

Weekly descriptive metrics are robust weekly RAP, return, Sortino, Calmar,
maximum drawdown, turnover, costs and coverage. PSR/DSR require their own
minimum sample contract and remain Sharpe diagnostics. Paper observations do
not replace the offline promotion panel's statistical gate.

All facts are OLAP views or immutable events over existing canonical ledgers.
No parallel operational database becomes a source of truth.

### S5. Paper/Demo authority and notice

Within an already authorized Paper/Demo seat and unchanged asset, timeframe,
venue, route, order family and risk envelope, a challenger that passes S1-S4
may succeed through the deterministic gate without a per-event confirmation
phrase. Telegram sends a pre-switch packet and a post-switch result. This is
the owner's standing no-idle Paper/Demo direction.

Explicit owner approval is still required for the first activation of a
venue, asset, route or order family; any risk-cap increase; and any future
Live-capital use. A failed, stale or timed-out gate leaves the incumbent active
or places the seat on hold. It never forces a switch.

## Part II - Difficulty Curriculum

Every model artifact carries an immutable curriculum tag:

1. `easy_no_margin_call`: training dynamics may relax early insolvency
   termination while all losses remain in fitness;
2. `normal_realistic`: realistic fees, financing, margin and margin-call
   dynamics; and
3. `hard_pessimistic` (optional): measured adverse costs and latency beyond
   nominal conditions.

Relaxation applies only to training episode dynamics and only after document
19's termination-cause instrumentation and controlled ablation pass. Train
tail, promotion panel, validation, protected test and every Paper/Demo/Live
route always use realistic solvency, fees and protection. No job-0/job-1 or
active-chain mutation is authorized here.

For the owner-activated ETH campaign, phases 1 and 2 execute inside every outer
genome stage and candidate: easy learning first, then continued normal training
from the learned weights before realistic selection and advancement. A matched
normal-only DOIN domain provides the causal control. The measured question is
whether this changes raw weekly return, drawdown, safety and sim-to-Paper
divergence. Harder training is not assumed to be better, and no opaque composite
may be presented as a business return.

## Part III - Regime-Specialist Track

### R0. Causal headroom measurement

R0 performs no policy training. It may reuse a regime detector only when that
detector was fitted inside the training cutoff and emits filtered posterior
probabilities using information available at each decision time. Smoothed
full-sample labels, hindsight clusters and future transition knowledge are
forbidden. If no qualifying detector exists, R0 is blocked until one is fitted
and frozen on CPU.

Join existing frozen policy traces to the causal posterior and report:

- per-state posterior mass, effective sample size, weeks and independent
  contiguous episodes;
- state-transition and low-confidence coverage;
- incumbent and candidate paired weekly RAP by state; and
- a routable-headroom estimate using the posterior available at decision
  time, not an oracle hindsight label.

R0 advances only when every target state has at least three independent
episodes and eight eligible weeks, the one-sided 95% lower bound of routable
net improvement exceeds zero, and the point estimate exceeds the largest of:

- two times measured incremental routing/turnover cost;
- 10% of the incumbent's absolute robust weekly RAP; or
- one basis point per week.

Failing coverage or effect size records a useful negative result and stops the
track without GPU work.

### R1. Sequential controlled comparison

If R0 passes, enqueue three equal-budget arms:

- **A, specialists:** one specialist per state, trained on contiguous episodes
  with transition context and sample weights. Regime windows are never
  concatenated into artificial time series.
- **B, conditioned generalist:** one model receives the causal regime
  posterior as an input, avoiding policy fragmentation and hard switching.
- **C, plain generalist control:** same architecture/search budget without
  regime input or specialists.

All arms use identical splits, promotion panel, seeds, costs and compute
budgets. They are replicated jobs in the one canonical campaign queue. The
fleet executes one collaborative DOIN swarm, one seed and one chain at a time;
no parallel independent domains or chains are permitted. Active jobs are not
preempted.

### R2. Standing shadow challenger

The R1 winner becomes a zero-order shadow challenger, not an automatic
trader. It is evaluated on matched eligible windows and again at each champion
succession. Promotion eligibility requires superiority under S2 in two
consecutive eligible matched windows, not merely two sparse succession events.

First deployment, if later approved, uses posterior blending with the plain
generalist as fallback and reduced exposure when confidence is low. Hard state
switching is not the first deployment.

Default placement is an Omega CPU systemd service limited to 25% CPU and 512
MiB memory, with no GPU, broker credential or order socket. It reads persisted
input packets and writes shadow facts only. Killing it cannot interrupt live
execution.

### R3. Future Paper pilot

R3 requires R2 evidence, an independent audit, an unchanged Paper/Demo risk
envelope and owner approval of the new policy role. It is not authorized by
this document.

## Part IV - D1-D5 Disposition

| ID | Disposition |
| --- | --- |
| D1 | S1-S2: exact seat key, 52-week independent promotion panel, 26 common weeks, paired four-week-block bootstrap with 10,000 resamples, one-sided 95% simultaneous bound, frozen max-stat comparison family, three seeds, seven days plus 90% shadow coverage. |
| D2 | S4: RAP is primary; DSR/PSR are Sharpe-only secondary diagnostics; safety faults have zero tolerance; cost/drift alerts require a versioned baseline and remain unavailable before it. |
| D3 | R0: causal filtered posterior, three episodes/eight weeks per state, positive lower bound and practical effect above costs/relative RAP/one-basis-point floor. |
| D4 | R2: isolated Omega CPU shadow service, 25% CPU, 512 MiB, no GPU, credentials or order sockets. |
| D5 | Routine same-seat Paper/Demo succession uses deterministic gate plus pre/post notice; new capability/risk and all Live capital remain owner-gated. |

## Part V - References Verified 2026-08-03

- Bailey and Lopez de Prado, "The Deflated Sharpe Ratio: Correcting for
  Selection Bias, Backtest Overfitting and Non-Normality," *Journal of
  Portfolio Management* 40(5), 2014, SSRN 2460551. It supports DSR for Sharpe,
  not direct deflation of arbitrary RAP.
- Shu and Mulvey, "Dynamic Factor Allocation Leveraging Regime-Switching
  Signals," arXiv:2410.14841, 2024.
- Bucci and Ciciretti, "Market Regime Detection via Realized Covariances,"
  arXiv:2104.03667, 2021.
- Gupta et al., "A forest of opinions: A multi-model ensemble-HMM voting
  framework for market regime shift detection and trading," *Data Science in
  Finance and Economics* 5(4), 2025, doi:10.3934/DSFE.2025019.
- Zhang, "FR-LUX: Friction-Aware, Regime-Conditioned Policy Optimization for
  Implementable Portfolio Management," arXiv:2510.02986, 2025.
- "Deep reinforcement learning portfolio model based on mixture of experts,"
  *Applied Intelligence*, 2025, doi:10.1007/s10489-025-06242-6.
- "Mixture-of-Experts Meets In-Context Reinforcement Learning," NeurIPS 2025.

These references motivate hypotheses and controls. None substitutes for this
project's causal, out-of-sample and live-divergence evidence.
