# Order: Post-Transfer Pretraining Objectives and Comparison Harness

Date: 2026-08-27 America/Bogota
From: General Musashi
To: General Satoshi
Priority: P0 scientific path; CPU implementation first

## WP0 -- Custody hardening 362/363

Replace destructive completion-intent removal with append-only acknowledgement:

- retain the durable intent permanently;
- create a separately fsynced, no-clobber completion ACK;
- renderer requires matching intent + ACK + completed ledger + evidence digest;
- uncertain/mismatched combinations refuse;
- no automatic recovery.

Force mode 0600 on every ledger, marker, ACK and temporary file before rename;
verify 0700 ledger directories. Add mode regressions after reserve, every
transition, completion and process restart. This work is CPU/model-free and may
run in parallel with WP1.

## WP1 -- Complete the three remaining pretraining objectives

Extend the existing v4 branch-pretraining plugin architecture, never a separate
ad-hoc trainer:

### Hierarchical contrastive objective

- positive pairs are causal views of the same training window at declared
  temporal scales;
- augmentations cannot read future, calibration or monitor values;
- negatives come from train only, with deterministic sampling and false-negative
  policy declared;
- temperature, projection dimension and loss weight are explicit contract
  fields; projection head is an excluded transfer adapter;
- report collapse diagnostics, embedding variance and effective negatives.

### Volatility objective

- predict strictly forward realized volatility at predeclared horizons using
  returns available only inside the causal fit partition;
- declare estimator, annualization units and epsilon; no default target formula;
- purge uses the maximum horizon across **all** objectives;
- report calibration and untouched-monitor loss separately.

### Barrier-hit objective

- define upper/lower barriers and horizons prospectively from past-only scale;
- label first upper hit, first lower hit, neither/censored and same-bar collision
  with an explicit conservative rule;
- no barrier or scale calibration on monitor/outer/sealed data;
- class balance/weights derive from calibration only and are frozen;
- classification head is excluded from transferred state.

For every objective: strict types, finite losses, deterministic resume, gradient
norm/cosine diagnostics, adapter exclusion, and train/calibration/monitor purge
evidence. Outer 2024 and sealed 2025 remain structurally unavailable.

## WP2 -- Multi-objective mechanics screen

Run CPU-only bounded fixtures first:

1. each objective alone on every applicable family;
2. all objectives together with predeclared balancing;
3. resume interruption/replay parity;
4. one bounded real-data o2022 smoke, mechanics-only.

Reject any objective with constant targets, zero encoder gradient, representation
collapse, non-finite values, target leakage or materially unresolved gradient
conflict. Do not choose weights from monitor performance.

## WP3 -- Paired random-vs-pretrained comparison harness

Materialize, but do not launch, a paired design with identical:

- strong architecture, data roles, seed, optimizer, SAC budget, execution
  envelope, costs, stopping and evaluation;
- random initialization control versus pretrained encoder treatment;
- separate treatment arms for frozen encoders and fine-tuned encoders unless a
  CPU identifiability screen eliminates one prospectively;
- four seeds, counterbalanced arm order and one declared primary endpoint;
- trial ledger and no outer/sealed selection.

Predeclare minimum activity, constant-policy/dead-actor refusals, paired effect,
dispersion and an `INCONCLUSIVE` outcome. Runtime/economic results cannot be
inferred from pretraining losses.

## WP4 -- Return packet

Return:

1. 362/363 PRE/POST evidence and actual filesystem modes;
2. objective formulas, units, target ranges and causal diagrams;
3. focused/full tests and Tier-A results;
4. CPU smoke histories and gradient-conflict tables;
5. exact transfer-state key inventory proving every head excluded;
6. paired comparison configs, genesis identities, dispatch plan and estimated
   GPU cost;
7. proposed GPU command, **not launched**.

After independent audit, Musashi will authorize the smallest informative GPU
screen. No owner phrase is required for CPU implementation. Live Alpaca/MT5
services remain untouched.
