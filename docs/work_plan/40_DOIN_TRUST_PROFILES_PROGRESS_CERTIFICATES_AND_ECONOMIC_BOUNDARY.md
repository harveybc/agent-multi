# 40. DOIN Trust Profiles, Progress Certificates and Economic Boundary

Status: owner-directed design boundary; current implementation differences are
recorded explicitly and are not relabeled as completed protocol behavior
Version: 1.0.0
Date: 2026-08-15

## 1. Purpose

DOIN already provides useful coordinated optimization in a fleet operated by
one trusted owner. A future network with mutually untrusted participants needs
additional verification, identity and economic mechanisms. These two systems
must not be described with the same present-tense claims.

This document separates:

1. the trusted system that is operating now;
2. the conditional untrusted research profile;
3. prototype economics currently present in `doin-core`; and
4. the owner's target concept of one progress certificate as the issuance
   unit.

It does not activate a coin, modify the running chain or authorize a public
permissionless deployment.

## 2. Trust Profiles

### 2.1 `trusted_consortium`

The current profile assumes one trusted operator or a cooperating consortium.
It still verifies candidate identity, ancestry, artifact integrity, duplicate
claims, chain consistency and lineage. Performance re-evaluation may use a
declared real domain criterion or may be explicitly disabled when the operator
accepts the report.

Skipping performance re-evaluation is a named profile capability. It is not a
claim that verification, hashes or lineage are unnecessary.

No native coin is required for this profile to be useful.

### 2.2 `untrusted_generated_gate`

This profile remains conditional research. A domain cannot use an untrusted
optimization acceptance gate until all of the following exist:

- authenticated participants and an explicit Sybil/collusion model;
- commit-before-challenge ordering and post-commit entropy;
- an admitted, content-addressed challenge generator;
- deterministic reconstruction of each evaluator's distinct draw;
- signed evidence, quorum rules and calibrated aggregation tolerance;
- a falsified-or-bounded attempt to optimize against generator artifacts; and
- a declared rule for comparing progress within and, if applicable, across
  domains.

Document 39 owns the synthetic-challenge admission and calibration program.
No domain currently passes that complete program. Therefore no current domain
has nonzero authority under this untrusted profile.

## 3. Generator Identity and Draw Custody

Two hashes have different meanings and both may be required:

1. `generator_identity`: a manifest hash over generator code, model weights,
   configuration, training-data references/hashes, runtime contract and
   dependency lock;
2. `draw_custody`: a hash of the generated challenge plus the post-commit seed
   derivation inputs used by one evaluator.

The generator hash identifies the challenge mechanism. The draw hash proves
which event-specific sample was evaluated. A draw hash must never replace the
generator identity, and a generator hash alone is insufficient to replay a
vote.

Evaluators may use distinct draws. Each draw must be reproducible after the
fact from its recorded `seed_i` and immutable manifest. Quorum therefore
aggregates a calibrated ensemble statistic; it does not require different
draws to have equal hashes or identical raw scores.

## 4. Ledger Liveness, Progress and Issuance

Three controls are separate:

| Control | Purpose | Current target |
| --- | --- | --- |
| Ledger liveness | Preserve ordered events and operational continuity | Event/heartbeat blocks may carry zero issuance |
| Progress bin | Define how much verified useful improvement fills one certificate | Fixed quality contract; it does not become easier to satisfy merely to meet wall-clock cadence |
| Issuance/distribution | Decide whether and how value is created and divided | One unit per completely filled verified progress certificate; zero for an empty progress bin |

The unit `1` is an owner-directed normalization target, not a statement about
the current code. It is not a salary per elapsed block. Partial contribution
inside a filled certificate affects distribution shares; fractional issuance
for an unfilled bin is not accepted without a later explicit economic
experiment.

## 5. Current Artifact Versus Target

As of 2026-08-15, `doin-core` implements:

- `INITIAL_BLOCK_REWARD = 50` and Bitcoin-like halving constants;
- a time-targeted adjustment of the optimization threshold;
- a weighted sum of raw domain increments without a formal cross-domain
  numeraire;
- a `0.5` verification-strength fallback for domains without synthetic data;
- `EVALUATION_SERVED` accounting that influences a task-count proxy but does
  not directly pay the inference service; and
- generator/optimizer/evaluator reward allocation, including prototype empty
  block and transaction-fee behavior.

The transaction-fee path also contains a reproduced conservation defect. With
block reward `50`, transaction fees `10` and no contributors,
`distribute_block_reward()` distributes `67.15` although only `60` is
available. This is a code defect, not an economic-policy choice. It must be
corrected under the invariant
`sum(outputs) == block_reward + transaction_fees` before any coin experiment.

Independent graph/code reproduction also found three prototype drifts that
must remain explicit until corrected in an isolated protocol branch:

- optimizer/quorum admission derives progress from `abs(reported_value)`
  rather than the improvement delta against the accepted current best;
- both the difficulty manager and consensus state can write the optimization
  threshold during block generation, so authority is not singular; and
- `ProofOfOptimization.record_evaluation()` has no runtime caller, while
  inference task completion records a task event rather than the distinct
  optimization-evaluation service fact.

These facts do not authorize a deployed consensus change during the active ETH
decision run. They require failing tests, typed trust/economic profiles and an
independently audited migration.

These are reproducible code facts. They are not accepted as owner-ratified
production economics. Papers and interviews must label them
`implemented_prototype` until a versioned replacement is implemented and
audited.

The following are implementation or documentation drift relative to the
current owner-directed design:

- presenting 50/halving/21M as canonical DOIN economics;
- lowering a quality threshold to maintain a block-time target;
- treating missing synthetic verification as half-trusted in the untrusted
  profile;
- calling task-count share a market price; and
- claiming cross-domain optimality from the current weighted raw sum.

## 6. Inference Service Boundary

Model discovery, artifact availability and execution of an inference request
are separate goods.

- A champion artifact may be publicly downloadable.
- A node may charge for timely hosted inference, availability, freshness or a
  service-level commitment when it accepts the client's bid.
- The current coinbase code does not implement that service payment.
- A future priority bid belongs to P14 and requires demand, spam, capacity,
  fairness and failure experiments before implementation.

`inference_tasks_completed / total_tasks` is named
`observed_on_chain_task_share`. It is a censored operational statistic, not a
price and not sufficient evidence of willingness to pay.

## 7. Cross-Domain Progress Is Open

The present code multiplies domain-specific raw increments by configured
weights and sums them. Accuracy points, Sharpe-like quantities and other
metrics are not automatically commensurable.

Until a normalization or exchange-derived numeraire survives adversarial
analysis, DOIN may claim:

- verified progress inside a domain under that domain's declared metric; and
- a configured multi-domain scheduling statistic.

It may not claim an economically optimal cross-domain allocation or a formal
composite price. P14 and P18 own this open research question.

## 8. Academic Speech Contract

Present tense is reserved for observed trusted-mode behavior. Conditional
tense is mandatory for the untrusted generator gate, cross-domain economics
and native coin target.

Permitted default phrase for P1 before the ratio experiment completes:

> a bounded verification evaluation compared with the search process that
> produced the candidate

After measurement, the paper must report the ratio with its workload, hardware,
domain, metric and uncertainty. It must not replace that measurement with
`cheap hash`, `free verification` or `unbounded search`.

## 9. Implementation Order

1. Preserve the current artifact as a replayable prototype profile.
2. Add typed trust/economic profiles and tests in an isolated branch.
3. Correct comments and READMEs to distinguish implemented facts from target
   design.
4. Correct and independently verify transaction-fee conservation without
   changing the declared prototype reward shares.
5. Complete document 39's generator admission spike.
6. Measure verification-to-generation cost on sealed evidence.
7. Specify progress-bin normalization and within-bin distribution separately.
8. Run adversarial economic fixtures before changing issuance behavior.
9. Require independent audit and owner ratification before deployment.
