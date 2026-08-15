# Musashi to General Satoshi III: Trustless Synthetic Challenge Audit Order

Date: 2026-08-15
From: General Musashi, technical lead and independent verifier
To: General Satoshi III, implementation lead
Authority: owner-approved work-plan addition
Execution priority: queued, non-interruptive; finish the currently assigned
corrected ETH L1 package first. This packet may use CPU as a sidecar only when
it does not delay that package or its fleet dispatch. It authorizes no GPU job,
active-campaign mutation or consensus-weight change.

## 1. Mission

Audit the existing DOIN synthetic-verification path against the threat model in
document 39 before proposing or writing consensus-critical corrections.

The business question is precise: when every participant knows the nominal
test data, can DOIN challenge a committed model on synthetic samples that were
unknowable at commitment time, then reconstruct and verify the result with a
calibrated error margin?

Do not answer from comments, class names or intended architecture. Trace and
execute the real runtime path.

## 2. Mandatory Reading and Teach-Back

Read in order:

1. `docs/work_plan/39_TRUSTLESS_SYNTHETIC_CHALLENGE_VALIDATION.md`
2. `docs/work_plan/05_DOIN_TRADING_DOMAIN_INTEGRATION.md`, sections 1, 2 and 4.3
3. `docs/work_plan/33_ETH_DECISION_RESEARCH_AND_MULTI_ASSET_ROADMAP.md`, R2-R6
4. `docs/work_plan/23_SOCIAL_INTELLIGENCE_AND_OPERATIONAL_CONTINUITY.md`,
   section 8
5. `doin-core/src/doin_core/consensus/deterministic_seed.py`
6. `doin-core/src/doin_core/models/commit_reveal.py`, `domain.py`, `quorum.py`
   and `task.py` or the current task-model owner
7. `doin-core/src/doin_core/plugins/base.py`
8. `doin-node/src/doin_node/unified.py`: reveal, task creation/claim,
   `_evaluate_task()` and completion/quorum paths
9. `doin-plugins/src/doin_plugins/predictor/synthetic.py` and
   `doin_plugins/trading/synthetic.py`
10. `synthetic-datagen` generator, held-out, memorization and audit contracts

The first output is a machine-readable teach-back containing exact repository
commit IDs and answers to these statements:

- scientific real validation, synthetic training and adversarial synthetic
  challenge are three different provenance classes;
- evaluator identity is public and is not secret entropy;
- candidate commitment bytes and nonce are optimizer-controlled and are not
  independent entropy;
- challenge reconstruction after evaluation is required;
- unpredictability before candidate commitment is required;
- synthetic challenge performance is not a real-market performance claim; and
- current trading profiles stay `synthetic_data_validation: false` throughout
  this packet.

Any disagreement or ambiguity is reported before implementation. Do not smooth
it over to satisfy the packet.

## 3. WP0: Executable Runtime Map

Produce one call/data-flow map with file/line references and runtime evidence:

```text
candidate optimize
  -> commitment
  -> reveal
  -> quorum selection
  -> verification task construction/flood/claim
  -> challenge seed derivation
  -> synthetic generator
  -> evaluator metric
  -> completion evidence
  -> quorum aggregation
  -> optimae acceptance/rejection
  -> Proof-of-Optimization/chain/OLAP facts
```

For each edge state the authoritative object, signer, persisted identity and
failure behavior. Mark intended-but-unwired methods explicitly.

## 4. WP1: Reproduce Before Correcting

Create a socket-free adversarial reproducer and focused tests. It must answer,
without mutating production code first:

1. Is the actual reveal `commitment_hash` copied into an immutable task field,
   or can `_evaluate_task()` fall back to candidate parameters/`optimae_id`?
2. Is the challenge chain anchor fixed in the task after commitment, or read
   from whatever local tip exists when each evaluator happens to execute?
3. Is that anchor finalized and provably later than the candidate commitment?
4. Can an optimizer enumerate evaluator IDs and precompute every seed when the
   effective chain tip is already known?
5. Can it grind candidate nonce/commitment values against a known challenge
   anchor?
6. Can a non-selected evaluator claim or complete the broadcast task?
7. Does one selected evaluator's task/evidence identify the full selected set
   and immutable challenge manifest?
8. What happens when the chain advances or reorganizes between quorum selection
   and `_evaluate_task()`?
9. Do different evaluator seeds intentionally produce different data hashes?
   If so, identify and test the actual quorum statistic; do not call it data-hash
   consensus.
10. Can missing synthetic plugin/data/hash/performance facts become `None`, an
    empty hash or a default `0.0` and still vote?
11. Can evaluators run different generator artifacts/configs yet produce votes
    that enter the same quorum?
12. Can a lazy evaluator fabricate performance/hash evidence without executing
    the generator and candidate?
13. Are seed basis, generator identity, path metrics and numeric environment
    persisted sufficiently for later reconstruction?
14. Can a candidate artifact observe the generator ID, seed, file path or
    environment and branch specifically for the challenge?
15. Do current config defaults risk enabling uncalibrated synthetic validation
    when a domain omits the field?

Include at least these adversarial fixtures:

- same commitment, two execution-time tips;
- optimizer-chosen `_don_commitment_hash` conflicting with the reveal envelope;
- known tip plus enumerated evaluator set;
- non-selected claimant;
- missing synthetic plugin;
- missing/empty data hash;
- missing performance key;
- mismatched generator artifact/config;
- replayed task from an earlier challenge round;
- honest per-evaluator samples with materially different metric variance.

Label every outcome `reproduced`, `not_reproduced` or `unavailable`, with the
direct fact supporting it. This work package closes nothing.

## 5. WP2: Contract and Protocol Options, No Code Yet

Based on WP1, propose the smallest backward-compatible design for:

- `SyntheticChallengeSpec` and `SyntheticChallengeEvidence` schemas;
- authoritative candidate commitment binding;
- a post-commit entropy source and finality/delay proof;
- one task-bound evaluator set and challenge manifest;
- seed-derivation versioning;
- generator artifact/config and numeric-runtime binding;
- signed per-evaluator evidence;
- robust ensemble aggregation and calibrated uncertainty;
- challenge replay prevention; and
- fail-closed unavailability.

Compare at least two post-commit entropy mechanisms:

1. a later finalized chain anchor with an explicit commitment/challenge delay;
2. evaluator multiparty commit/reveal mixed with a finalized anchor.

For each, analyze liveness, grinding, withholding, collusion, reorganization,
reconstruction and compatibility. Do not claim a VRF unless an actual verified
VRF implementation and key contract are proposed.

The design must not depend on secrecy of evaluator IDs. It must explain what
happens when no fresh block/finality event becomes available.

## 6. WP3: Calibration Plan

Prepare, but do not execute at scale, a calibration manifest using archived
predictor/trading candidates:

- valid strong and weak candidates;
- exact-public-test memorization controls;
- constant/no-trade controls;
- generator-artifact detectors or other deliberately brittle controls;
- multiple transparent generator families, seeds and regime mixtures; and
- later real chronological OOS outcomes when available.

The packet defines preregistered metrics for rank agreement, false acceptance,
false rejection, interval coverage, evaluator variance and compute cost. It
must show how a tolerance is estimated; `pick a margin that seems reasonable`
is an automatic rejection.

The first pilot is predictor-domain/socket-free. Trading remains shadow-only.

## 7. Forbidden Actions

This order does not authorize:

- changing `doin-core`, `doin-node` or `doin-plugins` consensus behavior before
  WP0-WP2 receive Musashi review;
- enabling synthetic validation in any active trading config;
- giving a challenge nonzero consensus weight;
- training a new neural generator;
- using synthetic rows as real validation/test evidence;
- interrupting or delaying corrected ETH L1 work;
- touching broker/live execution; or
- representing documentation comments as runtime proof.

## 8. Return Packet

Return one bounded packet with:

- repository commits and dirty-state hashes;
- teach-back;
- runtime map;
- reproducer and raw before-state evidence;
- finding table with severity and exact affected contract;
- protocol alternatives and recommendation;
- schema drafts marked `proposed`;
- calibration manifest;
- focused/full test counts; and
- explicit unknowns and owner decisions, if any.

End with a short execution proposal for the next packet. Do not begin protocol
implementation merely because the proposal is written. Musashi will reproduce
the counterexamples and decide the correction boundary while the approved ETH
compute continues independently.

