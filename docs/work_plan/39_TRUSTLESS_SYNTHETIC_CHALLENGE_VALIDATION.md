# 39. Trustless Synthetic Challenge Validation

Status: accepted research and protocol-hardening track; current trading
campaigns remain `synthetic_data_validation: false` until the acceptance gates
in this document pass
Version: 1.0.0
Date: 2026-08-15
Owner decision: synthetic validation is required as a possible answer to the
public-test overfitting attack in an untrusted optimization network

## 1. Purpose

DOIN participants may know every public training, validation and test dataset.
An adversarial optimizer can therefore train directly against a nominal test
set, report that memorized result and still let other nodes reproduce it. Exact
reproduction proves arithmetic agreement; it does not prove generalization.

DOIN already anticipates a second verification surface: after a candidate is
cryptographically committed, evaluators generate bounded synthetic challenges
that the optimizer could not know exactly when it committed the candidate.
This document turns that existing idea into an explicit trading-domain program.

It does not replace chronological evaluation on real data and it does not claim
that a synthetic market is the real market.

## 2. Two Validation Planes

The word `validation` has two different meanings in this system. They must
never share an unlabeled score or an ambiguous eligibility flag.

| Plane | Question | Data | Permitted use |
| --- | --- | --- | --- |
| Scientific selection | Does the candidate improve the trading objective on future real observations? | Point-in-time chronological real train/monitor/inner/outer splits; sealed real test for final reporting | Component selection, release evidence and real-performance claims |
| Adversarial challenge verification | Does the committed candidate generalize to an unpredictable, reproducible challenge drawn from a declared market-like generator? | Post-commit synthetic challenge ensemble | Detect fabricated reports, exact public-test memorization and brittle candidates; later contribute bounded Proof-of-Optimization weight only after calibration |

Synthetic samples may additionally be used for training, pretraining and stress
diagnostics under R6. That is a third provenance class. Training augmentation,
scientific selection and consensus challenge rows must remain separate in OLAP.

## 3. Security Objective and Limits

### 3.1 Objective

At candidate-commit time, the optimizer must be unable to know the exact
challenge trajectories, evaluator allocation or complete entropy that will
determine the challenge. After evaluation, an honest peer must be able to
reconstruct the challenge and verify the metric from committed evidence.

This raises the cost of an exact memorization attack: success must transfer
across a post-commit ensemble rather than one public file.

### 3.2 Non-claims

A passing synthetic challenge does not by itself prove:

- positive or stable real-market performance;
- equivalence to the sealed chronological test;
- immunity to generator-distribution gaming;
- robustness to a candidate that detects generator artifacts; or
- Byzantine safety of the surrounding quorum protocol.

Those are measured separately. Public documentation must use `challenge
verified`, not `real-world validated`, for this evidence class.

## 4. Commit-Before-Challenge Contract

The accepted design must bind this order:

1. The optimizer commits the complete candidate identity: parameters, model
   artifact, executable/evaluator contract, data contract and nonce.
2. That commitment becomes immutable at a declared chain position.
3. Challenge entropy is obtained only after the commitment. It must include a
   later finalized chain anchor or a bounded multiparty commit/reveal/VRF-like
   contribution whose honest portion was unavailable at candidate commitment.
4. The selected evaluator set and challenge manifest are bound to the task.
5. Every evaluator derives its challenge seed from the versioned derivation
   contract and records all reconstruction inputs.
6. The evaluator generates the challenge using one hash-pinned generator
   artifact/config contract, evaluates the committed candidate and signs the
   evidence.
7. Quorum aggregates a declared ensemble statistic and uncertainty, not an
   unlabeled equality comparison across different samples.

Unknown evaluator selection is not treated as secret entropy: evaluator IDs
are public and a small eligible set can be enumerated. Candidate-controlled
commitment bytes are also not entropy. A current or predictable chain tip is
insufficient unless the commitment is proven to precede the finalized anchor
mixed into the seed.

No entropy path may permit the optimizer, one evaluator or the task creator to
grind the final seed unilaterally. Failure to obtain the required post-commit
entropy fails closed to `challenge_unavailable`; it never falls back to a known
seed or public test.

## 5. Versioned Evidence Contracts

The contract spike proposes two schemas in `trading-contracts`; names are
provisional until the spike is accepted.

### 5.1 `SyntheticChallengeSpec`

- schema and seed-derivation versions;
- domain and candidate commitment IDs;
- commitment chain height/hash and finality evidence;
- challenge round and post-commit entropy/anchor evidence;
- selected evaluator IDs and selection evidence;
- generator family, artifact/config/code hashes and license/provenance;
- conditioning contract, horizon, path count and regime/event quotas;
- metric catalog and aggregation contract;
- calibrated tolerance model and calibration-artifact hash;
- resource limits and deterministic numeric-runtime contract.

### 5.2 `SyntheticChallengeEvidence`

- complete spec identity;
- evaluator identity and signature;
- derived seed and derivation inputs;
- generated-data hash, row/path counts and conditioning diagnostics;
- candidate artifact and evaluator code hashes;
- raw metric vector and per-path values;
- runtime/numeric environment and determinism diagnostics;
- refusal/unavailability reason when any required fact is absent.

Seeds are not secret after evaluation. Reproducibility after commitment is a
requirement; unpredictability before commitment is the security property.

## 6. Generator Ladder and Separation

The consensus challenge generator is a separate champion from the market-state
encoder and from any synthetic-pretraining generator. Sharing a schema is
allowed; silently sharing fitted state or scoring each other is not.

Challenge-generator ladder:

1. moving/block bootstrap with cross-feature and cross-asset alignment;
2. regime-residual bootstrap and fitted volatility/state baselines;
3. bounded stress overlays;
4. independently conditioned generators using declared point-in-time regime or
   event labels;
5. learned encoder-conditioned generation only after the lower-cost variants
   pass and circularity controls exist.

Generator fitting may use a disclosed reference distribution, including facts
derived from a public test corpus, because this plane is anti-memorization
verification rather than scientific holdout evaluation. That provenance must be
explicit and those scores can never be relabeled as real test results.

Candidate artifacts receive only their declared inference inputs. They receive
no dataset path, challenge seed, generator identity or arbitrary evaluator
metadata. Executable candidate code is restricted to the approved artifact and
inference interface so it cannot branch on environment inspection.

## 7. Calibration Before Consensus Weight

An arbitrary tolerance margin is prohibited. Calibration uses a fixed archive
of candidates spanning strong, weak, memorized, constant and deliberately
brittle controls.

For each generator family and metric, measure:

- rank and effect-size agreement with later real chronological OOS evidence;
- false acceptance of public-test-memorized controls;
- false rejection of candidates that remain valid on real OOS;
- interval coverage and evaluator-to-evaluator variance;
- sensitivity to generator family, seed, regime mix and path count;
- detection of generator artifacts by adversarial controls;
- compute and verification-to-optimization cost ratio.

Acceptance requires preregistered bounds for false acceptance, false rejection,
coverage and cost. The exact bounds are set from the pilot packet before results
are opened; they are not invented after seeing outcomes.

Promotion ladder:

1. `diagnostic_only`, zero consensus weight;
2. `shadow_challenge`, compared with real OOS but cannot accept/reject optimae;
3. `bounded_eligibility`, may reject an obviously non-generalizing candidate
   but cannot establish its real trading merit;
4. `calibrated_consensus_weight`, contributes a capped amount to Proof of
   Optimization while scientific selection remains real-only.

No direct jump from configuration existence to level 3 or 4 is permitted.

## 8. Current Implementation Facts Requiring Audit

The existing repositories contain substantial substrate, but the trading path
is not accepted merely because the symbols exist:

- `doin-core` provides `DeterministicSeedPolicy.get_seed_for_synthetic_data()`;
- `doin-node` calls it from `_evaluate_task()` using commitment/domain,
  evaluator ID and a chain-tip hash;
- `doin-node` supports `synthetic_data_validation`, evaluator plugins and
  `doin.synthetic_data` plugins;
- `doin-plugins` contains deterministic predictor and trading synthetic
  generators;
- current trading campaign profiles explicitly set synthetic validation false.

The audit must determine, with runtime traces rather than comments:

1. whether the challenge anchor is provably later than candidate commitment and
   finalized before use;
2. whether a candidate can grind commitment nonces or enumerate all effective
   seeds before committing;
3. whether only the quorum-selected evaluators can claim/complete the task;
4. whether the same task binds one immutable challenge manifest across peers;
5. whether a changing chain tip between selection and execution creates an
   unreconstructable or replayable challenge;
6. how quorum compares performance from intentionally different datasets while
   some documentation still refers to synthetic-data hash consensus;
7. whether seed basis, generator hashes and raw per-path metrics survive into
   signed chain/OLAP evidence;
8. whether missing plugins, hashes, entropy or calibration fail closed;
9. whether cross-hardware generation/evaluation is deterministic within a
   declared numeric contract; and
10. whether non-selected, lazy or colluding evaluators can fabricate evidence.

Until these questions pass adversarial reproduction, trading remains at
`diagnostic_only` and current campaigns keep `synthetic_data_validation: false`.

## 9. Execution Order

This track does not interrupt the corrected ETH L1 difficulty/learning-rate
factorial.

1. Audit the existing code path and produce executable counterexamples.
2. Freeze the two schemas and threat model before protocol edits.
3. Implement missing bindings additively with compatibility tests.
4. Run a socket-free multi-peer fixture with malicious controls.
5. Run a bounded predictor-domain pilot where synthetic verification already
   has historical precedent.
6. Run the trading shadow-challenge calibration against archived candidates.
7. Request independent audit and an explicit owner decision before any trading
   domain receives nonzero synthetic challenge weight.

R2 market-state encoder work and R6 synthetic-pretraining work continue under
their own downstream-real-utility gates. They may provide candidate components
to this track later, but they do not bypass this track's security/calibration
gates.

## 10. Acceptance Summary

The owner's proposal is accepted with this precise meaning:

- yes, post-commit synthetic challenges are a credible defense against exact
  public-test overfitting in an untrusted optimizer/evaluator network;
- yes, DOIN already contains important building blocks for that defense;
- no, synthetic scores replace neither real chronological validation nor the
  protected real test;
- no, the current existence of seed and plugin code proves the complete threat
  model is satisfied; and
- no, a tolerance margin receives consensus authority before empirical
  calibration and independent adversarial audit.

