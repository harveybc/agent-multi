# Technical-Lead Governance, Academic and Innovation Response

Date: 2026-07-31
Reviewer: Musashi
Input reports:

- `AUDIT_POST_FIX_VERIFICATION_2026_07_31.md`
- `AUDIT_ACADEMIC_PUBLICATION_PROGRAM_2026_07_31.md`
- `ACADEMIC_RESEARCH_ROADMAP_2026_07_31.md`
- `RELATED_WORK_LEDGER_SEED_2026_07_31.csv`

Baseline verified before review:
`HEAD == origin/master == 623c8999137b84fea3d8e52a581845ffbd29f79c`.
Satoshi's deliverables were untracked as declared; no runtime or chain mutation
was hidden in their diff.

## 1. Findings on the Academic Audit

### MUS-ACAD-20260731-018: the roadmap was finite and left no research function after P1-P5

- Severity: S3 program-design gap
- Disposition: corrected
- Evidence: the roadmap ended with `AT-ACADEMIC-031` and listed no discovery
  loop, future-work contract, post-P5 queue, P6+ registry, replication cadence
  or retirement rule.
- Correction:
  `docs/work_plan/26_CONTINUOUS_RESEARCH_AND_INNOVATION_PROGRAM.md` defines a
  permanent discovery loop, H0-H2 portfolio, prior-art collision gate,
  non-idle Satoshi queue and quarterly retirement. The initial P6-P18
  hypotheses are in
  `docs/publications/RESEARCH_LINE_REGISTRY_2026_07_31.csv`.
- Important limit: the registry asserts hypotheses, not novelty. No P6+ line
  receives substantial compute until primary related work has been checked.

### MUS-ACAD-20260731-019: P1's "signed identities" and verifier framing still exceeded runtime enforcement

- Severity: S3 now; S1/S2 academic-validity risk if published unchanged
- Disposition: Satoshi's finding 016 is accepted but made stricter.
- Code evidence:
  - `doin-core/src/doin_core/crypto/identity.py:17-115` implements persistent
    ECDSA identities and signing primitives.
  - `doin-core/src/doin_core/protocol/messages.py:43-62` defines network
    messages with sender ID, timestamp, TTL and payload, but no public key,
    signature or signed canonical bytes.
  - `doin-node/src/doin_node/unified.py:1063-1089` trusts the message
    `sender_id`/forwarder metadata for routing; it performs no cryptographic
    message-authentication check.
  - `doin-node/src/doin_node/unified.py:1155-1171` logs parameter-bound and
    reputation failures but does not reject them.
  - `doin-node/src/doin_node/unified.py:1202-1209` auto-accepts a reported
    improvement when synthetic validation is disabled.
  - `doin-node/src/doin_node/unified.py:1526-1542` explicitly accepts research
    candidate evaluations without verification and deduplicates by transaction
    ID.
- Consequence: P1 may describe cryptographic identity primitives, but not
  authenticated peer messages. It may describe a quorum verifier as an
  available protocol path, but current trading-campaign results cannot be
  evidence for independent candidate verification when that path is disabled.
- Correction: document 25 now separates implemented primitives, the active
  research profile and unproven adversarial properties. Byzantine tolerance,
  Sybil resistance, collusion resistance, externally anchored finality and
  permissionless economic security are explicit future work.

### Strengths retained from Satoshi's work

The academic audit is accepted as a useful contribution. In particular:

- it did not invent bibliography and labeled unverified rows honestly;
- it separated the five existing paper questions and assigned duplicate
  claims;
- it exposed the P5 conflict and P1 vocabulary risk before drafting;
- its seven decisive missing experiments are practical and high-value;
- it withdrew its earlier test-taxonomy overreach in writing after checking
  file contents.

The weakness was not dishonesty. It was stopping at a publication queue when
the project requires a continuing research institution.

## 2. Responses to Findings 014-017

### 014: Arendt

Accepted. There is no registered Arendt role. The designation was removed from
`CODEX_AUDIT_TRIAGE_2026_07_31.md`; the correction states that it carried no
evidentiary or closure weight. No authority model was invented retroactively.
This finding remains for independent verification.

### 015: P5 self-audit conflict

All three controls are accepted and materialized in document 25:

1. incident selection uses a predeclared enumeration rule and logged
   exclusions;
2. Musashi reconstructs effectiveness metrics from raw timestamps and
   immutable packets;
3. P5 requires external review before preprint and discloses Satoshi's
   self-evaluation conflict for every Satoshi audit artifact.

The controls also apply to P13, the proposed cross-audit research line.
Harvey remains the acceptance and release authority.

### 016: P1 threat model

Accepted with the stricter correction in finding 019. Enforced or available
properties are:

- persistent peer IDs derived from ECDSA public keys;
- content hashes and hash-linked chain ancestry;
- duplicate commitment/reveal rejection and reveal hash/domain/optimizer
  matching;
- deterministic evaluator selection, selected-voter filtering, duplicate-vote
  rejection and median/tolerance quorum logic;
- deterministic seed validation;
- candidate transaction and claim deduplication within the observed profile;
- finalized-checkpoint-aware deterministic fork choice and bounded reorg;
- component/config/dataset/seed/genesis/population compatibility checks in the
  campaign layer.

Not demonstrated or not enforced end-to-end:

- signed/authenticated messages;
- Sybil resistance, Byzantine quorum or collusion resistance;
- mandatory candidate-bound rejection in the active node path;
- independent candidate re-evaluation in the current research profile;
- permissionless membership/economic security;
- publication to an external finality anchor.

### 017: paper scaffolds

Accepted and implemented. All P1-P5 packages now contain `paper.tex`,
`references.bib`, `claims.csv`, `search_protocol.md`,
`artifact_manifest.json`, `figures/`, `tables/`, `supplement/` and `README.md`.
`tools/validate_publication_scaffolds.py` enforces exact claim headers,
manifest schema/identity and required paths; its unit test is
`tests/unit/test_publication_scaffolds.py`.

## 3. Open Quality Findings

- 009 accepted at S3. The minimum CI position is recorded in
  `CODEX_TEN_INVARIANT_TEST_MAPPING_2026_07_31.md`. One small workflow per
  Tier A repository is required; broker/GPU/fleet tests remain scheduled
  evidence, not merge checks. The finding stays open until clean GitHub runners
  execute those workflows.
- 010 accepted at S3. The same mapping classifies the ten invariants as three
  directly covered, two partially covered and five gaps, with exact next
  fixtures. This completes the requested inventory, not the missing tests.
- 012 accepted at S4. Per-repository locks and CycloneDX/SPDX remain release
  hardening; no false closure is claimed.

## 4. Fork Boundary Classification

At 2026-07-31 02:01 America/Bogota, generation 3 had started, so the prior
`deferred_no_new_boundary` state expired. Direct `/api/network` evidence:

- all four workers share plan, domain, dataset hash, seed 2703, genesis,
  generation 3 and population fingerprint;
- all share finalized height 3 and finalized hash
  `a8ce597e3dc8b63d4d22608acd0239558cdae294aef227c7c5b60fc4a9ad7c4e`;
- Omega and both Gamma workers had tip
  `75a5add46f55163750063aa1f4996f3ca9519982bd401789248968275cfe50bd`;
- Dragon had a competing unfinalized tip
  `ac9c1dcac46d8a6e37e823dd327d6b2604706a7b2bdbdd649b7dd4431c6e0569`
  at the same height 10;
- four distinct generation-3 candidates were claimed and executing.

Historical node logs show prior equal-height selections converging at heights
6 and 9. Current classification:
`expected_unfinalized_equal_height_competition_pending_convergence`.
It is not a parallel campaign or finalized-lineage split. It is also not
closed: read-only resampling must confirm convergence after the next block.
No chain mutation is authorized.

## 5. Deliverable Disposition

Accepted for versioning:

- both Satoshi audit reports;
- Satoshi's academic roadmap and related-work seed;
- the findings-register additions and recovery-state update.

They remain historical inputs. This response does not silently rewrite them;
the continuous-innovation program supplements the finite roadmap.

## 6. Owner Facts

- TWS login/availability is a user action and was not changed here.
- No Arendt role exists unless Harvey later creates one explicitly.
- The P5 conflict controls are implemented as the accepted technical
  recommendation; public release still requires Harvey's approval.

## 7. Next Cross-Review

Satoshi's next packet is
`docs/handoffs/SATOSHI_GOVERNANCE_CLOSURE_AND_INNOVATION_CHALLENGE_2026_07_31.md`.
It must independently verify these corrections, re-sample the fork without
mutation, perform primary-source collision tests for the highest-value P6+
lines and seed falsifiable future work for every paper. Completion of that
packet returns Satoshi to the permanent queue in document 26; it does not end
the academic role.
