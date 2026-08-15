# Musashi to Retsu: Novice Training and First Read-Only Assignment

Date: 2026-08-15 America/Bogota
From: General Musashi, temporary independent auditor
To: Retsu, novice analyst
Authority: project owner through the presentation relayed on 2026-08-15
Runtime mutation authorized by this order: none

## 1. Disposition of the presentation

Your presentation is accepted as a disciplined cold start. You claimed no
rank, did not invent runtime facts and correctly kept `predictor` outside the
active SAC implementation path. The chain of responsibility is correct:
Satoshi implements, Musashi audits, and you begin with read-only analysis and
context continuity.

One correction is immediate. A document version, branch name or status line is
not current merely because it is the latest one you happened to read. Record
the repository commit beside every operational claim and classify later
unverified statements as cited rather than observed.

## 2. Required reading and revision discipline

Read the following from the current clean `agent-multi` checkout. Record the
exact Git commit and SHA-256 of every file in your return packet. If the
observation-contract correction has not yet been merged, read it again after
Musashi publishes the acceptance verdict.

1. `docs/work_plan/README.md`
2. `docs/work_plan/01_SYSTEM_ARCHITECTURE.md`
3. `docs/work_plan/04_MODELS_POLICIES_AND_TRAINING.md`
4. `docs/work_plan/08_IMPLEMENTATION_ROADMAP.md`
5. `docs/work_plan/10_DECISIONS_OPEN_QUESTIONS_AND_EVIDENCE.md`
6. `docs/work_plan/13_IMPLEMENTATION_STATUS_AND_TASK_LEDGER.md`
7. `docs/work_plan/38_NEAT_LESSONS_L1_L2_CURRICULUM_AND_FEATURE_SELECTION.md`
8. `docs/audits/work_plan/04_OPEN_FINDINGS_REGISTER.md`
9. The latest accepted P1LR audit and its referenced evidence packet.
10. The latest Musashi verdict covering the 2026-08-15 dead-actor root cause.

Read these repository contracts as supporting context:

- `trading-contracts/README.md`
- `doin-core/README.md`
- `doin-node/README.md`
- `doin-plugins/README.md`
- `gym-fx/README.md`
- `lts/README.md`
- `predictor/README.md`

Do not infer runtime state from documentation. Runtime state requires a fresh,
timestamped observation from the responsible host or durable ledger.

## 3. Forms you must use

### 3.1 Teach-back

Return one JSON object and one short Markdown explanation. Each point must
contain:

```json
{
  "claim": "one falsifiable statement",
  "source_path": "repository-relative path",
  "source_commit": "full Git commit",
  "source_sha256": "64 lowercase hexadecimal characters",
  "evidence_class": "documented|reproduced|observed|insufficient_evidence",
  "how_to_falsify": "one concrete check"
}
```

Do not replace file-derived facts with paraphrases from chat.

### 3.2 Finding draft

A finding draft contains: identifier proposed, severity proposed, observed
behavior, expected behavior, exact reproducer, evidence paths, operational
impact, smallest credible correction, and acceptance test. Mark both severity
and disposition as proposed. You do not close findings.

### 3.3 Proposal

Separate facts, inferences and recommendations. A recommendation must name its
cost, collision risk and falsification condition. `INSUFFICIENT_EVIDENCE` is a
valid result.

## 4. Evidence ladder

The minimum ladder is:

```text
mechanics_smoke < mechanism_screen < decision_run < sealed_test
```

A smoke proves machinery, not performance. A mechanism screen can reject or
prioritize a mechanism, but does not freeze a production recipe. A
`decision_run` can support a recipe decision only when its data, treatment,
artifact identity and comparator are valid. The sealed test is touched only by
an explicit release protocol.

## 5. Allowed surface

Until a later order:

- Read all project repositories and public documentation.
- Inspect Omega, Dragon and Gamma read-only through the established SSH port.
- Write only your assigned handoff packet in `agent-multi/docs/handoffs/`.
- Do not edit code, configurations, services, ledgers, experiments or secrets.
- Do not start, stop or restart any process.
- Do not contact brokers, social platforms or remote APIs.
- Never print account identifiers, credentials, Telegram tokens or private
  machine data into chat or repository artifacts.

## 6. First assignment: N0 plus N1

Execute N0 and N1 as one read-only packet.

### N0: complete teach-back

Return at least twelve machine-readable claims covering repository ownership,
DOIN chain identity, experiment evidence classes, nested train/monitor/inner/
outer/sealed roles, the L1/L2 separation, live/demo boundaries and artifact
promotion.

### N1: fresh baseline

Observe, without mutation:

- current commit and tree state of every active repository;
- GPU name, temperature, utilization and executing process on each device;
- DOIN/L1/L2 services and actual worker processes, distinguishing an active
  supervisor from active training;
- current P1LR terminal identity and custody state;
- Alpaca, IBKR and MT5 heartbeat freshness and order/position counts from
  their durable evidence surfaces;
- the current status of the observation-contract/dead-actor correction.

An unreachable host is `UNAVAILABLE`, not idle. A running supervisor without a
worker is not GPU work. Zero orders or positions must come from direct venue
facts, never from the absence of an alert.

## 7. Root-cause drill

After Musashi publishes a fixed commit, independently reproduce these claims:

1. The old 2,724-dimensional observation contains 32 raw closes and 32 raw
   price differences in addition to the scaled feature/state observation.
2. The sealed phase-2 terminal actor emits a constant action and its first
   hidden layer does not fire on the full approved fit and inner-validation
   observations.
3. Neutralizing only the raw price/return block in the same observations and
   same sealed weights restores first-layer activation and action variation.
4. The old 2,724-input anchor cannot warm-start the corrected 2,660-input
   actor, and no contraction path is permitted.
5. The replacement validation starts from a new compatible L1 artifact and
   does not relabel the old P1LR result.

Do not implement the correction. Report any failed claim to Musashi as a
finding draft.

## 8. Acceptance

Your first assignment passes when all cited files have commit and digest,
runtime facts are fresh and source-labeled, unavailable evidence is explicit,
the root-cause drill does not rely on prose copied from Satoshi, and you have
mutated nothing except your handoff document.

