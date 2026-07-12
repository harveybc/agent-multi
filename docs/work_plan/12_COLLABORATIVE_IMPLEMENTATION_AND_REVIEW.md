# 12. Collaborative Implementation and Independent Review

## 1. Purpose

This document defines how Codex and a separately operated Claude coding agent
may collaborate without weakening architecture, reproducibility, security or
acceptance standards.

The collaboration is asymmetric by design:

- Codex is the technical lead, primary implementer, plan owner, integration
  owner and final reviewer.
- Claude is an optional implementation contributor for bounded work packages.
- The user may relay a task packet to Claude and paste its response back to
  Codex.
- A Claude completion report never proves that the implementation is correct.
- Only independently reproduced evidence can close a task or phase gate.

Claude is a capacity multiplier, not a second source of architectural truth.

## 2. Decision Rights

| Decision | Owner | Claude participation |
| --- | --- | --- |
| System and repository architecture | Codex | May identify risks or alternatives |
| Contract and schema semantics | Codex | May implement an approved schema |
| Fitness, risk and metric definitions | Codex | May implement frozen equations and tests |
| Leakage and walk-forward rules | Codex | May add tests under an explicit invariant set |
| DOIN protocol or trust semantics | Existing DOIN contracts plus Codex | No independent redesign |
| Phase sequencing and experiment priority | Codex | May estimate implementation effort |
| Bounded module implementation | Codex or Claude | Claude only with a task packet |
| Cross-repository integration | Codex | Claude may prepare isolated adapters |
| Acceptance and merge readiness | Codex | Claude report is supporting evidence only |

The user retains product authority and may change priorities or business
requirements. Codex translates those decisions into architecture and verified
implementation work.

## 3. Delegation Strategy

Codex should implement directly when a task:

- changes architecture, ownership boundaries or canonical contracts;
- affects trading accounting, future-information firewalls or test isolation;
- changes L1/L2/L3 objectives, early stopping, candidate selection or release;
- alters DOIN consensus, Proof of Optimization, identity, signatures, quorum,
  flooding, chain state or champion migration;
- spans several repositories with coupled behavioral changes;
- requires interpreting ambiguous research or business behavior;
- is on the critical path and review would cost as much as implementation.

Claude is a useful candidate when a task is:

- isolated behind an already approved interface;
- parallelizable while Codex works on a different dependency;
- mechanical but substantial, such as fixtures, translators or config schemas;
- testable with deterministic commands and explicit expected outcomes;
- limited to a small, named file set;
- valuable as an independent implementation or test-generation pass.

Examples of appropriate packets include:

- implement a legacy-to-canonical JSON translator against frozen examples;
- add schema fixtures and invalid-config tests;
- implement one thin `doin-plugins` adapter against existing base classes;
- add deterministic property tests to a settled multi-asset ledger contract;
- build one report/query module from an approved OLAP metric catalog;
- document an existing API from source without changing its behavior.

## 4. Required Task Packet

Codex must provide a self-contained specification. A packet is not ready for
delegation unless every required field below is concrete.

### 4.1 Header

```text
Task ID:
Title:
Objective:
Priority:
Owning phase:
Primary repository and branch:
Base commit:
Related repositories and commits:
```

### 4.2 Context and authority

The packet must state:

- why the task exists and which phase gate it supports;
- authoritative documents, contracts and code locations;
- which source wins if prose and implementation disagree;
- assumptions already decided by Codex;
- questions Claude may raise but must not decide independently.

### 4.3 Scope boundaries

The packet must list:

- files or directories allowed to change;
- files that may be read but not changed;
- explicitly forbidden repositories and modules;
- public APIs that must remain backward compatible;
- generated files and runtime artifacts that must not be committed;
- whether migrations or configuration translations are allowed.

An instruction such as "implement the transformer" is insufficient. The
packet must define inputs, outputs, shapes, masks, fitting cutoff, artifact
format, failure behavior and deterministic seed handling.

### 4.4 Behavioral requirements

Requirements must be observable. Depending on the task, they include:

- exact input/output contracts and schema versions;
- equations, units, aggregation period and missing-value behavior;
- chronological cutoffs and leakage prohibitions;
- determinism requirements and seed derivation;
- device selection and CPU fallback;
- compatibility and migration behavior;
- idempotency, retry and failure semantics;
- performance or memory ceilings;
- security and secret-handling constraints;
- logging, metrics, manifest and lineage requirements.

### 4.5 Required tests

The packet must specify:

- exact existing test commands to preserve;
- exact new unit, property, integration or regression cases;
- expected pass counts only when stable and meaningful;
- negative tests and expected error messages where relevant;
- smoke or dry-run commands using bounded resources;
- prohibited long-running training or production actions.

Tests must target behavior, not merely line coverage. A delegated task that
touches causal training requires at least one future-mutation or cutoff test.
A task that touches metrics requires reconstruction from atomic facts.

### 4.6 Deliverables and report format

Claude must return:

1. concise implementation summary;
2. complete changed-file list;
3. contract or behavior changes;
4. exact commands executed and their results;
5. tests added and the failure each test prevents;
6. unresolved assumptions, limitations and blockers;
7. migration or rollback notes;
8. confirmation that no unrelated files were changed;
9. current branch and commit status;
10. no claim of acceptance or production readiness.

Claude should not commit, push, deploy, start long jobs, change machine services
or modify secrets unless the packet explicitly authorizes that action.

## 5. Canonical Delegation Prompt Template

```text
You are implementing a bounded task in the Adaptive Multi-Asset Trading plan.
Do not redesign the architecture or expand scope.

TASK ID AND OBJECTIVE
<task identifier and measurable objective>

AUTHORITATIVE SOURCES
<documents, interfaces, code files and base commits>

APPROVED DESIGN
<settled behavior, equations, data flow and ownership>

ALLOWED CHANGES
<exact repositories and file globs>

FORBIDDEN CHANGES
<contracts, repositories, protocol behavior, generated data and services>

FUNCTIONAL REQUIREMENTS
<numbered observable requirements>

NON-FUNCTIONAL REQUIREMENTS
<determinism, leakage, performance, security and compatibility>

TEST REQUIREMENTS
<exact tests and commands>

ACCEPTANCE EVIDENCE TO REPORT
<changed files, command output summary, limitations and git status>

Stop and report a blocker instead of guessing when an authoritative source is
missing or contradictory. Do not claim final acceptance; Codex will inspect and
independently verify the contribution.
```

Each real prompt expands every placeholder. Codex keeps the final packet in a
versioned planning or handoff document when the task is material enough to
affect later reconstruction.

## 6. Independent Verification Protocol

When the user returns Claude's response, Codex performs the following review.

### 6.1 Establish provenance

- capture repository, branch, base commit and current commit;
- inspect `git status`, staged state and the complete diff;
- distinguish Claude changes from pre-existing user changes;
- reject unexplained generated files, logs, databases or artifacts.

### 6.2 Compare implementation to specification

- map each numbered requirement to code and tests;
- inspect all changed production code, not only Claude's summary;
- verify that no contract or ownership boundary changed implicitly;
- check config fields are actually consumed rather than merely documented;
- search for hard-coded paths, defaults, secrets and silent fallbacks.

### 6.3 Challenge correctness

Codex adds or runs adversarial checks appropriate to the component:

- future-data mutation and train-cutoff tests;
- deterministic replay across seeds and machines;
- malformed and unknown configuration fields;
- empty, partial, NaN and boundary-period data;
- asset-order permutation and clock alignment;
- accounting invariants, margin and cash conservation;
- metric reconstruction from weekly/order facts;
- crash recovery, idempotency and duplicate messages;
- CPU/GPU device selection and unavailable-device behavior;
- backward compatibility against frozen fixtures.

### 6.4 Reproduce evidence

Codex reruns the most relevant commands independently. Claude-reported output
is not copied as if it were observed evidence. For expensive tests, Codex runs
a deterministic bounded case and inspects artifacts before authorizing a wider
run.

### 6.5 Integrate and close

A delegated task is accepted only when:

- every requirement is implemented or explicitly deferred;
- independent checks pass;
- regressions and scope drift are absent;
- manifests, docs and configuration agree with runtime behavior;
- no protected test information entered optimization or selection;
- the affected phase gate has objective evidence;
- remaining limitations are recorded in the decision/evidence log.

Codex may repair the contribution directly, issue a corrective packet, or
discard it. A second Claude pass uses a new or revised task ID and includes the
specific findings from the failed review.

## 7. Review Depth by Risk

| Risk | Typical work | Minimum independent review |
| --- | --- | --- |
| Low | Documentation, fixtures, non-semantic generators | Diff inspection, format/link checks, targeted tests |
| Medium | Config loaders, adapters, OLAP queries, artifact utilities | Full module review, negative tests, integration smoke |
| High | Simulator, training, fitness, risk, allocation, causal features | Full diff, adversarial tests, frozen regression, bounded replay |
| Critical | DOIN trust/protocol, broker execution, live deployment, secrets | Codex implementation preferred; complete security and integration review |

Risk is determined by behavioral blast radius, not by line count.

## 8. Git and Workspace Rules

- Preserve unrelated user changes in dirty worktrees.
- Record the base commit before delegation.
- Prefer a dedicated branch or worktree for a substantial Claude packet.
- Never accept bulk formatting or dependency churn outside packet scope.
- Do not commit databases, checkpoints, raw logs, credentials or obsolete run
  outputs.
- Version schemas, examples, migrations, compact manifests and reproducible
  analysis code.
- Commit and push only after Codex review unless the user explicitly requests a
  different workflow.

## 9. Parallel Work and Dependency Control

Codex maintains a task ledger with these states:

```text
draft -> ready -> delegated -> returned -> reviewing -> accepted
                                              |
                                              +-> correction_required
                                              +-> rejected
```

Only `accepted` work may unblock a dependent phase. Parallel packets must not
edit the same ownership surface unless they use separate branches and Codex has
defined an integration order.

For each delegated task, the ledger records:

- task ID and phase;
- owner and reviewer;
- base commit and allowed paths;
- dependency and dependent tasks;
- date delegated and date returned;
- review evidence;
- accepted commit or rejection reason.

## 10. Initial Delegation Opportunities

The current plan may use Claude for these bounded tasks after their interfaces
are frozen:

1. canonical configuration JSON Schema fixtures and legacy translators;
2. per-machine unified `doin-node` config generator fixtures;
3. one isolated trading optimization plugin scaffold;
4. multi-asset ledger property-test expansion;
5. OLAP metric catalog SQL views and reconstruction tests;
6. artifact manifest validation and portability tests;
7. heuristic-strategy frozen-fixture adapters;
8. documentation generated from authoritative schemas.

Codex should retain direct ownership of the first vertical slice, canonical
metric equations, future-information firewall, portfolio accounting, optimizer
fitness, release gates and end-to-end DOIN/LTS integration.

## 11. Acceptance Checklist

- Codex remains the named owner and final reviewer.
- Every Claude task has a versioned, self-contained packet.
- Scope, forbidden changes and authoritative sources are explicit.
- The packet includes exact tests and observable acceptance criteria.
- Claude's response includes provenance, changed files, commands and limits.
- Codex inspects the actual diff and independently reruns critical tests.
- Claims are checked against runtime behavior and stored facts.
- Unexplained changes or unverifiable results are rejected.
- Only accepted commits can advance roadmap dependencies.
- Delegation decisions and outcomes remain reconstructible from the task ledger.
