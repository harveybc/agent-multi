# 24. Independent Audit and Continuous Improvement

Status: specified, baseline audit complete, academic audit scheduled
Date: 2026-07-31

## 1. Objective

Add an independent, evidence-driven audit function across three active runtime
fronts and one cross-cutting preservation front without creating a second
orchestrator or weakening ownership:

1. optimization and research: data, preprocessing, models, DOIN campaigns,
   artifacts and portfolio construction;
2. execution reality: simulation, LTS, Alpaca Paper, IBKR Paper, OANDA MT5,
   reconciliation and the social-trading boundary;
3. social intelligence and continuity: Hermes, Telegram, source collection,
   Moltbook, local-model cost routing, backups and recovery.
4. academic preservation and reproducibility: contribution boundaries,
   related work, claims, artifacts, disclosures and publication maintenance.

The audit function exists to find contradictions, defects, missing evidence,
security exposure, invalid assumptions and wasted work early. It is not a
ceremonial gate and does not require profitable results from screening
experiments. Findings must be actionable, reproducible and proportional to
their real impact.

## 2. Responsibility Split

| Role | Authority |
| --- | --- |
| User | Product, business, risk-tolerance and priority authority |
| Codex | Technical lead, architecture owner, primary implementer, runtime orchestrator, integration owner and final verifier |
| Claude ("Satoshi") | Independent read-mostly operational auditor and academic research lead; owns literature strategy, novelty review and paper architecture |
| Hermes agents | Machine-local telemetry, deterministic alert delivery and bounded analysis |
| Deterministic services | Scheduling, health checks, watchdogs, OLAP writes, broker reconciliation and fail-closed controls |

Claude does not become a second campaign supervisor. It may not independently
launch or stop DOIN jobs, select a champion, alter a chain, enable broker
orders, change credentials, deploy services, direct Hermes, commit, push or
merge. A separate bounded implementation packet under document 12 is required
before Claude changes production code.

Academic leadership does not grant runtime or submission authority. Satoshi
leads scholarly direction and drafts; Codex supplies and independently verifies
technical evidence; Harvey remains the human author and release authority.

Codex does not accept a Claude report as proof. Codex reproduces material
findings, decides the correction, implements or delegates it, and records
independent closure evidence.

## 3. Audit Dimensions

### 3.1 Business and product alignment

- The implemented workflow answers the current business objective.
- Paper accounts, personal accounts, signal distribution, copy trading,
  advisory structures and pooled customer capital are not conflated.
- Cost, time and compute are spent on decisions that affect the next usable
  artifact or deployment gate.
- Research gates do not imitate rigor while blocking useful screening.
- Claims shown to the user use consistent periods, units and evidence.

### 3.2 Data and machine learning

- Point-in-time availability, train-only fitting and protected-test isolation.
- Dataset, source, vintage, transformation and feature lineage.
- External paid-data coverage and explicit missing-source handling.
- Feature-selection, preprocessing and observation-window evidence.
- L1 convergence, patience, epoch budget, seed and device semantics.
- L2 genome coverage, inactive-gene handling and fitness sensitivity.
- Action collapse, insufficient activity and false flat-fitness detection.
- Train, validation and protected-test metrics are never silently mixed.
- Model weights, decoded parameters and resolved configuration reproduce the
  champion.

### 3.3 Trading and portfolio science

- Long and short capability matches the instrument and broker contract.
- Every risk-increasing order carries mandatory stop loss and take profit.
- Market, limit, stop and MIT behavior includes fills, expiry, missed
  opportunity, adverse selection and fallback costs.
- Commission, spread, slippage, financing and rollover are visible.
- Weekly return, annual return, weekly RAP, annual RAP, drawdown, activity and
  coverage are reconstructed from atomic facts.
- Portfolio optimization uses frozen per-asset artifacts and accounts for
  concentration, correlation, liquidity and marginal diversification.
- Rush and event conditioning is causal or probabilistic evidence, not a label
  invented after seeing protected outcomes.

### 3.4 Distributed systems and DOIN

- Existing DOIN contracts remain authoritative; trading extends them through
  plugins.
- All workers in one campaign share semantic domain, seed, dataset, config,
  genesis, population and required component revisions.
- Candidate ownership, leases, duplicate prevention, stage barriers and crash
  recovery behave as declared.
- Completion archives the exact champion before any successor starts.
- Successor startup cannot create parallel swarms or resurrect completed work.
- Chain and OLAP evidence remain decentralized; dashboards are projections,
  not the source of truth.
- Artifact references are content-addressed and sufficiently replicated before
  relying on them for inference.

### 3.5 Software architecture and quality

- Repository ownership follows document 01.
- Shared contracts remain dependency-light and versioned.
- JSON fields are consumed by runtime code, not merely documented.
- Defaults, migrations and backward compatibility are explicit.
- Tests exercise behavior, boundaries, failure paths and recovery.
- Generated artifacts, databases, logs and secrets are not committed.
- Dependencies, environments and component revisions are reproducible across
  machines.

### 3.6 Security, privacy and trust

- Secrets never enter Git, chat, reports, Telegram, OLAP or blockchain.
- Logs and audit evidence redact account and customer identity.
- External social content is hostile input and cannot invoke tools.
- MT5 bridge authentication, nonces, timestamps, firewall scope and demo-only
  mode fail closed.
- No chain-to-broker, social-to-broker or model-to-broker direct authority
  exists outside LTS risk and reconciliation.
- Package, model-weight and binary provenance is recorded.
- Credential rotation, revocation and least privilege are testable.

### 3.7 Operations and reliability

- Machine reachability, process identity, GPU use, GPU memory, temperature,
  RAM, swap, disk and OOM evidence are measured directly.
- Stale heartbeats are not reported as live work.
- Monitoring distinguishes CPU preprocessing from a stalled GPU candidate.
- Alerts are deduplicated but remain active until recovery is measured.
- A workstation loss does not corrupt campaign or broker-observation state.
- Backups, restore drills and artifact integrity hashes are verified.

### 3.8 Observability and OLAP

- Every important decision is recoverable from normalized facts and hashes.
- Dashboards label time basis, split, units and coverage.
- Reports identify the exact query, schema and source snapshot.
- Status answers include anomalies and missing evidence instead of hiding them.
- Retention removes redundant bulk without deleting unique lineage.

### 3.9 Academic contribution and reproducibility

- Each contribution is distinguished from prior art by a documented search.
- Research questions and claims are falsifiable.
- Every claim maps to an immutable artifact or a verified citation.
- References are opened and checked; no bibliographic field is invented.
- Baselines, ablations, uncertainty, negative results, limitations and threats
  to validity are explicit.
- Protected-test outcomes cannot choose the method, narrative or title claim.
- Manuscripts remain `outline` or `evidence_incomplete` until their paper gate
  passes.
- AI use, human responsibility, conflicts, licenses and availability are
  disclosed according to the target venue's current rules.
- Reproducibility packages do not redistribute paid or restricted raw data.

## 4. Evidence and Source Hierarchy

The auditor uses the same source hierarchy as the main plan:

1. immutable data, configuration, code and artifact hashes;
2. accepted DOIN candidate and verification records;
3. atomic and normalized OLAP facts;
4. resolved runtime configuration and manifests;
5. direct process, service, GPU, network and broker-observer evidence;
6. tests and bounded deterministic reproductions;
7. generated reports and plots;
8. prose.

When documentation conflicts with runtime, the auditor reports documentation
drift. It does not silently rewrite history or assume runtime is correct.

Every finding distinguishes:

- `observed`: directly reproduced from code, command, fact or artifact;
- `inferred`: conclusion supported by stated observations;
- `hypothesis`: plausible explanation requiring a named test.

## 5. Permission Model

### 5.1 Allowed by default

- Read repositories, work-plan documents, schemas, tests and bounded logs.
- Run non-destructive Git inspection.
- Run existing unit tests and bounded deterministic checks that do not train,
  deploy, trade, publish or consume material paid resources.
- Execute read-only machine inspection when access already exists:
  `hostname`, `uptime`, `free`, `df`, `nvidia-smi`, `ps`, bounded
  `journalctl`, `systemctl status`, `git status` and config/hash comparison.
- Query read-only SQLite/OLAP views.
- Produce an audit report under `docs/audits/` when requested.

### 5.2 Requires a Codex task packet

- Production-code, schema, configuration or test changes.
- A new dependency or package upgrade.
- Remote file changes or a dedicated audit worktree.
- New deterministic monitoring or an audit snapshot collector.
- Any test that launches training, simulation at material scale, a node,
  network traffic, a VM or a broker session.

### 5.3 Forbidden without explicit user and Codex authorization

- `sudo`, package installation, service restart, process termination or reboot.
- Campaign start/stop/advance, candidate release, chain repair or migration.
- Broker login, credential entry, order activation or account changes.
- Reading, printing, copying or rotating secrets.
- Sending Telegram messages, publishing social content or replying to DMs.
- Commit, push, merge, tag, release or deployment.
- Destructive cleanup, database mutation or artifact deletion.
- Access to private personal continuity material unrelated to the technical
  audit.

If a finding requires a forbidden action to verify, Claude records the exact
minimal request instead of attempting the action.

## 6. Continuous Improvement Cycle

```text
capture provenance
       |
collect changed evidence
       |
test invariants and contradictions
       |
issue evidence-backed findings
       |
user/Codex triage
       |
bounded correction
       |
independent reproduction
       |
close or reopen finding
       |
add regression or monitor
```

An audit does not end at criticism. Every accepted defect should produce one
or more of:

- a regression test;
- a deterministic invariant or preflight;
- an observable alert;
- a schema or lineage correction;
- a documented business decision;
- an explicit accepted risk with owner and review date.

## 7. Cadence

LLM audits are change-driven. Deterministic monitoring remains continuous.

| Cadence | Mechanism | Scope |
| --- | --- | --- |
| Every 5 minutes | Existing deterministic watchdogs | Worker, GPU, temperature, chain, broker heartbeat and exposure anomalies |
| Hourly | Deterministic summary, Telegram only on state change or active risk | Fleet and execution health; no full LLM review |
| Event-driven | Claude audit requested after a material event | Campaign transition, champion archive, incident, contract/fitness/risk change, broker activation or security alert |
| Every 24 hours while active work changes | Delta audit | Changed commits/configs, open high findings and one rotating front |
| Every 72 hours | Runtime-front coverage requirement | Each of the three runtime fronts receives at least one focused audit |
| Weekly | Full cross-front review | Architecture, ML/data, distributed behavior, execution, security, OLAP, cost and backlog |
| Monthly | Recovery and supply-chain review | Restore drill, secret inventory/expiry, dependency provenance, storage, cost and governance |
| Paper outline, evidence freeze and pre-submission | Academic review | Novelty search, claim ledger, baselines, ablations, reproducibility, licensing and disclosure |

Claude cannot remain alive between chat turns and must not claim continuous
monitoring. Cron/systemd and Hermes may collect deterministic snapshots. Claude
reviews those snapshots when invoked.

## 8. Severity and Finding Contract

| Severity | Meaning | Expected response |
| --- | --- | --- |
| `S0` | Active safety, credential, live-financial, chain-corruption or unrecoverable-data risk | Stop affected boundary and notify immediately |
| `S1` | High probability of invalid results, duplicated/parallel work, missing champion or major outage | Correct before dependent work continues |
| `S2` | Material defect, observability gap or avoidable cost with bounded impact | Schedule in current work cycle |
| `S3` | Localized weakness, maintainability debt or incomplete evidence | Prioritize against other work |
| `S4` | Improvement opportunity | Backlog with expected value |

Each finding has:

- stable ID: `AUD-<FRONT>-YYYYMMDD-NNN`;
- severity, confidence and status;
- affected front, repository, commit/config/artifact and environment;
- observed evidence with file/line, query or bounded command;
- business and technical impact;
- minimal reproduction;
- proposed correction and regression evidence;
- owner and dependency;
- closure evidence recorded by someone other than the original reporter.

The report leads with findings ordered by severity. It does not bury defects
under an executive summary or praise.

## 9. Hermes Boundary

Hermes is not supervised through free-form agent commands. It remains a local
telemetry and delivery layer.

Recommended progression:

1. Keep existing production watchdogs deterministic.
2. Add a read-only audit snapshot generator only through a reviewed task
   packet.
3. Let Hermes deliver hashes and concise state changes to Telegram.
4. Let Claude review exported snapshots, code and OLAP evidence when invoked.
5. Never let Claude or social content issue commands to Hermes that affect
   campaigns, brokers, services or secrets.

A future dedicated Hermes audit agent may summarize already-redacted evidence.
It must have no broker, publication, campaign-control or unrestricted shell
capability and must use an isolated identity and budget.

### 9.1 Model-cost delegation boundary

The independent auditor retains all judgment involving severity, closure,
security, consensus, novelty, causal/statistical interpretation, architecture
or final recommendations. This paragraph grants no delegation or tool
authority. An auditor may propose a bounded mechanical task, but only Harvey
or the technical lead may authorize it through a reviewed task packet.
Lower-cost Hermes/OpenCode models may then perform only that packet's
mechanical work, such as hashes, file inventories, schema checks,
deterministic command execution, formatting, deduplication and frozen-regex
event extraction.

Every delegated task records its exact prompt, model/provider, token ceiling,
inputs, output hash when practical, verification and whether it changed a
decision. The independent auditor verifies every delegated result before use.
A delegated task class is retired after two consecutive runs with no
decision-changing output. Token reservations and actual billed cost are
reported separately.

Invocation prose can never create capabilities, change permissions or
authorize delegation. Capability changes require a versioned task packet and
Harvey's approval when they alter the standing authority model.

## 10. Technical-Lead Recovery

The durable recovery prompt is:

```text
docs/handoffs/CODEX_TECHNICAL_LEAD_RECOVERY_PROMPT_2026_07_30.md
```

It rebuilds the Codex technical-lead role after chat loss by reading versioned
sources and re-verifying runtime. It contains no secrets and no private
personal context.

Claude audits this recovery prompt:

- after a new work-plan document or active front is added;
- after repository ownership or decision rights change;
- after orchestration, status or safety contracts change;
- weekly while the project is active.

Claude may propose a replacement recovery prompt in its report. It may not
silently change role authority. Codex reviews and versions accepted changes.
The previous prompt remains in Git history.

## 11. Audit Output and Storage

Versioned audit reports live in:

```text
docs/audits/
```

Reports contain compact evidence and links, not raw databases, credentials,
private messages or large logs. Machine-generated snapshots stay outside Git;
reports record their hash and local retention location.

The report format and lifecycle are defined in `docs/audits/README.md`.

## 12. Initial Audit Sequence

### A0: context and provenance

- Read the index, architecture, implementation ledger, all three runtime-front
  documents and the academic-preservation contract.
- Capture branch, commit, dirty state and remote tracking for active repos.
- Produce a repository and responsibility map.
- Identify stale or contradictory status statements.

### A1: optimization and research

- Audit protected-entry v2 contracts, current campaign identity and artifacts.
- Reconstruct fitness and canonical metrics from atomic evidence.
- Inspect genome coverage, inactive parameters, L1 convergence and leakage.

### A2: execution reality

- Audit LTS global ledger and venue boundaries.
- Verify Alpaca/IBKR observers and MT5 bridge remain paper/demo and fail closed.
- Compare simulation cost, order and protection behavior with observed broker
  capabilities.

### A3: social intelligence and continuity

- Audit the source/prompt trust boundary, paid-model budget and publication
  controls before activation.
- Audit backup, restore, human-maintainer and credential-revocation design.

### A4: integrated risk review

- Trace one champion from data/config through DOIN, artifact loading,
  portfolio intent, LTS planning and paper execution.
- Verify no layer silently changes units, asset identity, risk or authority.

### A5: automation proposal

- From observed recurring gaps, propose the minimum deterministic monitors.
- Do not implement them without a task packet.

### A6: academic preservation

- Audit the five-paper program in document 25.
- Separate engineering, integration and scientific novelty.
- Build a verified related-work ledger and claim/evidence matrix.
- Reject submission readiness when decisive evidence, controls, licensing or
  disclosure is missing.

## 13. Acceptance Criteria

- Claude can reconstruct the three runtime fronts and academic-preservation
  contract without relying on chat memory.
- Findings cite reproducible evidence and distinguish facts from hypotheses.
- No auditor action can alter a campaign, broker, service, secret or published
  content by default.
- Codex independently verifies every accepted `S0` through `S2` finding.
- Each front is audited at least every 72 hours during active development.
- The Codex recovery prompt passes a weekly context-loss tabletop review.
- Audit overhead does not materially reduce DOIN throughput or delay broker
  observation.
- Findings produce regressions, monitors, decisions or explicit accepted risk,
  not an endlessly growing criticism document.
- No academic claim or submission state can advance without independent
  citation/evidence review and human approval.
