# Codebase Memory MCP Operating Specification

Date: 2026-08-03 America/Bogota
Version: 1.0.0
Owner: Harvey (Gran Loto Blanco)
Prepared by: General Musashi
Applies to: Satoshi technical-lead sessions and General Musashi/Codex sessions
Authority class: development-navigation procedure, not runtime or trading authority

## 1. Purpose

Use `codebase-memory-mcp` as a local, derived knowledge graph for faster code
discovery, call tracing and impact analysis across the project repositories.
It is an accelerator for reading current source code. It is not project memory,
runtime state, evidence of correctness or a source of trading authority.

The authoritative sources remain, in order appropriate to the question:

- current Git commits and source files;
- executable tests and independently reproduced runtime evidence;
- canonical JSON contracts and configuration files;
- the versioned work plan, handoffs and findings register;
- DOIN blockchain/OLAP facts for optimization lineage; and
- direct broker/demo evidence for venue state.

No graph result may override any of those sources.

## 2. Safe Deployment Boundary

The graph is a development-only cache on Omega. It must never:

- run as part of DOIN consensus, optimization or candidate evaluation;
- become an LTS order authority or an input to an order decision;
- be required for broker recovery, reconciliation or risk controls;
- be treated as a decentralized source of truth;
- contain credentials, private keys, raw account identifiers or tokens;
- be committed to Git; or
- be copied to Dragon or Gamma merely for symmetry.

Use one-shot indexing. Keep automatic watchers and persisted repository
artifacts disabled unless the owner later approves them after resource tests.

## 3. Canonical Project Names

The following projects were indexed in `moderate` mode on 2026-08-03:

| Project name | Repository | Primary role | Nodes | Edges |
| --- | --- | --- | ---: | ---: |
| `agent-multi` | `agent-multi` | agents, pipelines, optimizers, campaign supervisor | 1,789 | 7,886 |
| `lts` | `lts` | live/demo execution, venue adapters, journals and controls | 2,539 | 11,937 |
| `trading-contracts` | `trading-contracts` | canonical cross-component execution contracts | 1,006 | 2,382 |
| `doin-node` | `doin-node` | decentralized node and collaborative optimization runtime | 1,315 | 6,904 |
| `gym-fx` | `gym-fx` | training/simulation environment | 440 | 2,294 |
| `predictor` | `predictor` | legacy/current predictor plugin reference | 1,406 | 7,433 |
| `prediction-provider` | `prediction_provider` | inference-provider integration reference | 2,106 | 7,781 |
| `doin-core` | `doin-core` | DOIN shared core | 862 | 3,964 |
| `doin-plugins` | `doin-plugins` | DOIN plugin interfaces and implementations | 321 | 1,464 |
| `heuristic-strategy` | `heuristic-strategy` | prior strategy/backtesting integration reference | 1,756 | 5,455 |

Use these exact project names in graph calls. Do not index
`/home/harveybc/Documents/GitHub` as one giant repository.

## 4. Measured Limitations

The first acceptance run established these facts:

1. The graph correctly found the live execution path
   `L1OutboxConsumer._consume_entry -> BracketExecutor.submit_bracket`, then
   traced capability consumption, durable effects, bracket translation and
   broker submission calls.
2. `agent-multi` architecture correctly identified the campaign supervisor,
   RL pipelines, adaptive order router and optimization components.
3. The default index excluded `docs/`, `examples/`, `scripts/`, `tools/`,
   generated runs, artifacts and several test directories. Therefore use
   direct file tools for work plans, handoffs, JSON, shell scripts and other
   non-indexed content.
4. Automatic cross-repository intelligence found zero edges across the ten
   projects. Our important boundaries are mostly Python package installation,
   plugin entry points, JSON configuration, subprocesses and broker/network
   contracts. Reconstruct those boundaries from canonical contracts and
   configuration; never claim the graph proved their absence.
5. Broad natural-language searches can return hundreds of matches. Inspect
   `total` and `has_more`, narrow by project/label/file pattern and paginate
   rather than accepting the first page as exhaustive.

## 5. Required Discovery Workflow

For code questions, use this sequence:

1. `get_architecture` for an unfamiliar repository or subsystem.
2. `search_graph` with a narrow natural-language query or name pattern.
3. Inspect `total`, `has_more`, project, qualified name and file path.
4. `trace_path` for callers, callees, data flow or impact.
5. `get_code_snippet` using the exact qualified name returned by the graph.
6. Read the current source file directly around the relevant lines.
7. Inspect canonical configs/contracts when behavior is configuration-driven.
8. Run focused tests or a read-only reproducer before reaching a conclusion.

Use `search_code`, `rg` or direct file reads when searching:

- string literals and error messages;
- Markdown, JSON, YAML, shell scripts and systemd units;
- dynamic plugin names or entry points absent from the graph;
- Git history and commit identifiers; or
- generated evidence and runtime logs.

## 6. High-Risk Claim Rule

For trading, security, distributed consensus, artifact lineage, broker state or
finding closure, every conclusion must report:

- project and current Git commit;
- graph symbol/path used for discovery;
- direct source/config evidence used for confirmation;
- focused test or runtime evidence, when applicable;
- index limitations or unresolved dynamic boundaries; and
- whether the statement is observed, reproduced, inferred or proposed.

Absence from the graph is never proof that code, a caller, a route, a plugin or
a safety control does not exist.

## 7. Refresh Protocol

Do not reindex continuously.

Reindex a project in `moderate` mode when:

- a code-changing commit lands in that repository and the next task depends on
  the changed call graph;
- a branch/worktree is switched;
- graph source does not match current source; or
- the graph returns missing symbols that direct source inspection proves exist.

After reindexing, run one known-symbol query and one trace before trusting the
new graph. Record the project name and Git head in any consequential report.
Do not enable `persistence` or commit `.codebase-memory` artifacts.

## 8. Resource Discipline on Omega

Omega has a history of OOM interruption. Therefore:

- index repositories sequentially, never concurrently;
- check available RAM and swap before a full refresh;
- use `moderate` mode by default;
- do not index experiment results, models, checkpoints, databases or logs;
- stop indexing if available memory falls below a conservative operating
  margin or swap grows unexpectedly; and
- never allow indexing to restart or mutate a running DOIN campaign.

## 9. Satoshi-Specific Rules

Satoshi must use the graph proactively for code discovery, especially after a
cold start, but must not spend the session rebuilding knowledge already stored
in versioned handoffs.

At session start:

1. read the current cold-start/resumption packet and work-plan status directly;
2. verify the needed graph projects with `get_architecture`;
3. index only missing or demonstrably stale projects;
4. use the graph to reconstruct the code paths relevant to the active task;
5. compare those paths with the canonical contracts and current findings; and
6. begin implementation rather than remaining in exploratory narration.

Satoshi must not create or modify ADRs through the MCP. Architectural decisions
remain ordinary reviewed Markdown and Git commits. Satoshi must not claim that
graph indexing preserves conversation history or owner decisions.

## 10. Musashi-Specific Rules

General Musashi uses the same graph for independent reproduction, impact
analysis and finding audits. Musashi must independently inspect current source
and tests rather than accepting Satoshi's graph queries or conclusions.

The graph is especially useful for finding omitted callers and sibling
implementations, but a finding must always cite stable source/test evidence,
not a mutable local cache.

## 11. Acceptance Criteria

The integration remains accepted while all of the following hold:

- all canonical projects answer architecture and symbol queries;
- a known critical path can be traced and confirmed in current source;
- no credentials or large generated artifacts enter the graph cache;
- no graph artifact enters Git;
- indexing does not cause OOM or interfere with active computation;
- both agents explicitly distinguish graph discovery from evidence; and
- work continues correctly when the MCP is unavailable.

Failure of the MCP is a tooling degradation, never a trading, consensus or
recovery failure.
