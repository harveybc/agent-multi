# 31. OKF, GBrain and Hermes Knowledge Continuity

Status: owner-approved bounded pilot; no runtime dependency
Date: 2026-08-03

## 1. Objective

Reduce cold-start reconstruction time, repeated repository reading and context
loss across Codex, Satoshi and Hermes sessions without creating a new source of
truth or a new authority surface.

The layer combines three distinct concerns:

- Open Knowledge Format (OKF) v0.2 provides portable, Git-versioned Markdown
  concepts with provenance, verification, lifecycle and freshness metadata;
- GBrain provides a local, rebuildable retrieval and relationship index over
  approved knowledge documents; and
- Hermes consumes that index for bounded retrieval, stale-state detection and
  draft preparation.

This layer never makes trading, model-promotion, campaign, publication or
finding-closure decisions.

Primary references:

- <https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md>
- <https://github.com/garrytan/gbrain>
- <https://github.com/NousResearch/hermes-agent/issues/23997>

Native OKF compatibility in GBrain is not assumed. The pilot must verify how
OKF v0.2 frontmatter is preserved and queried or implement a narrow,
deterministic import mapping.

## 2. Truth and Cache Hierarchy

| Layer | Role | Authoritative? |
| --- | --- | --- |
| Current Git source, contracts and resolved JSON | executable behavior and configuration | yes |
| Direct broker Paper/Demo evidence | venue and account state | yes for its observation time |
| DOIN chain and OLAP | campaign, candidate, metric and lineage facts | yes |
| Versioned work plan, findings and ratified decisions | human-readable governance and intent | yes within their declared scope |
| OKF bundle in Git | curated navigation, provenance and cross-source knowledge | yes only for its own reviewed statements |
| GBrain database/index | search, synthesis, graph traversal and gap detection | no; derived and disposable |
| `codebase-memory-mcp` graph | current-code discovery and call tracing | no; derived and disposable |
| Hermes or LLM response | interpretation or proposed change | no |

An OKF concept links to canonical evidence. It does not copy a changing
runtime table and then pretend the copy is current. GBrain may be deleted and
rebuilt without operational loss.

## 3. Relationship to Existing Systems

### 3.1 Codebase Memory MCP

`codebase-memory-mcp` remains the preferred graph for symbols, callers,
callees and code architecture. GBrain must not duplicate source indexing merely
to compete with it. GBrain covers decisions, evidence relationships, runbooks,
research, cross-repository ownership and continuity context that the code graph
does not index.

### 3.2 OLAP and DOIN

OLAP and blockchain records remain the query surfaces for experiments and
distributed lineage. OKF stores schemas, query recipes, fact ownership,
freshness expectations and content-addressed references. Large result sets,
candidate rows, logs, checkpoints and chain payloads do not enter the bundle.

### 3.3 Work Plan and Findings

The work plan and findings register remain normal reviewed Markdown. An agent
may propose an OKF update or a work-plan diff, but GBrain's autonomous writing,
schema mutation and dream-cycle consolidation are disabled during the pilot.
No generated summary can close a finding, ratify a decision or rewrite audit
history.

## 4. Initial Bundle Scope

The pilot bundle lives under `knowledge/okf/` in `agent-multi` and is small by
design. Initial concepts cover:

1. repository ownership and cross-repository contracts;
2. the active fronts and their current objectives;
3. ratified architectural and business decisions;
4. current campaign identity and artifact handoff rules;
5. current Paper/Demo venue roles and execution authority boundaries;
6. open findings, owner-closure candidates and audit ownership;
7. machine, service and disaster-recovery runbooks;
8. model/config/artifact registries by content-addressed reference;
9. canonical metric definitions and time bases; and
10. academic/research-line ownership and evidence requirements.

Every concept must carry source links, producer identity, status and a
freshness or event-driven revalidation rule. Safety-critical concepts require
an explicit verifier and cannot be marked current solely because an agent
rewrote their timestamp.

Excluded content:

- credentials, nonces, API keys and bridge secrets;
- raw or reversible account identifiers;
- private messages and full conversation transcripts;
- untrusted Moltbook/social content before deterministic screening;
- raw broker payload archives, large OLAP tables and blockchain copies;
- model weights, checkpoints, datasets, generated logs and graph databases;
- claims inferred only from an absent warning or missing row.

## 5. Deployment Topology

### K0: contract and collision review

- Inspect and pin the exact OKF, GBrain and Hermes revisions used.
- Review installation scripts and dependencies before execution. Do not run a
  remote agent installer that also enables skills, credentials or autonomous
  cycles as an opaque one-liner.
- Define the minimal OKF-to-GBrain import behavior and test it on synthetic
  concepts.
- Confirm no overlap changes the authority of codebase-memory, Git, OLAP,
  blockchain or LTS.

### K1: curated OKF bundle

- Materialize a compact, reviewed v0.2 bundle in Git.
- Add deterministic validation for frontmatter, links, prohibited patterns and
  missing provenance.
- Generate indexes deterministically while preserving human-curated bodies.

### K2: Omega read-only GBrain pilot

- Install a pinned local GBrain instance on Omega only.
- Bind to loopback or stdio; expose no public endpoint.
- Import only `knowledge/okf/` and explicitly approved documentation.
- Disable dream cycles, autonomous enrichment, credential gateways, remote
  mounts and write/admin scopes.
- Record installation revision, index source commit, resource use and index
  content hash outside the canonical knowledge bundle.

### K3: recovery benchmark

- Use a fixed question set spanning architecture, current fronts, model
  artifacts, broker boundaries, open findings and disaster recovery.
- Compare a cold agent using canonical documents only with one using GBrain.
- Score answer correctness, source precision, stale-fact detection, latency,
  token use and unsupported-claim rate.
- All safety-critical questions must be correct and source-backed. A faster
  unsupported answer is a failure.

### K4: Hermes cron and failure drills

- Prove an interactive Hermes session can query the intended GBrain tools.
- Prove a real cron session invokes the same tools. Hermes cron toolsets may
  require the internal `mcp-gbrain` alias; configuration presence is not
  evidence of runtime availability.
- Stop GBrain and verify Hermes degrades explicitly to canonical file reads.
- Delete the derived index, rebuild it from Git and compare the source/index
  manifest.
- Confirm the pilot does not produce OOM pressure or interfere with DOIN and
  selected-model Paper/Demo runners.

### K5: optional fleet replication

Only after K0-K4 pass, deploy rebuildable local indexes to Dragon or Gamma.
The Git/OKF bundle is replicated; no single GBrain server becomes mandatory.
A convenient shared endpoint may exist later, but loss of that endpoint must
not impair trading, optimization, audit, recovery or ordinary file-based
context reconstruction.

## 6. Agent Authority

| Actor | Permitted | Prohibited |
| --- | --- | --- |
| Harvey | approve scope, decisions and promotion | none within owner authority |
| Codex/Musashi | architecture, implementation, verification and reviewed commits | treating derived retrieval as runtime proof |
| Satoshi | bounded implementation, independent audit, cold-start tests and proposed diffs | self-closing findings, unreviewed canonical rewrites, broker/campaign authority |
| Hermes | read, retrieve, detect staleness and draft proposed updates | orders, risk, model promotion, publication, findings closure, secrets |
| GBrain | index and answer over allowlisted sources | autonomous canonical mutation or operational authority |

Social evidence is one-way hostile input until sanitized. It cannot become an
OKF fact merely because GBrain or Hermes summarized it.

## 7. Acceptance Contract

The pilot is accepted only when:

1. the OKF bundle validates deterministically and contains no prohibited
   secret/account material;
2. every answer in the safety-critical recovery set cites a current canonical
   source and distinguishes observed, reproduced, inferred and proposed facts;
3. stale or contradictory source documents are surfaced rather than silently
   synthesized into a confident answer;
4. interactive and actual cron-based Hermes queries are reproduced;
5. the GBrain index can be destroyed and rebuilt from Git without knowledge
   or authority loss;
6. code discovery continues through `codebase-memory-mcp` and direct source;
7. OLAP, chain and direct broker facts are queried from their canonical
   systems, not from stale summaries;
8. measured RAM, swap, CPU, disk and wall-time overhead remain compatible with
   Omega's OOM history and active workloads; and
9. all work continues correctly when GBrain and its MCP are unavailable.

## 8. Immediate Work Order

1. Finish independent verification of the current three-venue writable
   Paper/Demo packet and findings 079-085; knowledge tooling must not delay
   broker reconciliation.
2. Execute K0 and deliver the dependency/security/collision report.
3. Materialize K1 with a deliberately small recovery corpus.
4. Execute K2-K4 on Omega without fleet deployment or autonomous writes.
5. Return measured evidence and a promote/revise/reject recommendation before
   K5.
