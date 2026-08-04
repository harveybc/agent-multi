# Musashi to Satoshi III: Runtime Audit and OKF/GBrain Pilot

Date: 2026-08-03 America/Bogota
From: General Musashi, temporary independent auditor and architecture reviewer
To: Satoshi III (Mujuro Utsutsu), temporary technical lead
Owner decision: OKF/GBrain/Hermes bounded pilot approved

This is one ordered work packet. Live Paper/Demo truth comes first. Knowledge
tooling may proceed only in a bounded CPU/local lane and must not delay or
interfere with broker reconciliation, the DOIN campaign or machine health.

## 1. Required Reading

Read these files directly before acting:

1. [three-venue runtime packet](/home/harveybc/Documents/GitHub/agent-multi/docs/handoffs/MUSASHI_THREE_VENUE_WRITABLE_RUNTIME_PACKET_2026_08_03.md)
2. [findings register](/home/harveybc/Documents/GitHub/agent-multi/docs/audits/work_plan/04_OPEN_FINDINGS_REGISTER.md)
3. [implementation ledger](/home/harveybc/Documents/GitHub/agent-multi/docs/work_plan/13_IMPLEMENTATION_STATUS_AND_TASK_LEDGER.md)
4. [continuous demo operations](/home/harveybc/Documents/GitHub/agent-multi/docs/work_plan/29_CONTINUOUS_DEMO_TRADING_OPERATIONS.md)
5. [knowledge continuity contract](/home/harveybc/Documents/GitHub/agent-multi/docs/work_plan/31_OKF_GBRAIN_HERMES_KNOWLEDGE_CONTINUITY.md)
6. [codebase-memory operating spec](/home/harveybc/Documents/GitHub/agent-multi/docs/handoffs/CODEBASE_MEMORY_MCP_OPERATING_SPEC_2026_08_03.md)
7. [audit and continuous-improvement contract](/home/harveybc/Documents/GitHub/agent-multi/docs/work_plan/24_INDEPENDENT_AUDIT_AND_CONTINUOUS_IMPROVEMENT.md)
8. [social intelligence and continuity](/home/harveybc/Documents/GitHub/agent-multi/docs/work_plan/23_SOCIAL_INTELLIGENCE_AND_OPERATIONAL_CONTINUITY.md)

External specifications, to be pinned by exact revision in your report:

- <https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md>
- <https://github.com/garrytan/gbrain>
- <https://github.com/NousResearch/hermes-agent/issues/23997>

Use `codebase-memory-mcp` for code discovery under its existing operating
spec. Use direct file reads for Markdown, configs, Git history and runtime
evidence. Never use absence from either graph as proof.

## 2. Priority Zero: Independent Three-Venue Audit

The baseline at 2026-08-04T00:24:24Z reported protected, selected-model
Paper/Demo operation on Alpaca, IBKR and OANDA MT5 with zero consolidated
alerts. It is historical evidence, not a promise about the moment you read
this.

Without placing, cancelling, replacing or altering any order:

1. independently reproduce corrections 079-085 from their parent revisions
   or exact fixtures;
2. run the focused suites and complete LTS suite at the current clean head;
3. resample direct venue facts and account-bound runner heartbeats;
4. verify every open position or order has current model, instrument, account,
   environment, quantity and native SL/TP reconciliation;
5. verify restart/replay cannot duplicate MT5, IBKR or Alpaca effects;
6. test stale, wrong-account, wrong-instrument, missing-protection and count
   mismatch paths without touching brokers;
7. distinguish queued, accepted, partially filled, filled, protected, closed,
   unavailable and stale states precisely; and
8. report each finding separately. Do not close work implemented by Musashi.

Also reconcile register state:

- 069-074 are independently verified and await owner closure;
- 075-078 were independently verified by your previous packet and await owner
  closure;
- 079-085 await this independent verification.

Prepare one compact owner-disposition table, but do not grant closure or infer
owner authorization.

## 3. Priority One: K0 Collision and Security Review

Act as a senior knowledge-systems architect, distributed-systems engineer,
security reviewer and machine-learning infrastructure engineer. Be concrete;
do not repeat product marketing.

1. Pin the exact revisions and licenses of OKF v0.2, GBrain and the installed
   Hermes version.
2. Inspect GBrain's install and dependency paths before installing anything.
   Do not execute its broad remote `INSTALL_FOR_AGENTS.md` flow because it may
   add skills, credentials, autonomous cycles or services outside this packet.
3. Determine whether GBrain preserves/query-maps OKF v0.2 frontmatter. If this
   is not directly supported, specify the smallest deterministic adapter and
   test it. Do not claim compatibility from generic Markdown import.
4. Produce a collision matrix against Git/work plan, `codebase-memory-mcp`,
   OLAP, DOIN blockchain, runtime evidence and Hermes memory.
5. Threat-model memory poisoning, stale synthesis, secret ingestion,
   unauthorized canonical writes, exposed MCP endpoints, dependency supply
   chain, cron tool disappearance and single-index failure.
6. Return a promote/revise/reject decision for K1-K2. If a material security
   issue exists, stop before installation and report it.

## 4. Priority Two: K1-K4 Bounded Pilot

If K0 does not expose a blocker, implement the owner-approved pilot exactly as
document 31 specifies.

### K1: OKF bundle

- Create `knowledge/okf/` in `agent-multi` using OKF v0.2.
- Start small: repository map, active-front map, authority boundaries,
  current campaign/artifact handoff, current Paper/Demo roles, findings state,
  recovery runbooks and metric definitions.
- Cite canonical local sources. Include status, producer, verification and
  freshness semantics.
- Add deterministic validation for schema/frontmatter, links, duplicate IDs,
  prohibited secret/account patterns and missing sources.
- Never import raw chat, social posts, broker payloads, OLAP tables, secrets,
  weights, logs or databases.

### K2: Omega-only GBrain

- Install from a reviewed, pinned revision on Omega only.
- Use local PGLite or the smallest equivalent local engine; stdio/loopback
  only, no public listener.
- Import only the approved OKF bundle during the first run.
- Disable dream cycle, autonomous enrichment, remote mounts, credential
  gateway and all write/admin access exposed to Hermes.
- Keep databases and caches outside Git. Record source commit, index manifest,
  resource use and rebuild instructions.

### K3: cold-start benchmark

- Materialize a fixed, versioned recovery-question corpus covering all fronts,
  authority, open findings, artifacts, current runtime semantics and failure
  recovery.
- Compare canonical-file-only recovery with GBrain-assisted recovery.
- Score correctness, exact source citation, freshness, unsupported claims,
  latency and token use. All safety-critical answers must pass.
- Include adversarial stale and contradictory concepts; the correct behavior
  is to report the conflict, not blend it away.

### K4: cron and loss drill

- Prove both interactive Hermes and a real cron job can call the intended MCP
  tools. Inspect the actual cron run record/tool trace; `hermes mcp list` is
  insufficient.
- Check whether the installed Hermes revision requires the `mcp-gbrain`
  toolset alias described in issue 23997.
- Stop GBrain and prove explicit fallback to canonical files.
- Delete the derived index and rebuild it from Git.
- Measure RAM, swap, CPU, disk, wall time and any effect on Omega's DOIN and
  selected-model services. Stop on memory pressure or operational impact.

Do not deploy GBrain to Dragon or Gamma. K5 requires a later owner decision
based on this evidence.

## 5. Absolute Boundaries

- No Live broker or real-capital action.
- No broker order mutation during this audit/pilot.
- No GBrain, Hermes or LLM authority over orders, risk, model selection,
  campaign control, publication or finding closure.
- No credentials, bridge secrets, account IDs or private material in Git,
  OKF, prompts, reports or GBrain.
- No GBrain dependency in LTS, DOIN, watchdog, recovery or model inference.
- No replacement of SQLite/OLAP, blockchain, direct broker evidence,
  `codebase-memory-mcp` or canonical JSON contracts.
- No opaque installer, unpinned dependency or autonomous write cycle.
- Do not self-close findings or describe a configuration listing as runtime
  proof.

## 6. Deliverables

Return one evidence packet containing:

1. runtime audit dispositions for 079-085 and an owner-closure table for
   eligible 069-085 items;
2. exact repo commits, tests, timestamps and direct read-only venue evidence;
3. K0 dependency, license, collision and threat-model results;
4. K1 bundle manifest and validator results;
5. K2 installation/index manifest with no secrets;
6. K3 benchmark inputs and results;
7. K4 interactive, cron, failure-recovery and resource evidence;
8. explicit unknowns and new findings; and
9. a promote/revise/reject recommendation for fleet replication.

Use clickable absolute repository links in the report. Keep raw generated
indexes, databases and large logs out of Git. Commit and push only reviewed
source, tests, compact evidence and documentation. Do not let this knowledge
pilot interrupt continuous protected Paper/Demo operation or the active DOIN
campaign.
