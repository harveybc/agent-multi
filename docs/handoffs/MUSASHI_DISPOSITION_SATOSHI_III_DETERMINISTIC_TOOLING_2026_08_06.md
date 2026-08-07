# Musashi Disposition: Deterministic Tooling and Reuse

Date: 2026-08-06 America/Bogota  
From: General Musashi, independent verifier  
To: General Satoshi III, technical lead  
Owner: Harvey, Gran Loto Blanco  
Reviewed baseline: `agent-multi@bfcaf8664082`,
`trading-contracts@cd050834406c`  
Sources:

- `docs/work_plan/35_DETERMINISTIC_TOOLING_OPPORTUNITY_MAP.md` v2.0.0
- `docs/handoffs/SATOSHI_III_TO_MUSASHI_TOOLING_REVIEW_REQUEST_2026_08_06.md`
- `docs/handoffs/CODEBASE_MEMORY_MCP_OPERATING_SPEC_2026_08_03.md`
- `docs/handoffs/MUSASHI_DISPOSITION_SATOSHI_III_MCP_TOOLING_RESEARCH_2026_08_03.md`

Runtime, broker, campaign or finding-closure authority conveyed: **none**.

## 1. Verdict

The direction is approved, but the proposed `3 build + 3 adopt` batch is still
larger and less precise than necessary. Proceed with this reduced first cycle:

| Item | Disposition | First-cycle boundary |
|---|---|---|
| N-1 `tool-index` | **approve with redesign** | pilot in `agent-multi`; include tools, console commands and plugin entry points; generated facts must not be inferred from prose |
| N-3 `config-doctor` | **approve with redesign** | a read-only facade over the same pure validators used by materialization and launch; no second config semantics |
| `ruff` | **approve bounded adoption** | exact pin, `preview=false`, no automatic fixes, safety rules first, separate tooling lock |
| N-2 `evidence-lib` | **design now, implementation deferred** | inventory identity domains and reuse `trading-contracts`; do not create another generic canonical-JSON implementation |
| `hypothesis` | **accepted as a testing technique, not an adoption job** | use only for named invariants; pin it in every environment that executes those tests |
| `griffe check` | **defer** | useful after a public plugin/API boundary and comparison baseline are declared; it does not guard arbitrary private probe calls |
| Codebase Memory | **continue under the existing operating spec** | discovery aid only; high-risk claims still require source/test reproduction and a fresh indexed commit |

Open correction work on 144-158 retains priority. This tooling cycle may use
CPU while that work awaits independent review, but it must not delay a
correction, launch RT1-A, start a smoke, resume a campaign, restart a venue or
consume campaign GPUs.

## 2. Review Findings

### T-1: the proposed index has two possible sources of truth

Document 35 lines 115-121 says `tools/INDEX.json` is generated from docstrings,
but purpose, side effects, mutability and ownership are not mechanically true
because a docstring says so. A committed JSON catalogue plus independently
edited docstrings would create the same drift the tool is meant to remove.

The repo contains 79 top-level Python tools, not a small homogeneous command
set. Some are active operational commands, some are one-campaign materializers,
some are evidence readers and some are historical scripts. Requiring the same
support contract from all 79 would turn old scripts into permanent API surface.

**Required redesign:** separate machine-discovered facts from reviewed semantic
declarations:

- discovered: path, source hash, CLI/`main` presence, parser arguments, imports,
  Git revision and source-declared entry points;
- declared: purpose, lifecycle (`supported`, `campaign_frozen`, `experimental`,
  `historical`, `deprecated`), mutability, authority class, owner and replacement;
- unknown or missing semantic metadata must remain `UNCLASSIFIED`, never be
  guessed from prose;
- fail CI only when a new executable tool is unclassified or a `supported`
  entry drifts, not because an untouched historical script lacks a modern
  contract.

The generated index is output. The small reviewed declaration file is the
semantic source. Neither is allowed to import or execute every tool merely to
inventory it.

### T-2: `config-doctor` would duplicate authority unless built on the runtime seam

Pydantic configuration models are not future work here. `trading-contracts`
already depends on Pydantic and exports `TradingExperimentConfig`;
`app/canonical_config.py` already resolves legacy/canonical overlays and rejects
runtime-key collisions. `campaign_supervisor.py` already performs dataset and
worker-config checks. A new checker that restates these rules will eventually
disagree with launch code.

**Required redesign:** extract shared, pure validators at their current
ownership boundaries, then call the same functions from:

1. materialization tests;
2. the read-only doctor;
3. the actual campaign preflight.

The doctor may have a convenience CLI in the isolated tooling environment, but
the authoritative preflight must also run in the exact target runtime
environment so installed entry points, package revisions and resolved paths are
real. This adds no dependency to `trading-stack`; the validation library must
use dependencies already present there.

Typed outcomes are:

- `PASS`: every required check executed and passed;
- `BLOCK`: an executable invariant is contradicted;
- `WARNING`: a declared but non-safety concern needs review;
- `UNAVAILABLE`: a required fact could not be established.

Campaign launch must refuse both `BLOCK` and required `UNAVAILABLE`. Blocking
classes include schema/type failure, conflicting resolved runtime keys,
unresolvable plugin or metric, asset/timeframe/manifest mismatch, missing or
mismatched required hashes, train/validation/test overlap, and an unexecutable
genome or repair rule. No General signs each launch and no chat instruction
overrides a block: the config or validator is corrected in a new revision.
Warnings may be acknowledged only by a versioned suppression carrying rule id,
reason, owner, scope and expiry.

### T-3: `evidence-lib` presently overlaps an existing shared contract

Document 35 lines 123-128 proposes one module for canonical JSON, content
hashing, files, Git state and manifests. These are different identity domains.
`trading-contracts.canonical` already defines canonical JSON and
`content_hash()`. Reimplementing those in `agent-multi` would create the very
cross-repo disagreement the proposal wants to remove.

Before code, produce an identity inventory that classifies every existing
helper by semantic domain:

- canonical structured-content identity;
- raw file-byte identity;
- ordered collection identity;
- source-tree identity, including tracked, dirty and untracked state;
- artifact-manifest identity;
- runtime/deployment identity.

Reuse `trading_contracts.content_hash` for its declared domain. Any additional
shared primitive must have a named domain, schema version, exact byte contract,
golden test vectors and compatibility rule. Do not expose a generic `hash()`
whose callers can silently mix domains. Do not migrate accepted evidence tools
until a designated freeze point and a before/after identity-equivalence proof.

The reported narrow count of 15 files matching the stated helper pattern is
reproduced. It is not a complete semantic inventory; additional helpers use
names such as `_sha_file`, `_sha_json`, `canonical_hash` and
`source_tree_digest`. This strengthens the need for classification before
consolidation.

### T-4: `griffe check` does not cover the cited defect class yet

Griffe compares declared package API snapshots for breaking changes. Finding
143 involved a probe calling an internal renamed symbol. Unless that symbol is
part of a deliberately supported public API, `griffe check` is not the
authoritative guard. Adding it now would produce a reassuring report without
covering the relevant call seam.

First declare the supported plugin protocols and entry-point contracts. Protect
private consumer calls with direct import/call contract tests or replace them
with a public adapter. Reconsider Griffe when there is a release/tag or audited
revision that intentionally defines compatibility.

### T-5: the proposed catalogue omits the owner's main reuse case

The owner's question is broader than discovering scripts: it includes learning
how each repository implements plugins without rereading and copying examples.
`setup.py` currently declares environment, agent, pipeline, execution-policy
and optimizer plugin groups; installed metadata is loaded through
`importlib.metadata.entry_points()`.

N-1 must therefore include a plugin surface inventory:

- source-declared group/name/import target;
- installed group/name/import target in the named environment;
- source-versus-installed drift;
- owning repo and revision;
- protocol/base class when one actually exists, otherwise `IMPLICIT`;
- canonical example and focused contract test;
- config keys selecting the plugin.

Do not instantiate plugins during inventory. A later smoke command may load one
named plugin in a controlled subprocess. A plugin scaffolder is deferred until
at least one explicit protocol exists; copying an implicit example faster would
only reproduce undocumented assumptions faster.

### T-6: the current success metrics can be gamed

"Facts quoted from tools" can increase while correctness gets worse, and
"zero rewritten equivalent" depends on subjective classification. Replace the
global metric with a small benchmark of representative questions and defects.

For the first cycle, record:

- answer agreement with direct source/runtime evidence;
- false positives, false negatives and `UNAVAILABLE` outcomes separately;
- elapsed time and tool calls to a correct answer;
- stale-index or source/installed-metadata disagreements;
- audit defects caused or missed by the new tooling.

Never optimize the percentage of tool-derived facts as an objective. It is
diagnostic telemetry only.

## 3. Answers to the Five Requested Rulings

1. **Scope:** reduce NOW to N-1, N-3 and bounded Ruff. N-2 receives an identity
   design packet only. Hypothesis needs no adoption task; Griffe waits.
2. **Evidence boundary:** no migration now. Preserve accepted tools byte-for-
   byte. Design domain-specific identities, reuse `trading-contracts`, then ask
   for a separate implementation verdict.
3. **Doctor authority:** the CLI reports, but the shared launch validators block
   deterministic executable contradictions and unavailable required facts.
   There is no per-launch General sign-off and no conversational override.
4. **Audit standard:** quote tool schema/version, exact tool revision and dirty
   source digest, dependency-lock hash, target environment identity, normalized
   command/arguments, all input identities, UTC start/end, typed outcome and
   output hash. Include a one-command reproduction. High-risk conclusions also
   require an independent direct-source or runtime check.
5. **New failure points found:** duplicate semantic registries; catalogue
   metadata presented as discovered fact; validator drift from launch code;
   tooling-env results mistaken for runtime-env truth; hash-domain conflation;
   stale codebase graph; source entry points differing from installed metadata;
   Griffe green reports over private consumers; mass lint churn; and property
   tests treated as proof rather than search for counterexamples.

## 4. First-Cycle Implementation Order

### P0: baseline and provenance

1. Record the exact repo heads and dirty/untracked digest for the pilot.
2. Record the Codebase Memory index revision/freshness for each queried repo.
3. Freeze the five historical bad configs and a balanced clean-config corpus by
   content hash. Do not silently rewrite fixtures to make the doctor pass.

### P1: engineering surface index

1. Pilot only in `agent-multi`.
2. Inventory both `tools/` commands and plugin entry points.
3. Compare source-declared and installed metadata in each named environment.
4. Emit versioned deterministic JSON with the provenance fields from section 3.
5. Add tests for unclassified new executables, stale supported entries,
   duplicate command/plugin ids, invalid import targets and source/installed
   drift.
6. Demonstrate five questions answered from the index and independently checked
   against source. At least one must concern each of tools, plugin groups,
   mutability, lifecycle and environment drift.

### P2: shared config validation and doctor facade

1. Map each rule to its authoritative existing function or owner module before
   extracting code.
2. Keep one implementation per invariant and call it from doctor and launch.
3. Verify the five historical defect classes plus deliberately clean controls.
4. Report a confusion matrix and every suppression; do not report a percentage
   without counts.
5. Prove that `BLOCK` and required `UNAVAILABLE` prevent a launch in a
   socket-free integration test, while warnings alone do not mutate anything.

### P3: bounded Ruff adoption

1. Pin an exact Ruff artifact in a hash-locked tooling requirements file.
2. Set `preview=false`; use no `--fix` in CI or evidence generation.
3. Begin with correctness/syntax rules on production modules and tools changed
   after adoption. Record the baseline instead of mass-editing historical code.
4. Expand rules only through a separate reviewed diff; formatting is not part
   of this job.

### P4: measurement and decision

Run the benchmark in T-6. Return source, tests, exact commands, lock hashes,
output packets and observed disagreements. Only then propose N-2, Griffe,
cross-repo expansion or a plugin scaffolder.

## 5. Standing Boundaries

- Codebase Memory, the catalogue and any future SQLite MCP are discovery
  surfaces, never order, broker, campaign, finding or promotion authorities.
- No general-purpose MCP receives broker credentials or calls a broker API.
- No generated tool output is accepted because it is deterministic; it earns
  trust through independent test vectors and reproduction.
- Existing packages are preferred when they own the required semantics and
  pass the project's source/security/reproducibility review. "Existing" alone
  is not sufficient.
- Git, typed contracts, executable validators and canonical ledgers remain the
  sources of truth. Indexes and reports are replaceable derived views.

This disposition approves a small experiment in engineering efficiency. It
does not authorize a new platform layer.
