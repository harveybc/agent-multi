# 35. Deterministic Tooling Opportunity Map

Status: proposal — v2.0.0, 2026-08-06 — REVISED after self-review,
submitted to General Musashi for verdict before ANY adoption or build.
Author: General Satoshi III · For: Owner and General Musashi
Purpose: replace repetitive agent reading/derivation with deterministic
tools, and adopt existing packages instead of reimplementing
architecture. Grounded in what this correction campaign actually cost.

v2 changes: self-critique section (§0b) added; scope cut from 8 built
tools + 12 adoptions to 3 + 3 NOW, the rest explicitly DEFERRED or
WITHDRAWN; the overcounted sha256 claim corrected (15 tools define
their own helper, not 43 — 43 merely reference the string); every
"would have prevented finding X" counterfactual downgraded to "would
have made X less likely"; environment-isolation rule added (§0c).

## 0. The honest premise, including its danger

A deterministic tool is cheaper, faster and repeatable. It is also
**more dangerous than an agent when it is wrong**, because it fails
identically every time and carries an air of authority. This campaign
already produced that exact failure: `after_probe.py` mapped every
exception to "corrected" and confidently reported 8/8 passes that
Musashi demolished (finding 143).

So every tool proposed here must ship with: typed outcomes (never a
bare boolean), fail-closed defaults, its own tests, a version string in
its output, and no authority to conclude — only to **report facts**.

## 0b. Self-critique of v1 (defects I found in my own proposal)

1. **v1 contradicted its own evidence.** My documented failure was NOT
   a missing tool — it was not using tools that already existed
   (`multifront_status.py`, the indexed codebase graph). v1's answer
   was to build MORE tools, including `fleet-report`, a new wrapper
   around the very tool I ignored. A new wrapper does not cure
   non-use; a catalogue and discipline do. `fleet-report` is
   **withdrawn**; `multifront_status.py` gets documented in the
   catalogue instead.
2. **Twenty new components is itself a failure surface.** 8 tools + 12
   adoptions each need tests, versioning, and independent audit. The
   token cost of Musashi auditing twenty components exceeds the near-
   term token savings. v2 cuts to 3 built + 3 adopted NOW.
3. **I overcounted the duplication evidence.** 43 tools *mention*
   sha256; only **15** define their own hashing helper. The case for
   `evidence-lib` survives at 15, but v1 stated the larger number as
   fact. Corrected.
4. **"Would have prevented finding X" was counterfactual advocacy.**
   A contract dump only prevents the 100× error if its semantics
   annotation is *correct* — which requires exactly the understanding
   that failed. Every such claim is downgraded to "would have made X
   less likely by moving the fact from N inferences to 1 declaration."
5. **`contract-dump` as designed could become a lie generator.** If
   generated from annotations, it can drift from the code and then
   fails identically every time with an air of authority — the §0
   danger, built by me. Redesigned: it must derive from **runtime
   introspection** (instantiate the env, step it, report the actual
   `info` keys and spaces) plus a CI test asserting dump == runtime.
   That is more work than v1's "1 session" estimate, which was
   optimistic. Moved out of the NOW set.
6. **Third-party adoption is a supply-chain and reproducibility
   risk v1 understated.** The evidence chain today depends on stdlib +
   git + SQLite — fully auditable. Putting `dvc` inside the
   replica-authority path would replace ~40 audited lines with a large
   external system Musashi cannot cheaply verify, days after he issued
   finding 151 against that exact surface. `dvc`, `great_expectations`,
   `evidently`, `optuna`, `omegaconf` are **deferred or withdrawn**
   from any evidence-bearing path.
7. **Mass-refactoring 15 audited tools onto `evidence-lib` churns
   verified code.** v2 rule: `evidence-lib` is mandatory for NEW code;
   existing tools migrate only when touched for another reason; tools
   referenced by accepted audit packets are never retroactively edited.
8. **v1 had no acceptance criteria.** v2 adds §6b: each NOW item ships
   with a measurable claim Musashi can verify or refute.
9. **Process defect: I committed and pushed v1 as if approved.** It
   was a proposal, and its tone was advocacy. This revision and the
   accompanying review request to Musashi are the correction.

## 0c. Environment isolation rule (new, non-negotiable)

Nothing is installed into the `trading-stack` runtime environment.
That env pins numpy 2.5.1 / pandas 3.0.3 / torch 2.13.0+cu130; a
tooling dependency that drags a transitive upgrade could silently
change numerical results and invalidate every baseline. All adopted
dev tools live in a separate `tooling` venv with a hash-locked
requirements file; anything that must run inside `trading-stack`
(e.g. `hypothesis` in tests — already installed there) is adopted
only after an explicit dependency-freeze check.

## 1. Evidence: what the agents actually spent tokens on

Measured from this campaign, not guessed:

| Repeated act | How it was done | Cost signature |
|---|---|---|
| "What does `info` expose?" | grep + sed over `gym-fx/app/env.py` | 2–3 tool calls per question, repeated across sessions |
| "Is `position` a direction or a quantity?" | 3 greps through `bt_bridge.py` — and I got it **wrong anyway** (finding 152, the 100× error) | wrong answer survived into shipped code |
| "Which plugin entry points exist?" | grep of `setup.py`/`pyproject.toml` per repo | repeated in every campaign |
| "Is this config self-consistent?" | discovered at **runtime failure** (`continuous_action_threshold` collision, `dataset_manifest_file`, dormant `train_years`) | each cost a failed launch or an audit finding |
| Fleet status | hand-written `curl | python -c` one-liners, ~8 times | `tools/multifront_status.py` and `fleet_status_context.py` **already existed and I did not use them** |
| RT OLAP inspection | hand-written sqlite queries, ~6 times | inconsistent columns each time |
| Dataset facts (rows/splits/hash) | recomputed ad hoc in 3 places | 15 tools define their own hashing helper (43 reference sha256) |
| API drift in probes | `_score_interval` → `score_interval` broke a probe silently | became finding 143 |

**The largest single waste found:** ten repositories are already
indexed in the codebase-memory graph (agent-multi 2,239 nodes / 9,779
edges; lts 3,224 / 17,116; gym-fx 454 / 2,375 …) — a deterministic
structural query surface that I used almost not at all, preferring
grep. That is a discipline failure, not a missing tool.

## 2. Tools to BUILD

### NOW (3 — pending Musashi's verdict, none started)

**N-1 `tool-index`** — `tools/INDEX.json`: name, one-line purpose,
inputs, outputs, read-only or mutating, owner; generated from
docstrings, validated by a test that fails when a tool lacks an entry.
*Fixes the actual root cause of this campaign's waste* (existing tools
went unused because nothing catalogued them). Cheapest item; also the
prerequisite for deciding future builds honestly, because it exposes
what already exists. Read-only; zero runtime risk.

**N-2 `evidence-lib`** — one module for content hashing, canonical
JSON, git HEAD + dirty/untracked source digest, manifest emit/verify.
Mandatory for NEW code only; existing tools migrate opportunistically;
tools cited by accepted audit packets are never retroactively edited.
*Risk removed:* the same identity computed two different ways — the
pattern behind findings 130/141/149/151.

**N-3 `config-doctor`** — resolves a canonical config and reports:
runtime key collisions across sections, dormant/contradictory fields,
metrics no implementation resolves, foreign asset tokens, missing
manifest references, genome/stage inconsistencies, repair rules that
cannot execute. Read-only, typed findings, never mutates the config.
*Track record argument:* findings 108, 110, 113, 126, 142 are five
members of this one defect class; a checker would have made each less
likely. Most checks already exist scattered in
`materialize_eth_curriculum_configs.py` and want extracting — this is
consolidation more than construction.

### DEFERRED (build only after the NOW set proves itself)

- **D-1 `contract-dump`** — runtime-introspected (instantiate the env,
  step it, emit actual spaces/`info` keys with units), with a CI test
  asserting dump == runtime. Deferred because the annotation-based v1
  design was a lie-generator risk (§0b-5) and the honest design is
  more work.
- **D-2 `rt-report`** — read-only OLAP reporter (interval table,
  succession-chain check, persisted p50/p95, coverage). Wanted before
  the next RT packet so both generals quote identical numbers.
- **D-3 `causality-check`** — future-append invariance, train-only
  fit, warm-up sufficiency. Required by roadmap R1 **before any
  wave-2 feature work begins**; deferred only until that work is
  actually scheduled.
- **D-4 `probe-lib`** — extract `correction_probe_v2`'s four typed
  outcomes into a shared module. Low risk, but touching the probe
  Musashi just verified should wait for his consent.

### WITHDRAWN

- **`fleet-report`** — a new wrapper around `multifront_status.py`,
  which already exists and which I failed to use. The cure is the
  catalogue (N-1), not another tool (§0b-1).

## 3. Packages to ADOPT instead of implementing

All adoptions obey §0c: separate hash-locked `tooling` venv, nothing
enters `trading-stack` or any evidence-bearing runtime path.

### NOW (3 — all dev/test-time, read-only, outside the evidence chain)

| Need | Adopt | Why |
|---|---|---|
| **API-drift detection** | `griffe check` | Deterministically diffs the public API between two revisions; the finding-143 class (probe calling a renamed symbol) becomes mechanically detectable. Dev-time only, reads code, writes nothing. |
| **Property-based tests** | `hypothesis` (**already installed** in trading-stack) | Musashi repeatedly demands property tests; we hand-write example tables. Zero new dependency. |
| **Static checks** | `ruff` (+ `mypy` later) | Catches unused/renamed/None-flow defects before an audit does. Dev-time only. |

### DEFERRED (real value, but each adds a dependency or a decision)

- `pandera` — dataset schemas in tests; adopt with the first
  dataset-manifest work, after a dependency-freeze check.
- `pydantic` config models — makes the collision class structurally
  impossible, but migrating configs mid-campaign is churn; adopt at
  the next natural config-schema change. (`omegaconf` withdrawn —
  a second config system is its own failure point.)
- `utilsforecast`/`statsforecast` — rolling-origin baselines for R6;
  adopt when RT1 analysis actually starts.
- `sqlite-utils` — convenience only; adopt with D-2 if at all.
- TSFM packages (`chronos-forecasting`, `timesfm`, `uni2ts`) — only
  when roadmap option B is ratified; never reimplement, exactly as
  the owner directs.
- `evidently` — only when the live drift-monitoring window opens.

### WITHDRAWN

- **`dvc` in the replica path** — would replace ~40 audited stdlib
  lines with a large third-party system inside the very surface
  Musashi just audited (finding 151). Wrong direction for
  auditability (§0b-6). May be revisited for bulk *dataset*
  versioning only, never for decision evidence.
- **`great_expectations`** — heavy; `pandera` covers the need.
- **`optuna`** — a second optimizer beside DOIN invites divergence
  and confusion; the cheap-screen need can be met with plain grids.
- **`yq`** — marginal; `jq` (installed) suffices.

### Zero-cost, pure discipline (no verdict needed — already available)

The **codebase-memory graph** already indexes ten repos
(`search_graph`/`get_code_snippet`/`trace_path`). My commitment,
effective immediately: structural code questions go to the graph
first, `grep` second. This is the single largest token saving found
and it requires building nothing.

## 4. Per-front opportunities (long view — nothing here is scheduled)

**Data/preprocessing** — `causality-check` (D-3) plus `pandera`
schemas per dataset; a `dataset-manifest` generator so row counts,
splits, hashes and warm-up are emitted once and quoted everywhere.

**Feature engineering** — a deterministic **feature registry**: name →
builder → causal window → warm-up → parameters → provenance, so the
GA references registry ids rather than free strings.

**Models/training** — `contract-dump` (D-1, runtime-introspected);
`griffe` in CI so a signature change breaks a test rather than a
probe; anchor manifests (finding 158) generated by a tool at champion
time rather than hand-written later.

**Optimization/DOIN** — `config-doctor` (N-3) before any campaign
launch; later a `genome-lint` and `campaign-preflight` if N-3 proves
the pattern works.

**Live/execution** — `controller_inventory` already exists (keep it);
a `parity-report` joining due-bar decisions to simulation is future
work for the parity track.

**Audit/evidence** — `evidence-lib` (N-2) + eventually D-2/D-4, so a
packet is *generated* from facts instead of transcribed by an agent.
This narrows the gap between what I claim and what Musashi reproduces.

## 5. What I would NOT automate

- **Judgement about whether a result means anything.** A tool can say
  "p95 = 26 s, 3 updates"; only a general should say "therefore the
  cadence is unproven".
- **Closing findings.** Deterministic reporting, human/auditor
  disposition.
- **Anything that would let "GPU busy" or "tests passed" stand in for
  evidence** — the zero-activity campaign is the cautionary tale.

## 6. Sequencing

1. **Nothing until Musashi's verdict on this document.**
2. If approved: `tool-index` (N-1) first — it is also the audit of
   what exists; then `evidence-lib` (N-2), `config-doctor` (N-3), and
   the three NOW adoptions into the isolated `tooling` venv.
3. The DEFERRED sets are re-proposed individually, each with the §6b
   evidence from the NOW set, at their trigger points.

## 6b. Acceptance criteria (measurable, refutable)

- **N-1**: every file in `tools/` has an INDEX entry; a test fails on
  omission; in the next campaign packet, zero instances of a
  hand-rewritten equivalent of an already-catalogued tool.
- **N-2**: zero NEW tools defining a private hashing/identity helper
  after adoption (baseline today: 15); enforced by a lint test.
- **N-3**: run against the five historical configs behind findings
  108/110/113/126/142 — it must flag all five defect classes; run
  against the current champion configs — every flag it raises must be
  either a true defect or a documented false-positive with a
  suppression annotation. False-positive rate is reported, not hidden.
- **Adoptions**: `griffe check` runs in CI between HEAD and the last
  audited revision and its report is attached to the next packet;
  `ruff` runs clean or with a committed, justified ignore list.
- **Global**: the next correction campaign's packet reports how many
  facts were quoted from tool output vs re-derived by an agent — the
  measure your instinct predicts should move.

None of this touches the paused fleet, the frozen contracts, or any
open finding. It is CPU-only work whose whole purpose is that the next
campaign spends its tokens on judgement instead of on rediscovery.
