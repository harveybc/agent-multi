# 35. Deterministic Tooling Opportunity Map

Status: proposal — v1.0.0, 2026-08-06
Author: General Satoshi III · For: Owner and General Musashi
Purpose: replace repetitive agent reading/derivation with deterministic
tools, and adopt existing packages instead of reimplementing
architecture. Grounded in what this correction campaign actually cost.

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
| Dataset facts (rows/splits/hash) | recomputed ad hoc in 3 places | 43 tools independently implement `sha256` |
| API drift in probes | `_score_interval` → `score_interval` broke a probe silently | became finding 143 |

**The largest single waste found:** ten repositories are already
indexed in the codebase-memory graph (agent-multi 2,239 nodes / 9,779
edges; lts 3,224 / 17,116; gym-fx 454 / 2,375 …) — a deterministic
structural query surface that I used almost not at all, preferring
grep. That is a discipline failure, not a missing tool.

## 2. Tools to BUILD (small, fail-closed, agent-facing)

Priority order by (tokens saved × defect risk removed) ÷ effort.

### P0-1 `contract-dump` — one command, all runtime contracts
Emits JSON: env observation/action spaces and every `info` key with its
semantics, plugin entry points per group with declared params and
defaults, pipeline/optimizer/agent plugin names, and the selection
metric branches implemented in `app/metrics`.
*Replaces:* every "what does X expose?" grep chain.
*Would have prevented:* finding 152 (direction vs quantity) if the dump
declared units.
*Effort:* 1 session. *Consumers:* both generals, every campaign.

### P0-2 `config-doctor` — refuse a bad config before the fleet burns
Resolves a canonical config and reports: runtime key collisions across
sections, dormant/contradictory fields (`train_years` vs explicit
dates), metrics that no implementation resolves, foreign asset tokens,
missing manifest references, genome/stage inconsistencies (a stage
param that no gene declares), and repair rules that cannot execute.
*Replaces:* the launch-crash discovery loop.
*Would have prevented:* findings 108, 110, 113, 126, 142 — five audit
findings from one class of defect.
*Effort:* 1–2 sessions; most checks already exist scattered in
`materialize_eth_curriculum_configs.py` and want extracting.

### P0-3 `evidence-lib` — one implementation of the boring primitives
A tiny module (not 43 copies): content hashing, canonical JSON, git
HEAD + dirty/untracked source digest, artifact load-proof, replica
observation, manifest emit/verify.
*Replaces:* duplicated `sha256`/identity code across 43 tools.
*Risk removed:* the same identity computed two different ways in two
tools — exactly how findings 130/141/149/151 arose.

### P1-4 `fleet-report` — the status contract, canonical
Wrap the existing `multifront_status.py` into one CLI with stable JSON:
per-worker chain/tip/generation/claim, GPU temp/util, candidate epoch/
steps/trades, code revisions, alerts, and the §4 status-contract fields
Musashi mandates.
*Replaces:* hand-written one-liners; also fixes the **discoverability**
failure by being the single documented entry point.

### P1-5 `rt-report` — OLAP queries with a fixed schema
Per-run interval table, succession-chain check, handover reconciliation,
persisted p50/p95, coverage completeness. Read-only.
*Replaces:* ad-hoc sqlite; guarantees every packet quotes the same
numbers.

### P1-6 `causality-check` — the leakage gate as a command
Given a feature builder and a dataset: future-append invariance
(value at `t` unchanged when rows after `t` are appended), train-only
fit verification, warm-up sufficiency, NaN/inf and monotonicity.
*Replaces:* ad-hoc checks; **required** by roadmap R1 before any
decomposition/calendar feature is admitted. This is the highest-value
new tool for the research track.

### P1-7 `probe-lib` — typed counterexample harness
Extract `correction_probe_v2`'s four outcomes
(`postcondition_pass`/`expected_refusal`/`fixture_error`/
`harness_error`) into a shared module both generals import.
*Prevents:* a repeat of finding 143 in either direction.

### P2-8 `tool-index` — machine-readable catalogue
`tools/INDEX.json`: name, one-line purpose, inputs, outputs, read-only
or mutating, owner. Generated from docstrings, validated in CI.
*Replaces:* an agent rediscovering that a tool already exists (my
concrete failure this campaign).

## 3. Packages to ADOPT instead of implementing

Your instinct is right: do not rebuild what exists.

| Need | Adopt | Why it beats our own code |
|---|---|---|
| **API-drift detection** | `griffe check` | Deterministically diffs the public API between two revisions. Finding 143 (renamed `_score_interval`) would have been caught mechanically. Cheapest high-value adoption on this list. |
| **Dataframe/data contracts** | `pandera` (schema) or `great_expectations` | Declarative column types, ranges, monotonic index, uniqueness — replaces hand-rolled dataset verification and gives failure *reasons*. |
| **Property-based tests** | `hypothesis` (**already installed**) | Musashi repeatedly demands property tests; we hand-write example tables. Hypothesis generates the adversarial cases for order-preservation, coverage cardinality, fee conservation. |
| **Static checks** | `ruff` + `mypy` | Catches the unused/renamed/None-flow classes of defect before an audit does. |
| **Config composition/validation** | `pydantic` (**installed**) + `omegaconf` | Typed config models make the collision/dormant-field class structurally impossible rather than detected. |
| **Artifact/data versioning + replica** | `dvc` | Content-addressed artifacts with declared remotes and verified pushes — precisely the finding-151 replica-authority problem, solved by a mature tool instead of my rsync+ssh hash. |
| **Time-series foundation models** | `chronos-forecasting`, `timesfm`, `uni2ts` (Moirai) | Roadmap option B. Frozen encoders as feature extractors; never reimplement the architecture. |
| **Forecast/eval utilities** | `utilsforecast`, `statsforecast` | Rolling-origin/prequential evaluation primitives and baselines — RT1 already needs them; also gives cheap transparent baselines for R6. |
| **Drift/regime monitoring** | `evidently` | The OOD/regime detector I proposed for the frozen champion, without writing it. |
| **Hyperparameter search primitives** | `optuna` | Not to replace DOIN, but its samplers/pruners are useful for the *cheap local screens* (my §4.7 dissent) where DEAP is overkill. |
| **CLI/JSON plumbing** | `sqlite-utils`, `jq` (**installed**), `yq` | Replaces hand-written sqlite/JSON glue in reports. |
| **Symbol/structure queries** | the **already-indexed codebase-memory graph** | Ten repos indexed; `search_graph`/`get_code_snippet`/`trace_path` answer "where is X, who calls it" without reading files. Zero adoption cost — pure discipline. |

## 4. Per-front opportunities

**Data/preprocessing** — `causality-check` (P1-6) plus `pandera`
schemas per dataset; a `dataset-manifest` generator so row counts,
splits, hashes and warm-up are emitted once and quoted everywhere.

**Feature engineering** — a deterministic **feature registry**: name →
builder → causal window → warm-up → parameters → provenance. Makes the
decomposition families (wavelet/multitaper/Hilbert/fracdiff) admissible
by construction and lets the GA reference registry ids rather than free
strings.

**Models/training** — `contract-dump` for observation/action spaces;
`griffe` in CI so a signature change breaks a test rather than a
probe; artifact manifests (the finding-158 anchor manifest) generated
by a tool at champion time rather than hand-written later.

**Optimization/DOIN** — `config-doctor` before any campaign launch; a
`genome-lint` (subset) proving every stage param is a declared gene,
every repair rule executable, every conditional gene masked; a
`campaign-preflight` that runs the whole gate list Musashi keeps
finding violations in.

**Live/execution** — `controller_inventory` already exists (keep it);
add `parity-report` joining due-bar decisions to simulation with fixed
columns; `evidently` for the divergence trend.

**Audit/evidence** — `evidence-lib` + `probe-lib` + `rt-report`, so a
packet is *generated* from facts instead of transcribed by an agent.
This also narrows the gap between what I claim and what Musashi
reproduces.

## 5. What I would NOT automate

- **Judgement about whether a result means anything.** A tool can say
  "p95 = 26 s, 3 updates"; only a general should say "therefore the
  cadence is unproven".
- **Closing findings.** Deterministic reporting, human/auditor
  disposition.
- **Anything that would let "GPU busy" or "tests passed" stand in for
  evidence** — the zero-activity campaign is the cautionary tale.

## 6. Suggested sequencing (cheap first, all CPU)

1. Adopt `griffe`, `ruff`, `pandera`, `hypothesis` (installed) — days.
2. Build `evidence-lib`, then `config-doctor` and `contract-dump`.
3. Build `fleet-report`, `rt-report`, `tool-index`.
4. Build `causality-check` before wave-2 feature work begins.
5. Adopt `dvc` when the replica requirement becomes operational.
6. Adopt TSFM packages only when roadmap option B starts.

None of this touches the paused fleet, the frozen contracts, or any
open finding. It is CPU-only work whose whole purpose is that the next
campaign spends its tokens on judgement instead of on rediscovery.
