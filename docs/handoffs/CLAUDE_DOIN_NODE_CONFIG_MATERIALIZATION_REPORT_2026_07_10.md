# DOIN-CONFIG-001 — Task Report

- Task ID: `DOIN-CONFIG-001`
- Spec: `CLAUDE_DOIN_NODE_CONFIG_MATERIALIZATION_TASK_2026_07_10.md`
- Repository: `/home/harveybc/Documents/GitHub/doin-node`
- Base commit: `826c713dda29bd3f27513b647e979b73c5505dd9` (unchanged — no commit made)
- Status: implemented and verified; not committed. Codex review pending.

---

## 1. Preflight evidence

```
$ git rev-parse HEAD        → 826c713dda29bd3f27513b647e979b73c5505dd9   ✓ matches required base
$ git branch --show-current → master                                    ✓
$ git status --short        → (empty)                                   ✓ clean
$ python --version          → Python 3.12.7                             ✓
$ python -m pytest -q tests/test_unified.py → 2 failed, 32 passed in 1.37s   ✗ (see blocker below)
```

**Preflight blocker (raised and approved before editing).** The baseline targeted test does **not**
fully pass. Two tests fail at the exact required base commit (worktree clean, so pre-existing by
definition):

- `tests/test_unified.py:324` `test_domain_without_synthetic_zero_weight` — expects `0.0`, gets `0.5`
- `tests/test_unified.py:447` `test_no_synthetic_means_zero_effective_increment` — expects `0.0`, gets `0.5`

Root cause is in **doin-core** (forbidden to modify) at
`../doin-core/src/doin_core/consensus/weights.py:116-123`: a domain without synthetic data is
deliberately assigned `verification_strength = 0.5` (comment: a zero weight would deadlock block
creation), while the two tests still assert the old `0.0` contract. This is entirely in the
VUW/consensus layer, orthogonal to the config loader. Per the spec this triggers the stop condition
"baseline tests fail for a reason unrelated to this task," and it makes "complete existing test suite
green" unachievable in scope (the only fixes — edit doin-core, or weaken existing tests — are both
forbidden). The maintainer chose **"Proceed, prove delta = 0."**

The full suite has a **third** pre-existing failure, also out of scope:
`tests/test_gossip.py::TestControlMessages::test_graft_rejected_when_full`. All three reproduce
deterministically at the base commit.

## 2. Implementation summary

- **`DomainRole`** (unified.py): added three backward-compatible fields — `inference_config: dict`,
  `synthetic_data_config: dict`, `metric_type: str` — all with empty/default factories. No existing
  field changed.
- **`cli.py`**: added five pure private helpers (`_parse_param_bounds`, `_parse_resource_limits`,
  `_parse_incentive_config`, `_parse_fee_config`, `_select_identity_path`); rewired `load_config()`
  to materialize **every** `UnifiedNodeConfig` field (12 previously dropped) and **every**
  `DomainRole` field (incentive_config + the 3 new ones previously dropped), referencing dataclass
  defaults instead of duplicated literals; rewrote `setup_plugins()` so optimizer/inference/synthetic
  each receive an **isolated copy** of the correct subtree with the declared fallback and injection
  rules; fixed identity precedence in `main()`.
- Loading does no filesystem/identity/plugin/network/GPU work in the helpers; `load_config` only
  reads JSON as before.

## 3. Files changed

| File | Why it needed modification |
| --- | --- |
| `src/doin_node/unified.py` | `DomainRole` lacked `inference_config`, `synthetic_data_config`, `metric_type` (R01) — the loader has nowhere to put those JSON fields without them. |
| `src/doin_node/cli.py` | `load_config()` dropped 12 `UnifiedNodeConfig` fields + `incentive_config`; `setup_plugins()` fed `optimization_config` to all three plugins and aliased the same mutable dict; `main()` bypassed `identity_file`. All the required behavior lives here. |
| `tests/test_cli_config.py` (new) | Focused loader/plugin/identity regression tests (R14, and evidence for R01–R13). Written before the production change and confirmed failing first. |

No other files touched. `doin-core`, `doin-plugins`, examples, protocols, consensus, chain, OLAP,
dependencies, metadata — all unchanged.

## 4. Requirement evidence matrix

| ID | Implementation (file / function) | Test name | Result |
| --- | --- | --- | --- |
| R01 | `unified.py` `DomainRole` (new fields) | `test_r01_r02_r03_r04_every_domain_field_materialized` | PASS |
| R02 | `cli.py` `load_config` (domain loop) | `test_r01_r02_r03_r04_every_domain_field_materialized` | PASS |
| R03 | `cli.py` `_parse_resource_limits` | `test_r01_..._materialized`, `test_r03_r14_invalid_nested_sections_fail_clearly` | PASS |
| R04 | `cli.py` `_parse_incentive_config` | `test_r01_..._materialized`, `test_r13_omega_example_values` | PASS |
| R05 | `cli.py` `load_config` (defaults reference) | `test_r05_every_top_level_field_materialized`, `test_r05_empty_object_matches_defaults` | PASS |
| R06 | `cli.py` `_parse_fee_config` | `test_r06_nested_fee_config_materialized` | PASS |
| R07 | `cli.py` `_select_identity_path` + `main` | `test_r07_identity_precedence` | PASS |
| R08 | `cli.py` `setup_plugins` (optimizer copy) | `test_r08_r09_r10_each_plugin_receives_intended_subtree` | PASS |
| R09 | `cli.py` `setup_plugins` (inference + fallback) | `test_r09_inference_fallback_to_optimization_config` | PASS |
| R10 | `cli.py` `setup_plugins` (synthetic + fallback) | `test_r10_synthetic_fallback_to_optimization_config` | PASS |
| R11 | `cli.py` `setup_plugins` (metric/bounds injection) | `test_r11_metric_type_injection_and_no_overwrite`, `test_r11_param_bounds_only_in_optimizer`, `test_r11_param_bounds_not_injected_when_key_present` | PASS |
| R12 | `cli.py` `load_config` copies + per-plugin `dict()` | `test_r12_role_configs_independent_from_raw`, `test_r12_source_dicts_unchanged_after_configure`, `test_r12_mutating_plugin_cannot_contaminate_others` | PASS |
| R13 | `cli.py` `load_config` / `setup_plugins` | `test_r13_omega_example_values`, `test_r13_gamma_example_three_distinct_subtrees` | PASS |
| R14 | `cli.py` `_parse_param_bounds` / nested parsers | `test_r14_malformed_bounds_fail_clearly`, `test_r03_r14_invalid_nested_sections_fail_clearly` | PASS |
| R15 | whole diff | full suite (delta = 0) + `git diff --check` + `git status` | PASS (see §7) |

## 5. Behavior and backward compatibility

- **Preserved:** all existing example values load unchanged (30/30 examples load with 0 errors);
  optimizer still gated on `optimize`, evaluator/synthetic on `evaluate`; existing
  `param_bounds`→optimizer injection and the "don't inject when a bounds key already exists" guard are
  byte-for-byte preserved; plugin exception handling / startup-continuation and all log/print messages
  unchanged; legacy JSONs with no `inference_config`/`synthetic_data_config` fall back to
  `optimization_config`.
- **Fallback (R09/R10):** inference and synthetic each get an independent `dict(optimization_config)`
  copy when their own subtree is empty — never the same object.
- **Injection (R11):** domain `metric_type` injected into optimizer/inference **only** when that
  subtree lacks `metric_type`; never injected into synthetic; param_bounds never injected into
  inference/synthetic.
- **One deliberate, spec-mandated change:** when `data_dir` is **absent** from JSON, the loader now
  yields the dataclass default `"./doin-data"` instead of the previous divergent literal
  `"./don-data"`. This is required by R05 / "Top-level default preservation" ("reference the current
  `UnifiedNodeConfig` defaults … loading `{}` must match `UnifiedNodeConfig()`") and is covered by
  `test_r05_empty_object_matches_defaults`. No existing test relied on `./don-data` (full suite
  delta = 0). Every example specifies `data_dir` explicitly, so it is unaffected.

## 6. Tests added

`tests/test_cli_config.py` — 19 tests, all passing; no node start, no network, no real plugins, no
model fitting, no identity generation. Loader tests (top-level materialization, nested FeeConfig, all
DomainRole fields, omega values, gamma distinct subtrees, legacy fallback defaults, malformed bounds
×7 cases, invalid nested sections ×4, empty-object-vs-defaults, raw-independence). Plugin-setup tests
via recording fakes + a minimal fake node (subtree routing, inference/synthetic fallback, metric
no-overwrite, bounds-only-in-optimizer, sources unchanged, hostile-plugin isolation). Identity test
exercises `_select_identity_path` precedence purely.

## 7. Commands executed and exact results

```
# BEFORE change (baseline)
python -m pytest -q tests/test_unified.py        → 2 failed, 32 passed  in 1.37s
python -m pytest -q                              → 3 failed, 287 passed in 12.13s
python -m pytest -q tests/test_cli_config.py     → 18 failed, 1 passed  in 1.42s   (tests fail first, as required)

# AFTER change
python -m pytest -q tests/test_cli_config.py tests/test_unified.py
                                                 → 2 failed, 51 passed  in 1.67s
python -m pytest -q                              → 3 failed, 306 passed, 15 warnings in 12.14s
python -m pytest -q tests/test_cli_config.py     → 19 passed           in 1.09s
python -m compileall -q src/doin_node tests/test_cli_config.py → OK
git diff --check                                 → clean (no whitespace errors)
git status --short                               → M cli.py, M unified.py, ?? tests/test_cli_config.py
```

**Delta proof:** full suite went `3 failed / 287 passed` → `3 failed / 306 passed`. The failing set
is **identical** (`test_graft_rejected_when_full`, `test_domain_without_synthetic_zero_weight`,
`test_no_synthetic_means_zero_effective_increment`); passed count rose by exactly the 19 new tests.
**The change introduces zero new failures.** The 3 remaining failures reproduce on the untouched base
commit and are all in forbidden territory (doin-core VUW weighting; gossip control messages).

The single pre-change passing test (`test_r11_param_bounds_not_injected_when_key_present`) guards
*preserved* existing behavior; it was inspected per the spec and left intact rather than weakened.

## 8. Self-review findings

- **Unrelated changes:** none. Diff is 2 production files + 1 new test file; no reformatting of
  untouched code, no metadata/version/dependency edits, no examples/doin-core/doin-plugins edits.
- **Mutable-config aliasing:** none found. `load_config` stores shallow copies of the three plugin
  subtrees (independent from raw JSON — `test_r12_role_configs_independent_from_raw`); `setup_plugins`
  builds a fresh `dict()` per `configure()` call; the hostile-plugin test proves one plugin's mutation
  cannot reach the role or another plugin. Caveat: copies are shallow (matching the spec's `copy(...)`
  contract), so a plugin mutating a *nested* object would still be shared — the isolation guarantee is
  at the top level, as prescribed.
- **Ignored dataclass field:** none. All 33 `UnifiedNodeConfig` fields and all 17 `DomainRole` fields
  are materialized; verified by the "every field" tests and the empty-object-vs-defaults test.
- **Weakened assertion:** none.
- **Untested fallback:** none. Both inference and synthetic fallbacks have dedicated tests plus an
  end-to-end check on the real gamma example.

## 9. Assumptions, limitations and blockers

- **Blocker (pre-existing, out of scope):** 3 baseline test failures at the required commit (2 VUW in
  `test_unified.py`, 1 in `test_gossip.py`). They stem from doin-core/consensus and gossip code,
  cannot be fixed without forbidden changes, and are unrelated to this loader task. Proceeding per
  maintainer instruction; delta = 0 demonstrated.
- **ResourceLimits error semantics:** `ResourceLimits` is a pydantic model with `extra='ignore'`, so
  an *unknown* key is silently dropped by pydantic (doin-core must not change). "Clear domain/section
  errors" for that section therefore fire on non-dict sections and invalid *value types*
  (ValidationError, wrapped) — both tested. Unknown-key rejection is available naturally for the
  dataclass sections (`incentive_config`, `fee_config`), which is where it is tested.
- **`$doc` / `$schema`:** examples also carry a documentation-only `$schema` key (alongside `$doc`).
  Per spec both remain harmlessly ignored; no strict rejection added.
- **Adjacent improvement not implemented (listed, not done):** `README.md` was left untouched to keep
  the diff minimal; a short section documenting the now-supported JSON fields could be added if
  desired (spec permits this only "if needed").
- No commit/push/deploy/node-start/chain-reset/optimization was performed.

## 10. Git branch, HEAD and status

```
branch: master
HEAD:   826c713dda29bd3f27513b647e979b73c5505dd9   (unchanged — no commit made)
status:
 M src/doin_node/cli.py
 M src/doin_node/unified.py
?? tests/test_cli_config.py
```

Not claimed as accepted or production-ready — the diff is ready for Codex to inspect against every
requirement and rerun the critical tests. The one item warranting reviewer attention is the
deliberate `data_dir` absent-default change (§5), which is spec-mandated by R05 but is a real
behavior change for configs that omit `data_dir`.
