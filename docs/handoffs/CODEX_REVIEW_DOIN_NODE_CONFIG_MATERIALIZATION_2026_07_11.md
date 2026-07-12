# Codex Review: DOIN-CONFIG-001

Date: 2026-07-11
Repository: `/home/harveybc/Documents/GitHub/doin-node`
Base commit: `826c713dda29bd3f27513b647e979b73c5505dd9`
Decision: **accepted locally after corrections**
Commit/deploy status: not committed, not pushed, not deployed

## 1. Scope Reviewed

The review inspected the actual working-tree diff, the Claude report, all
requirements `R01` through `R15`, both representative predictor node configs,
the complete example directory, and the focused and full test suites.

Reviewed files:

- `doin-node/src/doin_node/unified.py`
- `doin-node/src/doin_node/cli.py`
- `doin-node/tests/test_cli_config.py`
- `agent-multi/docs/handoffs/CLAUDE_DOIN_NODE_CONFIG_MATERIALIZATION_REPORT_2026_07_10.md`

## 2. Findings and Corrections

### High: nested plugin configuration was not isolated

The implementation used `dict(...)`, which copied only the outer dictionary.
A plugin mutating a nested list or dictionary changed the stored
`DomainRole.optimization_config`; inference and synthetic fallback then
received the contaminated value. This violated the operational meaning of
`R08`, `R09`, `R10`, and `R12` for real configs containing nested optimization
stages and parameter structures.

Correction:

- use `copy.deepcopy(...)` when materializing each plugin subtree;
- use a fresh deep copy for every `plugin.configure(...)` call;
- add an adversarial nested-mutation regression test.

### Medium: unknown `ResourceLimits` fields were silently discarded

`ResourceLimits` is a Pydantic model configured to ignore extra fields. A typo
such as `max_epohcs` therefore loaded successfully and disappeared, contrary to
the required clear failure for invalid nested keys in `R03` and `R14`.

Correction:

- compare input keys against `ResourceLimits.model_fields` before
  construction;
- reject unknown fields with domain, section, and field names;
- add the missing negative regression case.

### Low: nested types were imported through an incidental re-export

`FeeConfig`, `IncentiveConfig`, and `ResourceLimits` were imported from
`doin_node.unified`, where they happened to be module globals. The loader now
imports each type from its owning `doin-core` module, avoiding an accidental
API dependency.

## 3. Accepted Behavior

After correction, the implementation satisfies the task contract:

- every current `UnifiedNodeConfig` and `DomainRole` field is materialized;
- empty JSON preserves dataclass defaults;
- fee, incentive, resource, bounds, and identity semantics are explicit;
- optimizer, inference, and synthetic plugins receive independent configs;
- legacy inference/synthetic fallback remains supported;
- metric and bounds injection follow the declared precedence;
- all 30 existing JSON examples load successfully;
- no consensus, network, OLAP, example, dependency, or metadata file changed.

The corrected `./doin-data` absent-field value is accepted because it matches
the authoritative `UnifiedNodeConfig` default. Existing examples provide an
explicit `data_dir` and are unaffected.

## 4. Verification Evidence

Focused config tests:

```text
python -m pytest -q tests/test_cli_config.py
20 passed
```

Focused config plus unified-node tests:

```text
python -m pytest -q tests/test_cli_config.py tests/test_unified.py
2 failed, 52 passed
```

Full suite:

```text
python -m pytest -q
3 failed, 307 passed, 15 warnings
```

The three failures are unchanged at the clean base commit. A detached worktree
at `826c713` produced the same failures:

- `TestControlMessages::test_graft_rejected_when_full`
- `TestVUWIntegration::test_domain_without_synthetic_zero_weight`
- `TestFullFlow::test_no_synthetic_means_zero_effective_increment`

Additional checks:

```text
python -m compileall -q src/doin_node tests/test_cli_config.py
git diff --check
load_config over examples/*.json: 30 passed, 0 failed
```

## 5. Residual Issues

The three baseline failures must be handled as separate maintenance tasks.
They do not block acceptance of config materialization, but the repository
cannot yet claim a globally green test suite. No node was started and no chain,
identity, optimization, or deployment state was changed during this review.
