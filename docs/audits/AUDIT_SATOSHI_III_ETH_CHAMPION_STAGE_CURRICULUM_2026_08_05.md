# Audit: Satoshi III ETH Champion and Stage Curriculum

Date: 2026-08-05 America/Bogota
Auditor: General Musashi, temporary independent auditor
Delivery reviewed:
`docs/handoffs/SATOSHI_III_ETH_CHAMPION_STAGE_CURRICULUM_DELIVERY_2026_08_05.md`

Owner order:
`docs/handoffs/MUSASHI_TO_SATOSHI_III_ETH_CHAMPION_STAGE_CURRICULUM_ORDER_2026_08_05.md`

## 1. Verdict

**REJECTED; the ETH-EN swarm was paused and disabled.**

The train-only solvency continuation, dataset freeze and easy-to-normal
weight continuation are useful implementations. They do not make the running
campaign valid. The materialized runtime trains/checkpoints against the old
`risk_adjusted_return` composite, the outer optimizer does not implement the
configured `lexicographic_weekly_v1` metric, and its scalar compatibility
encoding can reverse the authoritative lexicographic order. Candidate
evaluation failures have already been recorded as champion-eligible `-1e9`
results and accepted into blocks. The fleet also remained on three competing
tips at the same height.

The current chain is evidence of a failed campaign and must never be resumed
as the corrected ETH experiment. Corrections require a new plan ID, domain ID,
semantic hash, genesis and shared population.

## 2. Findings

### AUD-F1-20260805-108 — S1 — the configured objective cannot complete

The canonical config declares
`objectives.selection_metric=lexicographic_weekly_v1`, but
`training.selection_metric=risk_adjusted_return`. Canonical flattening gives
the training value to the pipeline. Live logs independently confirm every
normal epoch uses the old risk-adjusted train-tail/validation composite.

The same runtime sets `optimization_metric=lexicographic_weekly_v1`, but
`app.metrics.compute_optimization_fitness()` has no such branch and raises
`ValueError`. A candidate that survives the activity/collapse gates therefore
trains for hours and fails exactly when it should return fitness.

Evidence:

- `examples/config/phase_2_eth_curriculum/optimization/phase_2_eth_en_v1.json:359,945`
- `app/canonical_config.py:228-311`
- `app/metrics.py:16-57`
- `pipeline_plugins/rl_pipeline_with_validation.py:932-1042`
- reproducer: `selection_metric=risk_adjusted_return`,
  `optimization_metric=lexicographic_weekly_v1`, then
  `ValueError: unknown optimization_metric='lexicographic_weekly_v1'`
- Omega log line 37360: `risk_adjusted_return composite=...`

### AUD-F1-20260805-109 — S1 — failed candidates can become champions

`default_optimizer._evaluate()` catches pipeline failures and returns a finite
`-1e9` with `evaluation_error`, without the candidate-rejection contract.
DOIN only rejects `candidate_rejected[_reason]`; it therefore treats these
failures as eligible evaluations. Four invalid preprocessing candidates were
recorded with `candidate_rejected=false`; an accepted transaction with
`verified_performance=-1000000000.0` exists in block 2 of Omega's archived
chain.

This is not merely a dashboard label. A failed candidate produced an accepted
block and became the initial domain champion.

Evidence:

- `optimizer_plugins/default_optimizer.py:1239-1264,1301-1323`
- `doin-node/src/doin_node/unified.py:102-109,2492-2643`
- archived chain query reproduced by the evidence script
- Dragon logs contain three deterministic
  `feature-aware observation contract ... got 'none'` failures

### AUD-F1-20260805-110 — S1 — ETH artifacts target the USDCAD namespace

The ETH config still identifies itself as
`phase_1_asset_policy_usdcad_4h_protected_easy_v2` and routes optimizer
champion, parameters, resume, history and statistics to
`${ARTIFACT_ROOT}/protected_easy/usdcad_4h/...`. The campaign handoff expects
`${ARTIFACT_ROOT}/eth_curriculum/en/...`.

A successful ETH candidate could overwrite or contaminate historical USDCAD
artifacts, while campaign archival would look in a different directory and
fail to materialize the promised ETH champion.

Evidence:

- config lines 339 and 805-882
- independently resolved runtime paths in the evidence script

### AUD-F1-20260805-111 — S2 — the fleet was not on one blockchain

At 2026-08-05 11:31 COT all workers reported height 3, but Omega and Dragon
used tip `ebe4e3fb...`, Gamma-5070Ti used `1966ec2c...`, and Gamma-5090 used
`5f965f37...`. Every supervisor had raised `swarm_health: workers report
competing blockchain tips at the same height` continuously since about 04:09
COT.

Gamma's direct log records an `IndexError: list index out of range` at
`peer_blocks[-1]` during equal-height resolution. The existing single-peer
unit test passes, but it does not reproduce simultaneous block production,
peer rollback during fetch, or deterministic convergence among four workers.

Evidence:

- `doin-node/src/doin_node/unified.py:4474-4589`
- three supervisor `/api/status` payloads sampled directly
- Gamma-5070Ti log line 1577 and surrounding sync traceback

### AUD-F1-20260805-112 — S2 — the transport scalar is not lexicographic

The declared authoritative tuple is
`(mean_weekly_return, -max_drawdown, total_return)`, but the transport value is
`weekly - 1e-4*drawdown + 1e-8*total`. This weighted sum can reverse the tuple.
The reproducer constructs A=(0.01000,-0.9,0) and B=(0.00995,0,0): A wins
lexicographically, while B has the larger transport scalar (0.00995 vs
0.00991). DEAP and DOIN therefore need an authoritative tuple-aware comparator
or an explicitly bounded/quantized encoding that is proven order-preserving.

Evidence: `pipeline_plugins/_lexicographic_selection.py:28-29,73-78` and the
socket-free counterexample.

### AUD-F1-20260805-113 — S3 — the genome emits forbidden candidates

`preprocessing_mode` includes `none`, while the same config requires a
feature-aware preprocessor, declares no precomputed causal feature contract
and has an empty repair-rule list. Three Dragon candidates and one Gamma
candidate failed immediately for this deterministic contradiction. Invalid
configurations consumed pool slots and then triggered finding 109.

Evidence: config lines 510-529, empty `mixed_genome_repair_rules`, and worker
logs.

### AUD-F1-20260805-114 — S3 — protected-test and fixture evidence handling

The mechanism fixture forcibly enables `evaluate_test_split=True` for all N,
E and EN arms and the delivery reports all three protected-test returns. Even
though EN was chosen from validation, exposing all three test outcomes lets
future human decisions condition on the protected period. The referenced
`eth_fixture_full/fixture_report.json` is not present in the repository and no
absolute path or report hash was supplied.

The mechanism fixture must use train/train-tail/validation only. The 2025
period is now disclosed for this curriculum comparison and must be labelled as
such; a later untouched period is required for a genuinely protected final
comparison.

Evidence: `tools/eth_curriculum_fixture.py:77,90-93,126-133`.

### AUD-F1-20260805-115 — S3 — campaign stop leaves compute workers alive

Stopping `doin-campaign-supervisor.service` on all three hosts left all four
`doin_node.cli` children running at full load because the unit uses
`KillMode=process`. Direct SIGTERM and SIGINT did not interrupt the blocking
candidate evaluations; explicit SIGKILL was required. A service reported as
inactive can therefore leave the swarm consuming GPUs and mutating state.

The supervisor needs one explicit, tested pause operation that stops and
verifies every worker process group, with a bounded graceful interval and an
observable escalation path.

### AUD-F2-20260805-116 — S3 — the claimed LTS suite is time-dependent

The reported `652 passed` is not reproducible after 12:00 UTC on the same day:
`test_report_counts_and_labels` creates a lifecycle with the wall clock, then
queries a fixed `as_of=2026-08-05T12:00Z`. After that instant, the newly
created lifecycle is in the query's future and the assertion reads zero.
Independent result: **651 passed, 1 failed**. The fixture must inject or persist
the event timestamp instead of depending on execution time.

Evidence: `lts/tests/unit/test_rolling_evidence_report.py:21,39,52-66`.

## 3. Verified Strengths

- Dataset SHA-256 is exactly
  `1b447c66e68495e826c53e2ab2b08ecd3922c8fdc735747628f8d0435ebe440f`.
- Dataset is monotonic with zero duplicate timestamps and exact split counts:
  train 13,699; validation 2,196; disclosed test 2,190.
- `normal_realistic` and train-only
  `easy_chronological_continuation` are explicit; recapitalization is recorded
  as debt and validation/test reject easy dynamics.
- The curriculum saves a post-easy artifact, warm-loads actor/critic weights,
  resets the replay buffer and then trains/evaluates under normal dynamics.
- Satoshi correctly refused to claim Gates 8 and 9: the current MT5 authority
  is still the linear model, the live 87-feature SAC observation builder does
  not exist, and no model-originated protected SAC order has occurred.
- The SAC selection scaffold is fail-closed and its eight focused tests pass;
  it has no production caller yet.

## 4. Independent Verification

```text
agent-multi full suite: 511 passed, 2 warnings
gym-fx full suite:      82 passed, 48 warnings
lts SAC focused:         8 passed, 1 warning
lts full suite:        651 passed, 1 failed, 1 warning
doin-node current equal-height unit test: 1 passed
audit reproducer:      completed; network_used=false
```

Reproducer:
`docs/audits/evidence/SATOSHI_III_ETH_CURRICULUM_REPRO_2026_08_05.py`

## 5. Runtime Containment

The three supervisors were stopped and disabled. Because worker children
survived the stop operation, all four were subsequently terminated and their
absence was verified with direct process inspection. Evidence was copied on
each host to:

`~/.local/state/agent-multi/doin-campaigns/phase-2-eth-curriculum-invalid-audit-20260805/`

This archive is diagnostic evidence, not a resumable scientific campaign.
Live/Paper trading services were not changed by this containment action.

## 6. Acceptance Boundary

Do not enable the campaign supervisors until every S1/S2 correction has an
independent reproducer and a four-worker, one-generation smoke campaign proves:
no failed candidate is champion-eligible, the authoritative ordering is
preserved, artifact namespaces are ETH-only, all workers converge to one tip,
and pause leaves no worker process behind. Gates 8 and 9 remain separately
open until a real current-stack champion traverses the exact live feature
builder and one mandatory-SL/TP Paper/Demo order lifecycle.
