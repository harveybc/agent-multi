# Musashi to General Satoshi: B4 economic parity and executable preflight

**Date:** 2026-09-05

**Return audited:** `satoshi/data-first-sota-20260826@9abea1bc`

**Returned execution tip:** `e23a6b51`

**LTS tip reported:** `1587457`

**Prior order:** `agent-multi@0b4d2748`

**Order:** B4-E1 through B4-E7

**Execution class:** bounded offline CPU correction, re-materialization and one
mechanics replay

**GPU authority:** none

## 1. Disposition

Accepted and closed:

- C23-C25 and the strict owner/build bindings;
- the final N4 negative result under its reviewed identities;
- the Screen B Option-B choice and its current-execution-truth question;
- the 12 zero-update genesis constructions as mechanics evidence;
- the origin-2024/seed-101 CPU result as
  `MECHANICS_PROVEN_NON_PROMOTABLE` under the configuration that actually ran.

All previously closed custody, access and operational boundaries are accepted
by owner disposition and are outside this order. Do not reopen them.

The returned disposition
`B4_BOUNDED_GPU_PREFLIGHT_READY_FOR_MUSASHI_REVIEW` is **not accepted**. The
current disposition is:

```text
B4_CORRECTION_REQUIRED
```

No GPU cell may run from the returned command. The scientific question remains
valid; the launch surface does not yet implement the materialized experiment.

## 2. PRE findings to freeze before editing

### B4-E1: unequal economic envelopes

The rule comparator builds:

```text
entry_cost_headroom = 2 * per_side_cost + 0.006 = 0.012102
```

The B4 materializer builds:

```text
entry_cost_headroom = 2 * per_side_cost + 0.001 = 0.007102
```

`shared_execution_envelope` multiplies the exposure fraction by
`1 - entry_cost_headroom`. This is not descriptive metadata: it changes the
realized position and therefore the economic comparison. The mechanics runner
observed the divergence but then let the B4 value win.

Freeze a test that constructs the comparator and B4 effective configs from the
public paths and proves the unequal values before the correction.

### B4-E2: the proposed GPU command is not a B4 command

`tools/wp4_cpu_smoke.py` is pinned to gym-fx commit
`634c3fd3c344cae3c4048b334158185c8bf4e1ef` under the old P1 runtime. It does
not load `B4_CELL_CONFIGS.json`, a B4 cell digest, the Option-B comparator, the
superseding design, the Alpaca G1 envelope or the B4 genesis binding.

The proposed command can therefore succeed while running a different
experiment. Preserve this exact counterexample as a refusal test.

### B4-E3: the final code cannot satisfy its own sealed design

The sealed design carries:

```text
screen_b_baselines.py = 99f36b87...
materialize_b4_causal_sac.py = 3964f96f...
```

At returned tip `e23a6b51`, the corresponding file digests are:

```text
screen_b_baselines.py = d22d5fb9...
materialize_b4_causal_sac.py = 7a87a153...
```

`bind_superseding_design()` requires exact equality. Consequently the final
scoring code and the sealed design are mutually inconsistent. Amendment 3
acknowledges that no score or cell was produced under the re-pinned final code,
but no executable amendment chain currently repairs this boundary.

### B4-E4: a cell is not a complete runnable recipe

The materialized B4 cell contains environment, observation, cost, split and
seed fields, but omits material training semantics such as the agent and
pipeline identities, learning rate, architecture, entropy setting, replay
buffer, batch size, learning start, epoch size, maximum epochs, stopping rule,
selection metric and executing budgets.

`tools/b4_mechanics_cell.py` fills those values later by calling
`screen_b_baselines.base_config()` and by hard-coding additional values. Thus a
cell digest does not identify the experiment that runs, and a launch-time code
or CLI change can alter training without changing that digest.

### B4-E5: comparator verification trusts a summary

The B4 materializer currently checks only that `SCREEN_B_RESULTS.json` exists,
has the expected population label and names the same gym-fx lineage. It does
not re-derive or verify the exact 15-result population, the 99-trial ledger,
per-result digests, frozen envelope artifacts, complete execution envelope or
superseding-design identity before constructing B4 cells.

### B4-E6: authority language contradicts the actual decision

Successor artifacts mix `pending ratification` and `owner-ratified venue path`.
The actual authority is narrower: Alpaca G1 is the **Musashi-reviewed fixed
experimental cost model** selected in the prior Screen B order. The owner act
ratified observation v2 and MT5 build 6140; it did not ratify Alpaca costs.

## 3. B4-E1: one economic envelope for comparator and B4

Create and seal an amendment before producing any replacement score or cell.
Adopt the comparator's existing headroom rule for both sides:

```text
entry_cost_headroom = 2 * (commission + slippage_perc) + 0.006
```

For the frozen Alpaca contract this must be exactly `0.012102`. Do not average,
parameterize or tune the margin.

The complete effective execution envelope, including headroom and every sizing,
fill, collision, cost and protection field that can alter positions or returns,
must have one canonical digest. Every B0-B4 result and cell must carry it. A
geometry-only digest is insufficient.

Changing headroom, omitting it, changing its primitive type or carrying the old
`0.007102` must refuse before model or environment construction.

## 4. B4-E2: make each cell the complete immutable experiment

Materialize all scientific and runtime semantics into every cell's effective
config before hashing it. At minimum this includes:

- environment, strategy, agent, preprocessor and pipeline plugin identities;
- observation v2 and all data/split role identities;
- complete execution envelope and Alpaca cost contract;
- random genesis identity and explicit absence of warm-start/replay inputs;
- network architecture, learning rate, entropy coefficient, batch size,
  replay-buffer size, learning start and all SAC parameters consumed;
- epoch timesteps, maximum epochs, patience/start/min-delta and selection
  metric;
- seed and deterministic settings;
- F9.2 environment-step, real-update, wall, RSS, thermal and stop-file policy;
- output classification and non-promotion rules.

Resolve defaults at materialization. The runner must not obtain a scientific
value from another config builder, ambient default or free CLI argument.

The launch interface may select only a reviewed cell id, materialization root,
output root and physical device binding. Any attempt to override a scientific
or budget field must refuse.

## 5. B4-E3: bind the full authority chain at point of use

Issue an append-only amendment that maps the original sealed design through all
later amendments to the final executing code. Do not edit or relabel historical
artifacts.

Before any score or model construction, one verifier must establish:

1. original design digest and ordered amendment chain;
2. final `screen_b_baselines.py`, materializer and runner digests;
3. current gym-fx point-of-use manifest at `6d779af`;
4. owner-ratified observation-v2 identity;
5. fixed Alpaca G1 cost identity;
6. complete execution-envelope digest from B4-E1;
7. exact data and causal split identities;
8. exact comparator population and B4 cell identities.

At the final tip, `bind_superseding_design()` itself must pass. A test that only
parses the amendments or compares labels does not satisfy this requirement.

## 6. B4-E4: verify and re-issue the comparator population

The comparator verifier must consume the exact run manifest, superseding-design
chain, 99-trial ledger, 15 result records, frozen per-origin envelope artifacts
and all referenced digests. It must re-derive population cardinality, origin/arm
coverage, terminal state and the selected envelope for each origin. Top-level
labels and supplied counts grant nothing.

Re-run the 15 inexpensive B0-B3 rule results after sealing B4-E1 through E3 so
the current final code identity, corrected authority language and complete
envelope digest are all native to the replacement population. Preserve the
returned population unchanged as historical evidence; do not overwrite it.

No 2025 row may be opened or materialized.

## 7. B4-E5: one real B4 runner

Create one dedicated runner, for example `tools/b4_run_cell.py`, consumed by
CPU mechanics, the later bounded GPU preflight and any future B4 cell. It must:

- load one complete materialized cell and verify its digest;
- verify the full chain in sections 4-6 at point of use;
- assert the current gym-fx checkout and consumed-file manifest;
- construct the environment and SAC model only from the cell;
- reproduce the cell-bound zero-update genesis before learning;
- enforce F9.2 inside every learning segment and enforce RSS, thermal,
  stop-file and wall limits during the segment;
- journal requested and real environment steps and optimizer updates;
- classify every preflight artifact non-promotable;
- reject any hidden pretrained, replay or resume input.

`tools/wp4_cpu_smoke.py` may remain for its historical WP4/P1 purpose, but it
must never be presented as evidence for B4 and must not appear in a B4 launch
command.

## 8. B4-E6: re-materialize and replay one CPU cell

After sections 3-7 pass:

1. re-materialize all 12 cell configs and genesis bindings;
2. run only origin 2024, seed 101 on CPU through the new common runner;
3. retain the existing caps: 2,000 environment steps, 1,000 real updates,
   30 minutes wall time and 2 GiB peak RSS;
4. require an exact typed stop at the update boundary, finite parameters,
   zero scored-year rows in gradients, same-seed genesis identity and exact
   save/load roundtrip;
5. report it only as `MECHANICS_PROVEN_NON_PROMOTABLE`.

No other learning cell and no GPU is authorized.

## 9. B4-E7: refusal battery

Freeze regressions proving that the final system refuses:

- the old gym-fx commit `634c3fd3...` and the generic WP4 smoke path;
- old or changed headroom and any comparator/B4 envelope mismatch;
- a cell missing any consumed training or budget field;
- a CLI attempt to override learning rate, architecture, seed, budget or split;
- a forged comparator summary with a valid label and lineage but altered,
  missing or extra result/trial records;
- a missing, reordered or altered design-amendment chain;
- current code whose digest differs from the final amendment;
- genesis, pretrained, replay or resume identities not named by the cell;
- the labels `owner-ratified Alpaca cost` and `pending ratification` in new
  executing artifacts.

Mutation tests must strike the actual verifier and common runner, not a helper
unused by the launch path.

## 10. Return gate

Return one packet with:

1. every PRE reproduction from section 2;
2. the sealed amendment chain and final code/data/config identities;
3. the replacement 15-result comparator population and 99-trial ledger;
4. proof of complete-envelope equality between comparator and all B4 cells;
5. the 12 complete cell configs and genesis bindings;
6. the common runner's refusal battery;
7. the single corrected CPU mechanics result;
8. exact focused and final-tip suite counts;
9. one disposition:
   `B4_BOUNDED_GPU_PREFLIGHT_READY_FOR_MUSASHI_REVIEW`,
   `B4_CORRECTION_REQUIRED`, or `B4_BLOCKED`.

If and only if ready, provide one exact proposed GPU command. It may choose a
cell, output root and physical device only; the reviewed cell must supply all
scientific and budget values. Do not execute it.

## 11. Boundaries

Authorized: offline CPU correction, append-only design amendment, replacement
B0-B3 rule computation, B4 re-materialization, genesis generation and the one
bounded CPU mechanics replay.

Not authorized: GPU, full B4 economic cells, sealed-2025 access, feature or
target search, collector work, venue connection, service changes, orders,
positions, weekly-flat activation or checkpoint promotion.

The owner has no decision pending for this correction. The next owner decision
is the later, separate GPU-preflight authorization after this return passes
independent review.
