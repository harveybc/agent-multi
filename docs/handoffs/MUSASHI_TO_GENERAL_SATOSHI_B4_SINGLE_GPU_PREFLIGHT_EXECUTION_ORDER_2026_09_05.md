# Musashi to General Satoshi: execute one bounded B4 GPU preflight

**Date:** 2026-09-05

**Return audited:** `satoshi/data-first-sota-20260826@0b85113a`

**Execution tip audited:** `446360ee`

**Prior order:** `agent-multi@61622469`

**Owner authorization:**
`OWNER_AUTHORIZATION_B4_SINGLE_GPU_PREFLIGHT_2026_09_05.json`

**Owner authorization SHA-256:**
`7426a0bfc9cdb6c609730755512c0936f58fdc45d40cd17c0bb5a83ce00cdf82`

**Order:** B4-P1 through B4-P5, then execute exactly one GPU preflight

**Execution class:** one bounded GPU mechanics-and-throughput run

## 1. Disposition

B4-E1 through E7 are accepted in their completed CPU scope. In particular:

- the economic headroom is now equal at `0.012102`;
- the v6 comparator supersedes v5 without changing its scores;
- the 12 cells contain the intended scientific recipe;
- the corrected CPU mechanics replay is accepted as non-promotable evidence;
- N4 remains closed and no feature or target search is reopened.

The owner has approved that work start. The approval means **one** bounded GPU
preflight for `o2024_seed101`, not the 12-cell campaign.

The returned runner cannot yet execute that decision. Correct B4-P1 through P4
and then run the preflight automatically under P5. No further owner decision is
needed for that single attempt.

## 2. PRE counterexamples to preserve before editing

### P1: no positive authorization path exists

`tools/b4_run_cell.py` unconditionally refuses every non-CPU device. It does
not read an authorization artifact. After that branch it also always selects
`cpu_mechanics_replay` and assigns:

```text
run_cfg["device"] = "cpu"
```

Therefore the proposed command cannot execute on GPU even after the owner says
yes. The only committed authorization test proves refusal; there is no positive
test proving that the exact owner decision enables exactly one bounded run.

### P2: a caller can self-rebind a different scientific cell

The materialization root is caller-selected. `load_cell()` accepts a cell when
its self-declared config digest and genesis-binding digest agree, but neither is
compared with an external reviewed identity.

The public reproduction changed `learning_rate` from `0.0003` to `0.123`,
recomputed the cell digest, updated `GENESIS_BINDING.json`, and called
`load_cell()`. Observed:

```text
ACCEPTED_SELF_REBOUND_CELL 0.123
```

Learning rate does not alter the zero-update tensors, so the later genesis
check would not detect this substitution.

### P3: complete-envelope verification is presence-only

In a copy of the real v6 comparator, replace the first result's
`complete_envelope_digest` with 64 zeroes and call the public comparator
verifier. Observed:

```text
ACCEPTED_FORGED_COMPLETE_ENVELOPE_DIGEST 15 99
```

The verifier checks that the field exists but does not re-derive its value from
the frozen geometry and fixed cost bytes.

### P4: the materialized GPU mode is a campaign budget, not a preflight

The current `gpu_economic` mode allows `40,020,000` environment steps,
`40,020,000` updates and 16 hours. It also uses a generic 95 C thermal value and
contains no CUDA-memory limit. Those values cannot be inherited by the single
preflight authorization.

Freeze all four reproductions as regressions against the public executing
paths. Do not replace them with source-text assertions.

## 3. B4-P1: consume the exact owner authorization

Bring the owner record into the execution branch byte-for-byte and carry its
full SHA-256 as a fixed reviewed identity. The runner must verify the digest
before parsing it and then validate an exact schema and exact primitive types.
The authorization path and expected digest may not come from CLI, environment
variables, the materialization root or the output root.

The positive path must require exact agreement with:

- decision `APPROVE_ONE_B4_BOUNDED_GPU_PREFLIGHT_ONLY`;
- cell `o2024_seed101` and config digest `fb5a92d5...`;
- cell-population, materialization and genesis-binding digests;
- v6 run-manifest, results and 99-trial-ledger digests;
- design, amendment 4 and gym-fx identities;
- headroom `0.012102` and the fixed Alpaca cost authority;
- every limit and prohibition in the owner record.

Issue append-only amendment 5 for the execution-only change. It must name
amendment 4 and the owner-record digest, state that no scientific parameter or
data role changed, and pin the final authority module, runner and tests. The
live verifier must consume the complete chain through amendment 5.

A missing, edited, self-rehashed, substituted or caller-selected authorization
must refuse before CUDA initialization, model construction or output creation.

## 4. B4-P2: bind the exact materialization and cell

Before parsing a cell, hash and compare the exact files named by the owner
record:

- `B4_CELL_CONFIGS.json`;
- `B4_MATERIALIZATION.json`;
- `genesis/GENESIS_BINDING.json`;
- the selected cell's canonical config;
- the selected zero-update genesis container and tensor identity.

Only after those external comparisons pass may the internal schemas and
digests be evaluated. A self-consistent replacement tree grants nothing.

Permanent regression: repeat the `learning_rate=0.123` reproduction with all
internal digests repaired. It must refuse on the externally pinned population
or cell identity before constructing the model.

## 5. B4-P3: make comparator envelope verification factual

For every one of the 15 comparator results, reconstruct the complete envelope
from that origin's frozen geometry and the exact fixed Alpaca cost binding.
Recompute `complete_envelope_digest` and require exact equality with the result.
Also require all 15 derived digests to equal the matching B4 cells for their
origin.

Do not trust a supplied complete-envelope digest, cost binding, count or
top-level summary. Freeze the 64-zero counterexample and changes to headroom,
commission, slippage and one geometry field. Each must refuse by the factual
field that changed.

## 6. B4-P4: implement the bounded CUDA mode

Keep CPU mechanics behavior intact. For the authorized GPU path:

1. select a distinct `gpu_preflight` runtime mode derived only from the owner
   record; never select the full `gpu_economic` campaign budget;
2. preserve every scientific value from the externally pinned cell;
3. set the requested and effective model device to `cuda:0` under one explicit
   `CUDA_VISIBLE_DEVICES` binding;
4. refuse silent CPU fallback and record PyTorch CUDA availability, effective
   device name and physical identity;
5. enforce inside the learning segment: 20,000 environment steps, 20,000 real
   optimizer updates, 7,200 seconds wall, 8 GiB host RSS, 6 GiB peak CUDA
   allocation, 87 C GPU temperature and the external stop-file;
6. read actual GPU temperature from the effective NVIDIA device. CPU thermal
   zones are not GPU telemetry. Missing or ambiguous GPU telemetry refuses;
7. emit progress/heartbeat facts at least every 30 seconds;
8. persist requested versus effective limits and actual counters;
9. train only on the 2023 calibration role, with zero 2024 scored rows and zero
   sealed-2025 rows reaching gradients;
10. classify every output and terminal model as
    `B4_GPU_PREFLIGHT_MECHANICS_AND_THROUGHPUT_ONLY`, `g1_eligible=false` and
    non-promotable.

The preflight has one learning segment of at most 20,000 environment steps. It
does not run checkpoint selection, economic scoring or any comparison against
the comparator returns.

Before dispatch, record a fresh device inventory. If another substantial CUDA
compute workload is active, required free memory is unavailable, temperature
is already above the limit or any telemetry is unavailable, return
`B4_GPU_PREFLIGHT_RESOURCE_BLOCKED` without consuming the attempt.

Once CUDA/model construction begins, the one attempt is consumed. A crash,
typed stop or failed invariant is a result; do not retry without a new order.

## 7. B4-P5: execute the one authorized preflight

After the focused battery and point-of-use verifier pass at the final tip,
execute exactly:

```text
cell: o2024_seed101
device: one explicitly bound CUDA device exposed as cuda:0
attempts: 1
```

The command may choose only cell id, reviewed materialization root, output root
and physical device binding. It may not carry scientific settings or limits.

The expected terminal label is one of:

```text
B4_GPU_PREFLIGHT_MECHANICS_AND_THROUGHPUT_ONLY
B4_GPU_PREFLIGHT_RESOURCE_BLOCKED
B4_GPU_PREFLIGHT_FAILED_TYPED
```

None authorizes another cell or promotion.

## 8. Acceptance battery

At minimum prove through the real runner that:

- no authorization, wrong digest, wrong cell and an enlarged limit refuse;
- the exact authorization enables only `o2024_seed101`;
- the self-rebound learning-rate cell refuses;
- forged comparator envelope digests and changed envelope fields refuse;
- `cuda` requested with CPU effective refuses;
- another physical GPU than the recorded effective binding refuses;
- unavailable/ambiguous temperature telemetry refuses;
- environment-step, update, wall, RSS, CUDA-memory, thermal and stop-file limits
  each stop from inside a learning segment;
- a second attempt with the same authorization refuses;
- the full `gpu_economic` mode and all other 11 cells remain blocked.

Mutation tests must strike the positive authorization and GPU execution path,
not only its refusal branches.

## 9. Return packet

Return:

1. the four PRE reproductions and their POST refusals;
2. exact owner-record verification and amendment-5 chain;
3. final code and artifact identities;
4. the positive/negative authorization battery;
5. pre-dispatch and terminal device inventory;
6. environment steps, real updates, wall time, throughput, RSS, peak CUDA
   allocation and GPU-temperature series/peak;
7. training-role and sealed-period exclusions;
8. stop-file and exact-limit evidence;
9. all terminal artifact digests and their non-promotion labels;
10. final-tip focused and ordinary-suite counts;
11. one disposition:
    `B4_GPU_PREFLIGHT_ACCEPTED_FOR_RUNTIME_AUDIT`,
    `B4_GPU_PREFLIGHT_RESOURCE_BLOCKED`, or
    `B4_GPU_PREFLIGHT_FAILED_TYPED`.

## 10. Boundaries

Authorized: the narrow CPU code/test correction needed for P1-P4 and exactly
one bounded GPU preflight described by the owner record.

Not authorized: retry, another seed/origin, the remaining 11 cells, the
47-95 GPU-hour campaign, economic selection, sealed-2025 access, checkpoint
promotion, feature/target search, collector work, venue access, service changes,
orders or positions.

All previously closed operational boundaries remain outside this order and are
not reopened.
