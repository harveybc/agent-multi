# Selected70 Feed Review and Forward-Clock Repair

2026-11-08. Retsu submissions: LTS `e1ed101` (four integration corrections)
and `1cf6750` (historical70-column observation/model probe).
Musashi patch: LTS `4b5a542`, branch
`musashi/selected70-forward-clock-20261108`, based directly on Retsu's work.
This review does not modify Retsu's worktree or deploy broker services.

## Independently Reproduced

| Check | Result |
|---|---|
| Earlier F01-F04 counterexamples | WITNESSES_PASS;70 columns, unknown effect/halt survive reload, executor cash stays8500, no pre-entry stop |
| Retsu feed/account tests | 23 passed |
| Historical observation parity | 30859 rows,30234 valid, maximum error0 |
| CPU inference, final historical row | NOP; checkpoint079bef0e... preserved |
| Final CPU logits versus reported run | Maximum difference4.172325134277344e-7; same action |
| Unfiltered pinned store | 31116 rows; last bar starts2024-12-31T20:00:00Z |
| With forward-clock regressions | 30 passed |

Windows were compared across the complete historical support. Model inference
was rechecked on the two probe windows, not every historical account. These
are different scopes and must not be conflated.

## Repair Made

The new observation builder retained `test_closed_from=2025-01-01` in every
call. That is correct for replay, but prevents consuming2026 bars. The CLI also
pins the historical store, filters that split and intentionally accepts only
historical evidence. It is not a live feed driver.

LTS patch4b5a542 adds a separate `build_forward_observation` entry point:
same frozen recipe and constants; explicit decision time, expected finalized
bar and per-cell publication timestamps; no historical split cap; no fallback
to an older valid bar; future/incomplete/missing observations refused. Existing
historical behavior is unchanged. The patch does not fabricate current inputs,
fit producers or claim a running shadow account.

## Next Work Order for Retsu

Integrate commit4b5a542 (fast-forward from1cf6750 is possible while no later local
changes exist; otherwise merge normally without discarding work). Instructions
and test command are in LTS `docs/forward_model_pool/FORWARD_CLOCK_INTEGRATION.md`.

1. Bind the real observation/model callback and native entry planner to the pool.
2. Turn upstream lineage into an executable acquisition/producer manifest with
   per-column availability, exact frozen learned artifacts and warm-up/state.
3. Keep the immutable old store for replay; publish newly identified forward
   snapshots with the same model recipe and fixed TRAIN transforms.
4. Verify incremental producer outputs against batch prefixes. Twenty-five
   downstream rows are not enough to establish upstream warm-up for every family.
5. Produce the first actual hourly model decision and virtual-account event;
   then follow review222's MT5-specific preflight and authorized deployment.

The current inventory's `last_publication_bar_close_utc` is derived from a bar
close, not independently observed publication metadata for all70 sources.
Its `refresh_cadence` is null. Do not use either as proof of producer readiness.
Recover fitted Kalman/GMM/activity artifacts without silently refitting them.
No further owner model choice or VM reboot is required for this integration.

Status remains historical replay: no forward ledger event, no Paper/Demo
activation, zero submitted orders in these checks. Preserve existing services.
