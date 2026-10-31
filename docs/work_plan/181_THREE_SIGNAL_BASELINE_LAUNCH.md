# Three-Signal Baseline Launch

2026-10-31. Implements the owner-approved baseline in
[plan180](180_THREE_SIGNAL_SAC_BASELINE.md), before phase-2 floating selection.

## Immutable Execution

- Code: `606f3c6b`, published on `origin/master` before deployment.
- Campaign: `73892e698ce7a20c4a82d17eef047414ea199212860d38d03d9cdc5b7306e72d`.
- Parent input campaign: `a3db51b29e88560464d8004e903d9a0c52746e9d3880d11ef8f2633d16f36c13`.
- Primary accelerator: RTX5090, seeds7 then43, started18:29:58UTC.
- Secondary accelerator: RTX4090 Laptop, seed19, started18:29:59UTC.
- CPU collector started concurrently. Three persistent services are active,
  enabled across restarts, and initially have zero service restarts.
- One training process per host,6GiB service RAM limit. Resource admission
  requires4GiB available host RAM and3GiB free GPU memory before a worker starts.
- Per-seed sequence: sequential synthetic economic qualification, fresh
  financial SAC training, best-checkpoint reload, full TRAIN/INNER_A accounts,
  sealed trade/equity/output artifacts, then the next assigned seed.
- At most two process attempts. A contract failure is terminal for that unit;
  other assigned seeds continue. Interrupted fits restart fresh in a new attempt,
  not from a partial replay buffer. Completed authenticated units are reused.
- Collector authenticates expected campaign/seed partition and artifact hashes;
  it produces RESULTS.md, AGGREGATE.json, AUDIT_REQUEST.md and CLOSE.json.

No new tuning, phase-2 selection, strategy promotion or real trading is queued.
The smallest model is a baseline, not a claim that its topology is optimal.

## Prelaunch Evidence

26 focused tests pass (three-signal baseline, native environment, doubleTCN).
Scoped Ruff check and git whitespace check pass. Read-only review identified
and corrected a final close-diagnostic variable collision, missing flat-market
learning coverage and an inactive-policy patience exception. Both close
directions and the no-eligible-checkpoint outcome now have regressions.

CPU smoke and final-code RTX5090 smoke each executed128 real transitions and
8 SAC updates, exact checkpoint reload, and all6 chronological monitor blocks.
Maximum cash discrepancy was5.30e-12USD. Monitoring preserved training state
and RNG. These smokes are engineering checks, not trained financial candidates.

## Deterministic Synthetic Controls

The following use the known causal regime directly, NOT a trained network.
They verify that the same three-signal decoder can express the desired behavior
under the native economic engine. They do not establish financial performance.

| World | Net USD | Return % | Orders | Buy / sell | Early closes |
|---|---:|---:|---:|---:|---:|
| 91001 | 116.8866 | 1.1689 | 3 | 1 / 2 | 2 |
| 91002 mirrored | 144.5073 | 1.4451 | 3 | 2 / 1 | 2 |
| 91003 | 155.9946 | 1.5599 | 4 | 2 / 2 | 3 |
| Flat / NOP | 0 | 0 | 0 | 0 / 0 | 0 |
| Flat / forced BUY | -15.0884 | -0.1509 | 1 | 1 / 0 | 0 |

At18:30:35UTC the primary had4,864 native decisions and193 SAC updates;
secondary had5,120 decisions and257 updates. Both were in sequential synthetic
learning, with finite losses. No learned financial performance exists at this
snapshot. The financial fit starts only after qualification passes.

## Result Interpretation

120 phase-1 columns,31 groups,13 past observations at stride2; fixed TRAIN
scalers; market-only doubleTCN with branch width8 and core width16. One
categorical NOP/BUY/SELL actor, twin three-action critics. Fixed6h risk profile,
opposite signal closes only. No state features or rolling normalization.

Maximum1,048,576 financial decisions per seed, minimum131,072 before stopping,
evaluation every32,768, patience10. Monitor equally weights mean TRAIN-block
and internal-validation-block return per week. Activity is admissible only
above2 orders/week on average in each half, without an activity reward.

Gradient/replay period is2020..2023H1;2023H2 is repeatedly monitored internal
selection. Upstream phase-1 selection already saw2023H2, so these results are
development only. Neither2024 nor TEST is read. The final tables must report
these partitions explicitly rather than label internal selection as untouched
validation. Both active-best and unrestricted-best checkpoints are retained;
an inactive outcome is not promoted or relabeled as an active success.

The initial post-warmup throughput suggests hours, not minutes, for the
financial fits. Completion depends on qualification and patience; the hard
budget is a ceiling, not a promised stopping time. Progress records contain
measured decisions, updates, episode count and elapsed seconds for later ETA.
