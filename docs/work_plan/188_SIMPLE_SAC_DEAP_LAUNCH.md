# Simple SAC DEAP launch

Plan: [187](187_SIMPLE_SAC_DEAP_BEFORE_FLOATING.md).
Code: `136930f5`. Campaign identity:
`dc834f5e4efa368a14261f0e0e39120632490e0e33d06b7da46bf70a8386eae1`.
Started 2026-11-02 06:08:17 UTC (01:08:17 America/Bogota).

## Verified before launch

- 49 focused tests pass; correctness lint and diff checks pass.
- 19 consumed wavelet columns recomputed exactly from the pinned producer and
  source data, including upstream log/difference transformations. Nine checks
  across three parents/cuts prove prefix/future invariance and positive-control
  sensitivity. No 2024 or TEST observations read by this audit.
- Actual producer: trailing 256-bar db4 DWT, level 4, symmetric extension
  restricted to each trailing window. Extension is not future observation.
- Engineering smoke: baseline and upper-bound models; 1024 native transitions,
  four SAC updates, twelve monitoring accounts, exact checkpoint reload.
- Source, data, native engine and audit evidence hashes authenticated on both
  workers. The original pilot runtime and checkpoints were not modified.

## Execution

Population 20, maximum 2000 generations, paired seeds 7/19. Two isolated GPU
workers consume sequential queues, one fixed seed per worker. A persistent CPU
coordinator resumes completed evaluations by identity and hashes. The initial
generation contains 40 seed tasks; the unchanged baseline is first.

Both workers were observed at 4096 transitions with a real gradient update on
the synthetic qualification. Financial training follows automatically after
qualification. A failed learning qualification is an inadmissible candidate,
not a trading result of zero. Explicit nonfinite SAC loss is also recorded as
an inadmissible candidate. Broken contracts stop with a distinct error.

Per-worker resource admission checks free RAM and VRAM; scopes cap RAM at
6 GiB and swap at 1 GiB. Interrupted fits may restart once in a new attempt;
old attempts remain intact. This is NOT optimizer/replay mid-fit resumption.
Completed results are reused across generations; no unverified cache reuse.

Training patience 10 watches the mean of TRAIN and INNER_A block return/week;
DEAP has its own patience 10 on the two-seed combined score. Both scores and
counter scopes are explicit. Model checkpoints are restored before complete
TRAIN/INNER_A account reporting. The documented partitions and costs do not
change between candidates. The search does not add dropout or state feedback.

## Current interpretation

No completed DEAP candidate or new financial result existed at launch. No
completion ETA is inferred from warmup speed. Full paired fits make this a
multi-day or potentially multi-week campaign; the generation cap is not an
ETA. Report measured candidate durations as they become available.

The result is a model configuration for the next feature-selection experiment,
not a production strategy or an untouched-validation claim. Full floating
selection is NOT launched automatically; the owner receives the optimizer
results and intended evaluator first.
