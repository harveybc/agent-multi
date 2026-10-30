# Pragmatic contrast deployed

2026-10-30. Runtime code `6c9f597e`, independent immutable worktree.
Identity: `f5a82347b3181ee814fe07869b9e858965d52183b287293b144378eaced4e823`.
[Manifest](../audits/evidence/pragmatic_contrast_launch_20261030/MANIFEST.json)
and [runtime tests](../audits/evidence/pragmatic_contrast_launch_20261030/TESTS.json).

## Observed launch

The actual deployed runtime passed247 tests with zero skips/errors/failures.
Its full calibration started around10:58 UTC /05:58 Bogota. At the first
timed check8/36 calibration units were complete, each containing24 controller
accounts. Mean unit wall time22.28 seconds, about156 seconds of calibration
remaining at four workers. Confirmation duration depends on the frozen
representative count and reporter; first whole-chain estimate is8-15 minutes
from launch, not a measured confirmation completion time.

The persistent user service is enabled across session restart, with four CPU
workers, CPU quota400%, nice10, memory high4GiB and maximum6GiB. No GPU needed.
First measured total memory was about0.47GiB, with zero service restarts.
Transient process failures can retry at30s, with a bounded start limit; sealed
completed units are authenticated and reused. A semantic/integrity failure
remains an explicit FAILED result rather than being silently bypassed.

## No intermediate approvals

Tests ->864 paired calibration accounts -> frozen calibration-only selection
-> up to1,080 confirmation accounts across five reporter settings -> native
metrics/report/completion manifest. All three arms, all unsuccessful candidates
and all noise settings remain recorded. Source/config mutation rejects resume.

The two-order planner is deliberately a bounded comparison. Its finite set
of evaluated plans has an exactly enumerated non-dominated subset; conservative
ties, close persistence and urgent protection additionally govern the executed
first action. Neither that action nor its receding-horizon trajectory is
claimed to be the global optimum, or always the maximizer of the finite
lookahead score after those safeguards. The report generator's generic
"exact optimality" wording must be read with this explicit qualification.

The causal baseline is a prefix-only diagnostic with three paths, not a
learned or calibrated model. Actual RL output-head qualification and sequential
floating selection are later phases, not automatically released by completion.

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python tools/run_pragmatic_contrast.py \
  --engine-root <pinned-native-engine> \
  --output <new-run-root> --workers 4
```

Same command/root resumes authenticated work. No smoke flag for this run.
Use `STATE.json` for progress and measured phase ETA, `RESULTS.md` and
`REPORT.json` for final tables, `SELECTION.json` for the pre-confirmation
decision, and per-world envelopes for complete traces and planning vectors.
