# Automated multiscale contrast

Owner direction2026-11-06: execute the approved generalization sequence without
intermediate approvals. G1 is closed in204. This contract is written before G2
results. Preserve93 columns,30 groups, fixed scaling, four-year TRAIN,2024
validation, all baseline decision rows, trading rules, attention and categorical
SAC. No BN or dropout treatment is combined into this contrast.

## Implementation and tests before dispatch

The transport tensor packs two separate13x93 windows along its temporal axis.
This is NOT a26-step TCN: the encoder splits the tensor BEFORE extraction.
The original13xstride2 path is unchanged. The new13xstride6 path uses the same
groups, width4, two residual causal TCN blocks per group and a separate16-channel
projection. Each baseline slot receives only the latest context latent whose
observed-row index is not later than that slot. No interpolation or new scaling.
Add the projected context before the original attention/core. Its projection
starts at zero, preserving the original initialization exactly. Train the new
projection and branches normally; they are not permanently gated to zero.

Unavailable context is explicitly transported as NaN, never substituted in the
source store. The model masks that WHOLE context branch; internal zero padding
only makes arithmetic safe, and its contribution is zero. The original path,
native decision row and action set are retained. Availability is known at the
current decision timestamp. Prefix invariance for internal latent slots holds
conditional on the same branch-availability state; decision-level prefix
invariance, including missingness, is unconditional.

Acceptance: original-row/support identity; unavailable-context exact baseline
output; no future observation beyond decision time; finite-suffix causal
invariance; nonzero context gradients after initialization; true SAC twin update;
checkpoint reload; replay interruption parity; real CUDA checks before training.

## Queue

1. Primary GPU: G2 seed7, full training, max2000 epochs, patience20, delta1e-8,
   best checkpoint from epoch1, patience counting after epoch20. One epoch still
   traverses all TRAIN; no new per-tick validation or duplicated training score.
2. Compare with existing attention-only seed7 history under the SAME stopping
   rule replayed over its recorded40 epochs. No seed7 control retraining. Publish
   original selected control separately; it used a different checkpoint warmup.
3. If G2 has activity>2 orders/week in BOTH periods and improves validation
   return/week over that matched-monitor control by>1e-8, automatically run
   seeds19 and31: G2 sequentially on the primary GPU; attention-only controls
   sequentially on the secondary GPU. Otherwise close the pilot report without
   promoting it or spending the replication budget.
4. Collect authenticated metrics, checkpoints, learning curves and run identities;
   compare paired seeds, returns, Sharpe, DD, order activity/directions and closes.
   A pilot win is not a replicated conclusion. Never launch feature selection
   or combine interventions automatically.
5. CPU side task: synthetic producer coverage audit over all93 selected columns.
   G3 training is not allowed to start from a single-asset generator while
   external features remain unrelated. A missing joint producer is reported as
   an engineering task, not as an approval request or a successful synthetic run.

Persistent worker services resume at completed epochs; bounded retries handle
interruptions. A persistent coordinator polls states and dispatches dependencies,
without a model conversation. Exclusive locks prevent duplicate worker/queue
execution. Private fleet bindings live outside git. Memory caps protect hosts;
readiness checks run before each GPU task. Existing unrelated services are not
touched. An unrecoverable code or identity failure is recorded and stops only
dependent work, never silently changed into zero metrics.

Resource-failure fallback: after exhausted OOM/SIGKILL retries, stop and disable
only this pilot's primary service, synchronize its run/checkpoint while quiescent,
and resume the same identity on the secondary worker. Later replications then
share that worker sequentially. Never reduce batch size, inputs or window to fit.
Generic code/identity errors are NOT treated as memory failures.

Deployment correction: CUDA acceptance and the real-data canary passed, but
anonymous replay allocation triggered the host memory guard before an epoch
completed. The guard remains enabled. Both new arms now use file-backed replay
with unchanged capacity, batch size, sampling and update cadence. Recovery loads
checkpoint tensors with mmap and preallocates file-backed storage before restoring
the replay. A regression compares interrupted file-backed training with an
uninterrupted in-memory run, including weights and monitor state. Scratch replay
files are rebuilt from durable recovery and are not copied into collected results.
A guard-induced stop can be reported by systemd as success; the coordinator now
checks the guard journal against the worker's monotonic execution interval before
classifying that stop as a resource failure and moving the pilot.

Initialization correction found by tests: TorchRL expands twin parameters in
registration order. The new context otherwise changes common head weights even
with zero context projection. Build the corresponding no-context initializer
under an isolated RNG and copy its common actor/critic/target state; retain new
context weights and restore the reference RNG continuation. This is fresh paired
initialization, not transfer from fitted weights. Both actor and critics are
tested for initial equality; additional context gradients must become nonzero.

| Requirement | Acceptance evidence |
|---|---|
| Same93 inputs and support | real support hash plus context-packing test |
| No missing-value invention | NaN transport, explicit branch mask, exact baseline test |
| Temporal extraction | two separate branches, causal alignment and perturbation tests |
| New parameters learn | two updates and context gradient test |
| Reproducibility | same-device cold reload and interrupted replay test |
| No duplicate control | frozen history replay for seed7; new seeds only after pilot gate |
| No babysitting | persistent queue, locks, restart, collected reports and terminal status |

## Deployment Receipt

Immutable runtime `633d5cc5` is deployed to both GPU workers and the CPU
coordinator. The focused CPU suites passed50 tests with12 CUDA-only skips;
the primary worker passed7 CUDA tests and the real93-column canary. The pilot
entered actual SAC updates on the primary GPU with file-backed replay.

- G2 identity: `e1fd42873ee434118e736f52ff93e334e9b131ada3e5a525711a894016b02cc1`.
- R0 replication identity: `34083774dea0f74182df32d82db7269bdc60d8cc0da1a8b202f71c4b83956f07`.
- Automatic report: `/var/tmp/sac93-generalization-queue-v3-20261106/REPORT.md`.
- Queue state and receipts: same directory, `QUEUE.json` and `receipts/`.
- Synthetic audit: same directory, `SYNTHETIC_COVERAGE.json`; no synthetic
  training is admitted by that audit.

The first attempt ended before a completed epoch and is not a performance result.
The deployed pilot starts fresh; the existing attention-only seed7 control is
reused under the matched monitor. No TEST partition is opened by this queue.

The file-backed pilot also reached the host's low-available-RAM guard during
epoch1. It is moved to the larger-RAM worker without changing its data or model.
Migration regressions now cover two operational edges: resetting a healthy
unloaded destination unit must not fail, and a copied TRAINING marker must not
be mistaken for a crashed destination before that destination has been started.
The start obligation is persisted separately after copying and before launch.
