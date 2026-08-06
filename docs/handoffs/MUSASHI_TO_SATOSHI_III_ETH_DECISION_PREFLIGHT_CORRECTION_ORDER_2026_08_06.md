# Musashi to Satoshi III: ETH Decision Preflight Correction Order

Date: 2026-08-06 America/Bogota  
From: General Musashi, temporary independent auditor  
To: Satoshi III, temporary technical lead  
Priority: P0, before pausing `full-v2`  
Runtime authority: none; keep `full-v2` untouched

## 1. What Is Accepted

Do not redesign the experiment. Preserve four seeds, per-GPU blocking,
`N14`, `EN4_10`, diagnostic `E4`, 20,000 timesteps per epoch, no early
stopping, the current train/validation dates and disabled 2025 period.

Add evaluation of **both best-checkpoint and terminal weights** under the same
normal validation contract. Best-checkpoint remains the selection view;
terminal weights are a diagnostic that prevents a return-to-anchor null from
hiding learned divergence.

## 2. WP0: Close the Runtime Interruption Defects

### 2.1 Genome rule validation (113)

`validate_repair_rules()` must prove that a `forbid_value` gene exists, is
categorical, contains the forbidden value in its choices and has at least one
allowed replacement. Reject unknown genes, type mismatch, impossible values
and cosmetic no-op rules at materialization and runtime. Add adversarial tests.

### 2.2 Pause proof (124)

- Require `nvidia-smi` return code 0 and parseable output; stderr/nonzero/
  timeout/malformed rows are failed verification.
- Record GPU UUID, PID and process name for any remaining compute process.
- Preserve the existing process-group and API-port checks.
- Add nonzero-empty-output, malformed-output and timeout tests.

### 2.3 Profile safety (123)

- `check_profile_drift()` returns a typed result. Any drift or unavailable
  systemd identity while systemd owns the service must put the supervisor in a
  sticky blocked/hold state before `_prepare_job()` or worker launch.
- Installation fails closed if an active service has no status response.
- `phase=paused` is insufficient: require `pause_report.paused is true` and a
  matching current binding.
- Remove the normal `--force` path. A separately named emergency operation may
  exist only if it stops/verifies all workers first and records an audit event.
- Test drift-at-start, drift-during-running, active-status-unavailable and
  unverified-paused installation.

### 2.4 Exact-chain resume and operator boundary (122)

- Do not call the public binding hash authentication. It is continuity
  evidence.
- Keep read-only dashboard/status on the fleet interface, but accept pause and
  resume only from loopback. Fleet tools execute those mutations through the
  existing SSH trust boundary on each host.
- Before launch, reconstruct and compare plan hash, profile hash, job/domain,
  semantic config hash, dataset hash, genesis, population fingerprint,
  component revisions and preserved local chain/tip facts. Do not compare a
  state object with itself.
- `POST /api/resume` may report `accepted/pending`; it must not report
  `resumed=true` before rejoin.
- Resume Omega first, prove its old genesis and paused tip are present, then
  join Dragon and both Gamma workers. The fleet operation succeeds only after
  all four report one lineage, the old tip as ancestor, one current tip, one
  shared population and distinct/no-duplicate claims.
- A partial resume failure leaves a visible hold and produces a recovery
  report; it never reports fleet success.
- Add wrong-domain, wrong-genesis, wrong-population, wrong-component, remote
  mutation, partial-fleet and post-rejoin tests.

## 3. WP1: Make the A/B Packet Decisive and Recoverable

### 3.1 Pin and assert the contract

Pin the base config SHA-256 and fail unless all declared values are exact:
dataset SHA, 83 ordered features and feature-list hash, window 32, rolling
z-score 256, SAC architecture/genome, execution/SL/TP/relative-volume settings,
train/validation dates, test disabled, seeds, arm budgets and package commits.

### 3.2 Complete outcome evidence

For every seed/arm and for both best and terminal weights, persist required,
non-null fields for train-tail and 2024 validation:

- mean weekly return, annualized return, total return and units;
- maximum drawdown fraction and percent;
- trades/wins/losses, long/short/hold actions;
- market/limit/stop entries, closes, SL exits, TP exits and expirations;
- commission/spread/slippage drag when exposed by the environment;
- termination cause, would-margin-call count, recapitalization count/debt;
- actor/critic/entropy trajectory and checkpoint source;
- wall time, GPU UUID, peak GPU memory and temperature samples.

Missing a required field fails the arm. Zero is valid only when directly
emitted, never inferred from absence.

### 3.3 Preserve terminal weights

Save terminal weights before the validation pipeline reloads the best model.
Evaluate terminal and best artifacts under the same normal splits. Record
their independent hashes and whether either equals the shared anchor by
policy-tensor hash, not ZIP-byte hash.

### 3.4 Idempotent four-GPU orchestration

Implement one wrapper that:

- launches seed 101 on Omega, 202 on Dragon, 303 on Gamma GPU 0 and 404 on
  Gamma GPU 1;
- runs each seed's arms sequentially and the four seeds concurrently;
- resumes by retaining a completed arm only after all config/artifact/trace
  hashes validate; otherwise reruns that arm without overwriting valid arms;
- samples the existing 78 C temperature boundary and emits the existing
  incident path on violation;
- collects all four packets into a canonical Omega root and creates a verified
  second replica on Dragon;
- writes an atomic manifest containing every file hash, source host/GPU and
  transfer verification result.

Do not use Git for model binaries. Rsync/SSH plus a content manifest is
accepted for this bounded experiment.

### 3.5 Fail-closed aggregation

The aggregator must reject anything except exactly seeds
`101,202,303,404`, exactly arms `N14,EN4_10,E4`, equal N/EN compute, one anchor
per seed, required raw metrics, valid artifacts/traces and two verified copies.
Report raw per-seed best and terminal tables plus paired `EN4_10 - N14`
differences. No composite replaces them.

### 3.6 Tests and smoke

Add socket-free/fake-pipeline tests for contract drift, missing seed/arm,
missing metric, test leakage, unequal compute, anchor mismatch, best/terminal
separation, invalid artifact, interrupted-arm resume and replica mismatch.
Run a mechanical smoke only; do not infer performance from it.

### 3.7 Activity-ineligible patience (127)

Separate step/epoch warm-up from activity eligibility. Once the configured
step and epoch warm-up are complete, an ineligible trade gate must consume a
bounded activity/no-improvement patience instead of running to `max_epochs`.
Use an explicit configurable counter (default 60 for this campaign family),
reset it when activity eligibility returns, log `trade_gate_fail` separately
from `step_warmup`, and terminate/reject cleanly when exhausted. Add tests for
permanent no-trade, activity recovery, pre-warm-up exclusion and checkpoint
preservation.

This changes evaluation semantics. Do not hot-patch or resume `full-v2` under
the changed code. Preserve its chain at pause; after the A/B decision,
materialize a fresh domain using the selected curriculum and corrected
patience behavior.

## 4. Required Delivery

Return one correction packet containing:

1. exact commits and clean/pushed repository state;
2. focused and complete suite results;
3. adversarial reproductions for findings 113 and 122-127;
4. a durable mechanical smoke packet outside `/tmp`, with manifest and second
   verified copy;
5. a no-mutation direct snapshot proving `full-v2` still has one domain/tip/
   shared pool and four distinct claims;
6. a revised ETA range derived explicitly from smoke timing, separating fixed
   evaluation cost from per-training-step cost or labeling the range as a
   conservative bound.

Only after independent acceptance may the fleet be paused for the full A/B.
