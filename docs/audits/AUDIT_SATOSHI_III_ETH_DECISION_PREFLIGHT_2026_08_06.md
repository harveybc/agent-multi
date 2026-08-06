# Audit: Satoshi III ETH Curriculum Decision Preflight

Date: 2026-08-06 America/Bogota  
Auditor: General Musashi, temporary independent auditor  
Delivery reviewed: `docs/handoffs/SATOSHI_III_ETH_DECISION_PREFLIGHT_2026_08_06.md`  
Delivery head: `agent-multi@b6f6c351`  
Runtime mutation by this audit: none

## Verdict

**ACCEPTED IN EXPERIMENTAL DESIGN; EXECUTION NOT YET AUTHORIZED.**

The primary comparison remains correct and is not to be redesigned:

- `N14`: 14 normal epochs;
- `EN4_10`: 4 easy plus 10 normal epochs with a fresh replay buffer at the
  boundary;
- `E4`: 4 easy epochs followed by normal-condition inference only;
- four fresh seeds, one seed per GPU, with all arms run sequentially on the
  same GPU;
- 20,000 timesteps per training epoch, no early stopping, 2017-2023 training,
  2024 validation and no 2025 selection or metric payload.

The smoke proves that the three code paths execute and share one anchor. It
does not yet prove that an overnight packet would be complete, interpretable,
recoverable or that `full-v2` can be paused and resumed without false claims.

## Findings

### AUD-F1-20260806-122 (S2): resume is neither authenticated nor an exact-chain proof

`request_resume()` binds domain, genesis, population fingerprint and component
versions into the pause record, but recalculates only `plan_hash` and the profile
file hash before accepting resume. An independent reproducer changed the
persisted domain and component versions after pause; resume still returned
`resumed=true` and changed the phase to `starting`.

The binding hash is returned by unauthenticated `/api/pause`; `/api/resume` is
also unauthenticated and the supervisor listens on `0.0.0.0`. It is a continuity
token, not authentication. Finally, `resumed=true` is returned before any worker
has rejoined the old genesis, tip or shared population.

Evidence:

- `app/campaign_supervisor.py:3250` records the broad binding;
- `app/campaign_supervisor.py:3409` verifies only plan/profile;
- `app/campaign_supervisor.py:3425` reports success before worker rejoin;
- `app/campaign_supervisor.py:3501` exposes mutation routes without an
  authorization boundary;
- `examples/campaigns/phase_2_eth_anchored_full_fleet_v2/omega_profile.json`
  sets `listen_host=0.0.0.0`.

This keeps finding 121 open and blocks the fleet interruption.

### AUD-F1-20260806-123 (S2): profile drift is alerted but not blocked and installation fails open

`check_profile_drift()` only emits an alert and returns `None`; `tick()` then
continues through validation and worker launch. Its message says restart is
blocked, but no block exists. The installation tool also proceeds when systemd
is active but its status endpoint is unavailable, accepts `phase=paused`
without requiring `pause_report.paused=true`, and exposes a `--force` bypass.

Evidence:

- `app/campaign_supervisor.py:2934` ignores the drift-check result;
- `app/campaign_supervisor.py:3445` implements alert-only behavior;
- `tools/install_campaign_profile.py:52` through `:58` guards only when a
  status payload was successfully returned.

This keeps finding 119 open.

### AUD-F1-20260806-124 (S3): a failed NVIDIA query is still accepted as GPU-clear

The pause path reads `stdout` without checking `returncode`. A reproduced
`nvidia-smi` result with return code 1, empty stdout and an error on stderr
produced `paused=true`, `gpu_owner_pids_remaining=[]` and no failure reason.
The existing test covers only a missing executable exception.

Evidence: `app/campaign_supervisor.py:3312` through `:3337` and
`tests/test_operator_pause.py:169`.

This keeps findings 115/121 open.

### AUD-F1-20260806-125 (S3): the decision packet cannot yet explain a null or a margin effect

The validation pipeline reloads the best checkpoint before final evaluation;
the runner does not preserve or evaluate terminal weights. The smoke selected
the common anchor for every arm, so all three reported identical validation
outcomes despite non-identical training paths. Without terminal-weight
evaluation, a full run can manufacture an uninterpretable null by returning to
the anchor.

Normal final summaries also expose `would_margin_call_count`, recapitalization
and termination cause as absent/null. `_raw()` silently omits absent metrics.
The packet therefore cannot distinguish zero margin events from missing
instrumentation and cannot apply the promised attribution/ablation rule.
Action/order diagnostics are likewise omitted from `splits_raw` despite the
declared evidence contract.

Evidence:

- `pipeline_plugins/rl_pipeline_with_validation.py:1306` reloads the best
  artifact before final evaluation;
- `tools/eth_curriculum_decision_experiment.py:59` declares optional extraction
  and `:106` silently omits missing values;
- reproduced smoke records under Satoshi's temporary `decision_smoke2` root
  contain null normal margin fields.

This keeps findings 114/120 open.

### AUD-F1-20260806-126 (S3): the four-GPU run is not yet an idempotent, complete evidence transaction

The runner hashes the dataset but reads an unpinned mutable base config and
does not assert the 83-feature/window/scaling/execution contract described in
the preflight. No automated tests were added for the new runner or aggregator.
The aggregator accepts any number of seeds and missing arms without failing.
The documented rsync replica is not implemented by any wrapper, and the smoke
exists only inside a temporary Claude scratch directory. A host restart would
also rerun and overwrite completed arms rather than validating and retaining
them.

Evidence:

- `tools/eth_curriculum_decision_experiment.py:42` through `:47` uses a mutable
  base file without an expected hash;
- `tools/aggregate_curriculum_decision.py:25` through `:44` has no completeness
  gate;
- repository search found no decision-fleet/replica wrapper and no runner tests;
- the smoke packet was reproduced only under `/tmp/claude-1000/.../decision_smoke2`.

This blocks execution but not the accepted experiment design.

### AUD-F1-20260806-127 (S2): activity-ineligible epochs never consume patience

The active fleet exposed a separate resource/liveness defect. Once an epoch
fails the trade gate, `_update_l1_checkpoint_state()` returns without
incrementing `no_improve`. The log renderer then labels every ineligible epoch
as step warm-up even after millions of timesteps. At the audit snapshot all
four active candidates had `trade_gate=FAIL` and no patience progress:

- Omega epoch 127, 1,016,000 timesteps;
- Dragon epoch 270, 2,160,000 timesteps;
- Gamma 5070 Ti epoch 204, 1,632,000 timesteps;
- Gamma 5090 epoch 199, 1,592,000 timesteps.

With `max_epochs=2000`, a collapsed/no-trade candidate can consume the entire
budget while contributing no trainable checkpoint. The already saved warm
baseline protects result integrity, but does not justify the wasted updates.

Evidence: `pipeline_plugins/rl_pipeline_with_validation.py:207` through `:224`
and `:1154` through `:1249`, plus direct four-worker logs.

This does not mutate or invalidate already evaluated `full-v2` candidates. It
does mean the correction is a domain-semantic change: after the A/B decision,
the recommended production campaign is a fresh domain rather than silently
mixing corrected patience behavior into `full-v2`.

## Appended State of Prior Findings

| Finding | State after this audit | Basis |
| --- | --- | --- |
| 113 | `partially_corrected_open` | the real v2 rule executes, but schema validation accepts unknown genes and forbidden values that are not choices, so cosmetic rules still pass |
| 114 | `partially_corrected_open` | smoke artifacts exist locally; findings 125/126 prevent an accepted durable packet |
| 115 | `partially_corrected_open` | process-group stop works; finding 124 defeats GPU-clear proof |
| 119 | `partially_corrected_open` | an atomic installer exists; finding 123 defeats the blocking claim |
| 120 | `open` | the four-seed comparison has not run |
| 121 | `partially_corrected_open` | continuity token and sticky pause exist; findings 122/124 defeat authenticated exact-chain resume |

## Independently Reproduced Strengths

- `agent-multi`: **554 passed**, two pre-existing sklearn convergence warnings.
- `forbid_value` executes for the current v2 rule before environment creation.
- The mechanical smoke completed all three arms with exit 0, one common anchor,
  train/train-tail/validation traces, no 2025 metrics and loadable local model
  artifacts.
- `N14` and `EN4_10` have equal primary training budgets: 280,000 timesteps.
- `E4` is clearly labeled diagnostic and is not compared as equal compute.
- Direct fleet evidence during audit showed `full-v2` untouched: all four
  workers on domain `trading-asset-policy-eth-4h-anchored-full-v2`, one tip
  `22e0f314...`, four distinct claims and no supervisor alerts.

## Disposition

Satoshi must execute
`docs/handoffs/MUSASHI_TO_SATOSHI_III_ETH_DECISION_PREFLIGHT_CORRECTION_ORDER_2026_08_06.md`.
The current `full-v2` campaign remains running. No pause and no decision run is
authorized until the correction packet is independently reproduced.
