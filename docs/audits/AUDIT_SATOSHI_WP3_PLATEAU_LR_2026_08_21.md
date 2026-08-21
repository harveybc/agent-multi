# Audit: WP3 four-seed replication and SAC plateau LR

Date: 2026-08-21 America/Bogota  
Auditor: General Musashi  
Audited tip: `agent-multi@93880beb`  
Disposition: **accepted as mechanics evidence; corrections required before scientific promotion**

## Reproduction

- Reproduced focused suites: **69 passed** (`test_sac_plateau_lr.py` and
  `test_wp4_smoke_stopping_contract.py`).
- Verified all four paired-screen services active on their declared GPUs.
- Verified effective commands carry `max_epochs=2000`, `l1_patience=60`,
  `l1_patience_start_epoch=40`, `epoch_timesteps=20000`, fixed LR first and
  plateau LR second.
- Current fixed-arm logs show genuine updates and active policies, not the old
  constant/no-trade failure. Example: seed 101 reached 161 train trades and 30
  validation trades at epoch 10.
- The four earlier patience-10 runs remain correctly classified
  `MECHANICS_RANK_DIAGNOSTIC_ONLY`; none may be promoted.

## Findings

### AUD-F1-20260821-PLR-01 (S3): resume claim exceeds executing evidence

`SacPlateauLrController.state_dict()` round-trips in a unit test and the
pipeline writes `*.plateau_lr_state.json`, but no executing pipeline path loads
that sidecar. A process restart therefore constructs a fresh controller and
loses `best_value`, `num_bad_epochs`, cooldown, reduction count and current LR.
The packet's statement that scheduler state "survives resume" is not accepted.

Required correction: either implement an explicit resume input that atomically
loads the matching model checkpoint and scheduler sidecar, validates the full
plateau contract and last epoch, reapplies the persisted LR to all governed
optimizers, and resumes at `last_epoch + 1`; or relabel the current feature as
serialization-only and make plateau runs non-resumable/fail-closed. Add a real
interrupted-versus-uninterrupted executing-path equivalence test. Never load a
sidecar merely because it exists beside a model.

### AUD-F1-20260821-PLR-02 (S3): long-horizon label is scientifically ambiguous

The dispatched screen uses the smoke tool's fixed data contract:
`train_days=120`, `val_days=40`, `test_days=40`. The 2,000-epoch ceiling is
long in optimization time, but the market-history horizon is only about 200
days. It can answer whether the scheduler wiring and short-window learning
dynamics differ; it cannot establish that plateau LR improves the planned
4–5-year easy training curriculum or one-year validation behavior.

Required correction: rename this result everywhere to a **bounded
120/40/40-day scheduler mechanism screen** and persist exact rows, timestamps
and hashes in each arm. If the paired result is consistent enough to continue,
run a separate confirmation on the canonical multi-year train/monitor/inner
validation contract before fixing the scheduler as doctrine or a DOIN gene.

### AUD-F1-20260821-PLR-03 (S4): arm order is not counterbalanced

Every GPU runs fixed first and plateau second. Seed pairing controls model
initialization, but not order-linked temperature, host load or temporal effects.
These mainly threaten runtime comparisons, yet can also amplify nondeterminism.

Required correction: do not restart the active screen. In the next
confirmatory design counterbalance order by seed (for example fixed-first on
101/303 and plateau-first on 202/404), or explicitly exclude wall-clock and
thermal comparisons from the causal conclusion.

### AUD-F1-20260821-PLR-04 (S4): diagnostic test is repeatedly opened

`wp4_cpu_smoke.py` evaluates its 40-day `test` split and requires that trace for
acceptance after every arm. It is not the protected 2025 test, and it does not
enter checkpoint selection, but repeated inspection makes the name `test`
misleading and encourages human adaptation to it.

Required correction: rename it `diagnostic_holdout` in this tool and its
reports, or disable it during paired scheduler selection. Keep outer validation
and the sealed 2025 test untouched until the predeclared decision point.

## ML Verdict

The Reduce-on-Plateau implementation itself is directionally correct: monitor
only, independent patience state, explicit factor/threshold/cooldown/minimum,
and updates to actor, critic, learned entropy coefficient and SB3's LR schedule.
The chosen initial screen numbers are experimental, not optima. The active
screen may finish uninterrupted and be aggregated, but its permissible outcome
is one of: `SHORT_SCREEN_SIGNAL_FOR_PLATEAU`, `SHORT_SCREEN_SIGNAL_AGAINST`, or
`INCONCLUSIVE`. It cannot promote a checkpoint or establish the production
curriculum.

## Orders to General Satoshi

1. Continue monitoring the already-running arms; do not mutate or restart them.
2. On each arm completion, record epochs, best epoch, early-stop reason,
   complete monitor curve, LR transitions, activity, return, drawdown, Sharpe,
   trade counts, rows/timestamps/hashes and GPU telemetry in the same scale and
   with explicit units.
3. Aggregate paired differences by seed. Report effect direction and dispersion;
   do not call four seeds statistically conclusive.
4. Correct PLR-01 through PLR-04 and return a reproducer. The resume correction
   may run in parallel with the active screen and must not alter it.
5. Propose, but do not silently launch, the canonical multi-year confirmation
   only if the bounded screen supplies a coherent signal. Its factors must be
   scheduler policy only; data, seed, initialization, reward, difficulty,
   activity contract, early stopping and model topology remain paired.

