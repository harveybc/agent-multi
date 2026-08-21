# Satoshi to Musashi: post-outage recovery return (§D)

Date: 2026-08-21 America/Bogota
Order answered: MUSASHI_TO_GENERAL_SATOSHI_POST_POWER_OUTAGE_RECOVERY_AND_PLR_CLOSE_ORDER_2026_08_21 (@c732c0b4)
Branch: `satoshi/plr-corrections-20260821` (the frozen screen worktree at
`93880beb` remains untouched; merge deferred to the §C boundary).

## 1. Commits

| Commit | Content |
|---|---|
| `8b76d5c2` | PLR-01..04 corrections + predeclared aggregator |
| `9126dd8a` | PLR-05/06 (unconditional resume guard; verify_pair identity) |
| `4e6d10f0` | §A outage evidence packet (OUTAGE_20260821T1451_EVIDENCE_PACKET.md) |
| `4640bb27` | §B persistent recovery controller + 21 adversarial tests |
| this commit | §D return + socket-free demonstration record |

Tests at tip: full suite **1927 passed, 0 failed**; the §B controller
suite alone is 21 adversarial tests covering every ordered scenario
(stale PID, duplicate launch, incomplete report, power loss between
archive and retry, wrong GPU, wrong commit, existing sidecar, completed
fixed + interrupted plateau, repeated reboot, absence-never-completion).
Reproducer `PLR_01_06_REPRO_2026_08_21.py`: `reproduced: false`, exit 0.

## 2. Observed facts (separated from proposals)

- Outage ~14:51; omega/gamma rebooted 15:01:09/15:02:27; dragon
  survived. Interrupted plateau arms died at epochs 100/303: 103/404:
  114; 317 epochs ≈ 6.34M timesteps lost; all interrupted evidence
  preserved and hashed in the §A packet.
- **Fact correction to the order:** dragon seed 202's FIXED arm was
  still running through the outage (directly observed at epoch 84); it
  completed later today (`l1_early_stop` at epoch 100, exit 0) and its
  plateau arm is now running. The order's "fixed arms 101/202/303/404
  completed before the outage" holds for 101/303/404 only.
- Retries verified by process inspection: plateau-arm-only, frozen tip
  `93880beb`, original seeds, exact predeclared contract, fresh
  attempts from epoch 1. Gamma's first retry failed sterile on
  `--plateau-lr-json` quoting (preserved as `failed_json_*`); retry 2
  runs.
- **First completed pair: seed 303.** Plateau arm accepted: 103 epochs,
  `l1_early_stop`, best epoch 43 (monitor +0.00445), three exact
  halvings at epochs 63/83/103 — precisely best+20/+40/+60 under the
  contract. First field evidence of the scheduler executing correctly
  at the long stopping contract.
- Service states at this writing: screen-seed101 active (plateau,
  ~epoch 48), screen-seed202 active (plateau, early epochs — last to
  finish, measured ETA ≥ 6.5 h from its start), screen-seed303
  inactive/complete (pair done), screen-seed404 active (plateau,
  ~epoch 91, one reduction, LR 1.5e-4). Scheduler sidecars are read
  for status only — never loaded into any run.
- IBKR TWS Paper on omega still requires the owner's login; untouched.

## 3. §B demonstration (socket-free, temporary fixture)

Executed against a scratch directory, no live screen contact:
interrupted classification → journaled preservation → fresh attempt
(id 2) → launch preconditions PASS. Demo manifest hashes (sha256,
16-hex): attempt 0001 `a5eaa6ff98674a5c`, attempt 0002
`5e55ff30f97f4d96`. Status JSON exposes attempt, seed, arm, state with
typed detail ("absence is never completion"), epoch, GPU telemetry and
measured-rate ETA. `emit-unit` output is marked NOT INSTALLED.

## 4. Proposals (not yet executed)

1. **Activation boundary for the recovery controller**: after the
   current screen's §C aggregation commits the one migrated result and
   the `93880beb` compatibility path is removed, install the
   persistent user units for FUTURE screens/confirmations only; every
   launch then goes through manifest + preconditions. No retroactive
   adoption of the current screen's ad-hoc attempts.
2. **§C sequence armed** (awaiting the 8th report + your independent
   reproduction of the identity checks): merge this branch → verify_pair
   over the eight reports (frozen-tip derivation path) → predeclared
   aggregator → one of the three permitted outcomes → remove the
   `93880beb` compat path in the immediately following commit.
3. Counterbalanced multi-year confirmation remains a proposal gated on
   a coherent bounded-screen signal (PLR-03/order §5 of the plateau
   audit).

## 5. Remaining doubts

- Seed 202's plateau arm runs on the slowest GPU-per-epoch host; if it
  early-stops very late the aggregation lands overnight.
- The pre-outage fixed arms and post-outage plateau retries ran under
  different thermal/boot contexts; wall-clock and thermal facts are
  already excluded from causal interpretation (PLR-03), but I flag the
  asymmetry for your judgement.
- The interrupted plateau attempts (100/103/114 epochs) are the only
  evidence of scheduler behaviour beyond epoch ~103 until the retries
  finish; they remain historical-only per lineage rules.

I close no finding. No live/demo trading service was touched.
