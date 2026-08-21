# Outage evidence packet — power loss 2026-08-21 ~14:51 America/Bogota

Order: MUSASHI_TO_GENERAL_SATOSHI_POST_POWER_OUTAGE_RECOVERY_AND_PLR_CLOSE_ORDER_2026_08_21 §A.
Recorded by: General Satoshi III, from direct host inspection (not relayed
statements). GPU identities are 8-hex-truncated per the sensitivity gate;
full UUIDs and raw telemetry stay host-local.

## 1. Host survival and boot times

| Host | Fate | Boot time (uptime -s) |
|---|---|---|
| omega | REBOOTED | 2026-08-21 15:01:09 |
| gamma | REBOOTED | 2026-08-21 15:02:27 |
| dragon | SURVIVED | 2026-08-11 15:24:27 (unchanged) |

Outage instant bounded by the last interrupted telemetry sample on omega:
2026/08/21 14:51:10 (5-minute cadence).

## 2. Last durable epoch before loss, per interrupted arm

| Arm | Last epoch in interrupted log | Interrupted log sha256 |
|---|---|---|
| seed101 plateau (omega, GPU-612d1e0c…) | 100 | e6e006c9e7dd0c5434736ad8a4bcbecef6b1689f00fde336b5dfc24072b4b157 |
| seed303 plateau (gamma, GPU-b77fc3ad…) | 103 | f7c456ebe2e5b03e00f256521ba5dc57e1ea796536d2ca4ab1ecc0eb2f9a2db9 |
| seed404 plateau (gamma, GPU-a9f35631…) | 114 | d54fef256a0d6f1cc0eea6685d3267ef66cc9339b1eb0eaf72a773fd78e59fc3 |

No plateau report.json existed for any interrupted arm (reports are written
at arm completion); the partial output directories, logs and telemetry were
preserved under `*_interrupted_power_20260821T145x*` and were NOT resumed.

## 3. Completed fixed reports (preserved, useful work)

| Report (host-local under ~/.local/share/agent-multi/plateau_screen_20260821/) | sha256 (16-hex prefix) |
|---|---|
| seed101_fixed_report.json (omega) | ee90647fb42d7478 |
| seed303_fixed_report.json (gamma) | 3663c48ba7ec79d2 |
| seed404_fixed_report.json (gamma) | d359800d48950404 |

**Observed-fact correction to the order's §Current facts:** dragon's
seed 202 fixed arm had NOT completed before the outage and is STILL
RUNNING it — direct evidence at inspection time (~15:3x): log line
`[epoch 84/2000] L1 44/60 … best=+0.0054`, no `ARM_FIXED_EXIT` marker,
no seed202_fixed_report.json on disk. Dragon survived, so nothing was
lost there; the arm simply continues. Its plateau arm has not started.

## 4. Interrupted-attempt evidence (historical, never a continuation)

Preserved on their hosts, atomically renamed before retry:
- omega: `seed101_plateau_interrupted_power_20260821T1451/` (partial outputs),
  `seed101_interrupted_power_20260821T1451.log`,
  `seed101_gpu_telemetry_interrupted_power_20260821T1451.csv`
  (span 10:06:10 → 14:51:10, covers fixed arm + interrupted plateau).
- gamma: same pattern with `20260821T1454` for seeds 303 and 404; plus a
  FAILED first retry on both (`*_failed_json_20260821T1510*`): the relaunch
  command mis-quoted `--plateau-lr-json` and the tool refused at argument
  parsing with `json.decoder.JSONDecodeError` before any training
  (ARM_PLATEAU_EXIT=1). That failed attempt is preserved as its own
  historical evidence; retry 2 followed with corrected quoting.

## 5. Active retry attempts (fresh attempts, epoch 1)

All verified by direct process inspection: frozen tip `93880beb` worktrees,
original seeds, exact predeclared plateau contract
`{factor 0.5, lr_patience 20, min_lr 1e-6, threshold 1e-6, cooldown 0,
start_epoch 40}`, stopping contract `--l1-patience 60
--l1-patience-start-epoch 40 --max-epochs 2000 --epoch-timesteps 20000`,
monitor selection metric, **plateau arm only — no completed fixed arm is
being rerun**.

| Arm | Unit / invocation | Started | GPU |
|---|---|---|---|
| seed101 plateau retry | screen-seed101 / 141c68ea | 15:09:25 | GPU-612d1e0c… |
| seed303 plateau retry 2 | screen-seed303 / 0c8977a2 | 15:11:24 | GPU-b77fc3ad… |
| seed404 plateau retry 2 | screen-seed404 / 0136c7da | 15:11:24 | GPU-a9f35631… |
| seed202 fixed (uninterrupted) | screen-seed202 | 10:06:13 | GPU-a8bd1b2c… |

## 6. Lineage declaration

The interrupted attempts are HISTORICAL EVIDENCE ONLY. Each retry is a
FRESH ATTEMPT from epoch 1 — never a continuation: no model weights, no
replay, no scheduler state, no history rows cross the boundary (the
plateau controller is non-resumable by contract, PLR-01/PLR-05).
Histories are not merged. The aggregator cannot discover both: interrupted
directories carry no report.json, carry the `interrupted_power_*` /
`failed_json_*` names outside the `seed<S>_<arm>_report.json` pattern the
aggregator reads, and verify_pair refuses nonaccepted/incomplete arms.

## 7. Compute lost vs useful work

Lost (plateau attempts, epoch-1 restarts): 100 + 103 + 114 = 317 epochs
≈ 317 × 20,000 = 6.34M timesteps. Wall-clock lost ≈ the plateau portions
of 10:06→14:51 minus each fixed arm (~3.5-4.5 h per GPU), plus 19-20 min
of reboot+relaunch gap (14:51 → 15:09/15:11) and one sterile failed retry
(15:10, seconds, no training). Useful completed work preserved: fixed
reports 101/303/404 (§3), dragon 202 fixed progress (epoch 84, no loss),
all interrupted evidence and telemetry.

## 8. Live-service note

IBKR TWS Paper on omega requires OWNER reopening/login after the reboot —
broker authority is the owner's, not the recovery path's. No live/demo
trading service was touched by this recovery inspection.
