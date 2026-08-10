# IBKR Paper Reconciliation Packet — 2026-08-10

Prepared under the four-front correction order
(`docs/handoffs/MUSASHI_TO_GENERAL_SATOSHI_III_BLOCKCHAIN_AND_FOUR_FRONT_CORRECTION_ORDER_2026_08_10.md`,
WP4 item 2). This packet PREPARES evidence and the existing authenticated
owner command. It performs **no broker action**: no order was placed,
cancelled or modified; the hold was **not** cleared; no service was
restarted. Clearing the hold is exclusively an owner action behind the
Ed25519 signature boundary.

The live model on this seat is a linear CONTROL (baseline controller),
not a champion.

## 1. Current state facts (each with source and timestamp)

| # | Fact | Value | Source | As of (UTC) |
| - | ---- | ----- | ------ | ----------- |
| 1 | Execution halt state | `halt = hold` | `~/.local/state/lts/ibkr-model-execution.sqlite`, table `service_state`, key `halt` | read 2026-08-10T08:05Z |
| 2 | Hold origin | effect `l1e-f77144cb7c3af7d4` (bracket_entry): `protection_health_failure` with failures `take_profit: status 'Filled' not acceptable`, `stop_loss: status 'Cancelled' not acceptable`; `recovery_hold {"halt": "hold"}`; then `recovery_terminal {"state": "terminal_cancelled"}` | same ledger, `l1_broker_facts` seq 1170/1171/1172 | 2026-08-05T14:46:32.895 / .899 / :33.706 |
| 3 | Broker position (direct TWS, read-only observer preflight) | `open_positions = 0`, `open_orders = 0` — **flat** | `~/.local/state/lts/paper-execution-watchdog/latest.json` → `ibkr.latest_complete`, session `preflight-79239bb906ee4f40` | reconciliation_observed_at 2026-08-10T07:57:36.168979Z |
| 4 | TWS continuity | `tws_healthy = true`, `exposure_state = flat`, `n_restarts = 0` | `~/.local/state/lts/tws-continuity-monitor/state.json` | checked_at 2026-08-10T08:05:08Z |
| 5 | TWS socket | available, `127.0.0.1:7497`, connect_errno 0 | watchdog `latest.json` → `ibkr.socket` | generated_at 2026-08-10T08:02:41.706Z |
| 6 | Execution runner | active, `read_only = false` (write authority live), `state = decided`, freshest decision `rejected`, reason `halted:hold`, replayed | `~/.local/state/lts/ibkr-model-runner-heartbeat.json` (`lts.ibkr.model_runner.heartbeat.v1`) | observed_at 2026-08-10T08:01:53.013Z |
| 7 | Rejected due-bar decisions since the hold | 18 (first 2026-08-05T14:46:35Z, last 2026-08-10T08:00:51Z) | execution ledger, `decisions` where `outcome='rejected'` | read 2026-08-10T08:05Z |
| 8 | Account binding | fingerprint (masked) `0123456789abcdef`, `account_binding_verified = true`, environment `paper` | runner heartbeat + runner config `service.account_fingerprint` | observed_at 2026-08-10T08:01:53.013Z |
| 9 | Prior resume history | one prior owner resume at 2026-08-05T06:48:26.586Z for effect `l1e-d40d00c1ef40cda1`, evidence_sha256 `00f72c0a…93418` | execution ledger, `service_state` key `last_resume` | read 2026-08-10T08:05Z |
| 10 | Owner signer pin | `/etc/lts/resume_allowed_signers` present, `root:root`, mode 0644, 116 bytes | filesystem stat | modified 2026-08-05 01:47 local |
| 11 | Capability store | exactly one pair present: `~/.config/lts/ibkr-resume-capabilities/resume_09d5e9d7.json` + `.sig`, both dated 2026-08-05 01:48 local — this pair predates the current hold and belongs to the already-consumed 2026-08-05 resume (single-use nonce burned); it cannot clear the current hold | directory listing | read 2026-08-10T08:05Z |
| 12 | Model inference at the held seat | model `usdcad-4h-linear-live-v1` (linear CONTROL), action short, probability_up 0.49416, artifact `dc95edcb…c0fd`, input `24256b5e…607d`, output `d05eeef7…8c06` | runner heartbeat `inference` block, observed_at 2026-08-10T08:07:00.671Z | last_closed_bar 2026-08-10T04:00:00Z |

Facts 3 and 4 agree with fact 2's terminal state: the venue is **flat and
held**. No contradiction between direct TWS evidence and the ledger was
observed at assembly time.

## 2. The existing authenticated owner resume/hold-clear command

Verbatim per-resume flow from
`/home/harveybc/Documents/GitHub/lts/docs/security/OWNER_RESUME_SIGNER_SETUP_2026_08_05.md`
§2 and the CLIs `/home/harveybc/Documents/GitHub/lts/tools/mint_resume_capability.py`
and `/home/harveybc/Documents/GitHub/lts/tools/ibkr_resume_after_reconciliation.py`.
All three steps demand an interactive owner terminal; the sign step
demands the owner key passphrase no agent knows. Working directory:
`~/Documents/GitHub/lts`.

Step 1 — mint ONE single-use resume capability bound to the effect whose
recovery created the current hold:

```bash
python tools/mint_resume_capability.py \
    --profile ~/Documents/GitHub/lts/examples/configs/ibkr_usdcad_model_profile_v1.json \
    --resume-of-effect-id l1e-f77144cb7c3af7d4
```

(The tool prompts for the confirmation phrase
`resume ibkr paper after reconciliation` and prints the minted path,
`~/.config/lts/ibkr-resume-capabilities/resume_XXXXXXXX.json`.)

Step 2 — sign the minted capability with the owner key (passphrase
prompt is the human boundary; namespace must be `lts-ibkr-resume`):

```bash
ssh-keygen -Y sign -f ~/.ssh/lts_owner_resume -n lts-ibkr-resume \
    ~/.config/lts/ibkr-resume-capabilities/resume_XXXXXXXX.json
```

Step 3 — execute the bounded, fail-closed resume transition:

```bash
python tools/ibkr_resume_after_reconciliation.py \
    --config examples/configs/ibkr_usdcad_model_runner_v1.json
```

The CLI itself re-verifies, before anything else: root-pinned signature
over the exact capability bytes; exactly one capability in the store;
fresh direct read-only TWS evidence (positions, open orders, account
fingerprint) gathered immediately before the transition; no active
P0/P1 incident for `ibkr_paper`; single-use nonce burn, ≤15-minute
validity and exact venue/account/instrument binding inside one
`BEGIN IMMEDIATE` transaction (findings 093/094).

## 3. Pre-execution checklist (owner verifies before Step 1)

1. Watchdog `ibkr.latest_complete` is fresh (≤10 min) and still shows
   `open_positions = 0`, `open_orders = 0`.
2. `~/.local/state/lts/tws-continuity-monitor/state.json` shows
   `tws_healthy = true`, `exposure_state = flat`.
3. `/etc/lts/resume_allowed_signers` still `root:root`, mode 0644,
   non-empty.
4. Incident ledger has no active P0/P1 for `ibkr_paper`
   (`python tools/incident_ledger.py status --active --severity P0,P1 --json`
   in `~/Documents/GitHub/agent-multi`).
5. Dispose of the stale, already-burned pair
   `resume_09d5e9d7.json`/`.sig` (owner decision — the resume CLI
   refuses when the store does not contain exactly one capability, so
   the fresh mint must be the only file pair present).
6. Mint within 10 minutes of intended execution (default validity 600 s).

## 4. Post-execution verification checklist

1. CLI printed a sanitized result with `"applied": true` and an
   `evidence_sha256`.
2. `service_state`: `halt` is no longer `hold`; `last_resume` now
   references `l1e-f77144cb7c3af7d4` with a fresh timestamp and new
   `evidence_sha256`/`nonce_sha256`.
3. `l1_broker_facts` gained a fresh `resume_evidence` fact whose
   embedded broker evidence shows zero positions and zero open orders.
4. Incident ledger contains the `ibkr_hold_cleared` (P3) observe/recover
   pair reported by the CLI.
5. Next 4h due bar: runner heartbeat `decision.reason` is no longer
   `halted:hold` (a decision may still be legitimately rejected for
   other typed reasons; only the hold reason must disappear).
6. First new protected lifecycle after resume shows native SL and TP
   acknowledged before any fill (L1 contract), and the audit snapshot's
   broker authority block still reports both `observer_read_only` and
   `execution_write_enabled` truthfully (finding 205 fix).

## 5. What this packet does NOT do

- It does not clear the hold, mint, sign, or execute anything.
- It does not modify any ledger, heartbeat, or broker state.
- It contains no credentials; account identity appears only as the
  masked fingerprint already used by the runtime evidence.
