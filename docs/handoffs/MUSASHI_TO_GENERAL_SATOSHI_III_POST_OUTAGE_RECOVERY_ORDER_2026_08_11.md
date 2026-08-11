# Musashi to General Satoshi III: Post-Outage Recovery Order

Date: 2026-08-11 America/Bogota  
Priority: P0 runtime recovery, then immediate continuation of the accepted
209-223 work package  
Authority: standing owner anti-idle directive plus
`MUSASHI_TO_GENERAL_SATOSHI_III_209_223_VERDICT_AND_PHASE1_LR_ORDER_2026_08_11.md`

## 1. Role and Objective

Act as a senior Linux/NVIDIA fleet reliability engineer, distributed-systems
engineer, broker-integration engineer, machine-learning researcher and
evidence-custody implementer. Recover the four physical GPUs and three
Paper/Demo venue loops after the long electrical outage, prevent recurrence,
then continue the already-authorized phase-1 difficulty x phase-1 learning-rate
work. Do not ask the owner for a new phrase for work already ordered.

The outage did not corrupt the completed ladder: Omega and Dragon independently
recomputed the exact tree digest
`cdb6ef9947887992fc0a133a8c66adb76d64a4484cccb5cfc9f63fbea1c2ed8e`.
Preserve that collection and every legacy chain byte-for-byte.

## 2. Immediate Runtime Facts

Read first:

- `docs/audits/evidence/MUSASHI_POST_OUTAGE_RUNTIME_FACTS_2026_08_11.json`
- `docs/audits/AUDIT_SATOSHI_III_RETURN_209_220_2026_08_11.md`
- `docs/handoffs/MUSASHI_TO_GENERAL_SATOSHI_III_209_223_VERDICT_AND_PHASE1_LR_ORDER_2026_08_11.md`

At the observation time:

1. Omega's RTX 4070 is visible and healthy.
2. Dragon and Gamma booted `7.0.0-29-generic`, but only have NVIDIA 580 open
   modules through kernel `7.0.0-28`; `nvidia-smi` fails on both. The signed
   `7.0.0-29.29+1` module package is available.
3. Alpaca Paper is healthy and write-enabled with one protected exposure.
4. IBKR's runner is active and held, but TWS Paper is not listening on 7497.
5. Dragon's VM and Linux MT5 services are running, but `terminal64.exe` is not
   running in the interactive Windows session; bridge evidence is stale.
6. The old L1 factorial is complete 16/16. The old DOIN campaign is paused
   history, not an active job and not a valid successor.
7. Moltbook collection recovered after one transient HTTP 503. Preserve its
   timer and bounded retry behavior.

## 3. P0: GPU Recovery and Permanent Boot Guard

### 3.1 Current owner-assisted repair

After the owner installs the matching packages on Dragon and Gamma, verify
from an independent Omega process:

```text
uname -r == 7.0.0-29-generic
modinfo -k 7.0.0-29-generic nvidia succeeds
nvidia-smi succeeds
the expected GPU UUID set is exact
TensorFlow/PyTorch sees only the UUID assigned to each worker
no training process silently selected CPU
```

Expected fleet UUIDs:

- Omega seed 101: `GPU-612d1e0c-33de-d5cc-56eb-06c0ae424326`
- Dragon seed 202: `GPU-a8bd1b2c-26c4-f3a9-0fc0-fc3dfc6780f9`
- Gamma seed 303: `GPU-b77fc3ad-db77-b648-dc15-ec79b65e2519`
- Gamma seed 404: `GPU-a9f35631-d36a-6cc6-c23b-eb0b36d50fb8`

Do not dispatch a GPU job to a host whose expected UUID is absent. A broken
host does not stop healthy hosts from executing compatible queued work.

### 3.2 Durable prevention

Implement and test a rootless GPU readiness probe plus user systemd
service/timer integration. It must:

1. bind running kernel, `modinfo nvidia`, driver version, `nvidia-smi`, exact
   UUID set, temperature and compute visibility into one typed heartbeat;
2. inspect the newest installed boot kernel before a planned reboot and report
   whether its matching NVIDIA module exists;
3. refuse worker launch before Python imports the training framework when the
   assigned UUID is absent, the driver probe fails or CUDA falls back to CPU;
4. classify the host as `GPU_UNAVAILABLE_KERNEL_MODULE_MISSING`,
   `GPU_UNAVAILABLE_DRIVER`, `GPU_UUID_MISMATCH`, `GPU_FRAMEWORK_MISMATCH` or
   `GPU_READY`, never generic healthy/unhealthy text;
5. emit one deduplicated Telegram incident, not one message per polling cycle,
   and one recovery notice only after direct recovery evidence;
6. expose the exact remediation package for module-missing incidents without
   attempting unattended privileged installation; and
7. add tests for kernel advance without module, driver probe failure, UUID
   drift, framework CPU fallback, one-host loss and recovery.

Dispatch records must store kernel, driver, GPU UUID, framework build and
CUDA-visible-device values. Any mismatch makes that cell unavailable; it must
never be reassigned to CPU or counted as executed.

## 4. P0: Venue Recovery Without Weakening Controls

### 4.1 Alpaca

Do not restart the healthy Alpaca runner. Reproduce fresh heartbeat, direct
position/order facts and native protection facts. Preserve the current model,
account fingerprint and write-enabled Paper-only contract.

### 4.2 IBKR

The owner starts and logs into TWS Paper. The existing runner must reconnect
without duplicate submission. Reconcile direct broker position, orders and
protection facts. Keep the global hold set. Reconnection alone is not authority
to clear the hold; use the existing authenticated owner command only after
fresh flat/protected facts justify it.

Implement a post-boot TWS continuity state that distinguishes
`PROCESS_ABSENT`, `PORT_CLOSED`, `LOGIN_REQUIRED`, `SESSION_READY`, and
`RECONCILIATION_REQUIRED`. Telegram reports only unresolved transitions.

### 4.3 MT5

The owner opens the interactive Windows session, starts MT5, logs into the
OANDA demo account, enables Algo Trading and confirms only
`LtsMt5ModelBridge` is attached to ETHUSD H4. Verify version
`lts.mt5.ea.execution.v2`, `connected=true`, fresh bars, direct account mode,
position and native SL/TP facts before restoring the venue to healthy.

Add a Windows user-logon startup task for MT5, never a SYSTEM-session launch
and never plaintext broker credentials. The Linux side must distinguish VM,
guest agent, Windows session, terminal process, broker login, bridge freshness
and strategy attachment. A stale last-known protected position is not current
proof.

## 5. Resume Useful Compute

While the owner performs the privileged/interactive steps, start CPU work for
WP0, WP1 and WP3 from the 209-223 order. Do not wait for GPUs to write tests,
materializers or the typed handoff evidence.

Once all four GPU UUIDs are directly verified:

1. execute the same-artifact threshold replay from WP2;
2. complete the mechanics screen for the 2 x 2 phase-1 difficulty x phase-1
   LR factorial with seeds 101/202/303/404, one seed per exact GPU UUID;
3. run the full decision budget only if the screen's stated viability
   criterion is met;
4. never resume the paused 2026-08-06 chain; and
5. prepare the next distributed DOIN job only with one explicit v2 chain ID,
   genesis and identical component/data/config hashes on every node.

The supervisor being active does not prove useful compute. Status must name
the current package/cell/seed, assigned UUID, utilization, temperature,
checkpoint ETA and pool progress. If no training job is yet materialized,
report the exact CPU package being implemented and its ETA instead of calling
the host busy.

## 6. Acceptance Evidence

Return one packet containing:

- before/after GPU readiness facts for every host;
- exact package/kernel/driver/UUID/framework bindings;
- service/timer units and adversarial tests for the boot guard;
- direct post-recovery Alpaca, IBKR and MT5 venue facts with secrets and raw
  account identifiers redacted;
- proof that no duplicate broker action occurred across restart;
- fresh local and Dragon ladder digests matching the accepted digest;
- exact commits, clean trees and pushed refs;
- CPU package progress plus the dispatched phase-1 experiment identity; and
- any manual owner step still genuinely required.

Do not self-close findings and do not manufacture GPU utilization. General
Musashi independently reproduces the return packet.
