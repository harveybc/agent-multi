# Audit: Satoshi return 231-233 and runtime corrections

Date: 2026-08-12 America/Bogota  
Auditor: General Musashi (independent verifier and correction implementer)  
Subject: `agent-multi@f2ffebb2`, deployed correction code
`agent-multi@374dd2eb`, simulator `gym-fx@634c3fd`  
Runtime mutation: yes, bounded to the P1LR experiment and rootless user units;
no live-broker order was submitted by this audit.

## Findings first

### AUD-F1-20260812-234 (S2): active best checkpoint rejected as terminal

**Reproduced.** The corrected screen reached 15/16 records and seed 303
failed when an active cell's final checkpoint was also its best checkpoint.
`assert_cell_record_custody()` applied the inactive-only
terminal-as-best prohibition to every record. Distinct artifact roles and
paths with identical bytes are valid for an active cell that finishes on its
selected checkpoint.

**Corrected at `agent-multi@04414417`.** The recursive prohibition is now
applied only to inactive cells. Active cells instead require a complete best
checkpoint path/hash. Exact same-bytes active and missing-identity regression
tests were added. The old identity `1d3fc9df64987fb9` remains diagnostic.

### AUD-F1-20260812-235 (S2): one nonexistent terminal transition was scored

**Reproduced from the real environment.** Backtrader applied the action on
the last input bar, returned that transition, then required one additional
action only to release its waiting thread and discover `data_end`. Consumers
counted that empty release as step 2,191 for a manifest declaring 2,190 rows.

**Corrected at `gym-fx@634c3fd` and `agent-multi@04414417`.** The bridge now
marks the final real bar terminal before waking Gym, so the episode emits one
policy transition per input row. The executing internal selector and final
outer evaluation additionally refuse any scored-step count unequal to the
verified manifest. `gym-fx` 84/84 and the affected `agent-multi` suites pass.

### AUD-F1-20260812-236 (S3): role-file drift survived the manifest cache

**Reproduced by inspection and adversarial test.** The cache key covered only
the manifest's path, mtime and size. Rewriting a materialized role CSV after
the first verification left the cached role facts trusted.

**Corrected at `agent-multi@04414417`.** Cache custody now includes every
materialized role file's resolved path, mtime and size. Any ordinary rewrite,
removal or replacement forces full hash verification before scoring. The test
primes the cache, mutates the role CSV while leaving the manifest unchanged,
and proves fail-closed refusal.

### AUD-F1-20260812-237 (S2): the decision replay allocation can exhaust Gamma

**Reproduced from the materialized configuration and framework estimate.**
The inherited `buffer_size=200000` advertises about 21.80 GiB per worker for
the current observation space. Gamma has about 15 GiB physical RAM and owns
two concurrent workers. Launching the long decision run unchanged would make
host OOM a predictable outcome, not an unattended experiment.

**Corrected in the post-audit decision profile.** A versioned,
content-addressed execution profile applies uniformly to every seed and host,
retains two complete 20,000-transition pass-equivalents in a 40,000-transition
buffer and records the profile hash in the decision experiment identity and
every materialized config. `p1lr-decision@.service` enforces `MemoryHigh=5G`,
`MemoryMax=6G` and `MemorySwapMax=1G`, turning any regression into an
observable failed unit instead of a machine crash. The screen identity and
already-running screen are deliberately unchanged. Replay capacity is not
declared optimal; it remains an explicit future experimental parameter.

### AUD-F2-20260812-238 (S2): a filled IBKR parent can trigger false recovery

**Reproduced from the live recovery chronology and a socket-trapped fixture.**
After a parent fills, TWS may remove it from the open-order view while the
position and both native protection children remain valid. The previous exact
verifier interpreted the missing parent row as protection loss, cancelled the
children, flattened the position and set a global hold.

**Corrected at `lts@ea239a4`.** The controller retains append-only direct
execution proof for the filled parent while still requiring current direct
position, account, contract, TP and SL facts on every monitoring pass. A
filled parent that remains in the completed-order cache follows the same
position-proof path; cache presence cannot bypass exposure reconciliation.

### AUD-F2-20260812-239 (S3): MT5 replay heartbeat misstates command progress

**Reproduced.** Re-evaluating an idempotent decision could relabel an existing
MT5 command as freshly queued/replayed instead of reporting its durable
`pending`, `delivered`, `succeeded` or `failed` state.

**Corrected at `lts@ea239a4`.** Runner heartbeat state now derives from the
account-bound command ledger. A successful command is called flat only after
a newer direct MT5 snapshot proves zero positions and zero orders; unknown
states fail closed. The execution bridge and model runner were restarted on
Dragon and immediately resumed monitoring the existing protected ETHUSD
position.

### AUD-SEC-20260812-240 (S2): resume capability has a byte-swap TOCTOU window

**Reproduced by adversarial test.** Capability JSON could be parsed from a
different filesystem read than the bytes authenticated by the owner's
signature, allowing a replacement between verification and use.

**Corrected at `lts@ea239a4`.** Classification captures one immutable byte
snapshot and uses it for signature verification, SHA-256 and JSON parsing;
symlinked capabilities are ineligible. The broader finding-227 correction is
also integrated: unsigned, expired, consumed and malformed side files no
longer deny one valid owner-selected signed capability.

### AUD-F2-20260812-241 (S3): live seats cannot prove their loaded model

**Reproduced from the consolidated controller inventory.** Alpaca and MT5
heartbeats exposed a model label but omitted artifact, manifest,
configuration, feature-contract and preprocessing hashes. A fresh process and
open position therefore did not prove which bytes controlled the route.

**Corrected and deployed at `lts@4fcec85`.** All three model runners publish
the exact loaded artifact/config/manifest hashes plus canonical hashes of the
ordered feature contract and scaler parameters. Independent post-restart
inventory joins Alpaca, IBKR and MT5 to their local manifests with zero
blocking reasons. Evidence:
`evidence/MUSASHI_LIVE_MODEL_IDENTITY_AFTER_241_2026_08_12.json`, SHA-256
`25746b3cbf7e1a65d3461866d066efeb4e2c97e0716fd32d94f328dc3a9a5a25`.
The final LTS suite is **686 passed**.

### AUD-F1-20260812-242 (S2): bounded target replay still loads the unbounded source replay

**Reproduced during the first decision launch and stopped before useful work.**
The decision materializer correctly emitted `buffer_size=40000`, but the SAC
warm-start path first reconstructed the archived champion with its stored
`buffer_size=200000`. Stable-Baselines therefore advertised a transient
21.80 GiB replay allocation on every worker before the bounded target even
existed. The systemd memory ceiling would contain the process, but all four
workers would fail rather than train; Gamma's two workers also made host-level
pressure unacceptable.

**Corrected before relaunch.** Source archives used only for weight transfer
now load with a one-transition replay capacity, while the independently built
target retains the exact candidate/profile capacity. The same correction
applies to observation-space expansion, whose equal-dimension branch formerly
discarded the target contract and reloaded the complete archived trainer.
Boundary evidence records both source-transfer and target capacities. Real SAC
tests prove an archived 128/200-transition source loads with capacity 1, the
target retains its requested 128/40 transitions, optimizer state remains
fresh and replay remains empty. The diagnostic decision processes were
stopped immediately and are not promotion-eligible.

## Disposition of Satoshi findings

| Finding | Independent disposition | Evidence |
| --- | --- | --- |
| 231 | verified corrected | auditor reproducer false; adversarial equal-boundary tests; exact manifest horizon now enforced |
| 232 | implementation verified, then hardened by 234 | typed inactive flow passes; active same-bytes case corrected before replacement run |
| 233 | verified corrected and deployed for screen mode | mode-aware status reports identity `886b776e022d0d7c`; screen units and idle/readiness timers loaded on all hosts |

The unmodified auditor reproducer reports all three original
`finding_reproduced` flags false. Full pre-profile `agent-multi` evidence was
**1,297 passed**; the final profile-integrated suite is **1,300 passed**.
Focused decision/profile/idle-guard evidence is **164 passed**.

## Replacement runtime

All hosts were moved to the same detached runtime revisions:

- `agent-multi@374dd2eb1262ac9dbbf073c3ce112bb9b8b28e6c`
- `gym-fx@634c3fd3c344cae3c4048b334158185c8bf4e1ef`
- contract SHA-256
  `4a4e0f16b7da0783b3a0f3d1336474e8a286ec62acc88963599e762eedd00bd6`

Independent preflight for the replacement screen passed identically on
Omega, Dragon and Gamma:

- corrected screen identity: `886b776e022d0d7c`
- pre-profile decision identity: `2f5054dc59785e2a` (superseded before launch)
- training used by preflight: false

`p1lr-screen@101/202/303/404` are systemd-managed and running on their
contract-bound GPUs. `gpu-readiness-probe.timer` and
`p1lr-idle-guard.timer` are enabled on all three hosts; the guard is bound to
screen mode until the sealed screen verdict exists. Screen-to-decision
handoff must bind the new verdict, deploy the reviewed decision execution
profile on every host, derive one new common decision identity and switch the
guard to decision mode; no old verdict or old identity may be reused.

## Four-front audit snapshot

### Front 1

Replacement run active on all four workers, one common identity, no restarts.
The first observed records landed normally. The historical Gamma-to-Dragon
replica remains independent, restartable and in progress; it is not a result
gate substitute for the P1LR terminal collection.

### Front 2

- Alpaca Paper: write-enabled; one directly observed protected SPY exposure.
- MT5 OANDA demo: write-enabled; one directly observed ETHUSD short, volume
  0.01, with nonzero native stop loss and take profit.
- IBKR Paper: connected and flat, but held after a protection-monitor recovery.
  Finding 227 and the filled-parent evidence path are corrected, tested and
  deployed at `lts@ea239a4`; the existing historical hold remains fail-closed
  until the owner signs a fresh effect-bound resume capability. No hold is
  cleared by this audit.

The recovery-correction LTS suite was **685 passed** with network sockets blocked in the new
recovery, resume and MT5 fixtures. Omega and Dragon run the same LTS revision.

### Front 3

Collector and bounded enrichment timers are enabled. Latest enrichment runs
use `opencode-go/deepseek-v4-flash`; five consecutive observed runs completed
after the last isolated failure. Publishing remains human-gated; zero drafts
were sent.

### Front 4

The correction commits and this evidence are pushed. Findings 231-233 are
independently corrected pending owner closure. Findings 234-241 are corrected
by the auditor and therefore require a separate verifier or owner disposition;
they are not self-closed here.

## Residual operational boundary

The bounded replay profile prevents the known Gamma OOM configuration; it
does not establish 40,000 as the statistically optimal replay capacity.
Future replay-capacity comparisons must be registered as experimental work,
remain uniform across seeds and preserve the sealed split contract.
