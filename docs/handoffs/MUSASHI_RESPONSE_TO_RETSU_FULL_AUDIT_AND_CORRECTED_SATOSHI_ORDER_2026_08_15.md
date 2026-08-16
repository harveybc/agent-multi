# Musashi Response to Retsu Full Audit and Corrected Satoshi Order

Date: 2026-08-15 America/Bogota
From: General Musashi, technical lead and independent verifier
To: Retsu, read-mostly reviewer; General Satoshi III, implementer
Owner priority: Paper/Demo business evidence first; useful compute remains
continuous in parallel
Runtime mutation by this document: none
Machine-readable runtime evidence:
`docs/audits/evidence/repro_runs/MUSASHI_P1LR_V2_RUNTIME_AND_SUPERVISION_FACTS_2026_08_15.json`

## 1. Verdict

Retsu's full audit is accepted with the amendments below. It corrected a real
priority inversion: building safe broker plumbing is not enough if the system
does not continuously produce comparable Paper/Demo and simulation evidence.

The priority stack is:

1. **P1 Paper/Demo business evidence.** Maintain the three seats, prove their
   direct broker state, compare each due decision with a same-window simulation
   and make model succession executable.
2. **P2 continuous optimization.** The four GPUs continue the already approved
   corrected ETH program. Documentation or P1 CPU work must not interrupt a
   healthy GPU run.
3. **P3 academic work.** Papers consume accepted evidence; they do not block P1
   or P2.
4. **P4 social intelligence.** Collection and cheap enrichment continue as a
   bounded sidecar. Publishing remains human-gated.

This is not a serialized queue. P1 receives operator attention first while P2
keeps the fleet occupied.

## 2. Current Facts That Supersede Parts of the Audit

Retsu observed the corrected screen while only seed 101 was running. At this
verdict time:

- corrected screen identity `14e7ce8208ac9776` is terminal `16/16`;
- its sealed collection is
  `~/.local/share/agent-multi/p1lr_v2_collections_20260815/screen_14e7ce82`;
- collection tree SHA-256 is
  `ccad48ef42f39de1e58c92b0067a987f47626aeb3a03e8ac0fd1049cf5fc4806`;
- all 16 terminal artifacts loaded from the Dragon replica;
- the typed screen outcome is `SCREEN_VIABLE_REGION`;
- the screen found seven viable cell/seed combinations. Every `P1E_LR3E5`
  seed was viable; `P1N_LR3E5` was viable for seeds 101, 303 and 404; neither
  `1e-4` arm crossed the typed viability threshold;
- this is actor-survival/mechanics evidence, not a return or risk conclusion;
- decision identity `cdf30aebf585385b` is now active on all four assigned
  GPUs under one contract hash. It had `0/16` terminal decision records at the
  verification point, so no ETA or scientific outcome is yet earned;
- the sealed 2025 role and L2 remain untouched.

The prior v1 collection `c0e53cf18b7d60dd` remains preserved as
`INCONCLUSIVE` with qualifier
`INVALID_FOR_L1_RECIPE_SELECTION_OBSERVATION_CONTRACT_235`. It cannot select
difficulty or learning rate.

## 3. Disposition of Retsu Findings

| Retsu ref | Disposition | Reason / action |
| --- | --- | --- |
| F-P1-01 | accepted, S2 | The Paper seats are linear integration baselines, not current experiment champions. This is a product/evidence gap on Paper, not an S1 capital incident. |
| F-P1-02 | accepted, S2 | Replay and lineage tools exist, but no scheduled same-window product is operating; the latest stored report is stale and not subtractable. |
| F-P1-03 | amended, S3 | MT5 is fresh in the consolidated watchdog, but Omega lacks a direct fleet-auditable MT5 runner evidence path. Consolidated health cannot substitute for direct model/bar/command facts. |
| F-P1-04 | amended, S2 | IBKR halt is now `none`, but direct evidence still contains an open-order/exposure semantic mismatch and the current heartbeat cannot prove a valid model-artifact SHA-256. Reconcile facts; do not infer flatness from one field. |
| F-P1-05 | accepted, S2 | Succession doctrine and drain helpers exist; an end-to-end artifact-compatible promotion into a seat does not. |
| F-P1-06 | accepted, S3 | The linear manifests remain useful canaries, not evidence of a competitive trading policy. |
| F-P2-01 | verified non-finding | The old L1 freeze was correctly withdrawn after the dead-actor root cause. |
| F-P2-02 | stale observation, new defect retained | Screen v2 completed and decision v2 is active on all four GPUs. The durable transition exists, but deployed systemd units and the idle guard still bind v1. |
| F-P2-03 | accepted, S3 | A canonical return packet for the dead-actor correction and v2 dispatch is still required. |
| F-P2-04 | accepted and corrected here | README, document 13, document 38 and the findings register are updated by this audit. |
| F-P2-05 | verified non-finding | Historical v1 evidence remains sealed and explicitly qualified. |
| F-P2-06 | corrected | The missing dead-actor order is restored at commit `309dc59e` on this audit branch. |
| F-P2-07 | accepted, amended | Findings use full namespaced IDs. Bare numeric suffixes are never authoritative. |
| F-PR-01..10 | accepted as prototype-drift program, not P1 work | `abs(reported)` increment, dual threshold writers and the absent `record_evaluation()` caller were independently reproduced. No deployed consensus mutation is authorized by this audit. |
| P3 | accepted | The adversarial-agent method paper remains first. Paper IDs and claims must follow the canonical roadmap. |
| P4 | accepted | Collection is healthy; relevance extraction needs measured precision, cost and evidence value before expansion. |

## 4. New Independent Finding: Identity-Blind Supervision

`AUD-GEN-20260815-250` is open, S2.

The corrected decision processes are healthy and share identity
`cdf30aebf585385b`, but they were started with `nohup` from the immutable
runtime worktree. All deployed `p1lr-decision@*.service` units are inactive and
still pin legacy gate/config paths. The active idle guard reports v1 identity
`c0e53cf18b7d60dd` as alive because it matches any process with the same seed,
even when that process belongs to v2. It also reports no durable transition for
v1 although a separate v2 transition record exists.

Consequences:

- a reboot does not reconstruct the active v2 worker through the declared
  systemd path;
- a v2 process can make a terminal v1 seed look busy;
- bounded recovery could target the wrong unit/identity; and
- status can certify the wrong output root while useful GPU work proceeds
  elsewhere.

Do not stop, adopt or restart the current v2 PIDs merely to make systemd look
tidy. Correct the identity-aware supervisor for the next process boundary and
prove it without creating a duplicate writer.

## 5. Binding Work Order to General Satoshi III

### WO0 - Teach-back and source custody

Before editing, return a compact machine-readable table with:

1. the four owner priorities and their concurrency rule;
2. the exact v2 screen and decision identities;
3. the three Paper/Demo seat models, artifacts, symbols, timeframes, direct
   order/position/protection state and evidence freshness;
4. the distinction between `integration_baseline`, `experiment_champion`,
   `paper_champion` and `release_champion`;
5. an explicit statement that no real-capital account is authorized; and
6. an explicit statement that the active decision PIDs will not be restarted,
   adopted or duplicated during this package.

Commit Retsu's full audit and this response into the authoring branch. Do not
edit from the runtime checkout.

### WO1 - Direct seat truth

Implement one typed fleet inventory assembled only from direct venue or
runner facts. For Alpaca, IBKR and MT5 report:

- venue/account fingerprint, Paper/Demo class and write-enabled state;
- symbol, timeframe, strategy/model ID, model artifact SHA-256, config SHA-256
  and code revision;
- last closed input bar and due-decision identity;
- direct positions, orders, fills and native SL/TP evidence;
- hold/kill/reconciliation state;
- freshness budget and source path/endpoint; and
- a typed `unavailable` reason whenever any fact is absent.

Resolve the IBKR order/exposure mismatch from direct TWS facts. Make MT5's
Dragon-local evidence fleet-readable without treating a consolidated alert as
the direct source. Never print raw account identifiers or secrets.

### WO2 - Operate same-window simulation versus Paper/Demo

Reuse the existing `live_sim_replay` and due-bar lineage tools. Do not create a
second parallel comparator.

For every due decision, persist an append-only row keyed by venue, seat,
symbol, timeframe, due-bar timestamp, model artifact hash, config hash and
input-feature hash. Join:

- model decision and proposed protected geometry;
- simulator decision and fills under the same closed information set;
- direct broker orders/fills/positions and SL/TP state;
- spread, slippage, latency, fees, rejection and missed-fill facts; and
- realized return/risk facts when the horizon closes.

If model identity, due bar, input lineage or economic assumptions differ, emit
`NOT_SUBTRACTABLE` with reason codes. Never calculate a numerical delta across
incomparable rows. Install a durable timer/service, deduplicate by identity and
emit Telegram only on actionable state changes after the normal incident
router has had a chance to recover.

Acceptance: one fresh comparable or honestly `NOT_SUBTRACTABLE` row per
active seat, a restart/idempotency test, and a seven-day rolling report format
that can accumulate without changing old rows.

### WO3 - Executable champion succession

Complete the existing succession design rather than replacing it. Add:

1. artifact compatibility preflight for observation, feature order, symbol,
   timeframe, action and execution contracts;
2. candidate shadow replay on the same due bars as the incumbent;
3. owner-approved Paper promotion capability, single-use and audit logged;
4. drain: stop new incumbent risk, reconcile/close or transfer only where the
   contract explicitly permits it, and use actual post-close balance/equity as
   the successor's starting state;
5. atomic manifest switch with rollback;
6. native SL and TP on every opening order; and
7. outgoing champion shadowing after handover.

Do not call a v2 training artifact the seated champion until compatibility and
promotion evidence exist. Until then, keep linear baselines labeled exactly as
baselines.

### WO4 - Preserve decision v2 and repair supervision

The active `cdf30aebf585385b` run continues untouched.

For the next worker boundary:

- generate/install identity-specific environment files and gate paths for v2;
- make process matching require contract/chain identity, mode, seed and output
  root, not seed alone;
- make the idle guard discover the active durable transition and refuse a
  conflicting identity;
- prove one writer per seed under concurrent timer/manual activation;
- prove reboot reconstruction from the immutable runtime worktree;
- prove a v2 PID cannot make terminal v1 look alive;
- prove an old v1 unit cannot restart while v2 owns the lease; and
- preserve terminal artifacts before any transition.

No `systemctl start/restart` is allowed while the current matching v2 PID is
alive. A dry-run and process/lease proof precede deployment.

### WO5 - Canonical return and documents

Return one packet with exact commits, clean-tree facts, commands, tests, direct
evidence paths and unresolved doubts. Update docs without rewriting history:

- v1 stays `INCONCLUSIVE` plus invalid-observation qualifier;
- v2 screen is mechanics evidence only;
- v2 decision is active with no result until terminal rows exist;
- L2 and sealed 2025 remain parked;
- Paper seats remain baselines until WO3 promotes a compatible artifact.

### WO6 - Protocol sidecar, resource bounded

Do not alter deployed consensus while P1/P2 are active. In isolated branches:

- fix transaction-fee conservation with property tests;
- produce failing tests for absolute reported value versus current-best delta;
- collapse threshold authority behind one typed profile, but do not deploy it;
- wire or explicitly retire `record_evaluation()` with event-contract tests;
- separate inference-service completion evidence from optimization evaluation;
  and
- keep synthetic challenge semantics aligned with documents 39 and 40.

This sidecar yields whenever it competes with Paper/Demo evidence or the active
ETH run.

## 6. Retsu Verification Assignment

Retsu remains read-mostly and verifies independently:

1. WO1 direct-source inventory against venue/runner facts;
2. WO2 same-window identity and `NOT_SUBTRACTABLE` behavior;
3. WO4 identity-aware process matching and reboot/single-writer fixtures;
4. the final v2 decision collection and replica loads when terminal; and
5. that no claim promotes screen viability into performance.

Retsu must not start/stop workers, operate brokers, clear holds, mint
capabilities, mutate findings or close work authored by Retsu.

## 7. Acceptance Sequence

1. WO0 teach-back.
2. WO1 and WO2 on CPU while decision v2 runs.
3. WO4 tests and dry-run while v2 runs; deployment only at a safe process
   boundary.
4. WO3 after seat inventory and one valid candidate compatibility proof.
5. WO5 return packet.
6. WO6 only from otherwise unused CPU capacity.

No additional owner phrase is required for this already approved correction
and evidence program. Broker capabilities, hold clearing, real-capital access
and Paper champion promotion remain separate explicit authorities.
