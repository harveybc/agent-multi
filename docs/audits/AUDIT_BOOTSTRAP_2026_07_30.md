# AUDIT-BOOTSTRAP-001 — Read-Only Baseline Audit

Audit ID: AUDIT-BOOTSTRAP-001
Timestamp and timezone: 2026-07-30 20:05 America/Bogota (UTC-5)
Auditor: Claude independent continuous-audit agent ("Satoshi"), per
`docs/handoffs/CLAUDE_CONTINUOUS_AUDIT_AGENT_SPEC_2026_07_30.md`
Requested by: user (full authorization for the bootstrap baseline)
Scope: provenance capture for all 11 active repositories; context Layers A and
B; architecture/responsibility reconstruction; documentation-drift detection;
initial risk-ranked backlog; Codex recovery-prompt review; bounded read-only
runtime verification on Omega (processes, GPU, supervisor and node APIs).
Excluded scope: remote host inspection (Dragon/Gamma/MT5 VM), SQLite OLAP
queries, test execution, secrets, campaign/broker state changes. No private
personal continuity material was searched or read.

## Provenance

Captured 2026-07-30 ~19:58 COT. All repositories are clean (0 dirty files) and
in sync with upstream (behind/ahead 0/0).

| Repository/system | Branch | Commit | Dirty/runtime state |
| --- | --- | --- | --- |
| agent-multi | master | 21bcc427d8fc5a7f10eb911028384e17070a3aa2 | clean; deployed worker code `6a7bf5a` (delta to HEAD is docs-only) |
| trading-contracts | master | 534b0349d9a32768f5e81a0d5e96fea5ad38f11a | clean; matches deployed `534b034` |
| gym-fx | master | 40a5c844f22cd320d6cee66a0e75b9d87fc1b531 | clean; matches deployed `40a5c84` |
| heuristic-strategy | master | e388cefa4b96680b8c1e78cb7155f02779f61fba | clean |
| doin-core | master | 8573a874cbb8abcf9f07776068dd68d09fdf923f | clean; matches deployed `8573a87` |
| doin-node | master | 7c400f90c0789e3156202f231309b43890d245c2 | clean; matches deployed `7c400f9` |
| doin-plugins | master | f5fedf8a44656f4fccd36d8f453609fa62a94fe2 | clean; matches deployed `f5fedf8` |
| prediction_provider | main | ac4d9e2fa552b578e274a641253bad2a64d1b8c7 | clean |
| lts | main | 6c695c7059ce2f63251a2fdd15b97a5027e2fb1d | clean |
| financial-data | master | 298113425f30551f56f8a68c3947b5a5f2c275cd | clean |
| predictor | master | b1f8a74f19f6d164be6367b6d57e726c6ff6b95a | clean; last commits Apr–May 2026 (historical reference role) |
| Runtime: Omega | — | plan `phase-1-protected-execution-fleet-v2`, hash `b43844a7ebd7…` | supervisor + `doin_node.cli` running; job 0 `usdcad-4h-protected-easy-sac-shared-v2`, stage 1/4, gen 2 |

The spec's bootstrap table listed `agent-multi@163f8efb`; the only commit past
it is `21bcc427` — the commit that published the audit/recovery specs
themselves. Orientation drift is therefore self-referential and harmless.

## Findings

### AUD-GEN-20260730-001 — Task ledger reports the live v2 campaign as "pending deploy"

- Severity: S3
- Confidence: high (observed)
- Status: open
- Affected front: 1 (also cross-front status integrity)
- Repository/system: `agent-multi` `docs/work_plan/13_IMPLEMENTATION_STATUS_AND_TASK_LEDGER.md`
- Commit/config/artifact: doc at `21bcc427`; runtime plan hash `b43844a7ebd7c85a782c557a8c3459622e1cb353a5d33391816e85f107cb6b21`
- Observation: Document 13 (status timestamp 2026-07-30) lists Phase 4 as
  "protected v2 pending deploy", ledger tasks `ACTIVITY-GATE-002`,
  `PROTECTED-ORDERS-002`, `WEEKLY-METRICS-002` as
  `verified_local_pending_deploy`, and Immediate Next Tasks 2–4 (commit/publish
  v2, install v2 supervisors, run the USDCAD easy job) as future work. Observed
  runtime: the `phase-1-protected-execution-fleet-v2` campaign has been running
  since ~2026-07-29 18:15 COT (node uptime 92,595 s at observation), domain
  `trading-asset-policy-usdcad-4h-protected-easy-v2`, generation 2, 10/20
  evaluated, four workers with distinct claims, seed 2703, chain height 8.
- Inference: deployment tasks 2–4 were executed after the doc content was
  frozen; the ledger lags runtime by ~1 day.
- Business impact: a recovered Codex conversation or auditor following doc 13
  literally would attempt to re-execute deployment steps against a live
  campaign (supervisor install, config regeneration), risking disruption of an
  active swarm.
- Technical impact: status surfaces contradict the source-of-truth rule that
  active work must not be reported from stale prose.
- Minimal reproduction: read doc 13 §"Immediate Next Tasks" items 2–4; then
  `curl -s http://127.0.0.1:8795/api/status` on Omega and compare `phase`,
  `job_id` and worker states.
- Proposed correction: refresh doc 13's status stanza and ledger states to
  record the v2 deployment evidence (plan hash, seed, genesis hash
  `4e19257e8941…`, start date, component revisions) and mark tasks 2–4 done.
- Required regression or monitor: none beyond the existing update discipline;
  optionally have the supervisor's plan hash echoed in doc 13 at each status
  refresh.
- Owner: Codex (Musashi)
- Dependencies: none

### AUD-F1-20260730-002 — Job-0 wall-clock budget at measured throughput is ~3 weeks and undeclared

- Severity: S4
- Confidence: medium (observed Omega samples; fleet-level figure inferred)
- Status: open
- Affected front: 1
- Repository/system: `phase-1-protected-execution-fleet-v2` job 0
- Observed evidence: Omega candidate durations over 5 samples: p25 10,507 s,
  median 15,181 s, p75 15,972 s → 0.237 candidates/hour on the RTX 4070.
  `plan_jobs[0].planned_candidates = 480`; stage patience 4 over 6
  generations/stage, 4 stages.
- Inference: if the other three (faster) GPUs match or beat Omega's rate, fleet
  throughput is roughly 0.95–1.5 candidates/hour, so the full 480-candidate
  budget is ~13–21 days of wall clock unless L2 patience stops stages early.
  The v1 fleet measured 1.688 candidates/hour; protected brackets plus the full
  genome roughly halved Omega's rate.
- Business impact: compute-time expectation for the first eligible champion; the
  curriculum job (job 1) queues behind it.
- Technical impact: none; this is planning visibility, not a defect.
- Minimal reproduction: `curl -s http://127.0.0.1:8795/api/status` →
  `candidate_eta` block; multiply remaining planned candidates by observed
  median.
- Proposed correction: record an expected-duration/decision point for job 0 in
  doc 13 or the campaign plan (e.g., review after stage 1 completes), so an
  early stop versus full budget is a recorded decision rather than a surprise.
- Owner: Codex (Musashi) + user (accepted timeline)
- Dependencies: per-worker duration evidence from Dragon/Gamma (next cycle)

### AUD-GEN-20260730-003 — Recovery prompt's broad-recovery reading list omits four work-plan documents

- Severity: S4
- Confidence: high (observed)
- Status: open
- Affected front: cross-front (continuity)
- Repository/system: `agent-multi` `docs/handoffs/CODEX_TECHNICAL_LEAD_RECOVERY_PROMPT_2026_07_30.md`
- Observation: the "MANDATORY CONTEXT LOAD" broad-recovery list omits
  `08_IMPLEMENTATION_ROADMAP.md`, `11_DOIN_CONFIGURATION_PROFILES.md` (named
  "authoritative two-level configuration and generation procedure" by doc 05
  §10.1), `14_SIMULATION_ENGINE_SELECTION_2026_07_11.md` and
  `16_FLAT_FITNESS_ROOT_CAUSE_2026_07_19.md`.
- Impact: low — the README document map (read first) covers discovery — but 11
  and 16 encode contracts and incident context a fresh Codex needs before
  touching node configs or interpreting flat fitness.
- Proposed correction: add the four documents to the broad-recovery list in the
  next prompt revision; also update the "CURRENT SNAPSHOT WARNING" from
  "v2 active/prepared" to "v2 running since 2026-07-29".
- Owner: Codex (Musashi); auditor may draft the replacement text on request
- Dependencies: none

## Architecture Reconstruction (one page)

Authority: the user owns business goals/risk; Codex ("Musashi") is technical
lead, implementer, orchestrator and final verifier; this Claude conversation
("Satoshi") is the independent read-mostly auditor; Hermes provides
deterministic telemetry/Telegram delivery; deterministic services own
watchdogs, scheduling, OLAP writes and fail-closed controls.

Front 1 — optimization and research: `financial-data` (immutable data,
manifests, hashes) → `agent-multi` (mixed full genome, L1 training via
`gym-fx`/NautilusTrader simulation, local-first optimizer, metrics, artifacts)
→ `doin-plugins` thin adapters → one shared-population DOIN campaign on
`doin-node`/`doin-core` (four workers: omega, dragon, gamma-5070ti,
gamma-5090; one plan/seed/dataset/genesis/population/chain) → archived
champion (weights + decoded JSON + hashes + metrics) → frozen per-asset
library → static portfolio optimization → rush/event activation and weekly
retraining comparison last. Protected test is excluded from all selection.
Current state: job 0 `USDCAD@4h` protected easy_floor full-genome search is
live (stage 1/4, gen 2); job 1 (easy→nominal→stress curriculum, robust
weekly-RAP fitness) is queued and its config is generated fail-closed from
job-0 handoff artifacts.

Front 2 — execution reality: frozen artifacts → `prediction_provider`
(hash-verified inference) → `lts` (single global portfolio, NAV, risk, capital
reservations, venue router, reconciliation) → replaceable venue adapters:
OANDA Global Markets via MT5 EA (demo, read-only first; Windows VM on Dragon
being commissioned), Alpaca Paper (crypto observation, long-only control),
IBKR Paper (equities/multi-asset; observer live on Omega, TWS port 7497). The
REST-v20 Practice lab is preserved only for compatible OANDA divisions. Every
risk-increasing order requires SL+TP, fail closed. No chain/model/social path
to a broker bypasses LTS.

Front 3 — social intelligence and continuity: specified, not activated
(S0/S1 next): allowlisted sources → deterministic collection/hashing →
injection screening → tiered local/cloud model routing with hard budgets →
claim/evidence OLAP → Telegram review; publishing starts disabled. Social
content is hostile data and can never affect orders, campaigns or champions.
Continuity: reproducible services, encrypted tested backups, revocation, two
human maintainers; VPS is a collector/backup target, not an authority.

## Open Questions (evidence could not answer)

1. Dragon/Gamma `loginctl` linger status: is the temporary Omega-held SSH
   session bridge (doc 15 §2) still the availability dependency? Requires
   remote read-only check.
2. MT5 VM commissioning state on Dragon (doc 22 said "first Windows Setup
   screen" as of 2026-07-30) — not verifiable from Omega without remote access.
3. Is a ~2–3-week job-0 wall clock accepted, or is stage-level early stopping
   expected to be the norm? (Finding 002 decision point.)
4. Alpaca's five-minute preflight: no persistent process was observed; the
   scheduling mechanism (systemd timer/cron identity) was not verified this
   cycle.

## Verified Non-Findings

1. Fleet lineage consistency (checklist D): all four workers report the same
   plan hash `b43844a7ebd7…`, domain, generation 2, population fingerprint
   `4c85abd7a089…`, chain height 8, finalized anchor `8b7590aaf645…`, identical
   six component revisions, and four distinct candidate claims {10, 11, 12,
   13}; zero alerts, zero API/optimization errors. No parallel-lineage signal.
2. Deployed worker code `agent-multi@6a7bf5a` differs from HEAD `21bcc427`
   only by two documentation commits (`6a7bf5a→163f8efb` and
   `163f8efb→21bcc427`, both docs-only by file list). Doc 13's claim that the
   publication commit "is not falsely reported as deployed code" holds.
3. All 11 active repositories are clean and synchronized with upstream; there
   are no unknown user changes at risk.
4. `docs/audits/README.md` exists and defines the report contract referenced by
   doc 24 §11.
5. `examples/campaigns/phase_1_protected_execution_fleet_v2/` exists; job 1
   `planned_candidates=0` is consistent with the fail-closed curriculum
   materializer design (doc 19 §13), not a missing budget.
6. IBKR Paper observer (`app.ibkr_paper_cli`) is running read-only on Omega,
   consistent with doc 22 §12.1.
7. `predictor` recent commits are April–May 2026 predictor-domain history;
   consistent with its reference-only role; no competing trading lineage.
8. Omega GPU: RTX 4070, 36 % util, 47 °C — below the 78 °C alert threshold.
9. The gen-0 bootstrap population fingerprint (`295f1580…`) differing from the
   current gen-2 fingerprint (`4c85abd7…`) is per-generation fingerprint
   behavior, confirmed by `join_reason: "seed, genesis, population and
   component lineage match"` — not a divergence.

## Codex Recovery-Prompt Verdict

PASS — current and usable (version 1.0.0). It correctly reconstructs
authority, the three fronts, repository ownership, source-of-truth order, the
status contract, DOIN orchestration rules, multi-venue safety and the auditor
interface, and its snapshot section correctly instructs treating itself as
stale. Two minor improvements are recorded as finding 003 (add docs 08/11/14/16
to the broad-recovery list; refresh the snapshot warning to "v2 running"). No
full replacement text is required this cycle.

## Proposed Next Three Audit Tasks

1. AUDIT-V2-CONTRACT (Front 1, next 24–48 h): independently verify the
   protected-entry v2 eligibility contract in code and evidence — annual
   validation ≥ 12-trade floor, `-1e9` ineligibility sentinel, bracket-only
   entries failing closed — by reading the wiring and rerunning the focused
   `agent-multi` safety tests and `gym-fx` suite (bounded, no training), and
   reconstructing the current champion fitness `0.0004434357799592837` from its
   stored metric evidence to confirm weekly-RAP units and gap penalty. This is
   the exact contract whose v1 failure caused the 2026-07-29 incident, and the
   live campaign depends on it.
2. AUDIT-BROKER-BOUNDARY (Front 2, next 72 h): read-only fail-closed audit of
   the venue boundary — Practice client live-endpoint rejection, Alpaca/IBKR
   observers read-only, MT5 EA demo-only/HMAC/nonce logic from source, and
   read-only SQLite queries over the paper OLAP stores proving no raw account
   IDs or tokens persist. Gates the upcoming canary phases.
3. AUDIT-CHAMPION-ARCHIVE (Front 1, event-driven at job-0 convergence): verify
   the champion archive completes before job-1 startup — byte count, SHA-256,
   decoded genome, metric vector, five diverse elites, independent hash-load of
   the artifact, and that the curriculum config was generated identically on
   every host from local handoff artifacts with no v1 lineage contamination.

## Commands and Queries (all read-only)

```text
git -C <repo> rev-parse / status --porcelain / rev-list --left-right --count  (11 repos)
git -C agent-multi log --oneline 163f8efb..HEAD
git -C agent-multi show --stat HEAD
git -C agent-multi log -1 6a7bf5a; merge-base --is-ancestor 6a7bf5a HEAD
git -C agent-multi diff --name-only 6a7bf5a 163f8efb
git -C predictor log -8
ls docs/audits/ docs/work_plan/ examples/campaigns/
uname -n; date; uptime
ps aux | grep -E 'doin|campaign_supervisor|oanda|alpaca|ibkr|mt5|lts'
nvidia-smi --query-gpu=... --format=csv
screen -ls
curl -s http://127.0.0.1:8795/api/status  (GET only)
curl -s http://127.0.0.1:8795/api/network (GET only)
curl -s http://127.0.0.1:8470/api/monitor (GET only)
```

Documents read: work-plan README, 01, 02, 04, 05, 06, 07, 09, 10, 12, 13, 15,
17, 18, 19, 20, 21, 22, 23, 24; both handoff prompts; `docs/audits/README.md`.

## Artifacts and Evidence Hashes

- Campaign plan hash: `b43844a7ebd7c85a782c557a8c3459622e1cb353a5d33391816e85f107cb6b21`
- Genesis hash: `4e19257e8941caec2ec6a4d581a981a533ab5728c11e58c3d13040049e3cd6d5`
- Gen-0 population block: `8b7590aaf64541822559df3c1439a4cf906a8764648624b70ad4f5b7df38b3c0` (index 1)
- Gen-2 population fingerprint: `4c85abd7a089b727da1522bfed3eb03bfa4b21be534cd21bd6506b4fa6278d4c`
- Shared-population seed: 2703

## Next Audit Trigger

This auditor does not run between invocations. Next expected invocation: a
24-hour delta audit (target 2026-07-31 evening) or earlier on any of: job-0
stage transition or convergence, champion archive, MT5 EA compilation/bridge
activation, broker canary enablement, incident, or contract/fitness change.

## Change Confirmation

No code, configuration, service, machine, campaign, broker, credential or Git
state was modified. No commit, push or delete was performed. The only file
created is this report, explicitly authorized by the user; it is untracked and
left for Codex/user review before any commit.
