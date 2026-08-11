# Independent Audit: Satoshi III Return 209-220

Date: 2026-08-11 America/Bogota  
Auditor: General Musashi (Codex), independent auditor during the role swap  
Subject: `agent-multi@f6dc991d` on `satoshi/m0-aggregation-hardening`  
Component subjects: `doin-core@00397f5`, `doin-node@0821ec2`,
`predictor@e6b91b5`  
Runtime mutation by this audit: none

Evidence:

- `docs/audits/evidence/repro_runs/MUSASHI_209_220_ACCEPTANCE_REPRO_2026_08_11.py`
- `docs/audits/evidence/repro_runs/MUSASHI_209_220_ACCEPTANCE_REPRO_2026_08_11.json`
- `docs/audits/evidence/repro_runs/MUSASHI_209_220_RUNTIME_AND_ARTIFACT_FACTS_2026_08_11.json`

## 1. Verdict

**PARTIAL ACCEPTANCE.** Corrections 209-216, 218 and 219 are independently
verified. Finding 217 remains open because its checker reported a false zero
and the default `agent-multi` branch still contains two broken relative links.

The mechanism ladder executed real, useful work. Its defensible conclusion is:

> At seed 101, the v4 path handed an actually easy-trained policy into normal
> training, while the v3 positive control selected and handed the pristine
> anchor. Every easy-trained policy already had a near-constant raw action of
> magnitude below `0.015`; the normal action threshold is `0.1`, its immediate
> normal probes traded zero times, and normal training did not recover. The
> v3 anchor handoff remained active.

That does **not** yet isolate a pure "boundary handoff" mechanism. The one
configuration field changes checkpoint eligibility, selection objective,
normal-probe gating and fallback semantics together. The arms also trained
their easy epoch independently on different GPUs and produced different
policy hashes. The accepted label is therefore **phase-1 checkpoint-selection
and handoff bundle, with easy-phase action-amplitude collapse exposed by the
normal deadband**. Replay and optimizer transfer are ruled out as the observed
boundary difference: every arm reports no optimizer transfer and an empty
replay buffer at the boundary.

The real ladder artifacts are preserved: the auditor independently rehashed
and loaded all four terminal ZIPs on Dragon. The collector is nevertheless
unsafe for future evidence because it only requires `best_model_*`; inactive
arms have only `terminal_model_*`, so a collection with zero terminal files can
currently seal, replicate and publish.

Finding 220 is only partially satisfied. It delivered the ladder, then left no
successor job. At the runtime sample Dragon and both Gamma GPUs were at 0%, and
Omega had only desktop utilization with no training process. The running DOIN
supervisors were supervising a paused 2026-08-06 chain, not useful compute.

## 2. Independent Verification

| Surface | Result |
| --- | --- |
| `doin-core` full suite | 316 passed |
| `doin-node` full suite | 482 passed, 22 warnings |
| `predictor` full suite | 25 passed, 1 skipped |
| `agent-multi` full suite | 996 passed, 2 warnings |
| focused status tests | 51 passed |
| focused ladder/curriculum tests | 74 passed |
| focused predictor tests | 6 passed |
| ladder seal digest | `cdb6ef9947887992fc0a133a8c66adb76d64a4484cccb5cfc9f63fbea1c2ed8e` |
| Dragon terminal artifacts | 4/4 SHA matches and `SAC.load()` succeeds |

### 2.1 Findings 209-211

Accepted in code:

- malformed or incomplete height/tip metadata produces typed failure;
- a startup-verified append cursor binds metadata, tip and SQLite
  `data_version` inside the write transaction;
- transaction IDs are recomputed before Merkle/block validation;
- shared-population configuration refuses absent explicit chain identity; and
- history tamper through a second connection is refused before append.

Deployment disposition: preserve legacy chains read-only. The next DOIN job
starts a new v2 chain from the reviewed migration manifest with the same
explicit chain/genesis identity on every worker. No legacy hash rewrite and no
mid-chain rollout.

### 2.2 Findings 212-214

Accepted. A fresh `multifront_status.py --no-emit-alerts` run shows:

- L1 is terminal `16/16`, ETA zero and stale per-cell logs are not presented as
  current telemetry;
- aggregate ETA uses the maximum per-worker critical path; and
- IBKR is derived as write-enabled and operational-but-held, not missing or
  dependency-blocked.

### 2.3 Finding 215

The pre-push sensitivity control is accepted. The repository is already
private under the owner's explicit authorization. History rewriting remains a
separate destructive operation and is **not** authorized by this audit.

### 2.4 Finding 216

Accepted. Headerless CSV labels are normalized before date-column detection;
the strict xfail is gone and header/no-header regression tests pass.

### 2.5 Findings 217-219

- **217 remains open.** The committed checker says `broken=0` while two rows
  contain `README.md not in HEAD tree`; errors do not increment its failure
  count. It also checks local `HEAD`, not each repository's remote default tip.
  On current `agent-multi` `origin/master`, `_nested_splits.py` and
  `_paired_generalization.py` are linked from README but absent from that tree.
- **218 accepted.** Both supersession topics now name the systems described by
  their READMEs and the 20-topic invariant remains intact.
- **219 accepted.** The default-branch `causal-inference` README accurately
  distinguishes committed experimental code from the owner's dirty local
  analysis, which remains untouched.

## 3. Ladder Evidence

| Arm | Easy epoch-1 raw range | Immediate normal probe trades (tail/val) | Terminal normal result |
| --- | --- | --- | --- |
| D0 | exactly `0.010861` | `0 / 0` | active only because v3 selected/transferred epoch-0 anchor |
| D2 | `0.014293..0.014354` | `0 / 0` | constant `0.002390`, zero trades |
| D3 | `0.007986..0.007988` | `0 / 0` | constant `-0.000132`, zero trades |
| D4 | `0.012114..0.012130` | `0 / 0` | constant `0.001209`, zero trades |

Easy uses action threshold `0.0`; normal uses `0.1`. Counting every positive
`0.008-0.014` action as non-hold makes all four easy epoch-1 policies appear
active even though none can cross the normal deadband. Trade count alone is
therefore not a sufficient handoff-quality measurement.

The independently observed boundary facts are identical across arms:

- `optimizer_state_transferred=false`;
- `replay_size_at_boundary=0`; and
- `replay_transitions_transferred=0`.

The next experiment must not spend a factor on replay or optimizer carry.

## 4. New Findings

### AUD-F1-20260811-221 (S2): easy eligibility certifies near-constant policies that cannot act under normal dynamics

`easy_activity_eligible` only asks for easy trades/non-hold actions under a
zero threshold. It omits raw-action variance, range, observation sensitivity
and distance to the next phase's threshold. All trained easy policies pass
that easy label while their normal probes trade zero times. This blocks using
the current activity label for curriculum selection or a DOIN gene.

Required correction: emit and evaluate raw action distribution on train
monitor and inner validation under both easy and normal thresholds, add a
typed handoff-viability classification, and refuse promotion of a degenerate
constant policy. Calibrate any numeric floor from controls; do not invent one.

### AUD-F1-20260811-222 (S3): the ladder overstates a compound treatment as one pure boundary delta

`phase1_handoff_semantics` changes four behaviors: epoch-0 eligibility,
selection score, normal-probe gating and terminal fallback. Easy training was
also repeated separately on heterogeneous GPUs, producing different policy
hashes and raw ranges. Equal trade counts do not make easy behavior identical.

Required correction: relabel the current result as the v3-versus-v4
selection/handoff bundle; then replay one shared easy artifact under threshold
`0.0` and `0.1` without training, followed by a bounded paired design that
separates easy LR from threshold transition.

### AUD-F1-20260811-223 (S3): the ladder collector can certify a replica with no terminal models

The collector validates and builds replica expectations from
`best_model_path`/`best_model_sha256`. D2-D4 are inactive typed results and
carry only `terminal_model_path`/`terminal_model_sha256`. The production proof
therefore loaded only D0. The independent counterexample sealed, replicated
and published four records with **zero model files** and zero load
expectations.

Required correction: every training arm must have a terminal path/hash, the
staged exact terminal must hash-match, and the replica proof must contain one
successful load for every arm. Missing, duplicate, altered or unloaded
terminal evidence refuses sealing/publication.

## 5. Runtime and Front Status

At 2026-08-11 04:00 COT:

- **Front 1:** ladder terminal; no GPU training successor; 220 remains open.
- **Front 2:** Alpaca Paper and MT5 Demo are write-enabled with one exposure
  each; IBKR Paper is write-enabled, flat and held. No real capital.
- **Front 3:** 9,144 posts collected, 636 enriched, 315 backlog, zero drafts.
- **Front 4:** suites pass; 209-216 and 218-219 accepted; 217 and 220-223 open.

Temperatures are healthy: Omega 45 C, Dragon 30 C, Gamma 5070 Ti 24 C and
Gamma 5090 41 C. Temperature health does not turn idle workers into useful
work.

## 6. Acceptance Boundary

- 209-216, 218 and 219 may move to independently verified pending owner
  closure.
- 217 remains open.
- The ladder may be cited only with the corrected compound-mechanism wording.
- The current four artifacts are valid because the auditor independently
  loaded them; the collector implementation remains blocked by 223.
- No L2, feature-selection or broad DOIN campaign starts from this one-seed
  ladder.
- No new owner phrase is required for the focused correction and paired
  diagnostic in the accompanying order.

