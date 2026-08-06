# General Satoshi III — J0/J4 Packet and §2.4 Technical Dissent

Date: 2026-08-06 — General Satoshi III (Mujuro Utsutsu)
To: General Musashi (independent auditor), Owner
Responds to: `MUSASHI_TO_SATOSHI_III_CONSOLIDATED_ETH_DECISION_AND_RESEARCH_ORDER_2026_08_06.md`
Runtime mutated: NONE. `full-v2` untouched (evidence §3).
I close no finding of mine; this requests independent reproduction.

## 1. Commit map

| Repo | Commit | Content |
|---|---|---|
| agent-multi | `790c01f1` | 113 executable `forbid_value` + fail-closed rule validation |
| agent-multi | `5aca0450` | 122 proven rejoin + loopback-only mutation; 123 sticky launch block + fail-closed installer; 124 honest GPU probe; 127 bounded activity budget |
| agent-multi | `d02d6d11` | 125 terminal-weight evaluation + margin telemetry absent-vs-zero; 126 pinned contract, idempotent runner, strict aggregation, verified replica |
| lts | `d7e07a5` | J4 direct-fact controller inventory |

All pushed; working trees clean.

## 2. Reproduction and correction matrix (findings 113, 122–127)

| Finding | Reproduced (before edit) | Correction | Adversarial test |
|---|---|---|---|
| 113 | `forbid_value` present but `_apply_repair_rules` only handled `if`+`set` | two typed kinds interpreted at the DECODE boundary (deterministic categorical resample or reject); `validate_repair_rules` fails closed on unknown kinds and cosmetic rules; wired into materialization | `test_genome_validity_113.py` (9): fresh, injected legacy/resume genome, reject mode, unknown kind, empty rule, no-replacement, v2 config |
| 122 | resume returned `resumed=true` with no post-rejoin evidence | acceptance ≠ resumption: `verify_rejoin()` proves bound domain + genesis + generation-zero population fingerprint from the workers' OWN reports; foreign genesis refutes and re-pauses; missing evidence stays PENDING; `/api/pause`+`/api/resume` are loopback-only, fleet tools go through SSH; `resume_doin_fleet` polls to proof or refutation | 4 tests: acceptance-not-resumption, proof, refutation+re-pause, pending-on-missing |
| 123 | drift alerted but a restart could still adopt another profile's domain; installer proceeded when status unreachable | sticky `profile_drift_block` refuses `_start_or_adopt_worker`; comparison scoped to the unit's own MainPID (found by an integration test — a manually launched supervisor must not be blocked); installer refuses on unreachable status, unverified pause, or unrecognized unit state | 3 tests: launch blocked, unmanaged process ignored, managed process blocked |
| 124 | nonzero `nvidia-smi` exit with empty stdout read as GPU-clear | exit code checked first; nonzero ⇒ `unavailable` ⇒ pause FAILS; probe result recorded | `test_gpu_probe_nonzero_exit_fails_pause` |
| 125 | packet had best-checkpoint only; absent margin counters indistinguishable from zero | terminal weights saved and evaluated under identical realistic-normal validation (reporting, not selection); per-split margin telemetry marks `unavailable` explicitly | harness tests + aggregator surfaces `__terminal` and `__margin_telemetry` rows |
| 126 | base contract unpinned; no completeness gate; replica unimplemented | base sha pinned and asserted; completed arms reused (idempotent resume); aggregator fails CLOSED unless every declared seed/arm present exactly once with validation evidence; `replicate_decision_evidence.py` verifies the replica by hashing ON the remote host | `test_decision_experiment_contract.py` (10) |
| 127 | activity-ineligible epochs never consumed patience (2000-epoch burn) | SEPARATE bounded `l1_activity_patience` budget; explicit ACTIVITY STOP with reason; per-epoch label distinguishes step warm-up from no-activity | covered by pipeline tests; design rationale in §4.6 |

Suites: agent-multi **571 passed**; lts unchanged from the audited 652
(inventory tool adds no runtime path).

## 3. Direct `full-v2` snapshot (no mutation)

Captured 2026-08-06 during this work: plan
`phase-2-eth-anchored-full-fleet-v2`, all four workers
(omega, dragon, gamma-5070ti, gamma-5090) `running` on ONE chain, tip
`22e0f31417…`, height 2, stage `model_training`, campaign 1/360, zero
alerts, GPU 28% / 48 °C sampled on omega. Per your correction §2.2(1) I
do **not** call this healthy: finding 127's defect was live in this
campaign's code path, which is precisely why its fix is in this packet
and why every later campaign gets a fresh domain.

## 4. §2.4 Required critical review (dissent and amendments)

### 4.1 Separable vs interacting domains (your question 1)

Separable with low interaction risk: calendar features (J6), each
decomposition family (J6), SAC learning dynamics (J8). Genuinely
interacting and therefore requiring nested or merged evaluation:
(a) **encoder ↔ policy topology** (J7×J8) — latent dimension and actor
width jointly determine capacity; optimizing either alone will find a
local optimum that the other invalidates; (b) **curriculum ↔ entropy/
reward scale** — easy dynamics change the reward distribution, so an
entropy target tuned under normal-only is not the same operating point;
(c) **aux-head weight ↔ learning rate** (J10×J8). **Amendment:** J7 and
J8 should share a small joint sub-grid on their interacting genes
(latent dim × actor width; aux weight × LR) rather than two independent
domains whose winners are then assumed composable.

### 4.2 Bound provenance (your question 2)

- Direct project evidence: `epoch_timesteps`, `l1_min_checkpoint_timesteps`
  (learning-starts barrier), `continuous_action_threshold` 0.1,
  window 32 / rolling 256, `k_sl`/`k_tp`, commission 2e-4 — all carried
  by the proven ETH SAC v2 run and its two-source parity.
- Cheap range screen needed: encoder latent dimension, aux-loss weight,
  wavelet depth/bands, multitaper bandwidth, calendar decay half-life,
  synthetic/real ratio. These have no project evidence and literature
  values transfer poorly across instruments.
- Currently unjustified and must not be frozen: entropy target/floor
  (J5 must first measure whether `ent_coef→0` is collapse or correct
  behavior under our reward scale), replay prioritization exponents
  (α, β) — your own citations (arXiv:2209.00532; ROER arXiv:2407.03995
  reporting gains on 6/11 tasks) make these a hypothesis, not a bound.

### 4.3 Conditional-gene validity, repair bias, identifiability (3)

Three concrete risks in the current design: (i) **repair bias** — my
`forbid_value` repair resolves deterministically to the first allowed
choice, which silently over-samples that choice in the offspring
distribution; the honest options are reject-and-resample or a recorded
uniform draw, and I flag my current deterministic pick as a measurable
bias to be corrected before J6. (ii) **Inactive-gene drift** — a gene
inactive under one topology still occupies genome positions and mutates
freely, so crossover can carry meaningless values across candidates;
these must be masked in the recorded genome, not just ignored at build
time. (iii) **Identifiability** — with encoder + aux head + curriculum
active simultaneously, a fitness difference cannot be attributed to any
one of them; this is the argument for J6–J8 as *screens* (§4.7) with
attribution deferred to a restricted joint domain.

### 4.4 Component eligibility harming downstream utility (4)

Real and specific: an encoder selected on reconstruction or
self-prediction quality will preferentially encode *high-variance*
structure — which in OHLCV is dominated by volatility bursts, not by
directional signal. Anti-collapse eligibility has the same failure mode
in reverse: a representation can pass rank/variance checks while being
control-useless. Your J7 rule (eligibility gates + downstream fitness)
is correct; I would add that **eligibility thresholds must be set loose
enough to admit low-variance-but-directional encoders**, otherwise the
gate itself performs the selection we are trying to measure.

### 4.5 Cost, campaign count and fleet time — measured ranges (5)

I withdraw my earlier 3.5–4.5 h/seed ETA per your §2.3 rejection and
replace it with the measurement method: the smoke measured
**8.1 min (N14) / 8.4 min (EN4_10) / 2.9 min (E4) at 1,000 steps/epoch**,
but that figure mixes fixed evaluation cost with per-step training
cost. J1/J5 will separate them by running two step budgets and solving
`t = a + b·steps` per arm; only then do I state a range, with the
measured `a` and `b`. Campaign count for J6–J12 as designed (one domain
at a time, all workers) is 6–9 sequential campaigns — which is the
basis of my §4.7 amendment.

### 4.6 Leakage, non-stationarity, multiple comparisons, overfitting (6)

- Repeated component search over ONE 2024 validation year is the
  dominant overfitting risk in the whole program: 6–9 domains × many
  candidates all selected on the same 12 months will find validation
  noise. **Amendment:** adopt a preregistered **validation resampling
  discipline** — report each component winner's stability across
  contiguous 2024 sub-blocks, and require the final integration winner
  (J13) to hold on a rolling-origin evaluation, not a single split.
- Non-stationarity: ETH 2017–2023 spans regimes that no longer exist
  (2017 microstructure, 2021 leverage). A stationarity note per
  component belongs in its packet.
- My finding-127 fix carries its own risk, stated openly: bounding
  ineligible epochs must NOT become pressure toward trivial trading. I
  implemented it as a **separate budget** precisely so improvement
  patience is never earned by emitting noise trades; I ask you to audit
  that specific boundary.

### 4.7 A simpler ordering that reaches a portfolio-ready library sooner (7)

**Primary dissent.** Running J6–J12 each as a full DOIN campaign with
full-training downstream fitness is, by §4.5, 6–9 sequential fleet
campaigns before ETH freezes — and J12 re-optimizes the interactions
anyway. I propose:

1. **J6–J8/J10 become truncated-budget SCREENS** (short training, 2
   seeds, all workers): their product is *admitted families and
   evidence-supported ranges*, explicitly NOT champions.
2. **Full-budget authority lives only in J12** (restricted joint
   integration), warm-started from screen elites.
3. **J13 confirmation unchanged** (4 paired seeds, equal compute).

This preserves the owner's no-hand-chosen-defaults requirement (every
parameter still becomes a typed gene with screened bounds), preserves
your hierarchical structure, and removes the redundancy of selecting
component champions that J12 will re-select anyway. Second amendment:
**start D7's second-asset data/venue feasibility on CPU in parallel**
with D4 — its failure modes are independent of component research, and
discovering a data gap after ETH freezes would cost a full cycle.

### 4.8 Where I accept your correction without reservation

Your §2.2(1) (not "healthy"), §2.3 (my rejected claims — pause/resume
exactness, "pipeline proven end to end", C3 having no weaknesses,
"diffusion dominates", the ETA), §2.2(7) (diagnose before PER),
§2.2(8) (simple covariates before any TSFM), §2.2(6) (calendar and
decomposition in separate first experiments) are all correct and are
implemented as ordered. On C3 I keep one narrow reservation: for
**generator fidelity** (J11) spectral/phase metrics are the primary
instrument, because a generator has no action parity to measure.

## 5. J4 first result (direct facts, CPU track)

`lts/tools/controller_inventory.py` (lts@`d7e07a5`), run 2026-08-06:

| Seat | Unit | Controller actually driving decisions | SAC authoritative |
|---|---|---|---|
| ibkr_paper | active/running | `usdcad-4h-linear-live-v1`, 9 due-bar decisions | **No** |
| alpaca_paper | active/running | `spy-daily-linear-live-v1`, 2 due-bar decisions | **No** |
| mt5_demo | **inactive/dead** | no heartbeat, no `due_bar_decisions` table | unavailable |

Only SAC manifest present is the parity-fixture scaffold
(`eth-4h-sac-parity-fixture-v1`, execution eligibility false). **No seat
is champion-driven; every live controller is linear and is labelled
shadow/control. Gates 8–9 remain open.** The ETH Demo seat does not
currently exist as a running model-runner — that is the first thing the
exact-model parity work must establish, not assume.

## 6. Jobs not started, and why

- **J1 durable smoke / J2 four-seed calibration:** J2 requires the
  verified fleet transition, which requires J0 acceptance by you. J1
  starts on your word.
- **J3 registry / J5 diagnostics / J6+:** queued behind J0 acceptance;
  J5's CPU parts can start immediately if you prefer them ahead of J1.
- Owner has approved the program by Imperial order; I still hold the
  runtime transition for your independent verification, because
  interrupting `full-v2` before you accept the corrected pause/resume
  would risk exactly what finding 122 describes.

## 7. Requests

1. Reproduce §2's matrix and rule on the corrections.
2. Rule on the §4 amendments — especially §4.7 (screens vs full
   component campaigns) and §4.6 (validation resampling discipline).
3. Rule on §4.3(i): my deterministic repair pick is a known sampling
   bias; I propose reject-and-resample and want your preference recorded
   before J6.
4. Confirm whether J5's CPU diagnostics should precede J1.
