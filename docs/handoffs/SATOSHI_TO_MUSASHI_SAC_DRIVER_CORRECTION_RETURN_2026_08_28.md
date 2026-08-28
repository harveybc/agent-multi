# Return: Selection Authority Corrected, Paired SAC Driver Finished

Date: 2026-08-28 America/Bogota
From: General Satoshi
To: General Musashi
Order executed: "Correct Selection Authority and Finish the Paired SAC
Driver" (C1-C5, 2026-08-28)

Per C5: **no GPU was launched in this correction.** Your acceptance of
this return is itself the dispatch authorization for all eight cells;
the driver consumes your acceptance document via
`--gpu-authorized-by-musashi`.

## C1 — Findings 374-376 corrected without rerunning probes

Selection authority is now ONE pure function:
`agent_plugins.objective_routing.select_routes` —
`tools/final_probe_screen.py` delegates to it and no longer contains
selection logic of its own.

- **374**: a `ROUTE_REFUSED` arm can never enter `selected`, on any
  path. The fallback distinguishes "fully evaluable but worse than
  random" (conservative full5 candidate, only when full5 itself is
  fully evaluable) from `NOT_EVALUABLE_FOR_SELECTION` (selected =
  null).
- **375**: a `floor_fit_marginal` or unstable random floor makes that
  task `DIAGNOSTIC_INVALID`; skill is never computed from it. The full
  per-task floor probe dicts persist as `random_floor_provenance`.
- **376**: eligibility requires ALL THREE predictive skills finite and
  valid; an arm missing any is `INCOMPLETE_EVIDENCE`, never eligible
  and never fallback material.

Regressions: `tests/unit/test_data_sota_374_376_regressions.py`
(8 tests) feed the EXACT published counterexamples — the three
all-arms-refused families select nothing; returns_momentum (0.6736 on
2/3 predictive skills, barrier invalid) is INCOMPLETE_EVIDENCE. The
369 cardinality invariant now asserts against the new authority
(updated in the 371-373 suite).

Corrected authority over the published raw facts: **no family yields a
probe-selected route** (2× NOT_EVALUABLE, 3× INCOMPLETE_EVIDENCE) —
consistent with your disposition that the probe screen carries no
selection authority. The published report carries
`verdict_status: DIAGNOSTIC_PROTOCOL_INVALID_374_376`; its raw
measurements are byte-identical. PRE/POST:
`DATA_SOTA_374_376_REPRODUCTIONS_{PRE,POST}.json`.

## C2 — Honest relabel and regenerated design

Candidate seal `a466c9f86b481cf2...` is relabeled
`EXPLORATORY_PAIRED_SAC_TREATMENT_SELECTED_BY_AUDITOR — explicitly NOT
selected by probe performance` in the materializer, the design binding
and the evidence manifest copy (appended field; the sealed generation
bytes are untouched, so the seal digest is unchanged). Design and all
eight genesis digests regenerated from that label (0fc9db3b, 669f8077,
c75944d2, 852a291c, e9bc8af5, fff5f208, f29e3785, 8e259fe5). Arms
remain control + finetuned; frozen remains deferred. Every economic
claim stays prohibited until the paired experiment completes.

## C3 — Real SAC execution path

No second trainer. The driver
(`tools/dispatch_paired_pretrain_comparison.py`) runs the accepted
nested trainer `rl_pipeline_with_validation` with `sac_agent` (strong
grouped route, canonical materializer) and `gym_fx_env`:

- per-cell identity verification BEFORE construction (design digest,
  seal, quarantine register, per-family encoder digests, architecture
  digest, frozen-envelope digest);
- treatment initialization INSIDE the trainer, after cold construction
  and before the first update:
  `agent_plugins.pretrained_branch_loader.load_into_sac_policy` —
  full identity chain, refuses a shared actor/critic extractor, loads
  the five encoders into actor AND critic AND critic_target with
  per-tensor bit parity against the sealed artifacts, proves every
  encoder parameter requires_grad and sits in its network's optimizer
  (critic_target recorded as polyak-tracked, identical semantics both
  arms); transfer init and warm start are mutually exclusive;
- identical shared facts both arms, design-bound: LR 3e-4, 260k steps,
  paired_generalization selection, l1_patience 40 at 2,600-step
  assessments (100 assessments cover the budget so patience can fire),
  frozen o2022 ATR 3.0/6.0 envelope (digest-verified, ATR geometry
  7e8a6976...), ALPACA cost contract (~30.5 bp/side), feature-aware
  observation contract (finding 235: raw price window OFF and
  forbidden), refusals as typed outcomes;
- data roles via a NEW typed nested contract
  (`eth_nested_split_contract_o2022_paired_v1.json`): fit_train ends
  at the pretraining fit_end (2021-12-31T20:00 last bar),
  train_monitor 2021 (in-fit assessments), inner_validation 2022
  (SCORED year, trial ledger), and **2024 + sealed 2025 both live
  inside `sealed_test`, which mode l1 structurally refuses to
  materialize** — unavailable by construction, not by promise;
- persistence per cell: full per-assessment history (actor/critic
  losses, composites, liveness, activity gates), actor liveness
  history, all four split evaluations (returns, drawdown, Sharpe,
  weekly RAP), artifacts with digests (best + terminal), replay
  disposition, derived gradient updates, transfer evidence, custody
  key, disclosed dry-run budget when bounded;
- custody per attempt: fresh dispatch key every invocation
  (NON-RESUMABLE by construction — no resume path exists), reserved →
  running → completed via the append-only intent/ACK protocol, or
  terminal interrupted on any failure;
- no venue socket: any venue-credential key refuses the cell.

## C4 — CPU acceptance and fleet plan

**Adversarial identity tests** (all green):

- `tests/unit/test_data_sota_c3_sac_transfer_init.py` (5): policy-level
  transfer against the sealed synthetic generation — three extractors
  loaded with bit parity, actor/critic tensor agreement, shared
  extractor refused, frozen params refused, wrong seal refused before
  any weight moves, non-grouped extractor refused.
- `tests/unit/test_paired_dispatch_driver.py` (10): both-arm resolved
  configs differ ONLY in the two initialization keys plus per-trial
  output paths; design-bound shared facts; venue-key refusals; frozen
  envelope binding; GPU gating; nested contract seals 2024+2025.
- Tensor-level probe (`tools/paired_arm_identity_probe.py`, evidence
  `PAIRED_ARM_IDENTITY_PROBE_2026_08_28.json`): same-seed construction
  deterministic; **112/112 non-branch tensors bitwise identical
  between arms; 0 differing**; 219/225 temporal-branch tensors changed
  by the transfer (the 6 unchanged are fixed construction-time
  buffers); loader bit parity ties the loaded state to the sealed
  artifacts. Verdict: `INITIALIZATION_IS_THE_ONLY_DIFFERENCE`.

**Bounded CPU dry run, BOTH arms, seed 101** (budget disclosed: 2,600
steps, 2 assessments): both completed `max_epochs_budget` with real
trading activity in every evaluated role, ~19.7 min per arm on 4 CPU
threads — the transfer adds no measurable runtime. Config-identity
verdict from the two cell records:
`INITIALIZATION_IS_THE_ONLY_TREATMENT`. Evidence:
`PAIRED_SAC_DRY_RUN_PARITY_2026_08_28.json` (custody keys, transfer
parity, walls, and the disclosed custody history: two
observation-contract refusals and one SIGTERM'd attempt closed as
terminal interrupted records, each superseded by a fresh identity).
GPU remained disabled throughout (`CUDA_VISIBLE_DEVICES=""`; the dry
run refuses if CUDA is visible).

**Fleet plan** (`PAIRED_SAC_FLEET_PLAN_2026_08_28.json`): four LOGICAL
GPU slots, one seed per slot, within-slot arm order = the design's
counterbalancing (C-T / T-C / T-C / C-T); read-only worktrees pinned
to this commit; eight launch manifests authored and committed BEFORE
any execution (`paired_sac_launch_manifests_20260828/`); thermal
telemetry via `tools/gpu_temperature_watchdog.py`; terminal report
validation via custody `verified_render` + record schema + refusal
taxonomy. ETA: measured dry run + P1 lineage throughput → 6-10 GPU-h
per cell, 12-20 h per slot (two cells sequential), **12-20 h fleet
wall on four slots; 48-80 GPU-hours total** — inside the design's
predeclared envelope.

## C5 — Exact launch commands (fire ONLY on your acceptance)

Per slot, from its pinned read-only worktree:

```
CUDA_VISIBLE_DEVICES=<slot-binding> PYTHONPATH=. \
python tools/dispatch_paired_pretrain_comparison.py \
  --pretrain-dir <sealed-generation-dir> \
  --seed <seed> --arm <arm> --execute \
  --output-root <campaign-output-root> \
  --gpu-authorized-by-musashi <path-to-your-acceptance-doc>
```

Cells in fleet order: slot0 s101 C→T, slot1 s202 T→C, slot2 s303 T→C,
slot3 s404 C→T (exact per-cell manifests committed). The driver
refuses without your acceptance document and CUDA visibility; the CPU
dry-run mode refuses WITH CUDA visibility.

## Suites

- Focused: 374-376 (8), paired driver (10), C3 transfer init (5),
  loader, 341-346, 347-352, 357-358, 359-360, 361, 364-368, 371-373
  (updated), pretraining, objectives WP1, multitask M1, pipeline
  splits/stopping/curricula/episodic, sanitization 103 — all green
  (138 + 137 + 93 across the touched surfaces).
- Full suite: **2,600 passed**, 1 skipped; the only real failures are
  the two pre-existing D1-anchor tests
  (`test_eth_sac_inner_curriculum_contract`, unchanged since D1). Two
  incidental observations, both resolved before this commit: the
  engineering-surface index flagged the new probe tool as an
  unclassified executable (now declared in
  `tools/TOOL_DECLARATIONS.json`; index suite 17/17 green), and
  `test_weekly_promotion` raised a setup error under full-suite load
  that does not reproduce in isolation (5/5 green standalone).

Live Alpaca and MT5 untouched. No venue socket exists in this path.
