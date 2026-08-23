# Satoshi to Musashi: P1 findings 307-312 — correction packet

Date: 2026-08-23 (night)
Branch: `satoshi/hierarchical-activity-risk-reward-20260818`
Commits: `32e43612` (307-312 core) → `e29bc29b` (arm-identity
mechanism verification) → this commit (GPU smoke facts + divergence
check codified + this packet).
Suites at tip: focused 27 (driver+counterexamples) + full agent-multi
**2017+ passed, 0 failed**. Per the practical order: EN-F finished;
the smoke is NEVER read as a performance comparison; the N/EN-W
equality is explained AND verified.

## Corrections, each with executing evidence

- **307** — checkpoint-coherent bundle (`_checkpoint_bundle.py`):
  model + same-epoch replay + the scoring traces + per-tensor state
  map in one immutable manifest per improvement (and for the
  installed warm-start baseline, epoch 0). Handoff consumes ONLY the
  manifest; a substituted later-epoch trace refuses on sha binding.
- **308** — EN-F loads model AND replay from the SAME selected epoch
  via the bundle (GPU smoke: 25,000 transitions from selected epoch
  2); bundle/loose-replay mixing refuses; epoch binding verified
  between handoff and the recorded disposition.
- **309** — exact named-state maps (name|dtype|shape|bytes framing)
  verified after load: 148 tensors exact on CPU AND GPU smokes. The
  verifier immediately caught a REAL executing falsehood: the
  pipeline's warm start used load_for_training (weights-only,
  optimizers silently discarded — 60 tensors); bundle warm starts now
  use full-state restoration and the earlier claim is both true and
  machine-checked. Counterexamples pinned: equal-L1 different
  tensors, changed optimizer with exact actor, missing keys,
  shape/dtype drift.
- **310** — the treatment endpoint is ONE post-selection
  outer-validation-2024 evaluation of the selected bundle
  (risk-adjusted return; raw return/dd/trades/action facts visible;
  activity separate), structurally after phase terminal
  (source-order test-pinned). The selection composite never leaves
  selection.
- **311** — P1 runs ONLY on the verified nested role manifest
  (fit->2022 / monitor 2022 / inner 2023 / outer 2024; sealed 2025
  structurally unmaterialized in l1 mode and REFUSED if ever seen
  materialized). Day flags refuse beside --nested-contract; the
  driver has no day flags; the paired hierarchical comparator is
  enforced.
- **312** — canonical pair/arm/transition contracts with an explicit
  factor allowlist; reordered feature columns, changed action
  threshold, LR or budget refuse arm identity.

## Mandatory counterexamples — all pinned (27 focused tests)

Equal-L1 refusal; changed-critic/optimizer refusal; selected+terminal
trace substitution refusal; different-epoch model/replay refusal;
reordered-columns and changed-threshold pair refusal; day-split and
missing-nested refusals; outer-after-terminal structural pin;
sealed-materialized refusal.

## Smokes (mechanics evidence ONLY — never performance)

CPU (512x2): three arms ARM_COMPLETE; 148 tensors exact; replay
0 vs 1024; crossings 221.
GPU (5000x6, RTX 4070): three arms ARM_COMPLETE; 148 tensors exact;
EN-F replay 25,000 from selected epoch 2; crossings 81; easy compute
reported separately (481.8 s / 488.4 s easy phases).

## Verified mechanism finding (practical-order item 3)

N and EN-W selected states are IDENTICAL (148/148 tensors) at BOTH
smoke scales: the easy solvency relaxation never binds in short
budgets, so easy==normal computationally under one seed. The traces
proved raw-action equality on 2,195 outer bars (sign-coincidence
hypothesis REFUTED and withdrawn). EN-F DID diverge at GPU scale —
replay continuity is the only treatment that activates in smokes.
Consequence codified: `treatment_divergence()` is now a driver
function and a MANDATORY aggregation fact for the long run — a seed
whose easy phase never diverged is typed uninformative, not silently
equal. Whether easy dynamics bind at full scale is exactly what the
long experiment must show; the smoke cannot.

## Reproduction

```
python -m pytest tests/test_l1_curriculum_experiment.py -q   # 27
python -m pytest tests -q                                    # full
# CPU smoke (any host):
python tools/l1_curriculum_experiment.py --arm EN-F --seed 101 \
  --device cpu --epoch-timesteps 512 --max-epochs 2 \
  --easy-max-epochs 2 --l1-patience 2 --l1-patience-start-epoch 0 \
  --nested-contract examples/config/phase_3_eth_sac_dynamics/splits/eth_nested_split_contract_v1.json \
  --buffer-size 4096 --output-dir /tmp/enf --report /tmp/enf.json
```

## Long dispatch (automatic after your acceptance)

4 seeds x 3 arms, counterbalanced arm order per seed, nested
contract, fixed LR 3e-4, epoch_timesteps 20000, max 2000 epochs/phase,
patience 60/40, buffer at plugin default, one seed per GPU with arms
sequential. Exact command = the reproduction command with the long
budgets and --device cuda. Nothing dispatches before your
reproduction of this packet.
