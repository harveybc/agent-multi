# Audit: M0 Evidence, M1 Factorial and M0-X Proposal

Date: 2026-08-08 America/Bogota  
Auditor: General Musashi, independent verifier  
Audited head: `agent-multi@99bb7fff9c78999fee6ed9b5d5060a7860d61dae`  
Evidence root: `~/.local/share/agent-multi/eth_sac_inner_curriculum_m0_20260807_v1`  
Runtime mutation: none; no training, network request, queue launch or venue action

## 1. Verdict

**M0's `mechanism_pass` is rejected as evidence for an easy-to-normal
curriculum.** Every one of the 12 arms that declared an easy phase handed the
unchanged epoch-0 anchor to normal training. The trained easy epoch was rejected
by the normal-handoff gate. M0 therefore never tested easy-trained weights.

The 16 terminal artifacts are real, loadable policies and all 16 differ from
their anchors at all 32 policy tensors. The useful, narrower observation is:

> Starting again from the mature ETH anchor, one normal epoch at `3e-5` or
> `1e-5` retained activity in 4/4 seeds; one normal epoch at `1e-4` and two
> normal epochs at `1e-4` collapsed to zero trades.

That is evidence about normal fine-tuning rate and duration, not evidence for
easy pretraining and not an equal-effective-training comparison. Preserve the
records and raw metrics; withdraw the causal label and disable the queued
`mechanism_pass` successor.

M1 and M0-X must not launch from the current contracts. This does not require
idling the fleet: unrelated valid jobs may continue, while the corrected
handoff and one-seed CPU/GPU smoke are built and independently checked.

## 2. Findings

### AUD-F1-20260808-159 - S2 - All 12 easy arms hand off epoch-0 anchor weights

The easy pipeline evaluates and may save the warm-start baseline as epoch 0,
then selects only checkpoints that already pass both easy activity and a normal
handoff activity probe. In every M0 easy arm, epoch 1 was active under easy but
had zero normal train-tail/validation trades, so epoch 0 remained selected.
Normal training then reloaded that epoch-0 artifact.

Independent tensor comparison found `0/32` changed policy tensors and exactly
zero absolute delta between `model.post_easy.zip` and the anchor for all 12
easy arms. Every metadata file reports `best_easy_epoch=0` with an easy budget
of one epoch. The arm record also loses this fact because it asks for a
nonexistent `post_easy.probe`; all 12 report `post_easy_activity=null`.

Source: `pipeline_plugins/rl_pipeline_with_solvency_curriculum.py:324-365` and
`:422-437`; `tools/eth_sac_inner_curriculum_screen.py:197-210` and `:260-265`.
Executable evidence: `all_12_easy_handoffs_are_epoch_zero=true`,
`all_12_easy_handoffs_equal_anchor_weights=true` and
`all_12_records_omit_post_easy_activity=true` in the independent reproducer.

Impact: the aggregation statement at
`tools/aggregate_eth_sac_inner_curriculum.py:203-223` attributes survival to
"easy plus gentle normal fine-tuning" without any easy-trained handoff. The
generated successor currently says `launch_eligible=true`; that disposition is
invalid and must be quarantined append-only.

### AUD-F1-20260808-160 - S2 - ZIP hash inequality is not proof of changed weights

The runner defines `weights_changed_from_anchor` as terminal archive SHA being
different from anchor archive SHA, and `terminal_usable` consumes that result.
The aggregator then trusts the boolean. Re-saving the unmodified seed-101
anchor produced a different ZIP SHA while all `32/32` policy tensors remained
bit-identical.

Source: `tools/eth_sac_inner_curriculum_screen.py:223-273` and
`tools/aggregate_eth_sac_inner_curriculum.py:173-200`. The current 16 terminal
models did actually change at `32/32` policy tensors, so this finding does not
invalidate their raw terminal metrics; it invalidates the verifier used to
establish that fact.

Required correction: content-address the canonical policy state dictionary and
record numeric tensor distance for anchor -> post-easy and post-easy ->
terminal. Archive SHA remains an artifact-integrity field, never a weight-change
field.

### AUD-F1-20260808-161 - S2 - M1's proposed easy factor is confounded by a phase reset

M1 compares uninterrupted `N14` with `E4_N10`. The E arm reconstructs a fresh
SAC at the boundary, transfers policy weights, resets optimizer state and uses
a fresh replay buffer; N14 has no matched boundary. It therefore changes easy
dynamics, replay history, optimizer state, entropy/target state behavior and
effective gradient-update count together. A 2x2 table over schedule and LR does
not isolate an easy main effect.

Source: `pipeline_plugins/rl_pipeline_with_solvency_curriculum.py:426-448`,
`agent_plugins/sac_agent.py:361-445` and the M1 arm table in
`examples/config/phase_3_eth_sac_dynamics/m1_factorial_contract_M03.json`.

Required correction: use a matched normal-boundary control, for example
`{N4_R_N10, E4_R_N10} x {M1.0, M0.1}`, where both arms rebuild at the same
boundary and both exclude phase-1 replay from phase 2. Hand off the terminal
trained phase-1 policy, not a baseline selected with future normal activity.
An uninterrupted N14 arm may remain diagnostic but cannot be the causal normal
cell.

### AUD-F1-20260808-162 - S2 - The generic M0-X runner remains hard-bound to ETH

The v2 contract declares `asset=USDCAD`, 48 features and window 1, but
`run_m0_arm()` unconditionally calls the ETH D1 `_base_config()`. That helper
pins the ETH CSV, ETH base config, ETH data hash, ETH splits and the ETH
observation manifest (83 features, window 32). The USDCAD contract changes only
the anchor and rates; it does not bind an executable USDCAD data/preprocessing/
observation contract.

Source: `tools/eth_sac_inner_curriculum_screen.py:166-184` and
`tools/eth_curriculum_decision_experiment.py:45-64`, `:83-108`, `:327-352`.
Independent evidence reports `m0x_declared_asset=USDCAD` while the invoked base
data is `ethusdt_4h_tech_stat_full_model_ready.csv`.

Impact: M0-X cannot run as described and may fail by shape or, worse, emit an
ETH result labeled USDCAD.

### AUD-F1-20260808-163 - S2 - V2 has no executable decision aggregator or exact outcome contract

The only aggregator is fixed to M0 v1's contract, seeds, arms and interpretation.
It cannot aggregate M1 or M0-X. The v2 JSON uses undefined phrases such as
"materially worse", "comparably" and "pattern repeats"; no executable threshold,
paired estimator, missing-cell rule or deterministic successor is implemented.

The two M1 variants also share one output root. Execution identity is delegated
to the ETH D1 helper and does not include v2 contract hash, experiment, asset,
factor levels or winner selection. Common arms therefore collide across M0.3
and M0.1 variants.

Source: `tools/aggregate_eth_sac_inner_curriculum.py:29-35`, `:203-223` and
`:368-470`; `tools/eth_curriculum_decision_experiment.py:83-108`.

Required correction: one v2 aggregator must validate the exact factorial,
contract/data/code/artifact bindings, paired cells, tensor handoffs and explicit
`EASY_CONTRIBUTES | LR_ONLY | INTERACTION | INCONCLUSIVE` rules before emitting
any successor. Execution and output identities must include the full contract
SHA and asset/system manifest.

### AUD-F1-20260808-164 - S3 - V2 validation and artifact durability are incomplete

`validate_contract_v2()` accepts a one-cell "factorial" and accepts a
`winner_multiplier` that disagrees with its arm levels. It does not prove the
asset base contract, exact anchor-manifest hash, precondition re-evaluation,
worker/seed topology or unique output identity. The USDCAD anchor manifest
hashes only the model ZIP; its data, resolved config and dataset manifest are
described by sibling names/prefixes rather than exact hashes.

The reported 21/21 replicas are packets/records plus one manifest. The
replication function does not replicate any terminal or post-easy model ZIP.
Those observations are valid evidence-record replicas, but not durable model
artifact replicas.

Source: `tools/eth_sac_inner_curriculum_screen.py:52-103`;
`tools/aggregate_eth_sac_inner_curriculum.py:321-362`;
`docs/audits/evidence/eth_sac_inner_curriculum/USDCAD_SEED2703_ANCHOR_MANIFEST.json`.

### AUD-F1-20260808-165 - S2 - Proposed M0-X evidence horizon is insufficient

The proposed USDCAD system manifest declares only 1,604 training rows at 4h,
plus 1,609 validation and 1,589 test rows. It supplies no exact date bounds in
the immutable manifest. That does not establish the owner-required multi-year
training history or sufficient regime coverage for a cross-system mechanism
decision. A short run can still prove mechanics; it cannot falsify transfer.

M0-X is blocked until USDCAD (or a replacement second asset) has at least four
complete chronological training years at 4h, one full validation year and a
separate sealed test year, with exact row/date/hash and regime-coverage facts.
For data at 1h or finer, the owner's absolute minimum is one complete training
year, although the longer matched history remains preferred. No synthetic
padding or overlap may satisfy the requirement.

## 3. Evidence Preserved

- 16/16 terminal SAC archives load locally.
- Independent policy comparison: all 16 terminal models changed all 32 policy
  tensors relative to their anchors.
- 21/21 declared remote evidence-record observations were present and matched
  when sampled; these are not model replicas.
- M0 raw validation metrics remain usable with explicit units. At reduced LR,
  seeds 101/202/404 produced about `+0.0551% mean weekly return`, `+2.903% total
  return`, `2.838% max drawdown` and 122 trades. Seed 303 was negative
  (`-0.0127%` to `-0.0162% mean weekly`, `-0.722%` to `-0.910% total`, `3.212%
  max drawdown`, 92-125 trades). No profit claim or margin attribution follows.
- 2025 remained excluded from M0 selection evidence.

## 4. Verification

```text
independent reproducer: all 9 counterexample predicates true
network_used=false; training_started=false; runtime_mutated=false
v2 supplied contract tests: 13 passed
full isolated-worktree suite: 739 passed, 13 failed
```

The 13 full-suite failures are environmental in this isolated worktree: ignored
result fixtures such as the pinned ETH `config_out.json` and sibling
`/tmp/doin-node` templates are absent. They are disclosed and are not counted
as correction regressions or passes.

Canonical reproducer:

`docs/audits/evidence/SATOSHI_III_M0_M0X_REPRO_2026_08_08.py`

Executable repair specification:

`docs/handoffs/MUSASHI_TO_GENERAL_SATOSHI_III_EMERGENCY_M0_M1_REPAIR_SPEC_2026_08_08.md`

## 5. Runtime Disposition

1. Preserve all M0 raw records and artifacts.
2. Append a correction envelope to M0; do not overwrite historical evidence.
3. Disable/quarantine `queue/m0_successor_mechanism_pass.json` before any
   consumer can launch it.
4. Do not launch current M1 or M0-X contracts.
5. Keep unrelated valid pooled work running; this audit does not authorize an
   idle fleet or a mutation of another experiment.
6. A corrected one-seed handoff smoke must pass before the bounded four-seed M1
   run. M0-X follows M1 and never blocks valid independent work.
