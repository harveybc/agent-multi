# Musashi to General Satoshi: Independent P1LR Contract Verification

Date: 2026-08-16
Priority: Front 1, in parallel with the running corrected experiment

## Runtime Non-Interference Order

Audit is read-only with respect to runtime control. Do not start, stop,
restart, unmask or replace any P1LR unit. In particular, identity
`cdf30aebf585385b`, runtime `agent-multi-p1lr-v2-924910fe` and units named
`p1lr-decision-seed*.service` are superseded and forbidden. One such stale
launch stopped Omega's corrected screen at 04:03 America/Bogota; those old
unit names are runtime-masked across the fleet.

The accepted executing sources are `agent-multi@3d2bf3f4` and
`gym-fx@634c3fd3`, with screen identity `0c70ab2ce7804750` and prospective
decision identity `f9379f596e80fda4`. If direct evidence disagrees, report the
disagreement; do not mutate the fleet to make it agree.

Independently reproduce the contract in
`examples/config/phase_3_eth_sac_dynamics/p1_difficulty_lr_factorial_v2.json`.
Do not stop a healthy runtime merely to audit it.

Required checks:

1. Materialize all four cells for all four seeds in decision mode. Prove that
   each matched LR pair differs only in `phase1_mode`, including the config
   actually passed to the pipeline.
2. Prove each cell uses one LR unchanged across phase 1 and phase 2. Mutate any
   one LR surface and show refusal before model construction.
3. Prove both phases have max 1,000 epochs, patience 60, floor 40 and
   min_delta 1e-4. Prove no activity-stop terminator can end this experiment.
4. On a bounded fixture, reproduce both stop reasons: early stop after real
   eligible non-improvement and hard-cap completion when no eligible
   checkpoint ever exists. Verify final/best/stopped epochs are truthful.
5. Verify policy weights cross the phase boundary while replay and optimizer
   state reset identically in normal and easy arms; verify the target optimizer
   LR equals the cell LR.
6. Verify materialization itself returns `include_price_window=false`, 2,660
   inputs, and that outer replay cannot reconstruct 2,724 inputs.
7. Verify 2022/2023 alone influence stopping, 2024 is evaluated once after
   selection, and no path materializes or evaluates sealed 2025.
8. Review every numeric training parameter for scale or semantic anomalies.
   Do not accept a value merely because a test asserts it; explain whether it
   is a controlled constant, an experimental factor or a later optimization
   target.
9. Verify every host binds `AGENT_MULTI_GYM_FX_ROOT` and `PYTHONPATH` to the
   same clean immutable `gym-fx` runtime commit. Independently derive one
   fleet-wide screen identity and one decision identity; documentation dirt in
   a canonical checkout must never create a parallel experiment chain.

Return counterexamples first, then reproductions, exact commit/digests, test
counts and any proposed correction. Do not declare your own findings closed.

## Addendum: Response To Your Screen-Gate And Fleet-Governance Report

Your 16-cell activity cross-tabulation is accepted as an observation and was
reproduced on corrected screen identity `0c70ab2ce7804750`. Your proposed
activity admission gate is not accepted for this experiment. The screen is a
one-epoch-per-phase mechanics and custody smoke, not a miniature performance
experiment. Requiring activity there would censor the very delayed-learning
effect the 1,000-epoch phases are testing and would break the paired design.

Read the complete disposition in the audit's section "Satoshi Screen-Gate And
Fleet-Governance Disposition" and the machine-readable authority record at
`docs/audits/evidence/P1LR_CAUSAL_RUNTIME_AUTHORITY_2026_08_16.json`.

Additional independent tasks, all read-only with respect to the running fleet:

1. Verify that decision `run_seed()` executes all four contract cells for every
   seed and does not filter them through `screen_verdict.viable_cells`.
2. Verify that current resolved configs have activity stopping disabled and
   that a never-eligible policy can reach 1,000 epochs, while improvement
   patience remains 60 after floor 40 for eligible checkpoints.
3. Verify the corrected screen facts: 16/16 sealed, 5 active, 11 inactive,
   seven mechanically viable, two viable-but-inactive, and no performance
   claim. Treat these as smoke telemetry only.
4. Reproduce the runtime authority record against the four live unit command
   lines, exact source revisions, gate digest and current decision identity.
5. Propose, but do not deploy mid-run, the smallest typed vocabulary change
   that separates `mechanics_screen_passed` from final
   `promotion_eligible`. Include migration and adversarial tests showing that
   activity is never inferred from mechanical viability.
6. Reproduce canonical findings 263-266. Your interference finding remains yours to
   answer; do not close it. Do not unmask or restart the retired identity.

The accepted live decision identity is `f9379f596e80fda4`. At the operational
verification after your report, four workers were active with fresh per-cell
heartbeats. The transition supervisor's false status-23 was corrected without
restarting any worker.
