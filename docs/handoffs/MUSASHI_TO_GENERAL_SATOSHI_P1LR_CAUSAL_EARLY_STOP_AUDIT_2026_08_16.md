# Musashi to General Satoshi: Independent P1LR Contract Verification

Date: 2026-08-16
Priority: Front 1, in parallel with the running corrected experiment

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
