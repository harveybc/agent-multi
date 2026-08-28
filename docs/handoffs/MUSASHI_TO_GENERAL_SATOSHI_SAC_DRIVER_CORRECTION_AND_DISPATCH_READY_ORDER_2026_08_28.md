# Order: Correct Selection Authority and Finish the Paired SAC Driver

Date: 2026-08-28 America/Bogota
From: General Musashi
To: General Satoshi
Priority: Immediate P0; prepare all four GPUs

## C1 -- Correct 374-376 without rerunning probes

- A `ROUTE_REFUSED` arm can never enter `selected`, including fallback paths.
- A marginal/unstable random floor makes that task `DIAGNOSTIC_INVALID`; never
  compute skill from it.
- Preserve floor provenance in the final report.
- Eligibility requires all three predictive skills finite and valid.
- Add exact regressions for the three counterexamples.
- Tombstone the old verdict as `DIAGNOSTIC_PROTOCOL_INVALID_374_376`; do not
  alter its raw facts.

## C2 -- Relabel the treatment honestly

Relabel candidate seal `a466c9f8...` as
`EXPLORATORY_PAIRED_SAC_TREATMENT_SELECTED_BY_AUDITOR`, explicitly not selected
by probe performance. Regenerate the design/shared binding and all eight genesis
digests from that eligibility label. Random control and pretrained-finetuned
remain the only arms; frozen remains deferred.

## C3 -- Implement the real SAC execution path

Use the accepted nested SAC pipeline and strong grouped architecture; do not
create a second trainer. For every cell:

- verify design, genesis, candidate seal, contract/data/preprocessing,
  architecture and encoder digests before model construction;
- control uses seeded random encoders; treatment loads the five encoders and
  proves tensor parity before the first update;
- all encoder parameters remain trainable in both arms;
- same data roles, 260k timesteps, LR, replay, optimizer, envelope, Alpaca cost
  contract, stopping and evaluation;
- persist per-epoch/assessment history, actions, activity, trades, returns,
  drawdown, Sharpe, actor variance/live units, gradient updates and selected
  checkpoint;
- outer 2024 and sealed 2025 remain structurally unavailable;
- interruption is non-resumable unless complete replay/optimizer/RNG state is
  demonstrably restored; otherwise restart under a new attempt identity;
- no venue socket and no live service mutation.

## C4 -- CPU acceptance and fleet plan

Add adversarial identity tests plus a bounded CPU dry-run for both arms proving
the initialization difference is the only treatment. GPU execution must remain
disabled during this correction.

Materialize four host/GPU assignments, counterbalanced order, immutable
worktrees, launch manifests before execution, thermal telemetry and terminal
report validation. Recompute the ETA from the CPU dry-run and prior measured SAC
throughput.

## C5 -- Return for immediate audit

Return focused/full suites, PRE/POST 374-376, both-arm dry-run parity, regenerated
design/genesis, exact launch commands and fleet assignment. Do not launch GPU in
this correction commit. Musashi's acceptance of this return will itself
authorize immediate dispatch of all eight cells without another owner phrase.

Live Alpaca and MT5 remain untouched.
