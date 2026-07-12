# Phase 1 Asset-Policy Configuration

This phase mirrors the proven `predictor/examples/config/phase_*` operating
pattern while using the canonical `trading_experiment.v1` contract.

```text
phase_1_asset_policy/
  phase_1_asset_policy_solusdt_4h_sac_config.json
  inference/
    phase_1_asset_policy_solusdt_4h_sac_inference_config.json
  optimization/
    phase_1_asset_policy_solusdt_4h_sac_optimization_config.json
    phase_1_asset_policy_solusdt_4h_sac_smoke_optimization_config.json
```

The full optimization file is standalone because DOIN loads one common domain
config on every node. The smoke and inference files are overlays on the base
config and are composed by `examples/scripts/run_phase_1_asset_policy_local.sh`.

Incremental L2 stages are `action_behavior`, `critic_dynamics`,
`training_dynamics`, and `refinement`. Every stage freezes parameters outside
its declared `params`; improvements seed the next stage. L1 early stopping is
owned independently by `rl_pipeline_with_validation`.

During optimization:

- L1 watches risk-adjusted train-tail and validation performance;
- L2 maximizes the gap-penalized `train_validation_l1_score`;
- the protected test range is not evaluated;
- the exact optimizer champion is stored separately from the final retrain;
- resume state, candidate history, stage statistics and parameter JSON are
  written under `examples/results/phase_1_asset_policy/`.

This is a component-search phase. It cannot promote a live or annual champion
until the weekly walk-forward protocol evaluates the selected component.
