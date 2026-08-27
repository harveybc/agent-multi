from setuptools import setup, find_packages

setup(
    name="agent-multi",
    version="0.4.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "agent-multi=app.main:main",
            "doin-campaign-supervisor=app.campaign_supervisor:main",
        ],
        "env.plugins": [
            "gym_fx_env=env_plugins.gym_fx_env:Plugin",
        ],
        "agent.plugins": [
            "ppo_agent=agent_plugins.ppo_agent:Plugin",
            "dqn_agent=agent_plugins.dqn_agent:Plugin",
            "sac_agent=agent_plugins.sac_agent:Plugin",
            "project3_sac_actor_critic_agent=agent_plugins.project3_sac_actor_critic_agent:Plugin",
            "random_agent=agent_plugins.random_agent:Plugin",
            "buy_hold_agent=agent_plugins.buy_hold_agent:Plugin",
            "no_trade_agent=agent_plugins.no_trade_agent:Plugin",
            "momentum_agent=agent_plugins.momentum_agent:Plugin",
            "reversal_agent=agent_plugins.reversal_agent:Plugin",
        ],
        "pipeline.plugins": [
            "rl_pipeline=pipeline_plugins.rl_pipeline:PipelinePlugin",
            "rl_pipeline_with_validation=pipeline_plugins.rl_pipeline_with_validation:PipelinePlugin",
            "rl_pipeline_with_solvency_curriculum=pipeline_plugins.rl_pipeline_with_solvency_curriculum:PipelinePlugin",
            "rl_pipeline_with_execution_curriculum=pipeline_plugins.rl_pipeline_with_execution_curriculum:PipelinePlugin",
        ],
        "execution_policy.plugins": [
            "adaptive_order_router=execution_policy_plugins.adaptive_order_router:Plugin",
        ],
        "feature_branch.plugins": [
            "mlp_branch=feature_branch_plugins.mlp_branch:Plugin",
            "gru_branch=feature_branch_plugins.gru_branch:Plugin",
            "tcn_branch=feature_branch_plugins.tcn_branch:Plugin",
            "transformer_branch=feature_branch_plugins.transformer_branch:Plugin",
            "patchtst_branch=feature_branch_plugins.patchtst_branch:Plugin",
            "tft_branch=feature_branch_plugins.tft_branch:Plugin",
            "timesnet_branch=feature_branch_plugins.timesnet_branch:Plugin",
        ],
        "pretrain_balancing.plugins": [
            "inverse_initial_loss=pretrain_optimizer_plugins.balancing_inverse_initial_loss:Plugin",
            "frozen_gradient_norm=pretrain_optimizer_plugins.balancing_frozen_gradient_norm:Plugin",
        ],
        "pretrain_combiner.plugins": [
            "ordinary_sum=pretrain_optimizer_plugins.combiner_ordinary_sum:Plugin",
            "pcgrad=pretrain_optimizer_plugins.combiner_pcgrad:Plugin",
        ],
        "feature_fusion.plugins": [
            "concat_fusion=feature_fusion_plugins.concat_fusion:Plugin",
            "gated_fusion=feature_fusion_plugins.gated_fusion:Plugin",
            "cross_family_attention=feature_fusion_plugins.cross_family_attention:Plugin",
        ],
        "pretraining_objective.plugins": [
            "next_step_huber=pretraining_objective_plugins.next_step_huber:Plugin",
            "direction_bce=pretraining_objective_plugins.direction_bce:Plugin",
        ],
        "optimizer.plugins": [
            "default_optimizer=optimizer_plugins.default_optimizer:Plugin",
            "project3_full_genome_optimizer=optimizer_plugins.project3_full_genome_optimizer:Plugin",
        ],
    },
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "psutil",
        "requests",
        "gymnasium",
        "backtrader",
        "stable-baselines3>=2.3",
        "deap",
        "trading-contracts>=0.1.0",
    ],
    extras_require={
        "dev": ["pytest"],
    },
    author="Harvey Bastidas",
    author_email="your.email@example.com",
    description=(
        "Plugin-based RL trainer/optimizer that consumes the gym-fx environment."
    ),
)
