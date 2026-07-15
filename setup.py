from setuptools import setup, find_packages

setup(
    name="agent-multi",
    version="0.3.0",
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
        ],
        "optimizer.plugins": [
            "default_optimizer=optimizer_plugins.default_optimizer:Plugin",
        ],
    },
    install_requires=[
        "numpy",
        "pandas",
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
