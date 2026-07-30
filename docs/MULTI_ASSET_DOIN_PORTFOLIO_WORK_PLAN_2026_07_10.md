# Adaptive Multi-Asset Trading Work Plan

The work plan is maintained as a versioned composite document suite rather than
one monolithic specification.

Start here:

- [Master index](work_plan/README.md)
- [System architecture](work_plan/01_SYSTEM_ARCHITECTURE.md)
- [Contracts and configuration](work_plan/02_CONTRACTS_AND_CONFIGURATION.md)
- [Multi-asset simulation and execution](work_plan/03_MULTI_ASSET_SIMULATION_AND_EXECUTION.md)
- [Models, policies, and training](work_plan/04_MODELS_POLICIES_AND_TRAINING.md)
- [DOIN trading domain integration](work_plan/05_DOIN_TRADING_DOMAIN_INTEGRATION.md)
- [OLAP metrics and lineage](work_plan/06_OLAP_METRICS_AND_LINEAGE.md)
- [Serving, LTS, and OANDA](work_plan/07_SERVING_LTS_AND_OANDA.md)
- [Implementation roadmap](work_plan/08_IMPLEMENTATION_ROADMAP.md)
- [Testing, security, and operations](work_plan/09_TESTING_SECURITY_AND_OPERATIONS.md)
- [Decisions, open questions, and evidence](work_plan/10_DECISIONS_OPEN_QUESTIONS_AND_EVIDENCE.md)
- [DOIN configuration profiles](work_plan/11_DOIN_CONFIGURATION_PROFILES.md)
- [Collaborative implementation and review](work_plan/12_COLLABORATIVE_IMPLEMENTATION_AND_REVIEW.md)
- [Implementation status and task ledger](work_plan/13_IMPLEMENTATION_STATUS_AND_TASK_LEDGER.md)
- [Simulation engine selection](work_plan/14_SIMULATION_ENGINE_SELECTION_2026_07_11.md)
- [Distributed campaign lifecycle](work_plan/15_DISTRIBUTED_CAMPAIGN_LIFECYCLE.md)
- [Flat-fitness root cause](work_plan/16_FLAT_FITNESS_ROOT_CAUSE_2026_07_19.md)
- [Data and preprocessing evidence recovery](work_plan/17_DATA_PREPROCESSING_EVIDENCE_RECOVERY.md)
- [Full-genome per-asset optimization](work_plan/18_FULL_GENOME_PER_ASSET_OPTIMIZATION.md)
- [Execution curriculum and order routing](work_plan/19_EXECUTION_CURRICULUM_AND_ORDER_ROUTING.md)
- [Protected execution incident](work_plan/20_PROTECTED_EXECUTION_ACTIVITY_GATE_INCIDENT_2026_07_29.md)
- [OANDA Practice execution lab](work_plan/21_OANDA_PRACTICE_EXECUTION_REALITY_LAB.md)
- [Multi-venue paper execution and social trading](work_plan/22_MULTI_VENUE_PAPER_EXECUTION_AND_SOCIAL_TRADING.md)

The central architectural premise is that DOIN is already functional and is
the stable decentralized optimization/inference substrate. This project adds
trading domains through its existing plugin, configuration, verification,
champion migration, Proof of Optimization, and OLAP mechanisms.
