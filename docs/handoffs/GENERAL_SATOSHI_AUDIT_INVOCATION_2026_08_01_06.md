# General Satoshi Audit Invocation 06

Date: 2026-08-01
From: General Musashi, technical lead
To: General Satoshi, independent auditor and academic lead
Authority: project owner Harvey

General Satoshi, perform a strict independent delta audit of the new
social-trading business-reality loop. Hard criticism, alternative platforms
and concrete improvement proposals are explicitly welcome. Reproduce material
facts instead of accepting this invocation, prose documentation or another
agent's conclusion.

Do not mutate the active DOIN job, broker accounts, credentials, system
services, MT5 terminal, social-platform state or live facts. Do not open or
fund accounts. Findings are proposals until the owner accepts them.

## Required Reading

Read in this order:

1. `docs/work_plan/README.md`
2. `docs/work_plan/22_MULTI_VENUE_PAPER_EXECUTION_AND_SOCIAL_TRADING.md`
3. `docs/work_plan/27_REALTIME_FEATURE_AND_ASSET_PARITY.md`
4. `docs/work_plan/28_SOCIAL_TRADING_BUSINESS_REALITY_LOOP.md`
5. `docs/work_plan/08_IMPLEMENTATION_ROADMAP.md`
6. `docs/work_plan/10_DECISIONS_OPEN_QUESTIONS_AND_EVIDENCE.md`
7. `docs/work_plan/13_IMPLEMENTATION_STATUS_AND_TASK_LEDGER.md`
8. `docs/audits/evidence/MUSASHI_SOCIAL_TRADING_REALITY_2026_08_01.md`
9. `docs/audits/work_plan/04_OPEN_FINDINGS_REGISTER.md`
10. `docs/handoffs/CLAUDE_AUDIT_RECOVERY_PROMPT_2026_07_30.md`

Cross-repository implementation, frozen at `lts@db80d97`:

1. `/home/harveybc/Documents/GitHub/lts/app/social_trading_lab.py`
2. `/home/harveybc/Documents/GitHub/lts/app/social_trading_cli.py`
3. `/home/harveybc/Documents/GitHub/lts/examples/configs/social_trading_platform_registry_v1.json`
4. `/home/harveybc/Documents/GitHub/lts/examples/configs/social_trading_accounting_scenario_v1.json`
5. `/home/harveybc/Documents/GitHub/lts/tests/unit/test_social_trading_lab.py`
6. `/home/harveybc/Documents/GitHub/lts/docs/SOCIAL_TRADING_REALITY_LAB.md`
7. `/home/harveybc/Documents/GitHub/lts/docs/MULTI_VENUE_PAPER_EXECUTION.md`

Official sources must be opened directly, not inferred from our registry:

- https://help.ctrader.com/open-api/
- https://help.ctrader.com/ctrader-copy/faq/
- https://help.ctrader.com/ctrader-copy/investing-in-strategies/
- https://www.mql5.com/en/signals/rules
- https://www.mql5.com/en/signals/terms/provider
- https://www.darwinexzero.com/docs/what-is-darwinex-zero
- https://www.darwinexzero.com/assets
- https://www.darwinexzero.com/pricing
- https://www.etoro.com/en-us/copytrader/
- https://pamm.hfm.com/int/en/pamm-accounts/program
- https://pamm.hfm.com/int/en/fund-managers/fund-managers-faqs

## Audit Questions

### A. Accounting and copy-allocation correctness

1. Try to falsify unit accounting across manager/investor deposits,
   withdrawals, gains, losses and fee crystallization.
2. Verify whether proportional HWM adjustment on withdrawal and additive HWM
   on deposit are correct generic defaults. Identify platform-specific cases
   where crystallization order, equalization or subscription timing differs.
3. Test that recovery below a prior HWM cannot be charged again, and that
   manager capital cannot pay the manager performance/management fees.
4. Challenge whether management-fee proration, fee transfer, Decimal
   serialization and precision are sufficient for long-lived accounting.
5. Check zero-unit, full-withdrawal, loss-to-near-zero, late subscription,
   multiple fee periods, simultaneous flows and rollback/idempotency edges.
6. Verify copy volume ratio, floor/round-up policy, maximum rejection and
   tracking-error definitions. Identify instrument-unit or FX-conversion facts
   missing from the neutral contract.
7. Determine whether the SQLite facts are reconstructable and audit-grade, or
   whether append-only event hashes, balances before/after and idempotency keys
   are required before external platform reconciliation.

### B. Platform facts and alternative selection

1. Independently verify every capability, account environment, fee,
   instrument, provider requirement and SL/TP claim in the registry.
2. Specifically verify whether MQL5 demo accounts can be signal providers,
   what is required for free versus paid signals and whether copied SL/TP
   semantics satisfy subscriber-side protection in practice.
3. Verify cTrader's demo investor/provider split, equity-to-equity sizing,
   fee caps, stock exclusion, leverage/missing-instrument behavior and the
   claim that native Copy does not copy SL/TP.
4. Challenge the assumption that a custom cTrader Open API copier can always
   create valid account-local protection across brokers and instruments.
5. Verify Darwinex Zero's current asset coverage, virtual/live distinction,
   performance-fee/risk-index mechanics, Colombia eligibility and total cost.
6. Verify eToro Virtual capabilities and customer-entity/geography limits;
   confirm that it remains a manual UX control rather than an API dependency.
7. Verify HFM PAMM manager-capital, fee, allocation, rollover, withdrawal,
   country and legal requirements. Do not recommend funding it merely because
   the platform exists.
8. Search current official sources for a better API, social, PAMM, MAM or
   investable-provider laboratory. Recommend replacement/addition only when it
   supplies a missing experiment at lower expected cost/risk.

### C. Architecture and feedback loop

1. Verify that causal data, inference, portfolio/risk, venue execution and
   social accounting remain independent planes.
2. Try to find a path where platform availability incorrectly changes alpha
   ranking, or where a research-only asset reaches social execution.
3. Verify the provider/platform-observable/protected-executable set
   intersection and mandatory account-local SL/TP gate.
4. Challenge the four-disposition feedback contract: measurement,
   simulation gap, optimization variable and hard constraint.
5. Find any path by which live/social evidence could mutate an active DOIN
   chain, leak protected test information or select a model retrospectively.
6. Verify that after-fee investor metrics, cash-flow attribution and tracking
   error can be compared with provider/source metrics without horizon or unit
   ambiguity.
7. Challenge whether MQL5 Signals should precede cTrader/Open API given the
   current OANDA MT5 read-only state and mandatory 24-hour/canary gates.

### D. Security, legal and operational boundaries

1. Prove the local social lab has no broker client, secret reader, order path
   or hidden mutation capability.
2. Review whether pseudonymous IDs, database paths and scenario JSON can leak
   customer/account information when external adapters arrive.
3. Identify idempotency, replay, authorization, reconciliation, incident and
   audit-log controls required before copying or pooled capital.
4. Distinguish technical evidence from legal/tax/regulatory conclusions.
   Flag questions requiring a qualified professional rather than inventing an
   answer.
5. Challenge provider support, disclosures, marketing claims, strategy
   retirement, fee disputes and investor-flow stress coverage.
6. Verify that Hermes/social intelligence cannot activate subscriptions,
   publish providers, allocate capital, change fees or issue orders.

### E. Current front sanity check

1. Reproduce LTS tests and the deterministic scenario from a clean checkout of
   `db80d97`; verify `orders_submitted=0` and scenario/registry hashes.
2. Reproduce current DOIN job/worker/lineage/candidate facts read-only. Do not
   repair or redesign consensus during this audit.
3. Reproduce Alpaca, IBKR, MT5, shadow and social-intelligence freshness from
   current facts, not process existence.
4. Recheck Gamma storage and the equal-height tip warning with proportionate
   severity. Do not conflate them with the new social implementation.

## Required Output

Produce a dated report under `docs/audits/` and update the findings register
and recovery state only where reproduced evidence warrants it. Lead with
findings ordered by severity. For every finding include:

- stable ID, front, severity, confidence and status;
- observed fact versus inference versus proposal;
- exact reproduction and evidence path;
- effect on investor correctness, safety, profit/risk, reproducibility,
  compliance cost or operational continuity;
- smallest correction and the regression/acceptance evidence required.

Do not close findings you introduce and materially correct. Do not treat this
invocation as owner authority. End with:

1. the three strongest reasons to keep the selected platform order;
2. the three strongest reasons to change it;
3. one explicit recommended order with expected information value, owner
   effort, recurring cost and risk;
4. the five highest-value tests or code corrections before any external
   social account is connected;
5. what evidence would falsify each recommendation.

Return a concise closure message plus exact paths to every report/evidence
file the technical lead must review.
