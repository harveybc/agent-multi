# Audit: Champion Succession, Regime Research and P0/K0 Disposition

Date: 2026-08-03 America/Bogota
Auditor: General Musashi, temporary independent auditor
Audited inputs:

- `docs/work_plan/32_CHAMPION_SUCCESSION_AND_REGIME_RESEARCH.md` at
  `agent-multi@991739cc` before correction;
- `docs/handoffs/SATOSHI_III_TO_MUSASHI_SUCCESSION_REGIME_PROPOSAL_AND_P0_K0_2026_08_03.md`;
- `docs/audits/AUDIT_SATOSHI_III_THREE_VENUE_RUNTIME_AND_K0_2026_08_03.md`;
- `lts@6daf85e`.

Broker mutations by this audit: zero. Runtime checks were status, process and
read-only operational-status queries. No order was submitted, changed or
cancelled.

## 1. Verdict

The champion/challenger and regime-specialist directions are admitted after
five corrections. Document 32 version 1.1.0 is the governing proposal. It
keeps the incumbent trading while a challenger shadows, promotes only at a
flat boundary, and preserves one sequential collaborative DOIN campaign.

The P0 correction packet is independently reproduced at head: 40 focused and
544 full LTS tests pass. A fresh direct Dragon sample also verifies that the
two MT5 user services are active and the fleet-readable status reports:

- execution bridge v2, Demo, connected, `read_only=false` and
  `execution_enabled=true`;
- terminal build 6090 and a fresh heartbeat;
- one authorized position, zero unexpected positions and zero unexpected
  orders; and
- one successful model command plus the retained failed command fact.

This direct sample resolves Satoshi's "no first-hand Dragon/MT5 sample"
unknown. It does not independently reproduce MetaEditor's exact zero-error,
zero-warning compile output. Finding 079 therefore remains pending that narrow
piece of VM evidence.

K0's `PROCEED-WITH-REVISIONS` verdict is accepted. K1 may start in its bounded
local CPU lane. K2 remains gated on the frozen-lockfile/postinstall review and
must not execute a remote fetch-and-obey installer.

## 2. Findings and Corrections

| ID | Sev | Original defect | Correction in doc 32 v1.1.0 | State |
| --- | --- | --- | --- | --- |
| AUD-F1-20260803-086 | S3 | The proposal applied a "Deflated-Sharpe-style" trial correction directly to robust weekly RAP without a defined estimator. DSR is a Sharpe-specific statistic. | Separate 52-week promotion panel; paired weekly RAP differences; moving-block bootstrap; frozen max-stat comparison family; DSR/PSR retained only as Sharpe diagnostics. | corrected_pending_independent_verification |
| AUD-F1-20260803-087 | S3 | Regime labels could be hindsight-smoothed and R0 performance heterogeneity could be mistaken for an actionable router. | Train-cutoff fit, causal filtered posteriors, no smoothed labels, explicit coverage and routable-net-headroom gates. | corrected_pending_independent_verification |
| AUD-F2-20260803-088 | S3 | Seven calendar days was allowed to carry both operational and statistical meaning. | Seven days plus 90% expected coverage proves runtime compatibility only; offline promotion evidence carries superiority. | corrected_pending_independent_verification |
| AUD-F1-20260803-089 | S3 | R1 proposed parallel DOIN domains, conflicting with the owner's single-chain, one-swarm-at-a-time invariant. | Arms are replicated sequential jobs in the canonical queue; all workers share one seed, pool and chain at a time. | corrected_pending_independent_verification |
| AUD-GEN-20260803-090 | S4 | References were vague and the AIMS ensemble-HMM paper was dated 2026 although it was published in 2025. | Primary identifiers, venues, years and DOI/arXiv references are explicit; claims are narrowed to motivation. | corrected_pending_independent_verification |

## 3. D1-D5 Disposition

- **D1:** accepted as the exact-seat, independent-panel, paired-block,
  frozen-family and three-seed contract in document 32 S1-S2.
- **D2:** RAP is primary; PSR/DSR remain Sharpe-only diagnostics. Safety
  identity/protection/reconciliation failures have zero tolerance. Drift
  thresholds require baseline evidence and versioning.
- **D3:** R0 requires causal posterior coverage and a positive lower bound
  whose practical effect exceeds measured routing cost and the declared
  relative/absolute floor.
- **D4:** R2 defaults to an isolated Omega CPU shadow service with no broker
  credentials, order socket or GPU.
- **D5:** an unchanged exact Paper/Demo seat may rotate through the
  deterministic gate plus pre/post notice. New capabilities, increased risk
  and every future Live-capital route remain explicit owner decisions.

The owner can halt or reject a promotion. No owner or agent override bypasses
the deterministic safety contracts.

## 4. Implementation Consequences

1. The current champion continues trading while evidence accumulates; there
   is no planned idle gap.
2. Promotion requires a new evidence schema and deterministic gate, not a
   prose decision or an LLM judgment.
3. A model change drains the exact seat, records post-close balance/equity,
   and seeds the successor session from those facts.
4. Curriculum relaxation affects training dynamics only. Validation,
   promotion, Paper/Demo and Live use realistic solvency and costs.
5. Regime research begins with R0 on frozen causal traces. It consumes no GPU
   unless the headroom and coverage gates pass.
6. The active DOIN job and chain are unchanged. No parallel regime chain is
   authorized.

## 5. Independent Verification Requested

Satoshi III should attempt to falsify 086-090, verify document 32's D1-D5
contract against documents 15, 19 and 29, and report any contradiction before
implementing the gate. The reporter does not close these findings.

## 6. Reproduced Commands and Runtime Scope

```text
conda run -n trading-stack pytest -q \
  tests/unit/test_mt5_execution_bridge.py \
  tests/unit/test_paper_execution_watchdog.py \
  tests/unit/test_alpaca_l1.py \
  tests/unit/test_mt5_model_runner.py
=> 40 passed

conda run -n trading-stack pytest -q tests
=> 544 passed

systemctl --user is-active \
  lts-mt5-execution-bridge.service lts-mt5-model-runner.service
=> active / active
```

Operational probes must use `systemctl --user` for the Dragon MT5 units. A
system-scope lookup returns no unit and must not be interpreted as downtime.

