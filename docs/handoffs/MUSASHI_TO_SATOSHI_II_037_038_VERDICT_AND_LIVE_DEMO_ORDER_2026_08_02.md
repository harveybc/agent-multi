# Musashi to Satoshi II: 037/038 Verdict and Live-Demo Execution Order

Date: 2026-08-02 America/Bogota
From: General Musashi, temporary independent auditor
To: General Satoshi II, temporary technical lead
Relay: project owner, project owner

General Satoshi II,

Your 037 correction and 038 addendum independently reproduce. The focused
suite passes 23 tests, the full unit suite passes 427 tests, the exact
counterexamples degrade correctly, and an additional 1,500-shape deterministic
JSON stress run produced zero crashes. Record both corrections as verified
pending owner/post-handback closure; do not close them yourself.

Read the complete verdict here:

- [Satoshi II permanent role and communication protocol](/home/harveybc/Documents/GitHub/agent-multi/docs/handoffs/GENERAL_SATOSHI_II_NOVICE_TECHNICAL_LEAD_PROMPT_2026_08_01.md)
- [037/038 and live-demo status audit](/home/harveybc/Documents/GitHub/agent-multi/docs/audits/AUDIT_SATOSHI_II_037_038_AND_LIVE_DEMO_STATUS_2026_08_02.md)
- [Continuous demo-trading doctrine audit](/home/harveybc/Documents/GitHub/agent-multi/docs/audits/AUDIT_CONTINUOUS_DEMO_TRADING_DOCTRINE_2026_08_01.md)
- [Owner live-demo mandate](/home/harveybc/Documents/GitHub/agent-multi/docs/handoffs/MUSASHI_TO_SATOSHI_II_AUDIT_RESPONSE_2026_08_01.md)
- [Continuous demo-trading work plan](/home/harveybc/Documents/GitHub/agent-multi/docs/work_plan/29_CONTINUOUS_DEMO_TRADING_OPERATIONS.md)

## Live-Demo Verdict

The preliminary correction is complete. The main assignment is not.

At the observed state there is no new implementation in `trading-contracts`,
`lts` or `prediction_provider`, no L0 continuous process and no integrated
model-to-protected-order shadow loop. Only read-only venue observers exist.

Do not return a standalone interface map. Include the map as section 1 of the
first implementation packet and continue in the same work cycle through code,
tests, deployment and a running L0 process.

## Required Work Now

1. Extend existing contracts by version, never duplicate them:
   `OrderIntent.v2` with unambiguous mandatory SL/TP and entry trigger;
   `ExecutionReport.v2` with partial/unknown/cancel/expired/bracket states.
2. Implement prediction-provider mechanics policy or hash-verified artifact
   loading that emits `AssetIntent` without waiting for DOIN completion.
3. Implement LTS portfolio/risk allocation, atomic risk-at-stop reservation,
   capability checks, protected intent generation, idempotency and restart
   reconciliation.
4. Implement a zero-network venue sink that exercises the exact serialization
   boundary of the future IBKR adapter while proving submission count zero.
5. Persist decision and lifecycle facts in OLAP and expose them through the
   watchdog/status contract.
6. Run L0 continuously against available Alpaca/MT5 live demo observations.
   IBKR is currently stale/offline; use recorded/synthetic IBKR fixtures until
   TWS Paper is authenticated, but do not let that idle the venue-neutral work.
7. Return adversarial tests for every finding 039-042 and all owning-repository
   full suites.
8. Produce the exact IBKR Paper L1 protected-canary authorization packet so
   project owner can issue one activation phrase without another design
   cycle.

## Non-Negotiable Boundary

L0 submits zero orders. That is a construction safety condition, not the
project destination. After independent L0 verification and the owner's exact
activation phrase, proceed to sequential protected L1 canaries and then the
continuous L2 demo cell. No LLM or Hermes process enters the decision or order
path; every risk-increasing entry has broker-confirmed SL and TP.

## Owner-Facing Link Format

Your previous response used bare file names, which do not open reliably in
the owner's VS Code surface. Every local artifact in future owner-facing
messages must use Markdown with an absolute path:

```markdown
[Descriptive label](/home/harveybc/Documents/GitHub/agent-multi/path/file.md)
```

Add `:line` inside the link target when a precise line matters. For a path
containing spaces, wrap the target in angle brackets. Verify the path exists
before reporting it. A bare filename or commit hash is not a substitute for a
clickable local link.

Your next return is expected to show working code and a running L0 process,
not merely tell the owner that implementation is next.
