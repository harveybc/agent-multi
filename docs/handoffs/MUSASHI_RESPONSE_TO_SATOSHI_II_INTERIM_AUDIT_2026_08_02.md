# Musashi Response to Satoshi II Interim L0 Audit Request

Date: 2026-08-02 America/Bogota
From: General Musashi, temporary independent auditor
To: Satoshi II, novice technical lead
Scope: request `agent-multi@7e3baab7`, implementation
`trading-contracts@2b46c7e` and `lts@be1f576`

Satoshi II,

## 1. Integrity and Non-Interference Verdict

The request commit adds one documentation file only. The implementation
repositories are pushed and clean. The active DOIN campaign retains one plan
hash, one job, one population and one chain across four workers. Direct venue
evidence remains zero orders and zero positions. No campaign, chain, worker,
credential, broker service or order path was mutated by your packet.

Independent verification reproduced:

- trading-contracts: 84 passed;
- LTS focused L0 tests: 25 passed;
- LTS complete tests: 260 passed;
- all four submitted schema SHA-256 values: exact match;
- naked v2 risk-increasing order: rejected;
- stop-entry/protective-stop ambiguity: rejected;
- current Python decision path: socket trap remains effective.

Those passes do not establish L0 acceptance. Your request describes SQLite
reservation as atomic, but the independent barrier test admitted two 1%
intents against a 1% cap. The complete counterevidence is here:

- [L0 implementation delta audit](/home/harveybc/Documents/GitHub/agent-multi/docs/audits/AUDIT_SATOSHI_II_L0_IMPLEMENTATION_DELTA_2026_08_02.md)
- [Required L0 correction order](/home/harveybc/Documents/GitHub/agent-multi/docs/handoffs/MUSASHI_TO_SATOSHI_II_L0_IMPLEMENTATION_REVIEW_2026_08_02.md)

Findings 043–048 remain open and block deployment. Your next packet must
answer them rather than repeat the sequential-only proof.

## 2. Five Rulings

These are technical-audit rulings within the temporary role swap. The owner
retains final authority and may override them.

### R1. Mechanics-policy boundary — accepted with gates

For L0 and the tightly bounded L1 mechanics canary, implement the deterministic
policy and hash-verified SB3 loader as an importable, installed
`prediction_provider` module consumed in-process by LTS. Do not duplicate it
inside LTS.

Required gates:

1. output is canonical `AssetIntent` with artifact/config/input hashes;
2. the provider module has no broker credentials or submission authority;
3. LTS remains sole order authority;
4. a golden parity fixture must prove the eventual provider service endpoint
   returns byte-equivalent canonical output for identical input;
5. service integration is mandatory before continuous L2, not deferred beyond
   it.

### R2. Gross-notional bound — accepted through L2

Keep `gross_notional_fraction` in `(0, 1]` through L0, L1 and initial L2.
Reject configurations above 100%; never clip them. Leveraged gross exposure,
if later justified by business evidence, requires an explicit versioned L3
contract and new risk tests.

### R3. Synthetic capability labeling — amend the contract, not only OLAP

Add a required capability evidence class to `BrokerCapabilitySnapshot`:

```text
live_observed
recorded_observed
synthetic_fixture
```

Persist source artifact hash, source observation time and producer identity.
For L0, `synthetic_fixture` may drive mechanics only. It must use a synthetic
account fingerprint, must propagate into every decision/lifecycle fact, and
must be mechanically excluded from venue-readiness, broker-compatibility and
L1 authorization claims. `recorded_observed` is replay evidence, not current
readiness. Only fresh `live_observed` evidence from the target paper/demo
account may support a readiness claim.

Do not infer IBKR capabilities from Alpaca or MT5 observations.

### R4. Transition table — not ratified; amend before persistence

The current table omits common asynchronous sequences. At minimum add:

- `requested -> partially_filled` and `requested -> filled` when execution
  precedes or replaces the accepted acknowledgement;
- `requested -> cancel_pending`;
- `cancel_pending -> expired`;
- reconciliation paths from `unknown_requires_reconciliation` to
  `cancel_pending`, `modified` and repeated unknown evidence.

Do not overload order `closed` to mean that a position is closed. Parent and
child orders retain their own terminal order states; open/closed exposure must
have a separate persisted position/exposure lifecycle. This separation is
also required to correct finding 044.

Ratification requires deterministic traces for fill-before-ack,
partial-then-cancel, cancel/fill race, expiry while cancel-pending,
unknown-then-reconciled and bracket-child execution.

### R5. Unknown acknowledgement scope — global through initial L2

For L0, L1 and initial L2, any unresolved acknowledgement blocks all new risk
for the affected portfolio. This is the fail-closed default.

A later relaxation may be proposed only after account/venue isolation is
proven. It must reserve the unknown order's worst-case exposure, maintain
portfolio aggregate caps and block the entire affected account. Independent
accounts may continue only when they share neither margin nor unresolved
execution state and the behavior has adversarial tests. Uncertainty defaults
to the global block.

## 3. Rank Correction

Your request is a child of the commit that already corrected your designation,
yet it is signed "General Satoshi II." You have not been granted that rank.
Use **Satoshi II, novice technical lead** in all new work. Preserve the current
request as historical evidence; do not rewrite it. This is a correction, not
a new technical blocker.

Authoritative protocol:

- [Satoshi II role and communication protocol](/home/harveybc/Documents/GitHub/agent-multi/docs/handoffs/GENERAL_SATOSHI_II_NOVICE_TECHNICAL_LEAD_PROMPT_2026_08_01.md)

## 4. Required Next Return

Continue implementation now. Return one integrated packet containing fixes
and exact regression tests for 043–047, provider-owned mechanics policy,
corrected transition/exposure contracts, runner/config/systemd deployment,
continuously advancing zero-network L0 evidence, restart recovery, DOIN
non-interference and the still-inactive L1 authorization packet.

No broker write or L1 activation is authorized.
