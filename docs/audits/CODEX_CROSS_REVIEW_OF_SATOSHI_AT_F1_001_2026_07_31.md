# Musashi Cross-Review of Satoshi AT-F1-001

Review ID: `MUS-REV-20260731-AT-F1-001`
Date: 2026-07-31 America/Bogota
Reviewer: Musashi
Reviewed artifact:
`AUDIT_GS_COUNTER_RESPONSE_AND_AT_F1_001_2026_07_31.md`, SHA-256
`8f1cb79f03b23f98415ef2083150a638d32f5255ff6460f2ef95a9844e5d13e2`

## Findings

### AUD-F1-20260731-026: AT-F1-001 passed a non-weekly L2 as weekly

Provisional severity: **S2**. Confidence: high. State: open.

The audit task requires reconstruction of L2 and confirmation of
"weekly-fraction units end to end". Satoshi reconstructed the stored scalar
exactly, but did not verify its units. The active implementation resolves the
inner selection metric to `risk_adjusted_return`, and
`_selection_value()` computes that value as full-period `total_return` minus
full-period maximum drawdown. It does not use `mean_weekly_rap`.

Atomic block-8 evidence makes the mismatch explicit:

| Split | Weeks | Full-period RAP used by L2 | Mean weekly RAP |
| --- | ---: | ---: | ---: |
| train tail | 3 | `+0.00017968079917197056` | `+0.00005090569047665089` |
| validation | 53 | `+0.0013898804150448444` | `-0.00007597242102824658` |

The stored calculation is internally exact:

```text
mean(total-period RAPs) - 0.25 * gap = +0.00048223070314018903
```

The calculation required by the task's weekly-unit statement is:

```text
mean(mean-weekly RAPs) - 0.25 * gap = -0.00004425289315202221
```

This is not a display-only difference: the sign changes. The current job-0
campaign may intentionally use its documented full-period proxy before the
job-1 robust-weekly curriculum, but that intent conflicts with AT-F1-001's
explicit unit contract. Reproducing configured behavior does not prove
conformance to a conflicting requirement.

Impact: Satoshi's `PASS` is rejected. `AT-F1-001` is reported with a material
finding and cannot be closed until Harvey decides which objective contract is
authoritative. No live worker, chain, config or champion is changed by this
review; changing fitness semantics inside the active chain would invalidate
comparability.

### AUD-GEN-20260731-027: impossible report chronology

Provisional severity: **S3**. Confidence: high. State: open.

The report declares `2026-07-31 04:40 America/Bogota`, while it claims to have
audited commits created later:

```text
8e63b7dc author/committer time 2026-07-31 04:58:31 -0500
69d06a24 author/committer time 2026-07-31 05:16:19 -0500
```

The report file was created locally at `2026-07-31 13:31:23 -0500`. The
declared timestamp therefore cannot identify either the evidence cutoff or
the report-generation time. This is a provenance defect. Correct it with an
addendum that distinguishes `evidence_observed_at`, `report_started_at` and
`report_written_at`; do not rewrite the preserved original.

### AUD-GEN-20260731-028: the audit session failed its own handoff contract

Provisional severity: **S3**. Confidence: high. State: open.

The report says `AT-F1-001` is `reported`, opens
`AUD-GEN-20260731-025`, and confirms that exactly one file was written.
However:

- the backlog still listed `AT-F1-001` as `scheduled`;
- the open-findings register omitted `AUD-GEN-20260731-025`;
- the recovery prompt still named an older next invocation.

The audit lifecycle explicitly says a session that does not update all three
state files has failed its handoff duty, regardless of the report quality.
This cross-review reconciles the backlog and register so later sessions do not
inherit false state. Satoshi must still publish a correction addendum and
refresh his recovery state.

## Independently Verified Claims

The following parts of Satoshi's work survive cross-review:

- config file SHA-256 is
  `31b506e03926280836b12c30c321e6996f2bf3951a748ede3b555806328e3f10`;
- focused `agent-multi` tests reproduce as `9 passed, 390 deselected`, and the
  firewall test reproduces as `1 passed`;
- `gym-fx` reproduces as `75 passed`;
- the five cited GitHub Actions runs are successful at the exact reported
  commits;
- block 8 contains train-tail `5` and validation `166` completed trades, so
  both configured activity floors pass;
- decoding `_model_b64` from block 8 produces exactly `48,801,983` bytes with
  SHA-256
  `892fbee0f00ecf7b93ca6ed2bea05ca4db3cd09aa557a8cd16b9698fa7e09842`;
- SQLite `quick_check` returns `ok`.

The report's sentence "Trades_total = 166 (floors 1/12 exceeded)" was
insufficient by itself because it named only validation's total. The nested
chain evidence supplies the missing train-tail count, so this is corrected
evidence rather than another system finding.

## Disposition

Satoshi's authority retractions, the `doin-plugins` acknowledgment, test
counts, remote gates, artifact proof and arithmetic reconstruction are
accepted. His overall `AT-F1-001 PASS` is not accepted because arithmetic
identity was mistaken for unit-contract conformance.

Runtime disposition: **no mutation authorized**. The current swarm remains on
its existing chain and objective until Harvey makes the explicit metric
decision.
