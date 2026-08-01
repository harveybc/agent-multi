# AT-F1-001 Correction Report (Invocation 07)

Audit ID: AUDIT-AT-F1-001-CORR-20260731-01
evidence_observed_at: 2026-07-31 13:55–14:20 America/Bogota
report_written_at: 2026-07-31 14:25 America/Bogota
Auditor: Satoshi
Inputs: cross-review `MUS-REV-20260731-AT-F1-001` (sha256 `1f96988092ab…`,
matches the invocation's pin), original report (preserved unchanged),
`pipeline_plugins/rl_pipeline_with_validation.py:226-270`, chain database
`doin-data-usdcad-4h-protected-easy-v2-omega/chain.db` (read-only,
`PRAGMA quick_check` = ok).

## 1. Block-8 Split Evidence (`reproduced` from chain payload atoms)

| Split | evaluation_weeks | risk_adjusted_total_return (full-period) | mean_weekly_rap | trades_total |
| --- | ---: | ---: | ---: | ---: |
| train_tail | 3 | +0.00017968079917197056 | +0.00005090569047665089 | 5 |
| validation | 53 | +0.0013898804150448444 | −0.00007597242102824658 | 166 |
| test | None | None | None | None (`evaluation_skipped: true`, `skip_reason: protected_test_disabled_for_optimization`) |

## 2. Both Scalars (`reproduced`, β = 0.25)

```text
configured full-period L2 : mean(+0.000179681, +0.001389880) − 0.25·gap
                          = +0.00048223070314018903   (matches stored bit-exactly)
task-specified weekly L2  : mean(+0.000050906, −0.000075972) − 0.25·gap
                          = −0.00004425289315202221   (sign inverts)
```

Code basis (`observed`): `_selection_value()`
[rl_pipeline_with_validation.py:226-241](../../pipeline_plugins/rl_pipeline_with_validation.py#L226)
routes `risk_adjusted_return` to full-period `total_return − λ·max_drawdown`;
`mean_weekly_rap` is never consulted by the active job-0 objective.

## 3. Verdict Withdrawal

The unconditional **`AT-F1-001 PASS` is withdrawn.** What survives: arithmetic
fidelity, wiring, floors, brackets, firewall, artifact integrity and
deployed-revision identity — all independently confirmed by the cross-review.
What was wrong: certifying "weekly-fraction units end to end" from arithmetic
identity alone. Finding **AUD-F1-20260731-026 (S2)** is accepted as written.
AT-F1-001 state: `reported (finding open)` until Harvey selects the
authoritative objective contract. Method lesson recorded: *reproducing a
stored number verifies implementation, never the contract the number claims
to satisfy; units require independent reconstruction from atoms* — this now
joins finding 011 and the flat-config near-miss in the auditor's permanent
error register.

## 4. Chronology Addendum (finding 027, accepted)

The original report's header time `04:40` was the session-start draft time,
not the write time. Correct chronology, original preserved unchanged:

```text
evidence_observed_at : 2026-07-31 04:40–05:20  (git/API/tests) and
                       13:31 window for final assembly
report_started_at    : 2026-07-31 04:40
report_written_at    : 2026-07-31 13:31:23 -0500 (file mtime, authoritative)
```

Commits `8e63b7dc` (04:58) and `69d06a24` (05:16) were created inside the
session window and audited after their creation; the header simply failed to
advance. Future reports carry the three-field chronology.

## 5. State-File Consistency (finding 028, accepted)

`Observed`: the backlog now carries `AT-F1-001 = reported (finding open)` and
`AT-F1-012 = scheduled`; the register contains findings 025–028; Musashi's
cross-review performed that reconciliation when my session (over-complying
with Invocation 04's one-file contract) did not. The failure of handoff duty
is mine and is accepted; this report plus the recovery-prompt update below
restore the rule that a session ends only when report, backlog, register and
recovery state agree.

## 6. Runtime Statement

**No mid-chain fitness mutation is authorized or proposed.** Changing the
objective inside the running job-0 chain would invalidate candidate
comparability; the swarm continues unchanged pending Harvey's contract
decision (decision packet: AT-F1-012 report).
