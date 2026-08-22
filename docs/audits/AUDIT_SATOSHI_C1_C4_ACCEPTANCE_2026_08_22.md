# Audit: C1-C4 acceptance and early-intervention dispatch

Date: 2026-08-22 America/Bogota
Auditor: General Musashi, temporary independent auditor
Commit: `9fc76505`
Runtime mutation during audit: none

## Verdict

**ACCEPT C1-C4 and authorize the counterbalanced bounded-ETH early-
intervention screen.** No additional owner phrase is required.

Independent evidence:

- correction reproducer: `reproduced: false`, no survivors;
- focused suites: 85/85 passed;
- exact schema allowlist rejects the reproduced malicious suffix class;
- launch artifact now receives file fsync, atomic rename and directory fsync;
- scheduler preflight predicts intervention before the historical best in
  seeds 202, 303 and 404 (3/4), satisfying the declared dispatch rule;
- the historical diagnostic numbers remain unchanged and zero-authority.

## Residual limitation AUD-F1-20260822-PLR-09 (S4, observed)

The migrated historical reports still carry no explicit `pair_contract` or
`arm_contract`. Their full effective configurations cannot be proven identical
except for treatment from two different config hashes alone. The stricter
diagnostic proves the reported identity fields and a 33-field trajectory
prefix, which is adequate for this zero-authority exploratory artifact, but it
must not be described as cryptographic proof of complete historical config
identity.

This does not block the new screen. Every newly generated report must carry
explicit canonical pair/arm contracts and a config-minus-treatment identity
hash from materialization time; absence refuses aggregation.

## Dispatch constraints

- fixed-first on seeds 101/303; plateau-first on 202/404;
- plateau specification: start epoch 0, LR patience 8, factor 0.5,
  `min_lr=1e-6`, threshold `1e-6`, cooldown 0;
- same bounded 120/40/40-day data and all non-treatment factors fixed;
- no checkpoint promotion;
- negative result rejects only this bounded-ETH scheduler specification;
- durable reports, per-arm logs, thermal telemetry and paired identity facts;
- aggregate only after all eight arms are accepted and independently paired.
