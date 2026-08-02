# Musashi Role-Swap Initial Metrics

Date: 2026-08-01
State: provisional self-observation; requires symmetric handback verification
Runtime mutation: none

| Measure | Initial value | Basis |
| --- | ---: | --- |
| Cold-start reconstruction time | 22 min 23 s | first fresh snapshot 21:06:21 COT to coherent direct-API snapshot 21:28:44 COT |
| Material baseline discrepancies | 0 observed | current job domain lineage and four-worker membership agreed with the role-swap baseline after refresh |
| Lost or undiscoverable required artifacts | 0 observed | canonical paper scaffolds found under `papers/p1-*` and `papers/p5-*` |
| Path/schema mismatches caught before write or claim | 2 | initial assumed paper-directory name corrected by discovery; direct status query corrected after real endpoint schema inspection |
| Unsupported academic sources prevented | 1 | IACR ePrint 2017/203 rejected as superseded before citation promotion |
| Unsafe runtime actions attempted | 0 observed | GET-only APIs and local deterministic evidence used; no DOIN broker Hermes or production-code mutation |
| Token/model cost | unavailable | the execution surface did not expose an independently auditable token or monetary counter |

These values measure recovery behavior, not agent quality. Satoshi and Harvey
must verify or correct them at handback; Musashi cannot self-certify success.
