# Audit: P1 L1 Curriculum Terminal Aggregation

Date: 2026-08-25 America/Bogota  
Auditor: General Musashi  
Runtime mutation: none  
Sealed 2025 access: none

## Verdict

The fixed-LR P1 campaign is terminal: 12/12 arm reports are
`ARM_COMPLETE` and accepted. The predeclared outer endpoint is the 2024
risk-adjusted return; 2,196 scored rows and the same outer CSV digest are
bound in every report.

Neither easy arm provides evidence that relaxed solvency improved SAC:

| Arm | Per-seed delta vs N (101, 202, 303, 404) | Median | Direction rule | Informative easy seeds |
| --- | --- | ---: | --- | ---: |
| EN-W | -0.032365, 0, 0, 0 | 0 | INCONCLUSIVE | 0/4 |
| EN-F | +0.465593, 0, -0.215831, 0 | 0 | INCONCLUSIVE | 0/4 |

The stronger fact is mechanical: each easy selected checkpoint is
identical to its N checkpoint in all 148 named state tensors, for both
EN-W and EN-F, in all four seeds. The easy solvency treatment never
bound. EN-F's nonzero endpoints therefore test replay carry, not easy
difficulty. P1 authorizes no promotion and no claim that easy helps or
harms.

## Findings

### P1-316 (S3, observed): path-valued pair identity

Seed 101 N and EN reports differed only in the absolute path to the
same nested contract. Both files hash to
`2b31b7770f815b75b14d8234961d848787ae7c7fde9c03dbc494480fcb4130c6`.
The terminal aggregator replaces that path with the verified content
digest. Future materialization must persist the digest as authority and
keep the path descriptive only.

### P1-317 (S3, observed): mandatory divergence fact absent

Terminal arm reports did not embed the named state maps required to
re-derive `treatment_divergence()`. Aggregation required separately
retrieving the immutable selected-checkpoint manifests. Future arm
reports must embed or content-address those maps and fail closed when
the evidence is unavailable.

## Reproduction

```bash
PYTHONPATH=. python -m pytest \
  tests/test_aggregate_l1_curriculum_campaign.py -q

PYTHONPATH=. python tools/aggregate_l1_curriculum_campaign.py \
  --reports docs/audits/evidence/p1_curriculum_terminal_20260825/reports \
  --manifests docs/audits/evidence/p1_curriculum_terminal_20260825/manifests \
  --nested-contract docs/audits/evidence/p1_curriculum_terminal_20260825/nested_split_contract_v1.json \
  --output docs/audits/evidence/p1_curriculum_terminal_20260825/P1_TERMINAL_AGGREGATE.json
```

Focused result: 4 passed. The evidence directory contains the 12 reports,
12 selected manifests, contract, hashes and aggregate.

## Disposition

P1 is complete as a diagnostic experiment. It must not be repeated with
the same non-binding easy treatment. The post-P1 order advances to Screen
B rule baselines, followed by causally trained B4 SAC under the corrected
v2 observation identity.
