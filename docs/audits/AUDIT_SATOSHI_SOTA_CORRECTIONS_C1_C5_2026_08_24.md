# Audit: Satoshi SOTA Corrections C1-C5

Date: 2026-08-24
Audited series: `ef69fb0a..c7e5fdfb`
Auditor: General Musashi
Focused reproduction: 26 tests passed; registry validator passed
Runtime mutation: none

## Verdict

`PARTIAL_ACCEPTANCE_REVISE_BEFORE_MATERIALIZATION`.

C1-C5 substantially correct the prior defects. No B/A/R/C screen may be
materialized yet. One newly exposed observation-identity defect and three
remaining contract defects require correction. P1 should continue to terminal
because it is already 9/12; its executed 84-feature identity must be reported
honestly rather than relabeled as the intended 83-feature system.

## Findings

### S2 — SOTA-C01: executed P1 observation has 84 features, contract declares 83

Independent inspection of
`seed101_N/normal_report.launch_manifest.json` gives
`len(effective_config.feature_columns) == 84`. The committed system manifest
`ethusdt_4h_l1_system_v1.json` declares 83. The extra executed feature is
`typical_price`. The corrected warmup probe independently reports denominator
84 in all roles, confirming this is execution reality rather than prose drift.

Consequences:

- the documented flattened size `32*83+4=2660` is not the executed P1 size;
- grouped-family counts and observation hashes do not describe the actual run;
- P1 results remain diagnostic for their 84-feature identity, but cannot be
  promoted or compared as the declared 83-feature contract;
- post-P1 materializers must refuse this mismatch before model construction.

Do not stop P1. Seal it with `executed_feature_count=84`, the exact ordered list
and digest, and classify the intended-contract mismatch explicitly.

### S2 — SOTA-C02: Screen R still confounds cadence with total optimization

Every refresh receives 5,000 gradient steps. Therefore 12-hour adaptation gets
14 times the weekly arm's weekly compute. Reporting total compute as a covariate
does not make cadence the only treatment and cannot identify its causal effect.

Split the question:

1. causal cadence screen with equal total update budget over each scored period,
   allocating that budget across refreshes;
2. operational bundle screen, if desired, with fixed 5,000 steps per refresh,
   honestly labeled `cadence_plus_compute` and compared on value per GPU-hour as
   well as economic outcome.

### S2 — SOTA-C03: causal eligibility ignores `origin.fit_end`

`check_causal_eligibility()` computes only whether fit/selection information is
before `score_start`; `Origin.fit_end` is never used. A policy trained after the
declared fit boundary but before score start can pass. Date strings are also not
validated as ISO dates.

Require parsed dates and enforce at minimum:

- `policy.fit_data_end <= origin.fit_end`;
- selection information within a separately declared pre-score selection
  boundary;
- ordered, non-overlapping origin boundaries.

### S3 — SOTA-C04: release companion schema remains permissive

`check_release_packet()` rejects three named influence keys on report-only
companions, but arbitrary alternative keys can still encode selection or
fallback authority. Use a typed allowlist schema for report-only entries and
require the sole finalist's frozen artifact/config/code/ensemble digests.

### S3 — SOTA-C05: claim-level source validation is heuristic, not exhaustive

The validator catches numeric top-level bullets matching a keyword regex. It
does not cover Markdown table cells, prose paragraphs, nested bullets or numeric
claims outside its keyword list. Its `127 claims` means claims detected by the
heuristic, not all quantitative claims.

Accept it as a useful lint gate, but remove the exhaustive claim. Add explicit
claim IDs/source IDs to quantitative tables and a coverage manifest if complete
binding is required.

### S3 — SOTA-C06: statistical block-selection wording is incoherent

Doc 41 says Politis-White block length is computed on "the control arm's
differential series" before any candidate is examined. A differential requires
two series. Bind whether block selection uses control returns alone or each
predeclared candidate-minus-control differential. Do not select one block after
examining candidate outcomes.

## Accepted Corrections

- Frozen future-trained P1 policies removed from causal rolling origins.
- Exactly one decision-authoritative release finalist in principle.
- Action mapping, threshold calibration and close-head staged separately.
- TSMOM lookbacks retained as distinct trials.
- Volatility baseline parameters predeclared.
- GRU chosen before architecture results and +/-5% capacity tolerance declared.
- Warmup probe is now runnable, identity-bound and role-wise; the strong
  256-bar scaler-dead-zone claim is rejected by its evidence.
- DSR historical trial count correctly labeled a lower bound.
- No screen launched and P1 runtime preserved.

## Disposition

Correct C01-C06 on CPU while P1 finishes. The rule-only baseline formulas may
be unit-tested, but no result-bearing B/A/R/C materialization is authorized by
this verdict.

