# Musashi Audit: B4 C26 and T2 Public Utility Gate

Date: 2026-09-06

## Disposition

- B4 C26 append-only repair: `ACCEPTED_NARROWLY`.
- B4 campaign execution: `REVISE_C27_C28_BEFORE_FINAL_DISPATCH`.
- T1 v4: prior acceptance remains unchanged.
- T2 development harness: `REVISE`.
- T2 confirmatory execution: `PUBLIC_DATA_AND_CONFIRMATORY_IMPLEMENTATION_REQUIRED`.

The owner has authorized the B4 campaign in principle. That decision is now
recorded independently, but it does not erase an executable contradiction in
the current activation path and therefore is not yet a dispatch order.

## Findings

### P0: pinning the campaign authorization breaks amendment 10

`tools/b4_campaign_executor.py` currently has `CAMPAIGN_AUTH_SHA = None`, while
amendment 10 pins the exact current digest of that file. Replacing `None` with
the reviewed authorization digest changes the file digest from
`c852c3cc...` to another value, so `verify_amendment_chain()` rejects the very
code needed to consume the authorization.

This is a finite two-phase activation problem, not a reason to reopen the
scientific design. The repair must append amendment 11 after consuming the
reviewer-authored authorization record. Amendment 11 must name amendment 10,
name the authorization record, and pin the final execution surface. Amendments
9 and 10 remain byte-immutable.

### P1: B4 still contains an operator-specific absolute data fallback

`tools/b4_authority.py:1251-1254` falls back to a literal path under one
operator's home directory instead of resolving the accepted predictor root and
a logical relative data identity. The source digest still protects the bytes,
but this is needless machine coupling in a campaign intended to be replayable.
It must be removed in the same execution-only amendment.

### P0: T2 is not blocked only by missing public bytes

`tools/t2_assay_harness.py:351-357` rejects every `--confirmatory` invocation
unconditionally. Supplying the requested data cannot open that path. The code
also has no panel/TSF confirmatory loader, immutable-design consumer, attempt
ledger, task-family adjudicator, or fresh-process confirmatory verifier.
Therefore `PUBLIC_DATA_REQUIRED` is true but incomplete.

### P0: the T2 development loader has a future-looking missingness path

`tools/t2_assay_harness.py:104-108` converts malformed values to missing values
with `errors="coerce"` and then applies `ffill().bfill()`. Backfilling a leading
missing observation uses a later observation and violates the causal contract
before the T0 operator sees the series. The current future-row test does not
exercise this boundary.

### P1: the T2 estimand is not ready for heterogeneous task families

The pilot reports raw MAE/RMSE across differently scaled series, uses ridge
without an intercept or train-only scaling, and defines an "extreme" as the top
decile of absolute target level. On trending level series, that tail metric is
mostly a late-period detector. It is not a general extreme-event measure.

The only cost field is one aggregate for all arms. It cannot support an
incremental utility-per-cost decision. The validation role is allocated but
never consumed, so model selection and calibration behavior are unspecified.

### P1: the proposed support and license claims need a real manifest

Four dataset families is not, by itself, a defensible precision rule. The
minimum must follow from a predeclared task-level or hierarchical precision
criterion. Seeds and rolling origins remain nested observations, not sample
size.

Licensing cannot be inferred from an archive-wide label. The Monash archive
contains datasets from multiple original sources and explicitly describes
research use; each selected record and source must be checked separately.
ETDataset currently carries CC BY-ND 4.0, so it must remain a separately
reviewed candidate rather than being silently treated as an unrestricted
benchmark.

## Independent Reproduction

- Amendment 9 at `d97c3f62` hashes to `eb9d4970...`.
- The rewritten amendment 9 at `d8f25438` hashes to `01aeee95...`; only the
  C23-C25 execution/test pins differ.
- The current amendment 9 is byte-identical to the first Git object.
- Amendment 10 hashes to `c299d03e...`; all nine final code pins match the live
  checkout.
- `campaign_record_required_bindings()` and the v2 resource contract re-derived
  exactly.
- C26 focal tests: `8 passed`.
- T2 focal tests: `9 passed`.
- The system Python lacked project dependencies; both batteries were rerun with
  the established `trading-stack` environment.

## Reviewer Records

- Owner ratification:
  `docs/audits/evidence/OWNER_B4_CAMPAIGN_AUTHORIZATION_RATIFIED_2026_09_06.json`
  (`540fb175f0203338aa08a21bc91bba1dddf942008e011ab7b34c6695af776c63`).
- Campaign authorization record:
  `docs/audits/evidence/MUSASHI_B4_CAMPAIGN_AUTHORIZATION_RECORD.json`
  (`c58008cc5285365b4c64e2827a9b9d1a329e3b64f7c72a37b62c1c6e702ae55d`).

The second record approves the exact pre-activation amendment-10 snapshot and
resource limits. It becomes executable only through the C27 append-only
activation closure and a final review of that closure.

## Public-Data Sources Reviewed

- Monash Forecasting Repository: https://forecastingdata.org/
- Monash Zenodo community: https://zenodo.org/communities/forecasting/records
- ETT repository: https://github.com/zhouhaoyi/ETDataset
- ETT license: https://github.com/zhouhaoyi/ETDataset/blob/main/LICENSE

No dataset was downloaded and no confirmatory score was computed in this audit.

## Boundaries

No GPU workload, campaign cell, venue connection, service, key, position,
checkpoint promotion, financial score, or confirmatory T2 score was started.
