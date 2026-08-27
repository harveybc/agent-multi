# Acceptance Audit: DATA-SOTA-353..356

Date: 2026-08-27 America/Bogota
Auditor: General Musashi
Reviewed tips: `agent-multi@5d2d800a`, `lts@24aa620`

## Verdict

**ACCEPTED. DATA-SOTA-353, 354, 355 and 356 are independently verified as
corrected.** No new blocking finding arose from this return.

This acceptance closes no register entry by itself. It authorizes only the CPU
transfer-loader smoke already bounded in the correction order.

## Independent Reproduction

- Agent-multi focused suite with Tier-A real ETH H4 data: **86 passed**.
- LTS quote scheduler and real-SQLite regressions: **28 passed**.
- No GPU training, economic run, checkpoint promotion or collector activation
  was performed by the auditor.

## Verified Facts

### 353 -- purged partitions

- Purge is mechanically `max(horizons) = 12` at each boundary.
- The smoke proves train last target row 8094 precedes calibration anchor 8095.
- Calibration last target row 8706 precedes monitor anchor 8707.
- Twenty-four purged windows are digest-bound; additional embargo is explicitly
  zero.
- Input context and target ranges are separately represented.
- Purge and origin-plan identities bind resume.

### 354 -- canonical quote integrity

- Exact content replay is idempotent across sessions.
- Same canonical identity with changed payload raises `QuoteConflictError` and
  preserves the original fact.
- Canonical insertion and session membership commit or roll back together.
- `alpaca_quote_summary_olap` reads `quote_canonical` at venue+symbol grain.
- Legacy observations are exposed only through an explicitly legacy view.

### 355 -- TRM durability

- Manifest temporary file is fsynced, renamed atomically and followed by parent
  directory fsync.
- Injected file-fsync, rename and directory-fsync failures cannot acknowledge a
  successful manifest.

### 356 -- origin ancestry

- The ordered origin plan rejects duplicate, unknown and non-chronological
  origins.
- Every non-first origin names and verifies its immediate predecessor.
- A digest-valid o2022 decision cannot materialize o2024 while skipping o2023.
- The first-origin exemption is explicit.

## Scientific Scope

The regenerated o2022 v4 run remains `NOT_TRANSFER_ELIGIBLE`. Its decreasing
three-epoch monitor losses and zero quantile crossing establish mechanics only;
they do not establish useful representations, economic improvement or a winner.

## Authorized Next Step

General Satoshi may now implement and execute exactly one bounded CPU smoke of
the transfer loader under the companion dispatch. No owner phrase is required.
No GPU pretraining, B4, economic comparison, model promotion or quote-scheduler
activation is authorized by this acceptance.
