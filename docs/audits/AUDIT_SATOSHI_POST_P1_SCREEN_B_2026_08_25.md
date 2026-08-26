# Audit: Satoshi post-P1 Screen B and B4 materialization

Date: 2026-08-25
Auditor: General Musashi
Audited tip: `satoshi/post-p1-screen-b-20260825@bd3748c7`
Disposition: **REVISE BEFORE B4 DISPATCH**

## Findings

### AUD-F1-20260825-318 (S2) — the executed rule arms omit the shared protection envelope

Document 40 requires native SL/TP to be simulated identically in every arm and
requires separate `envelope_close` and `policy_close` facts. The executing loop
in `tools/screen_b_baselines.py` only supplies target-position actions and its
evidence contains neither close reason. The current B0-B3 results are useful
diagnostics, but they are not the accepted same-harness Screen B comparison and
cannot enter G1. B4 must not launch against these unmatched rules.

### AUD-F1-20260825-319 (S2) — B3 is not a 15% portfolio-volatility target

The formula computes a volatility multiplier correctly, but the environment
applies it to the inherited fixed `position_size` of one ETH. It therefore
targets a fraction of one coin, not a fraction of current equity. Independent
reconstruction of the declared-cost evidence found median actual notional/equity
exposure of 3.2%, 6.4% and 7.5% in 2022-2024 and annualized strategy volatility
of 3.24%, 2.84% and 4.89%, not 15%. B3 must convert target exposure into units
from lagged equity and lagged price, with the declared leverage cap, and persist
both requested and realized exposure.

### AUD-F1-20260825-320 (S2) — neither cost arm satisfies the shared cost contract

The P1 recipe is zero-cost. The proposed arm adds 5 bp commission and 1 bp
slippage, but no half-spread term, while document 40 requires taker fee,
half-spread and slippage. The constants are also proposals rather than values
calibrated from the Demo venues. Zero-cost remains a diagnostic sensitivity;
`declared_5bp` is not yet decision authority. G1 requires one predeclared,
source-bound primary cost contract and may retain zero/stress arms as
descriptive sensitivities.

### AUD-F1-20260825-321 (S3) — Screen B evidence omits required identity and timing facts

Document 40 requires effective-config hash, immutable code/launch identity,
inference latency and deadline evidence. The result rows contain none of these.
The trial ledger is append-only without a run identity or uniqueness check, so
rerunning the command appends the same 30 trials again. Add a canonical run
manifest, unique trial identities, idempotent registration and the required
per-arm facts before the results can feed DSR/SPA.

### AUD-F1-20260825-322 (S2) — B4 checks observation identity in the driver, not at application

The materializer proves a valuable v2 actor shape, `latent_pi.0.weight =
(256, 2660)`, and excludes sealed 2025. However, the pipeline application layer
still records the observation contract as undeclared, as the return packet
itself reports. A driver-only check does not protect alternate or resumed entry
paths. Bind and verify the v2 contract at the pipeline/model-construction seam
before any B4 GPU launch. Preserve the CPU smoke as evidence, but do not use it
as dispatch authority until this is corrected.

### AUD-SEC-20260825-323 (S4) — full GPU UUIDs remain in published branch history

The current packet is clean, but a published `wo4-*` remote branch contains full
GPU UUIDs. They are topology identifiers, not broker credentials, so no key
rotation is indicated. Create sanitized retained evidence and delete obsolete
remote branches only after proving no unique required commits would be lost;
do not rewrite shared default history casually.

## Verified corrections and non-findings

- P1-316/317 focused suite reproduced: 50 focused tests pass including Screen B
  formula/lag and post-P1 contract tests.
- Pair identity schema v3 rejects relocation/mutation and embeds state maps.
- Rule signals use strict lag and the three origin slices exclude sealed 2025.
- The v2 feature path constructs the intended 83-feature, 2660-wide actor input.
- The discovered short-rebalance sign defect was corrected before publication.
- B0-B3 evidence makes no G1 profitability claim; that restraint is correct.

## Disposition

Accept P1-316 and P1-317 as independently verified corrected, pending the normal
closure authority. Preserve the current B0-B3 artifacts under the explicit
label `DIAGNOSTIC_INVALID_FOR_G1_CONTRACT_MISMATCH`. Findings 318-322 block B4
dispatch and any G1 conclusion. No GPU campaign is authorized by this audit.

