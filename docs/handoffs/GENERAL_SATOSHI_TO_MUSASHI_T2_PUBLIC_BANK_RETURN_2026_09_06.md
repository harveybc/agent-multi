# General Satoshi to Musashi: T2 C1-C8 and public bank return

Date: 2026-09-06. Order:
`MUSASHI_TO_GENERAL_SATOSHI_T2_C1_C8_PUBLIC_BANK_ORDER_2026_09_06.md`.

CPU only at nice 15 (B4 coordination respected; no process
touched). No GPU, financial data, live action, DOIN gene,
checkpoint promotion — and **no confirmatory score exists**: the
chronology stopped exactly at step 5, before sealing the design.

## Chronology (exact, as ordered)

**Step 1 — C1-C8 code/tests + acquisition protocol
(commit `36d25deb`):**

- **C1** (`t2_bank.py`): future-looking fill is dead. Malformed
  numeric tokens REFUSE (no `errors="coerce"` anywhere); leading
  missing values drop by the predeclared `drop_leading_missing_v1`
  rule before roles are formed; interior gaps forward-fill only up
  to the per-unit declared maximum run (longer refuses; trailing
  refuses outright since forward fill has no later truth and
  backward fill is forbidden); the original mask, filled count,
  max run and policy identity travel on every unit. Frozen
  regression: mutating the first future observed value leaves
  every earlier accepted value byte-identical.
- **C2**: every dev unit now carries a parsed or mechanically
  reconstructed time index with duplicate/ordering/spacing facts
  and frequency/seasonal-period provenance; no claim about
  excluded future timestamps is made where none are parsed.
- **C3** (`t2_confirmatory.py`): the unconditional `--confirmatory`
  refusal is REPLACED by the real gate sequence — public-data
  manifest → immutable design sealed over that manifest → accepted
  T1 operator identity → exact task population/geometry → attempt
  ledger + resource contract → fresh-process verifier spec — each
  absence refusing with its own typed reason. The final gate
  requires your design-review record, which does not exist, so the
  path is built, tested and **closed**; supplying data alone now
  moves the refusal forward (PUBLIC_DATA_REQUIRED →
  DESIGN_REQUIRED → DESIGN_REVIEW_REQUIRED) instead of being
  swallowed.
- **C4**: a strict Monash `.tsf` panel parser (malformed structure
  refuses; non-UTF8 headers fall back to byte-bijective latin-1,
  RECORDED per panel), per-series units, GLOBAL cross-archive
  physical deduplication (byte-level series digest; duplicates
  count once — proven in-battery), deterministic id-hash
  subsampling (never by outcome), series=paired unit,
  origins/seeds=nested, dataset=family cluster.
- **C5**: MASE is the primary loss with a train-defined
  seasonal-naive denominator (mutating score rows cannot change
  it — regression); raw MAE/RMSE demoted to per-series
  diagnostics; three predeclared rolling origins cover the final
  40% of each series; the ridge gained an intercept (unpenalized)
  and train-only standardization; the small MLP uses the same
  train-only scaling, deterministic seeds and a temporally valid
  epoch rule on the FINAL slice of the fit rows (the validation
  role's explicit purpose); budgets identical across arms.
- **C6**: the extreme set is predeclared from train-scaled
  INNOVATIONS — proven on a trending fixture that "late" does not
  become "extreme"; interval coverage and width are reported
  together; candidate harm margins are frozen in the design draft
  before any score.
- **C7**: per-phase costs (denoise fit+transform, lag features,
  ridge fit+forecast, MLP fit+forecast per seed) recorded for
  every arm and origin; the old single aggregate is gone.
- **C8**: `adjudicate_confirmatory` — dataset/family outer
  cluster, series inner unit, Bonferroni-by-family, predeclared
  min-series/min-families, INCONCLUSIVE first-class; fixtures
  prove concentrated family harm defeats a favorable grand
  average, mixed consistency lands INCONCLUSIVE, and only
  broad-family cleared CIs yield the (still gated) candidate
  label. The "four families are enough" claim is retired.
- The statsmodels pilot is relabeled
  `DEVELOPMENT_MECHANICS_ONLY_REQUIRES_C1_C8_CORRECTION` (now
  cleared mechanically by the rerun; zero scientific authority),
  with co2's real 18-wide gap under a declared 25-run contract and
  nile's typed geometry refusal preserved.

**Step 2 — D0 acquisition (first authorized downloads):** Zenodo
forecasting-community census first (65 records, license read PER
RECORD), then 10 datasets, 51.1 MB total against the 2 GiB cap,
read-only HTTPS from official records, no credentials:

| logical id | family | license | admission |
| --- | --- | --- | --- |
| tourism_monthly | tourism | cc-by-4.0 | ADMISSIBLE |
| pedestrian_counts | urban_pedestrian | cc-by-4.0 | ADMISSIBLE |
| hospital | health_hospital | cc-by-4.0 | ADMISSIBLE |
| solar_weekly + solar_10_minutes | solar_energy | cc-by-4.0 | ADMISSIBLE |
| saugeenday | hydrology | cc-by-4.0 | ADMISSIBLE |
| us_births | demography | cc-by-4.0 | ADMISSIBLE |
| electricity_weekly | electricity_demand | cc-by-4.0 | ADMISSIBLE |
| weather | weather | cc-by-4.0 | ADMISSIBLE |
| etth1 | electricity_transformer | **CC-BY-ND-4.0 (license text digested)** | **REVIEW_REQUIRED** |

Per-file manifest: final URL, DOI/archival record, retrieval time,
byte size, SHA-256, upstream md5 where available, license id +
text digest, citation, logical id. Raw bytes live OUTSIDE Git at
`<state_root>/t2_public_raw`; only sanitized manifests are
committed. Monash financial/economic records (Bitcoin, FRED-MD,
Dominick) excluded by class; M4 excluded because its mixed
composition adds no independent non-financial family (reason
recorded). ETTh1 is not admitted and no transformed bytes are
redistributed pending your license review.

**Step 3 — bank census (commit `8aa7b064`):** the acquired bytes
through the C1/C2/C4 loaders with global dedup — **1404 admissible
series in 8 families** (tourism 299, pedestrian 66, hospital 300,
solar 137, electricity 300, weather 300, hydrology 1,
demography 1). solar_weekly was excluded whole (52-obs series
below the declared geometry, reason recorded) and replaced for the
family by solar_10_minutes; bounded 300-series/archive census
parse disclosed; per-dataset exclusion reasons committed.

**Step 4 — design DRAFT + precision calculation
(`0778ff58…`, committed sanitized):** exact task population (251
series via deterministic id-hash caps over 6 primary-gate families;
hydrology/demography sensitivity-only), role geometry, MASE
primary, candidate practical margin 0.02, harm margins frozen,
disclosed precision rule (planning sd 0.04 from the dev-mechanics
scale → n ≥ (z·sd/margin)² = 28 per family under
Bonferroni-by-family; declared minimum 28, families ≥ 4),
missing-unit and INCONCLUSIVE rules, resource contract, and the
fresh-process verifier specification. The draft is **NOT sealed**:
`design_review_record_sha256 = PENDING_MUSASHI_DESIGN_REVIEW`.

**Step 5 — stopped.** No design seal, no confirmatory score.

## Batteries and suites

- T2 battery: **17 passed** (C1 regression incl. the
  first-future-value freeze, C2 time facts, C3 typed gate walk
  with fixtures for every stage, C4 tsf/dedup/subsample, C5
  train-only normalization + MASE-denominator immutability, C6
  trending-series extreme fixture, C8 decision-rule fixtures,
  forbidden-claims and stage-aware gate closure).
- One battery evolution disclosed: after real acquisition the
  world moved the gate refusal from PUBLIC_DATA_REQUIRED to
  DESIGN_REQUIRED and the full suite caught my stale expectation;
  the test is now stage-aware (closed at every stage, zero
  scores).
- agent-multi full suite at the final tip: **3077 passed,
  2 failed, 1 error** — the preexisting D1-anchor pair and
  the known `test_weekly_promotion` collection-order flake
  (passes isolated at this tip; named for watch since the
  @9abea1bc era).

## Resource estimate for the sealed run (candidate)

Dev-pilot scale: ≈1.2 s/series-origin-arm set on one core at
nice 15 → 251 series × 3 origins ≈ 30-60 CPU-minutes for the full
confirmatory matrix, RSS well under 8 GiB. No GPU.

## Blocked / pending on you (via Musashi)

1. ETTh1 license disposition (CC-BY-ND-4.0 recorded exactly).
2. The design review: seal or reject the draft (`0778ff58…`);
   only your review record opens the C3 gate.

`T2_PUBLIC_BANK_AND_DESIGN_READY_FOR_MUSASHI_REVIEW`
