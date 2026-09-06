# General Satoshi to Musashi: T2 public utility gate return

Date: 2026-09-06. Order:
`MUSASHI_TO_GENERAL_SATOSHI_T2_PUBLIC_UTILITY_GATE_ORDER_2026_09_06.md`,
under your external review record
`MUSASHI_T1_V4_EXTERNAL_REVIEW_2026_09_06.json` (`a0414440…` —
verified byte-exact before starting).

CPU only, `nice -15` per the resource-coordination rule; no
process or service touched; no T3/T4, no DOIN adapter or gene, no
GPU, no financial confirmation, no live activity.

## Chronology (exact)

1. **T2.0 census** (`tools/t2_public_data_census.py`, committed at
   `da5e2af1`; output digest `a7e91696…`):
   - Preferred confirmatory sources — a bounded Monash subset,
     ETTh1, Weather/Jena, optional M4 — are **ABSENT from local
     storage**, and network retrieval sits outside my standing
     constraints (the no-download rule from the T-orders was never
     lifted).
   - The financial inventories on this machine (FX/S&P/VIX under
     `feature-eng/tests/data`, `examples/data`) and the
     self-generated synthetic sets are inventoried only as
     EXCLUDED classes per T2.0.3; `statsmodels.macrodata` is
     excluded as economic.
   - Three public NON-financial development units ARE physically
     present inside the installed statsmodels package and are
     bound by bytes digest + logical source + license + schema +
     frequency + missingness policy: `sm_co2` (weekly, 2284 rows,
     atmospheric chemistry), `sm_sunspots` (yearly, 309, solar),
     `sm_nile` (yearly, 100, hydrology).
   - Census verdict: **`CONFIRMATORY_BANK_UNAVAILABLE`** — three
     short univariate series across three families cannot satisfy
     the sealed gate's whole-family holdouts, task support and
     multiplicity precision.

2. **T2.1 harness** (`tools/t2_assay_harness.py`, same commit):
   one shared path for all arms — `X`, `D(X)` (ewma α=0.3,
   selected from your reviewed T1 v4 record as `LAB_CALIBRATED`,
   fixed before any T2 score existed), `[X, D, X−D]`, and a
   matched-capacity control (train-frozen independent channels).
   Assays: seasonal naive with PREDECLARED periods, lagged ridge
   (lags 8, λ=1.0), and a small MLP (fixed budget, seed tape
   11/12/13) — identical splits (60/20/20 by index), fit windows,
   budgets and score rows across paired arms. `D(X)` is fitted
   through the accepted T0 contract (train-interval-bound fit,
   causal batch transform), so nothing observes held-out rows or
   future timestamps. Records are content-addressed, the TASK is
   the statistical unit, and the only claim classes are utility /
   calibration / extreme-preservation / cost — noise, true-SNR
   and eligibility tokens refuse structurally
   (`check_record_schema`).

3. **T2.2 development pilot** (output digest `3dde54c0…`,
   authority `DEVELOPMENT_ONLY_ZERO_CONFIRMATORY_AUTHORITY`):
   mechanics verified end to end on the two geometrically viable
   dev units; `sm_nile` refused typed (too short for the
   lag/split geometry) and stays visible as a refusal record.
   Development-only observation (zero confirmatory authority, no
   margin frozen from it beyond mechanics): on both dev units the
   D and XDR arms improved ridge MAE over X (co2 0.3623→0.3503;
   sunspots 16.22→15.10) while the width control tracked X —
   the harness distinguishes arms as designed.
   **No immutable T2.2 design was sealed**: the order requires
   sealing exact datasets and whole-family holdouts, which cannot
   truthfully exist before the confirmatory bank does. Sealing
   comes after the bank.

4. **T2.3 confirmatory matrix: NOT RUN.** The gate refuses by
   construction: `--confirmatory` exits with
   `PUBLIC_DATA_REQUIRED` while the census verdict stands, and no
   sealed design exists to validate at the point of use.

5. **T2.4 package (what exists now):** census + pilot records
   (sanitized copies under
   `docs/audits/evidence/t2_gate_20260906/`, state digests bound),
   exact code identities via the tool registry, and the frozen
   adversary battery (`tests/test_t2_harness.py`, **9 passed**):
   bytes/frame decoupling, no-dataframe entry point, future-row
   leakage immunity (including the causal transform on the train
   role), forbidden claim tokens, cost omission, record digest
   tampering, the confirmatory PUBLIC_DATA_REQUIRED gate, task
   duplication, financial/synthetic exclusion, and
   no-authority-token asserts over the T2 tools. The fresh-process
   confirmatory verifier is specified by these contracts and will
   be completed against the sealed design once the bank exists.

## Return: PUBLIC_DATA_REQUIRED

Per T2.0.3, the exact deficit the operator must supply (bytes +
licenses; I will record digests on receipt and only then seal the
immutable T2.2 design):

| Source | What is needed |
| --- | --- |
| Monash repository | a bounded subset covering ≥4 independent task families (e.g. tourism, hospital, solar, pedestrian) as the original `.tsf` bytes, with per-dataset licenses |
| ETTh1 | `ETTh1.csv` exact bytes (ETDataset) |
| Weather/Jena | `jena_climate_2009_2016.csv` or the Autoformer `weather.csv` exact bytes |
| M4 (optional) | only if it adds an independent family |

Minimum: ≥4 independent non-financial task families with ≥1000
observations per series or ≥20 series per family. Financial data
and synthetic replicas were not, and will not be, used as
substitutes.

## Suites (final tip `da5e2af1` + packet commit)

- T2 battery 9 passed; T1 adversarial 57 passed and surface index
  17 passed at the same tip (T2 tools declared in the registry).
- agent-multi full suite at the packet tip: **3069 passed,
  2 failed, 1 error** — the preexisting D1-anchor pair and the
  known `test_weekly_promotion` collection-order flake (passes
  isolated; named for watch since the @9abea1bc era).
- Development-pilot resource use: ≈4.5 s CPU total at nice 15;
  RSS well under the operator contract.

## Disposition

The T2 gate is BUILT and closed at exactly the boundary your order
draws: census truthful, harness causal and claim-disciplined,
pilot mechanics proven with zero authority, confirmatory scoring
structurally impossible until lawful public bytes exist. No DOIN
adapter, no gene, no T3/T4.

`PUBLIC_DATA_REQUIRED`
