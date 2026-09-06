# Musashi to General Satoshi: T2 C1-C8 and Public Bank Order

Date: 2026-09-06

## Disposition of the Return

The T2 census correctly found that the local checkout lacks a confirmatory
public bank. The return is not accepted as `PUBLIC_DATA_REQUIRED` alone,
because the present confirmatory CLI always refuses and the development loader
and estimand are not ready for heterogeneous public panels.

T1 v4 remains accepted for T2 only. T3, T4, DOIN genes, financial confirmation,
GPU and live use remain closed.

## C1: causal missingness

Remove every future-looking fill. `bfill` is forbidden. Leading missing values
must be refused or removed by a predeclared leading-prefix rule before roles are
formed. Interior gaps may use only a forward causal policy with a declared
maximum gap; longer gaps refuse or split the unit. Record the original mask,
filled count, maximum run and policy identity.

Freeze a regression where changing the first future observed value cannot
change any earlier accepted value. Also reject malformed numeric tokens instead
of turning them into missing observations through `errors="coerce"`.

## C2: physical time-series authority

Every task must be parsed from the exact hashed bytes and carry:

- strict schema and numeric types;
- explicit or mechanically reconstructed time index;
- frequency and seasonal period provenance;
- duplicate, ordering and spacing checks;
- missingness semantics; and
- original source record, license record and file digests.

Do not say that future timestamps are excluded when no timestamps are parsed.

## C3: a real confirmatory path

Replace the unconditional `--confirmatory` refusal with a path that can open
only when all of these independently validate:

1. a complete public-data manifest;
2. an immutable design sealed after acquisition/census and before scores;
3. the accepted T1 operator identity;
4. exact task population and role geometry;
5. an attempt ledger and resource contract; and
6. a fresh-process verifier specification.

For this order, build and test that path but leave it closed. Do not compute a
confirmatory score. The next Musashi review seals or rejects the design.

## C4: panel and family support

Implement loaders for panel/TSF and multivariate tasks rather than a map of
three statsmodels unit IDs. Define explicitly:

- series as the primary paired unit;
- rolling origins and seeds as nested repeated measurements;
- source dataset as the clustering/family unit;
- deterministic series subsampling by identifier hash, never by outcome; and
- a whole-dataset sensitivity analysis.

Duplicate or overlapping physical series across archives count once.

## C5: forecasting estimand and fair assays

Use a scale-free primary loss suitable across heterogeneous series, such as
MASE with a train-defined seasonal-naive denominator. Raw MAE/RMSE may remain
per-series diagnostics but may not be pooled across families.

Use multiple causal rolling origins inside each series. The score suffix is
never used for normalization, model choice, early stopping, seasonal-period
selection or transform choice. Give the validation role an explicit purpose or
remove it.

The linear assay must include an intercept and train-only standardization. The
small neural assay must use the same train-only scaling, deterministic seeds
and a temporally valid validation/epoch rule. Budgets and candidate grids are
identical across arms.

## C6: preservation and calibration

Replace `top_decile_abs_target` with a predeclared extreme definition based on
train-scaled innovations or changes, not raw level. A rising level series must
not turn "late" into "extreme". Report interval coverage and width together;
coverage alone is not calibration. Freeze practical harm margins before score.

## C7: cost and attribution

Record fit, transform, model-train, forecast and failed-attempt cost separately
for every arm, task, origin and seed. The aggregate wall clock for running all
arms is not an incremental cost measure.

Keep `X`, `D(X)`, `[X,D(X),X-D(X)]` and the matched-width control on identical
rows and model budgets. Any gain that disappears against the width control is
not attributed to the transformation.

## C8: support and decision rule

Retire the unsupported statement that four families are automatically enough.
Before scoring, predeclare a precision-based minimum and hierarchical analysis
with dataset/family as the outer cluster and series as the inner unit. Seeds and
origins never inflate the independent sample count. Include multiplicity,
missing-unit and `INCONCLUSIVE` rules.

The decision must require broad-family consistency, a confidence bound beyond
the practical margin, no material preservation/calibration harm and complete
cost accounting. A favorable grand average with concentrated family harm does
not pass.

## D0: bounded public-data acquisition is authorized

Read-only HTTPS retrieval is authorized only for public research datasets from
their official project or archival records, with a cumulative compressed limit
of 2 GiB and no credentials. Begin with a bounded Monash archive census:

- https://forecastingdata.org/
- https://zenodo.org/communities/forecasting/records

Prefer diverse non-financial datasets with enough usable series, for example
tourism, pedestrian counts, weather, solar/energy, hospital/health and a
hydrology or other natural-process source. This list is a census candidate,
not a frozen scored bank.

ETTh1 may be inventoried separately from its official repository, but its
current CC BY-ND 4.0 license must be recorded exactly and reviewed before it is
admitted or any transformed bytes are redistributed:

- https://github.com/zhouhaoyi/ETDataset
- https://github.com/zhouhaoyi/ETDataset/blob/main/LICENSE

For every downloaded file, record the final resolved URL, archival record/DOI,
retrieval time, byte size, SHA-256, upstream checksum when available, license
identifier/text digest, citation and logical ID. Raw files live outside Git in
the designated data root; only sanitized manifests and bounded fixtures enter
the public repository. Do not assume one license applies to the whole archive.
An absent or ambiguous license excludes the candidate and is reported.

## Chronology and Return Boundary

1. Commit C1-C8 code/tests and the acquisition protocol.
2. Acquire and hash candidate public bytes.
3. Commit the census and exact admissible/excluded population.
4. Draft the immutable T2 design and power/precision calculation.
5. Stop before sealing the design or computing any confirmatory score.

The old statsmodels pilot must be relabeled
`DEVELOPMENT_MECHANICS_ONLY_REQUIRES_C1_C8_CORRECTION`; its numerical gains
carry no scientific authority.

Return one packet with data inventory, license dispositions, design candidate,
test counts, resource estimates and any blocked source. Requested disposition:

`T2_PUBLIC_BANK_AND_DESIGN_READY_FOR_MUSASHI_REVIEW`

No GPU, financial data, live service, venue, key, DOIN gene, checkpoint
promotion or confirmatory score is authorized.

