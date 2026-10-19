# Bounded Selected-Input Comparison

## Design Gate

Consumers are the selection owner and the static RL operator. The owner
supplies a newly selected, causally reproducible subset of the declared admitted
EURUSD population. Full financial inventory admission and all394/catalog
completion are explicitly out of scope. No selection, feature generation,
source recovery, macro clock inference, AE, predictor fit, DOIN or deployment
belongs in this adapter.

Preserve the unchanged baseline design's exact arm order: SAC_FLAT,
SAC_PER_FEATURE_CONV1D, NO_TRADE, PASSIVE_LONG, PASSIVE_SHORT, RANDOM.
Both SAC arms receive exactly the same ordered selected
variables, causal rolling normalization, 13 observations at stride 2, seed 7,
100000 transitions and unchanged SAC/cost/business-policy settings. Validation
is static inference from the corresponding TRAIN checkpoint, never refit.

The queue already accepts general names and checks split lineage. Gym's
feature-window preprocessor accepts general names and uses strictly preceding
rows for scaling. Its data feed accepts CSV only. The adapter may export
admitted parquet losslessly to CSV and record source/transport hashes; this is
not a new source admission. No gym implementation change is required for that
route. Direct parquet loading would require the gym feed and shared CSV-only
curated-input validator to be extended together.

## Test Design

Acceptance: a synthetic selected bundle yields twelve finite tasks (four
controls and two SAC arms, train/validation), identical selected information,
scoped identity and publication/copy continuation without running a financial
fit. System negatives must refuse before dispatch. Integration constructs the
existing real SAC models and checks independent trainable temporal encoders,
causal prefix invariance, observation order and 13-by-F shape without learning.
Component and unit negatives cover malformed scope, future availability,
stale selection, digest/order/column mismatch, bad support and bypassed budgets.

See `selected_rl_comparison/TRACEABILITY.md` and
`selected_rl_comparison/PROJECT_METHOD_STATE.json` for the separate extension
state. Existing raw campaign and method records remain untouched.

## Producer Interface

No selected names, hashes or availability facts are invented by this tool.
The synthetic `fixture` in `tests/unit/test_selected_rl_comparison.py` is a
complete runnable producer example, not a proposed financial selection.
All intervals below are UTC half-open intervals. Files must remain immutable
during a run. Hash pins protect bytes, not a malicious admission authority.

Every receipt, selection artifact and split manifest requires these fields:

```json
{
  "scope": "SELECTED_ADMITTED_POPULATION",
  "feature_selection_complete": true,
  "selection_complete": "WITHIN_DECLARED_ADMITTED_POPULATION",
  "full_inventory_complete": false
}
```

Request schema `selected_rl_comparison_request.v1` has exactly these keys:
`schema`, `base_design_file`, `base_design_sha256`, `receipt_file`,
`receipt_sha256`, `selection_sha256`, `population_sha256`,
`feature_columns_sha256`, `execution`, `publication`. Use absolute paths for
the design and receipt. Base design is the unchanged static baseline design
JSON, used only for policy/cost/SAC settings, never for RAW inputs or results.
Execution and publication are existing queue objects, including pinned gym
and publication-runtime revisions, bounded timeouts/attempts, and the existing
warehouse protocol. The publication runtime field remains `predictor_root`
for compatibility with the shared warehouse client; this does not invoke a
predictor fit. CUDA, if eventually authorized, retains the queue's evidence
and assignment gate. This adapter adds no GPU fallback.

Receipt schema `selected_rl_bundle.v1` requires `status:"ADMITTED"`, the
scoped flags above, `selection_file`, `selection_sha256`, `feature_columns`,
`feature_columns_sha256`, `population_sha256`, `calendar_sha256`,
`closure_intervals_file`, `closure_intervals_sha256`, and `splits` with exactly
`train` and `validation`. Each split has `input_file` (CSV or parquet),
`manifest_file`, `manifest_sha256`. Bundle members are relative paths confined
to the receipt directory, including after symlink resolution.

The receipt additionally requires `selection_close_file` and
`selection_close_sha256`. Point the request's `receipt_file` at the finalizer's
`DEPLOYMENT_RECEIPT.json`, retaining schema `selected_rl_bundle.v1` and all
original receipt fields. A sealed-but-unpublished receipt is insufficient.
The close must be pinned by its file-byte SHA256 and resolve directly inside
the receipt directory, not a nested or external directory, even via symlink.
Before reading source rows, the adapter requires:

```json
{
  "state": "FILTER_SELECTION_OLAP_AND_TWO_PHYSICAL_COPIES_VERIFIED",
  "selection_sha256": "<exact-selection-artifact-sha256>",
  "selected_count": "<integer-equal-to-selected-column-count>",
  "pending": 0,
  "mismatch": 0,
  "verified_files": "<positive-integer>",
  "full_inventory_complete": false
}
```

Placeholders above describe required values, not literal JSON strings to emit.
Counts must be actual integers, not booleans. The adapter binds the close hash
into the campaign identity and rechecks it on materialization, explicit check,
dispatch/resume and campaign publication. It consumes the owner's pinned OLAP
and two-copy attestation; it does not contact services or inspect mirrors to
recreate that attestation. Finalizer publication and copy verification must
finish before issuing this deployment receipt.

Selection schema `selected_rl_selection.v1` requires the scoped flags,
ordered unique `feature_columns`, `population_sha256`,
`causal_transforms_verified:true`, `validation_used_for_selection:false`,
`fit_period_start:"2020-01-01T00:00:00+00:00"`,
`fit_period_end:"2024-01-01T00:00:00+00:00"`, `availability_column`,
`availability_semantics:"row_max_selected_available_at"`, and normalization:

```json
{
  "mode": "rolling_zscore",
  "window_bars": 256,
  "fit_scope": "PAST_ONLY",
  "binary_columns": []
}
```

Binary column names, if any, must be a subset of the selected names and carry
only -1/0/1 values. Selected transforms must already be causal; this adapter
does not regenerate or fit them. Existing rolling scaling uses only rows
strictly before the decision origin, identically for both SAC arms. Static
means frozen policy, not frozen causal observation history.

Split manifests extend existing `curated_rl_input.v1` with the scoped flags
and `availability_column`. `input_sha256` pins actual source CSV/parquet bytes;
`inventory_sha256` identifies the declared admitted population, not an
unadmitted full inventory. The two manifests must bind the same selection,
calendar, population and ordered names. Periods are exactly TRAIN
`[2020-01-01,2024-01-01)` and VALID `[2024-01-01,2025-01-01)`; no sealed TEST
file is opened. Other existing receipt fields and expected timestamp digest
are described in `44_CURATED_RL_INPUT_INTERFACE.md`.

The source includes timezone-aware `DATE_TIME`, OHLC, `ASSUMED_SPREAD`, the
selected numeric variables, and the owner's selected row-max availability
timestamp column. Each availability timestamp must be no later than its
completed-bar timestamp. A boolean availability assertion alone is not enough.
The owner is responsible for computing that maximum over all selected
dependencies, including raw inputs; no macro clock is inferred here. Required
support comes from admission; the adapter never manufactures a digest from
delivered rows to repair incomplete coverage.

`feature_columns_sha256` is SHA256 of UTF-8
`json.dumps(feature_columns, separators=(',', ':'))` (Python's default ASCII
escaping, no newline). `selection_sha256` is the actual selection artifact's
file-byte hash. The request's expected selection pin is the owner's current
selection identity; stale receipts or subset files cannot silently supersede
it. Changing the selection requires a new request and comparison root.

For parquet, the adapter records source file and source manifest hashes,
exports all supplied columns without normalization, imputation or truncation,
and pins the transport CSV and derived transport manifest. CSV sources are
consumed directly. No source files or admission artifacts are rewritten.

## Operation And Ownership

`python -m tools.selected_rl_comparison materialize --request <request.json>
--root <new-comparison-root>` performs no dispatch. `check --root <root>`
revalidates every source and transport pin without environment construction.
Only after main finalizes/publishes the real selection may an authorized
operator use `run --root <root>`. It delegates to the existing queue, including
resume, six arms, twelve tasks, bounded attempts/timeouts, frozen validation,
ordinary publication and artifact checks. No monitor, daemon or retry loop is
added. `run` also attempts publication/close once after queue success;
`publish` retries only publication/copy, not learning. `status` and `results`
reuse the existing read-only report helpers. Status explicitly says input
integrity was not rechecked; use `check` for that.

Warehouse report tags carry the scoped ordered names and selection/population
pins through the existing structured `input_policy` tag; effective configs
also carry them. No warehouse schema change is required. Closure requires
all twelve expected task identities plus delivery/readback equality, then
uses the existing SQLite backup protocol and records a pinned physical copy.
The raw close function cannot be reused directly because it hardcodes
`feature_selection_complete:false`; the selected closure is deliberately
small and contains no execution or publication transport reimplementation.

Main-owned existing-file integration, not performed by this write set:

- Register `selected_rl_comparison.py` in `tools/TOOL_DECLARATIONS.json` with
  `purpose:"Bind scoped selected inputs to the existing static RL queue"`,
  `lifecycle:"experimental"`, `mutability:"mixed"`,
  `authority_class:"campaign_operation"`, `owner:"static RL comparison"`.
  Main reports this registration is now present; this adapter write set did
  not edit the declaration file.
- Optional follow-up: parameterize the raw publisher's closure metadata and
  extract its SQLite copy step into a shared helper, then delegate the selected
  close to it. No RAW behavior or active campaign contract needs changing.
- Gym adaptations required for this adapter: **none**. The real feed,
  preprocessor and trainable SAC architecture accept general names. Native
  parquet, if preferred over export, requires coordinated feed/validator work
  and is not implemented here.

Integration2026-10-19: executable registration is now present. The existing
closed collector also accepts `selected_rl_close.v1` only when scoped completion,
ordered selected binding and contract agree, with the same exact artifact and
closed SQLite checks. RAW behavior remains covered. Four regressions include
scope, selection-hash and selected-order mismatches. The focused comparison,
input/materializer/status/collector suite passes120 tests; no active fit code
or gym source changes are needed for collection. The scientific runtime remains
pinned independently of this later collector-only repair.

CPU-only verification in the existing dedicated trading-stack environment:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  GYM_FX_CHECKOUT=<matching-gym-checkout> python -m pytest -q \
  tests/unit/test_selected_rl_comparison.py \
  tests/integration/test_selected_rl_information_path.py
```

No financial fit, GPU use, service access, private data read, source admission
or broad inventory completion is established by these synthetic tests.

Verified 2026-10-19: 32 new tests passed in 5.16 seconds. Together with existing
curated-input, raw-materializer, raw-runner and closed-raw-collection unit
tests, 91 passed in 5.61 seconds. No learning was invoked. Full repository
suite and repository-wide lint were not run; executable registration remains
main-owned and required before repository tool-surface acceptance.

Publication-gate follow-up on the same date: main reports executable
registration completed. The adapter now requires the owner's pinned selection
OLAP/two-copy close. The same focused regression command, extended with 25
gate cases, passed **116 tests in 9.01 seconds**. The gate negatives were run
red before implementation. No fits, services, real-source rereads or commits
were performed for this follow-up.

Ordering correction: arm order is copied directly from `design['arms']`, not
the queue's controls-first development default. Exact CSV/parquet order tests
were observed red first; the full focused CPU set then passed 116 tests in
5.79 seconds. Task count remains twelve and all other settings are unchanged.
