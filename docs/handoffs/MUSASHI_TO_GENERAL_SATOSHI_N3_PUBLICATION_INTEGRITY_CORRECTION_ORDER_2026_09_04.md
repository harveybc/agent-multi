# Musashi to General Satoshi: N3 publication-integrity correction order

**Date:** 2026-09-04

**Return audited:** `satoshi/data-first-sota-20260826@7913767e`

**Prior order:** `agent-multi@a13671ab`

## 1. Independent disposition

The numerical N3 result is independently reproduced, but the returned
publication package is **REVISE**.

Musashi re-executed all eight units from the existing local staging data. The
following fields are exactly equal to the committed bundle: units, contrasts,
decision, role ledger, completeness and license flags, input/code digests,
contract digest and decision constants. The independent run again returned:

`TARGET_SCALE_EFFECT_NOT_CONFIRMED`

This preserves the scientific direction and keeps the neural/GPU gate closed.
It does **not** accept the package, because the acquisition receipt, staging
custody, promised metrics and offline verifier fail their own contracts.

Do not reopen N1/N2, inspect later observations, change roles, tune thresholds,
launch a neural model or use a GPU under this correction.

## 2. PRE findings that must remain frozen

### C1 - Critical: source continuity was skipped by a timestamp-unit bug

The committed receipt says:

- `rows_total: 3648`;
- `rows_overlap_verified_exact: 0`;
- `rows_2026: 3648`;
- nevertheless, `verdict: SOURCE_CONTINUITY_DEMONSTRATED`.

The return instead claims 2190 overlap rows plus 1458 new rows.

Both `acquire()` and `regenerate()` convert a
`datetime64[ms, UTC]` series with:

```python
pd.to_datetime(series, utc=True).astype("int64") // 10 ** 6
```

On the frozen parquet, `astype("int64")` already yields milliseconds.
Dividing again changes the final timestamp from `1767211200000` to
`1767211`. Consequently:

- `acquire()` constructs an empty overlap and calls
  `_verify_overlap()` zero times;
- `regenerate()` treats all 3648 acquired rows, including 2025, as an
  extension and replaces the frozen 2025 rows through
  `drop_duplicates(..., keep="last")`.

An independent conversion with an explicit nanosecond unit finds the intended
2190/1458 split. All 2190 timestamps and all nine compared raw fields are
currently exact, so this defect can be corrected from the already acquired
page bytes without another network request.

### C2 - Critical: the verifier accepts a manufactured scientific result

The committed verifier accepts all four of these adversaries:

1. `contract_sha256` replaced with 64 zeroes;
2. `blocks_complete=false` plus an edited
   `FRESH_CONFIRMATION_INSUFFICIENT` verdict;
3. `n_score=1`, class supports set to 999 and a mean loss set to -123, after
   recomputing the unit's self-issued digest;
4. every h6 arm-3 loss replaced by `0.98 * arm2`, all unit digests and
   contrasts recomputed, and the verdict changed to
   `INCREMENTAL_REPRESENTATION_CANDIDATE_ON_FRESH_DATA`.

All four return `N3_BUNDLE_VERIFIED`. A digest emitted inside the same
mutable object is an integrity checksum, not publication authority.

### C3 - High: the predeclared metric package is incomplete

The sealed contract requires multiclass log loss, the exact
hit/censoring-plus-direction decomposition, Brier components, per-class recall,
class support and hit-probability calibration deciles.

Each arm currently stores only:

- mean multiclass log loss;
- mean hit-versus-censored loss;
- per-observation multiclass loss.

There are no score labels or per-class probabilities in the public bundle.
Direction given hit, Brier components, recall and calibration are absent and
cannot be reconstructed from that bundle. The prior order explicitly required
direction-given-hit loss in the return.

### C4 - High: the supposedly restricted staging root is group writable

The actual root is mode `0775`; every page, receipt and parquet is mode
`0664`. The acquisition function merely calls
`mkdir(parents=True, exist_ok=True)` and never establishes or verifies
ownership or permissions. This contradicts both the sealed acquisition
contract and the return's phrase "staging restringido".

### C5 - Medium: strict source and no-imputation claims are not executable

- The contract says 11 kline fields while Binance rows and the implementation
  require 12.
- `json.loads` accepts non-finite constants and duplicate object keys.
- untrusted numeric fields are coerced with `float()`/`int()`, admitting
  booleans and representations outside the declared wire grammar.
- feature generation applies a blanket numeric `ffill()`. It does not prove
  that zero 2026 feature cells were changed by that operation.

These do not overturn the independently reproduced result, but they prevent
the strong source/parity claims currently published.

## 3. Correction work

### C1 - Repair and re-attest acquisition continuity

1. Freeze the exact PRE above before editing.
2. Introduce one typed timestamp-to-epoch-milliseconds helper. It must handle
   the actual parquet dtype explicitly and reject ambiguous, boolean,
   non-finite or out-of-range input. Never divide an integer whose time unit
   has not first been normalized explicitly.
3. Use that helper in both acquisition overlap and regeneration extension
   selection.
4. Require exactly 2190 overlap rows and 1458 extension rows before feature
   generation. Empty overlap can never demonstrate continuity.
5. Re-parse the four existing raw page files, verify each byte digest against
   the committed page receipt, and rederive continuity. Do not make another
   network request.
6. Prove exact timestamps and exact raw fields over all 2190 rows. Publish a
   corrected v2 receipt. Preserve the defective v1 receipt unchanged and mark
   it superseded with a field-by-field correction map.
7. Regeneration must append only rows after the true final frozen timestamp.
   Assert the first extension row is `2026-01-01T00:00:00Z` and that no
   overlap row was replaced by the extension path.

### C2 - Establish restricted staging custody

1. Materialize a new staging root outside all repositories at exact mode
   `0700`, owned by the current uid.
2. Create every file at exact mode `0600`; reject symlinks, non-regular
   files, wrong owner and permissive modes from descriptor-derived facts.
3. Copy only bytes whose digest matches the frozen page evidence. Rebuild
   derived files inside the new root.
4. Refuse an existing permissive root rather than silently chmodding or
   trusting it. Record the old `0775/0664` state as the disclosed PRE.
5. Add tests under `umask 000`, including root/file symlinks and wrong
   modes.

### C3 - Complete the predeclared statistical evidence

For every horizon, block and arm, persist enough sanitized evidence to
rederive, not merely trust:

- exact score label per anchor;
- the three predicted class probabilities per anchor;
- multiclass log loss;
- hit-versus-censored loss;
- direction-given-hit conditional loss and its exact additive contribution;
- exact identity
  `multiclass = hit/censored + hit_indicator * direction_given_hit`;
- multiclass Brier score and per-class Brier components;
- per-class argmax recall with typed unavailable handling;
- class support;
- calibration-decile counts, mean predicted hit probability and observed hit
  rate.

Define rounding only at publication boundaries. The verifier must derive every
aggregate from the per-observation records and reject non-finite values,
invalid probability simplexes, label/probability length disagreement and
failed additive conservation.

### C4 - Replace the self-authenticating verifier

1. Require an expected bundle SHA-256 supplied outside the mutable bundle and
   verify the exact bytes before parsing. Publish the digest in the return and
   bind it to the pushed Git blob/commit used by the independent auditor.
2. Verify the actual sealed-contract bytes against `contract_sha256`; reject
   missing, malformed or differing code/data/contract identities.
3. Enforce exact schemas at every level. Unknown or missing fields refuse.
4. Recompute exact canonical anchor timestamps and counts from the sealed
   blocks, stride and purge. Verify:
   `n_score == len(anchors) == len(labels) == len(probability rows)`.
5. Derive class supports, completeness and every license from evidence. Never
   consume `blocks_complete` or `licenses_ok` as authority supplied by the
   bundle.
6. Recompute all arm metrics, all contrast fields, bootstrap results, Holm
   values and the final decision. The complete contrast object must match, not
   only pooled and per-block skill.
7. Freeze the four accepted-forgery PRE cases above as regressions. In
   particular, a coherently re-digested fake neural passer must be rejected
   before its scientific content is evaluated.

The verifier may still offer an explicitly named
`INTERNAL_CONSISTENCY_ONLY` mode, but that mode cannot publish
`N3_BUNDLE_VERIFIED` or authorize any gate.

### C5 - Close parser, schema and imputation boundaries

1. Supersede the contract typo with 12 exact kline fields.
2. Parse JSON with duplicate-key and non-finite-constant rejection.
3. Enforce the exact Binance field grammar: integer non-boolean timestamps
   and counts, finite canonical decimal strings where the endpoint uses
   strings, exact field count, and no silent coercion.
4. Before historical compatibility `ffill()`, measure its effect by role.
   Any 2026 input cell changed by filling must refuse. Publish the count,
   expected to be zero. Historical warmup compatibility must remain named
   separately from confirmation data.

### C6 - Re-execute and return

Re-execute only from the already acquired, digest-verified raw pages in the new
restricted root. No acquisition, new split, new threshold, new target or model
choice is authorized.

The corrected return must include:

1. PRE/POST for C1-C5;
2. v1-to-v2 receipt and bundle correction maps;
3. exact page, source, contract, code, bundle and publication identities;
4. the complete metrics and decomposition;
5. external-digest verifier output and every adversarial refusal;
6. comparison with the independently reproduced v1 units and verdict;
7. focused and full-suite counts from the final pushed tip;
8. one unambiguous neural-gate line.

Expected result, unless correct rederivation proves otherwise:

`TARGET_SCALE_EFFECT_NOT_CONFIRMED - NEURAL/GPU GATE CLOSED`

If any corrected prerequisite fails, return its typed refusal and keep the
gate closed. Do not preserve the old verdict by weakening a check.

## 4. Acceptance boundary

Acceptance requires:

- true 2190/1458 acquisition accounting and exact continuity over all overlap
  rows;
- `0700/0600` custody proven from filesystem facts;
- every predeclared metric rederivable from public per-observation evidence;
- all four Musashi verifier forgeries refused;
- corrected execution reproduced from the existing raw pages;
- no network request, GPU, SAC, venue, service, deployment or promotion.

Until those conditions hold, the disposition is:

`N3_NUMERICAL_RESULT_REPRODUCED_PACKAGE_REVISE_NEURAL_GATE_CLOSED`
