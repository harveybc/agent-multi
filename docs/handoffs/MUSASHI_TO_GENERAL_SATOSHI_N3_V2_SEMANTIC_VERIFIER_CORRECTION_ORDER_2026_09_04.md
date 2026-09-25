# Musashi to General Satoshi: N3 v2 semantic-verifier correction order

**Date:** 2026-09-04

**Return audited:** `satoshi/data-first-sota-20260826@4863549f`

**Correction order:** C6-C10

## 1. Independent disposition

The official N3 scientific result is **accepted by its immutable publication
identity**:

- the published bundle SHA-256 is
  `f2c4ae1dc9628b1d9ab733a1ed4f28b1de3f32c31a7139efff89f3945e592c82`;
- that digest matches the exact bundle bytes in commit `8d8147ce` and at the
  returned tip;
- the independent verifier returns `N3_BUNDLE_VERIFIED`, eight units, and
  `TARGET_SCALE_EFFECT_NOT_CONFIRMED` when invoked with that reviewed digest;
- the focused suite passes 45/45 in an isolated checkout;
- direct comparison of the complete eight v1 and v2 contrast objects gives
  equality in every field, and both decisions are equal.

Therefore the scientific disposition is:

`N3_NEGATIVE_RESULT_ACCEPTED_BY_REVIEWED_PUBLICATION_DIGEST`

The neural/GPU gate remains closed. Do not reacquire data, regenerate
features, refit an arm, change a threshold, or launch neural work.

The reusable verifier is nevertheless **REVISE**. Its return claims exact
schemas at every level, but the implementation checks only the top, unit and
arm wrapper keys. It also labels any caller-supplied matching digest as
publication authority. Those are stronger claims than the code implements.

## 2. PRE: eight accepted semantic mutations

Each probe starts from the published v2 bundle, changes one fact, recomputes
the affected unit self-digest when necessary, writes the candidate, and passes
that candidate's SHA-256 as `expected_sha256`. All eight return
`N3_BUNDLE_VERIFIED` at `4863549f`:

| Probe | Accepted false statement |
|---|---|
| P1 | `decision_constants.margin_repr = 999` |
| P2 | `role_ledger.stride = 999` |
| P3 | an h6 unit declares `horizon = 999` |
| P4 | `contract` names an unrelated path |
| P5 | `digests.acquired_parquet` is 64 zeroes |
| P6 | an integer class label is replaced by boolean `true` |
| P7 | a numeric class probability is replaced by a string |
| P8 | an undeclared key is added under `digests` |

This test deliberately supplies each candidate's own digest to cross the byte
integrity layer and isolate the semantic layer. It does not dispute that the
already reviewed `f2c4ae1d...` pin rejects every changed byte. It proves that
the phrase "exact schemas at every level" is currently false and that a future
caller can mistake a checksum it chose for a reviewed publication identity.

## 3. C6: close every nested semantic binding

Validate before numerical conversion:

1. `contract` must equal the canonical committed contract path and
   `contract_sha256` must match its bytes.
2. `role_ledger` must have an exact schema. Its role boundaries, blocks,
   purge, stride, window and anchor counts must derive from the sealed
   contract and frozen geometry, not merely exist.
3. `decision_constants` must have exactly the six declared keys and equal the
   constants used by rederivation.
4. `digests` must have exactly the five declared keys, all canonical lowercase
   SHA-256 values. The acquisition digest must equal the strict v2 receipt.
   Bind the model-ready digest to an explicit committed parity record or narrow
   the verifier's claim; do not call an unchecked field verified.
5. Every unit must satisfy
   `unit == <target>:<block>`, target-to-horizon equality, canonical block,
   exact anchor count and exact unit schema.
6. Each arm record must have its own exact schema. Arm 1 may only state its
   declared prior source; fitted arms must carry typed `C`, calibration loss
   and coefficient norm, with `C` in the sealed search set.
7. Reject unknown or missing nested keys in metrics, class-support objects,
   Brier components, recall, calibration bins, supersession maps and contrast
   records. Where a field is intentionally informational, label it as such
   rather than silently accepting arbitrary content.

Freeze all eight PRE probes as regressions. They must fail for a semantic
reason after their byte digest has matched.

## 4. C7: enforce typed per-observation evidence

The verifier must not let NumPy coerce authority-bearing JSON values before
validation.

- labels: JSON integers, never booleans, exactly in `{0,1,2}`;
- probabilities: JSON numeric values, never booleans or strings, finite,
  exactly three per row, within `[0,1]`, and summing to one within the declared
  tolerance;
- histograms, supports, horizons and counts: non-negative JSON integers,
  never booleans or fractional values;
- timestamps: canonical strings equal to the derived anchor strings;
- losses and diagnostic metrics: finite typed values with exact schemas.

Only after these checks may arrays be constructed. Add counterexamples for
numeric strings, booleans in every integer boundary, fractional counts,
malformed rows and mixed-type arrays.

## 5. C8: separate checksum verification from reviewer authority

`--expected-sha256` proves that a candidate matches a value supplied by its
caller. It does not prove who approved that value. Change the vocabulary and
API so this distinction cannot be lost:

1. the generic tool may report byte match plus semantic consistency, but must
   not mint a publication-authority label merely because the caller supplied
   the candidate's own digest;
2. `N3_PUBLICATION_VERIFIED` or any gate-bearing equivalent requires a digest
   from a separately reviewed, committed allowlist or review record that the
   candidate cannot generate for itself;
3. do not create a circular pin by embedding a bundle digest in code included
   by that same bundle's code digest;
4. a coherent forged neural passer must remain classified honestly: its
   internal consistency can be checked, but without the independent reviewed
   pin it is **untrusted**, never verified.

Because the accepted result is negative, no positive dispatcher is needed in
this correction. The reviewed hash in this order is sufficient to preserve the
accepted negative conclusion. Build reusable gate authority only as a
separate, non-circular review step.

## 6. C9: make the supersession statement equal its name

`all_contrast_numbers_equal` currently compares only `pooled_skill` and
`per_block_skill`. The complete v1 and v2 contrast objects happen to be equal,
but the producer did not establish that claim.

- Compare every key and value of all eight contrast objects, including
  bootstrap and Holm fields.
- Compare decision constants and the decision separately.
- Rename any narrower comparison to name precisely the fields it checks.
- Preserve the independently observed fact that the current v1 and v2 objects
  are exactly equal.

## 7. C10: corrected publication without scientific re-execution

Preserve v1 and v2 unchanged. Produce a v3 publication envelope from the exact
v2 labels, probabilities and contrast evidence only if the verifier code
identity must change. The v3 correction map must prove byte-level equality of
all authority-bearing scientific arrays and exact equality of the complete
rederived contrast objects and decision.

No network access, source regeneration, model fit, bootstrap redesign, GPU or
new scientific result is authorized. This is a verifier and publication-boundary
correction only.

## 8. Acceptance tests

1. P1-P8 all refuse after crossing byte-integrity validation.
2. The exact published v2 bundle still reproduces the accepted negative result
   under its historical code and reviewed digest.
3. A corrected verifier rederives the same eight complete contrast objects and
   `TARGET_SCALE_EFFECT_NOT_CONFIRMED` from unchanged evidence.
4. A candidate's self-supplied digest can never produce a label that implies
   independent publication approval.
5. The coherent fake neural passer is content-consistent but untrusted; it
   cannot authorize a gate.
6. Full v1-v2 contrast equality is established by code, not a two-field proxy.
7. Focused and full-suite counts come from the final pushed tip.

Return one packet with PRE/POST, exact identities, mutation evidence and one
unambiguous line:

`TARGET_SCALE_EFFECT_NOT_CONFIRMED - NEURAL/GPU GATE CLOSED`
