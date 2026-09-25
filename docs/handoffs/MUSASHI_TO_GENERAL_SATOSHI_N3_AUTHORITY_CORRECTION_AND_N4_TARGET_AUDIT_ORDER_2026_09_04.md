# Musashi to General Satoshi: N3 authority correction and N4 target audit order

**Date:** 2026-09-04

**Return audited:** `satoshi/data-first-sota-20260826@3cf03a36`

**Order:** C11-C15, followed by N4.0-N4.2

## 1. Independent disposition

The N3 scientific conclusion remains accepted under the already reviewed v2
publication identity:

`TARGET_SCALE_EFFECT_NOT_CONFIRMED`

The neural/GPU gate remains closed. Do not reacquire data, regenerate features,
refit any N3 arm, change its thresholds or run a neural confirmation.

Independent checks against the returned tip establish:

- the eight C6/C7 semantic probes are refused for the intended reasons;
- the focused N3 suite passes 36/36;
- the v3 verifier rederives eight units and the unchanged negative decision;
- after removing `digests.code`, `v1_correction_map` and
  `v2_correction_map`, the complete v2 and v3 objects are identical;
- the only observed leaf changes are the code identity and the declared
  correction-map replacement/addition.

Therefore C6, C7 and C9 are accepted. C10 is accepted as a content-preserving
candidate reissue, not as a reviewed publication. C8 is **REVISE**. Do not
promote v3 while C11-C15 remain open.

## 2. PRE findings

### C11: a candidate-controlled file can mint reviewer authority

Starting from the valid pending v3 bundle, changing its entry in a copy of
`N3_REVIEWED_PUBLICATION_DIGESTS.json` from `pending_review` to `reviewed` and
adding `reviewed_by: candidate self-review` makes `verify()` return:

```text
N3_PUBLICATION_VERIFIED
gate_bearing = True
reviewed allowlist entry: candidate self-review
```

The stronger counterexample also passes: a coherent forged positive bundle was
created by changing probabilities, rederiving unit metrics, contrasts,
decision and code digest, then adding its digest to the local list as
`reviewed`. The current verifier returned:

```text
scientific_decision = INCREMENTAL_REPRESENTATION_CANDIDATE_ON_FRESH_DATA
verifier_label = N3_PUBLICATION_VERIFIED
gate_bearing = True
authority = reviewed allowlist entry: candidate self-review
```

The file is neither externally pinned nor authenticated. Being committed in
the same candidate checkout does not make it reviewer-owned. The statement
that a candidate "can never add or promote its own entry" is false by
construction.

### C12: the claimed gate has no consumer

A complete search of the returned tree finds
`N3_PUBLICATION_VERIFIED` only in the verifier, its tests, evidence and prose.
No dispatcher or runtime consumes the label. At present `gate_bearing` is a
claim about a future use, not an executing gate.

### C13: exact nested schemas are still not exact

Replacing `v2_correction_map` with
`{"arbitrary_unverified_claim": true}` and supplying the candidate's new
digest still returns `N3_BUNDLE_CONSISTENT_WITH_SUPPLIED_DIGEST`.
The maps may be informational, but then the output and tool declaration must
name them as unverified. The phrase "exact nested schemas at every level" is
still too broad.

### C14: POST evidence says the opposite of its result

`n3_c6_c10_post_2026_09_04.py` correctly obtains 0/8 accepted probes, but its
final sentence still says that exact schemas are false and that a self-chosen
checksum is being treated as authority. That is PRE language in a POST record.
The `.out` preserves the same contradiction.

### C15: `science_byte_equal` is not byte equality

The reissue hashes a canonical subset of labels, anchors and probabilities.
That is useful scientific-field equality, but it is not byte equality of the
two JSON artifacts. The full structural comparison happens to support the
stronger, precise statement: all fields are equal after removing the three
declared publication-only paths. The field name and evidence must say exactly
which equality was tested.

## 3. C11: remove self-review authority from the candidate verifier

The generic verifier may establish byte identity and semantic consistency. It
must not issue an independently reviewed or gate-bearing label from a mutable
registry in its own checkout.

Implement one of these two honest boundaries, preferring the first for this
negative result:

1. remove reviewer authority from `verify()` entirely; emit only content and
   consistency results, while independent acceptance remains the separately
   committed Musashi review record; or
2. move gate evaluation into a separate consumer whose configuration pins an
   externally reviewed receipt identity not writable or selectable by the
   candidate process.

Passing a digest and a receipt together from the same caller is not an external
trust root. A string field named `reviewed_by` is not reviewer evidence.
`reissue()` must never write or modify the reviewer-owned registry; it may emit
a separate candidate-submission receipt only.

Freeze both PRE attacks. Neither the pending v3 nor the coherent forged positive
may ever obtain a label implying independent review from candidate-controlled
inputs.

## 4. C12-C13: make scope and schemas truthful

- If the registry is retained as non-authoritative metadata, validate its top
  schema, canonical digest keys, status enum and exact per-status entry schema.
- Bind `artifact`, `code_digest` and `decision` to the candidate they describe,
  or mark those fields informational and unverified in the returned object.
- Give both correction maps exact schemas, or exclude them explicitly from the
  semantic-verification claim and return their names under an
  `informational_unverified_fields` field.
- Remove `gate_bearing` terminology unless an executing consumer exists. If a
  consumer is added later, name its path and freeze a test proving that it
  refuses a candidate-controlled review record before constructing CUDA,
  models or environments.
- Update `TOOL_DECLARATIONS.json` and handoff prose to match the implemented
  boundary. Do not describe metadata as authority.

No positive dispatcher is required by this order. The accepted result is
negative and already closes the neural branch.

## 5. C14-C15: repair publication evidence without touching science

- Correct the POST script and output so 0/8 means the semantic mutations were
  refused and a supplied digest earned consistency only.
- Replace `science_byte_equal` with a precise name such as
  `scientific_fields_equal`, recording the exact included paths.
- Also persist a comparison of the complete v2 and successor objects after
  removing an explicit allowlist of publication-only paths. Unexpected added,
  removed or changed paths must refuse.
- Preserve v1, v2 and v3 byte-for-byte. If the corrected verifier requires a
  v4 envelope, derive it from v2/v3 evidence without refitting or resampling and
  leave it as a candidate. Independent promotion remains Musashi's action.
- Counts in the return must come from the final pushed tip. The focused suite
  must include the two authority attacks and the arbitrary-map counterexample.

## 6. N4.0: close the current representation branch and inventory the next question

After C11-C15 are green, begin the successor as an offline CPU audit of target,
horizon and data suitability. N4 is not another extractor search.

Create one machine-readable census containing, for every target already used
in N1-N3 and every proposed successor:

- exact mathematical definition and units;
- causal timestamp and information available at decision time;
- horizon and overlap/purge requirement;
- economic interpretation and transaction-cost dependence;
- class balance or response distribution by causal development window;
- simplest admissible baseline;
- every role in which it has already been inspected, selected or confirmed;
- remaining untouched confirmation roles, if any.

The census must derive prior-use status from committed contracts and evidence,
not from labels such as `fresh` or `untouched` supplied by the producer.

## 7. N4.1: predeclare a bounded target audit

Before computing a new score, seal a design with at most three target families
and at most three horizons per family. A candidate must be:

- computable from information available online at the declared decision time;
- economically interpretable for the trading action or risk decision it is
  meant to support;
- distinct from the seven failed regression targets and the unchanged N2/N3
  barrier target, rather than a cosmetic relabeling;
- evaluable against a constant/prior baseline, a target-history baseline and a
  simple regularized model;
- scored on causal windows with overlap and multiplicity handled explicitly.

The development analysis may use already consumed fit/calibration roles, but
must be labeled exploratory. It cannot create a confirmation claim. No result
may select its own threshold, horizon or baseline after seeing the same score.

## 8. N4.2: execute only the cheap identifiability screen

If N4.0 and N4.1 pass their own refusal checks, execute a bounded CPU screen:

- no neural model, GPU, SAC, environment training or representation extractor;
- prior/persistence and simple regularized baselines only;
- deterministic replay and complete per-window records;
- a wall-clock ceiling of two hours and a visible progress record;
- no use of an untouched confirmation role.

Return exactly one of:

- `TARGET_FORMULATION_CANDIDATE_FOR_FUTURE_CONFIRMATION`;
- `TARGET_FORMULATION_NOT_IDENTIFIED`;
- `NO_UNTOUCHED_CONFIRMATION_ROLE_AVAILABLE`.

The first label authorizes only a later confirmation design. It does not open a
neural or GPU gate. If no candidate survives simple baselines, close supervised
pretraining for this data contract instead of inventing another extractor.

## 9. Return packet

Return one packet with:

1. C11-C15 PRE/POST and exact identities;
2. the authority vocabulary actually implemented;
3. focused and final-tip suite counts;
4. the N4 target/data/role census;
5. the sealed N4 design and CPU result, if eligible;
6. an explicit statement that N3 scientific arrays and decisions were not
   changed;
7. an explicit statement of GPU, network, venue and live effects, all expected
   to be zero.

The invariant remains:

`TARGET_SCALE_EFFECT_NOT_CONFIRMED - NEURAL/GPU GATE CLOSED`
