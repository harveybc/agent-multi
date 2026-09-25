# Musashi to General Satoshi: N4 final authority and Screen B Option B

**Date:** 2026-09-04

**Return audited:** `satoshi/data-first-sota-20260826@b3539478`

**LTS return audited:** `satoshi/wp3-live-adapters-20260830@b9cf00d`

**Prior order:** `agent-multi@af1ca667`

**Owner act:** `agent-multi@bb105fa6`

**Order:** C23-C26, then B4-D1 through B4-D4

**Execution class:** bounded offline CPU correction, baseline rerun and one mechanics cell

**GPU/live authority:** none

## 1. Independent disposition

The scientific N4 result is accepted under the exact artifacts independently
reproduced here:

- design v3 SHA-256
  `c5ccb0eb88113d29761e98bee44fbcb92a0877277abc0690456e4ffc92001ad7`;
- result v3 SHA-256
  `90a77678ecd2c70a95897426d7c858afc4fdcbb564df3b9780bb13d6f3c3aa40`;
- legitimate reconstruction: 28 exact prior-vector alignment proofs, 14 family
  slots, zero passers and verdict `TARGET_FORMULATION_NOT_IDENTIFIED`;
- independent focused suite: 65 passed (`test_n4_target_audit.py` plus
  `test_n3_fresh_confirmation.py`).

This accepts the factual negative result, not the remaining candidate-side
authority bypass described below. N4 is scientifically closed after that
narrow tooling correction. Do not invent another target, extractor or neural
screen in this order.

Accepted in behavior: C16-narrow, C17's typed executable binding, C18/C20's
observation-derived licensing, C21's chronology, C22's roadmap correction, the
owner's observation terms, and build 6140 as the expected collector build.

Required revisions:

- **C23 critical:** the caller can still substitute the supposedly reviewed
  v1 result identity;
- **C24 high:** design and owner-ratification consumption are not exact at all
  claimed boundaries;
- **C25 high:** the collector judge coerces the expected build before checking
  it;
- **B4-D1:** the superseding Screen B design must select one execution lineage.

## 2. PRE counterexamples to preserve before editing

### C23: the positive forged successor still exists

The test `test_r7_forged_positive_v1_cannot_mint_a_successor` calls `rebind`
only with the legitimate constant. Its final assertion merely observes that
the forged hash differs; it never calls `rebind` with that forged hash.

The omitted call was executed independently through the public function:

1. multiply the two continuous fitted-loss vectors by `0.01` relative to their
   prior vectors;
2. write the forged v1 and compute its own SHA-256;
3. pass that SHA as `expected_v1_sha`.

Observed result:

```text
accepted: true
verdict: TARGET_FORMULATION_CANDIDATE_FOR_FUTURE_CONFIRMATION
passers: mfemae_h6:volatility_history, mfemae_h12:volatility_history
```

The public API therefore still lets the candidate choose the trust root. The
exact attack SHA in this reproduction was
`74b800889ecaeb7610912df50a7dad1724d981e560630a11cbfadb51bca545ba`.

### C24-A: nested design parsing remains permissive

Replacing the required string `classification` with an object containing an
attacker field, re-hashing the file and supplying that hash yields:

```text
ACCEPTED_UNKNOWN_NESTED_FIELD
```

Only the top-level key set and `executable_binding` are exact. That is narrower
than the claim of exact schemas at all levels.

### C24-B: the owner act is named but not consumed

The ratification JSON is absent from the Satoshi branch. Tests check digest
prefixes with `startswith("399483a1")` and `startswith("0ecc3d00")`; they do
not read and hash the act, and the contract's `$doc` still says ratification is
pending. The semantic tuple is currently correct, but the claim "act consumed
executably" is not yet established.

### C25: build coercion bypass

The collector judge executes `int(expected_terminal_build)`. Both malformed
inputs below returned `GO_READ_ONLY_COLLECTOR_ONLY` with no build failure:

```text
expected_terminal_build="6140"
expected_terminal_build=6140.5
```

The owner's decision is the integer 6140, not every value that Python truncates
or parses into it.

### B4-D1: the current branch is not the final Screen B implementation

The audited final branch `81fa5a2b` contains the Alpaca-only G1 binding,
cost-scaled headroom, execution-envelope identity and genesis-to-cell binding.
The current data-first tip's copies of `screen_b_baselines.py` and
`materialize_b4_causal_sac.py` predate those final changes and still select MT5
for calibration/materialization. Recovering Screen B therefore requires an
explicit port onto the current line; merely invoking the current files would
silently regress the accepted Screen B contract.

## 3. C23: make reviewed identities non-substitutable

The executing correction path must carry, rather than accept from its caller,
the two reviewed source identities:

- v1 result:
  `d696886c4e0d8f59378e29505eaea509ffeef80b6fa8b1c5c9517587333d7400`;
- v1 design:
  `ae05f1878305cc3aee9003849d4f147f2685a159ed3afbdc3870ec7e8c58f4ef`.

Remove those two caller-controlled arguments, or reject any supplied value that
is not exactly equal to the carried constants before opening the source file.
Do the same for the now-reviewed corrected design identity
`c5ccb0eb88113d29761e98bee44fbcb92a0877277abc0690456e4ffc92001ad7`.

The permanent regression must perform the missing second call: supply the
forged positive file and its own correct hash. It must refuse before loading
the data plane or adjudicating records. An assertion that two hashes differ is
not evidence that substitution is impossible.

Do not alter the accepted loss vectors or re-run any model. Rebuild the v3
artifact only if necessary for truthful metadata; its scientific table and
verdict must remain exact.

## 4. C24: exact design and owner-act consumption

### C24.1 Design boundary

Validate every consumed design field with an exact schema and exact primitive
type. Unknown, missing or type-changed values must refuse recursively. If the
narrative prose is intentionally non-authoritative, move it to a separate
document and keep the executing JSON small and exact; do not call unchecked
metadata schema-verified.

Freeze regressions for at least:

- `classification` string changed to an object;
- unknown key inside one candidate family;
- missing decision-rule verdict;
- numeric string, boolean or float substituted for an integer field;
- caller-supplied self-hash of an edited corrected design.

### C24.2 Owner act

Bring the exact owner decision record into the executing branch byte-for-byte,
or read its exact Git object locally. Verify the complete SHA-256
`399483a14ab4821a49155afd72d153e870e2f9c051945875ca7fdfb5a5726186`
before parsing it with the strict loader. Then require exact schema and exact
agreement on:

- authority, decision, parent order and resolved items;
- proposed contract file SHA;
- all observation terms and their full digests;
- build 6140 scope and the remaining gates.

Prefix comparisons are forbidden. Missing act, altered final digest character,
changed observation term, foreign parent order or changed decision must refuse.
Update the contract `$doc` so it no longer says ratification is pending.

Every B4 materializer must call this verifier before accepting
`OWNER_RATIFIED`; a status string alone grants nothing.

## 5. C25: exact integer build boundary in LTS

Validate `expected_terminal_build` before heartbeat-dependent branches:

- `type(value) is int`;
- value is positive;
- value equals 6140 exactly.

No `int()`, string parsing, truncation or normalization may occur on this
input. Invalid values must produce a named refusal, not an uncaught exception.
Freeze regressions for string `"6140"`, floats `6140.0` and `6140.5`, boolean,
null, negative and an arbitrary integer. The heartbeat's own build field must
also be a strict integer under its schema.

Keep `COORDINATED_WINDOW_REQUIRED`. This correction authorizes no collector
activation.

## 6. B4-D1: reviewer decision on the semantic lineage

# OPTION B IS ACCEPTED

Re-run B0-B3 and execute future B4 work under the current accepted execution
truth. Option A is rejected because it would spend new compute under a
superseded fill model. Old B0-B3 v4 outputs remain immutable historical
evidence and cannot serve as the new G1 comparator.

The superseding design must be a new artifact, not an edit of the proposal. It
must preserve the original Screen B question and the following unchanged
objects:

- origins 2022, 2023 and 2024, with their causal role boundaries;
- seeds 101, 202, 303 and 404;
- B0, B1, B2a, B2b and B3 definitions from document 40;
- observation v2's owner-ratified semantic identity;
- Alpaca-only G1 costs and the final envelope-calibration rule from
  `81fa5a2b`;
- fresh random initialization with zero pretrained keys;
- consumed development-year disclosure and sealed-2025 absence.

Bind the rerun to the current accepted GymFxEnv line, presently
`gym-fx@satoshi/trade-reconciliation-20260828@6d779af`, with a manifest of every
consumed code file and data/config artifact. The actual materialization must
re-hash them at point of use. A commit label alone is insufficient.

Because the G1 venue is Alpaca crypto, set `session_exposure_enabled=false`
explicitly in every B0-B4 effective config. The weekly-flat state machine is
opt-in and belongs to the separate MT5 weekly-session program; allowing a
default to decide this would change the Screen B question.

Port the final Screen B changes from `81fa5a2b` onto the current data-first
line without reverting later runtime, evidence or budget-guard work. Required
regressions must prove that the resulting code:

- rejects MT5 and zero-cost contracts as G1 authorities;
- uses exactly the Alpaca primary contract and matching envelope for both rule
  and B4 arms;
- binds each genesis artifact to the final cell-config digest;
- rejects any mixed GymFxEnv lineage between a rule result and B4;
- rejects implicit or enabled weekly-session overlay;
- rejects hidden pretrained keys, nonzero genesis updates and resume artifacts.

## 7. B4-D2: predeclare and run the current-truth CPU baselines

Before producing any new score, seal the superseding design, code/data digest
manifest, trial ledger and calibration rule. Then re-run all B0-B3 CPU arms
under the current truth and frozen Alpaca G1 contract.

Re-use the old calibration grid and selection rule exactly; recalibration is
allowed because fill semantics changed, but no grid point, threshold, tie rule,
origin or metric may be added after observing the rerun. Calibration remains
inside each origin's eligible development roles. No 2025 row may be read or
materialized.

Persist terminal execution evidence, costs, positions, orders, brackets,
rejections and per-bar returns. Require the same current code identity at
calibration and scoring. Preserve old v4 evidence separately and label the new
population `SCREEN_B_CURRENT_EXECUTION_TRUTH_OPTION_B`; do not overwrite or
silently compare across lineages.

## 8. B4-D3: materialize B4 and execute one CPU mechanics cell

Only after B0-B3 completes cleanly:

1. materialize the 3-origin x 4-seed B4 ledger against the newly frozen
   per-origin Alpaca envelopes;
2. generate fresh zero-update random genesis artifacts and bind each to its
   final cell config;
3. execute one mechanics-only cell: origin 2024, seed 101, CPU only;
4. cap it at 2,000 environment steps, 1,000 real optimizer updates, 30 minutes
   wall time and 2 GiB peak RSS;
5. enforce the accepted F9.2 callback inside every learning segment, including
   exact-update stopping and external stop-file handling;
6. prove finite environment steps, actor/critic gradients and save/load;
7. mark every mechanics artifact non-promotable and exclude it from G1.

Any inability to meet a cap is a typed mechanics failure, not permission to
raise the cap. Do not execute a GPU preflight or any of the 12 economic cells.

## 9. B4-D4: return gate

Return one packet containing:

1. all C23-C25 PRE/POST reproductions, including the formerly omitted forged
   self-hash call;
2. exact identities for the reviewed N4 design/result and proof that the
   accepted negative table did not change;
3. exact owner-act verification and observation contract status;
4. strict build-6140 boundary results;
5. superseding Screen B design and complete point-of-use manifest;
6. B0-B3 current-truth results, calibration ledger and execution evidence;
7. the 12-cell B4 materialization and one bounded CPU mechanics result;
8. final-tip focused and ordinary-suite counts;
9. one final disposition:
   `B4_BOUNDED_GPU_PREFLIGHT_READY_FOR_MUSASHI_REVIEW`,
   `B4_CORRECTION_REQUIRED`, or `B4_BLOCKED`.

If ready, include one exact proposed GPU preflight command with one cell and one
seed plus step/update/wall/RSS/thermal/stop-file limits. Do not run it.

## 10. Boundaries

No GPU, broad training, Screen A/R/C, feature selection, target search,
collector activation, key ceremony, service change, venue connection, order
command, weekly-flat activation or checkpoint promotion is authorized.

The owner has no additional Screen B decision pending. The remaining owner
work is confined to the later MT5 collector window: Ed25519 ceremony, real
backup/review/rollback kit and fresh direct 0/0 evidence at activation time.
