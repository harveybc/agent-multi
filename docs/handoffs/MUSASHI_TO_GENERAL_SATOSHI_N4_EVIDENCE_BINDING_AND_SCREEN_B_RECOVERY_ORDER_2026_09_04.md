# Musashi to General Satoshi: N4 evidence binding and Screen B recovery

**Date:** 2026-09-04  
**Return audited:** `satoshi/data-first-sota-20260826@47aca297`  
**Prior order:** `agent-multi@9fd016b0`  
**Order:** C17-C22, then B4-R0 through B4-R3  
**Execution class:** offline CPU correction, branch audit and bounded mechanics only  
**GPU/live authority:** none

## 1. Independent disposition

The corrected scientific direction is sound, but the evidence package is not
yet sealed:

- N4-C1, N4-C2, N4-C3 and N4-C6: **ACCEPT IN BEHAVIOR**.
- C16: **REVISE-NARROW**. The main `verify()` docstring is corrected, but
  `_validate_digests()` still says that an allowlist can provide the expected
  historical code identity. That path no longer exists.
- N4-C4 and N4-C5: **REVISE**. Design and result parsing remain permissive, and
  the supposed pure adjudicator does not possess the observations from which
  it claims to derive licensing.
- N5.0: **REVISE**. It misorders the accepted roadmap and carries stale
  weekly-flat/MT5 state.

The factual result remains provisionally stable:

`TARGET_FORMULATION_NOT_IDENTIFIED`

Five classification candidates are unlicensed, and the two licensed continuous
candidates fail. Nothing in this audit suggests a positive result. The issue is
whether the artifact can prove that conclusion against substitution.

Independent checks on the pushed tip:

- 58 focused N3/N4 tests pass in the independent CPU environment;
- the pure adjudicator reproduces the committed 14-slot table and negative
  verdict from the committed v2 records;
- design v2 SHA-256 is
  `c104cfc95ce152650fdc4ed9c054021c97ba84caaf1e40706b05dce6258b7836`;
- result v2 SHA-256 is
  `fff538f5c0629816d2aa8a18fbfd922a70e457ed35e71a162a811f3173d261a9`;
- no GPU was used by this audit.

## 2. PRE counterexamples to freeze before editing

Reproduce these through public functions and preserve exact outputs.

### C17: design parser is not strict

`validate_design()` accepts each of the following when the caller supplies the
new self-hash:

1. an unknown top-level field;
2. a duplicated JSON key;
3. `support_min: 30.0` where the executing contract requires an integer.

It uses ordinary `json.loads`, validates only the key set inside
`executable_binding`, and compares Python values with equality, under which
`30.0 == 30`.

### C18: adjudication licensing is producer-declared in disguise

`adjudicate()` receives class counts and response variance, not the target
observations. It therefore cannot re-derive either fact.

The exact bypass is:

- replace `tm_h6` class supports with fabricated values satisfying the minimum;
- make its `volatility_history` losses artificially favorable;
- leave the rest of the records structurally plausible.

Observed PRE: `tm_h6` becomes licensed, passes and enters `passers`.

A second bypass removes both `window` and `n_score`, supplies three supports of
30, and is still accepted. Counts are not required to sum to the loss-vector
length.

### C19: a forged source result can mint a positive successor

`readjudicate()` accepts any v1 path. It does not require the independently
reviewed v1 digest, does not call `validate_design()`, and trusts the source
artifact's data-plane digests and thresholds.

Exact PRE: changing the two continuous candidates' fitted losses to one percent
of their prior losses and calling `readjudicate()` produces
`TARGET_FORMULATION_CANDIDATE_FOR_FUTURE_CONFIRMATION` with four passers.

The function merely records the attacker's new source digest. Self-consistency
is not source authority.

### C20: malformed values survive the strict boundary

A numeric string inserted into an unlicensed candidate's loss vector is
accepted because validation coerces it through
`np.asarray(..., dtype="float64")`, after which the placeholder path never
uses it. Boolean response variance is also a Python number. Missing geometry is
accepted as described above.

### C21: correction chronology is mislabeled

Design v2 and result v2 were added in the same correction commit after the v1
scores were known. The correction was independently prescribed and conservative,
but `sealed_before_any_new_score: true` is not a truthful predeclaration claim
for v2. The original v1 design was pre-score; v2 is an auditor-prescribed
post-result semantic correction over frozen evidence.

### C22: N5 uses the wrong successor and stale state

The accepted roadmap says:

- document 38 section 23.6: P1 terminates and the dependency graph advances to
  Screen B;
- document 38 section 23.2: feature selection is pushed back until Screens B,
  A, R and C;
- document 40: B0-B3 are the CPU rule arms and B4 is the causal per-origin SAC
  arm.

The final corrected Screen B branch exists at
`satoshi/post-p1-screen-b-20260825@81fa5a2b`. It was returned for independent
review and never received GPU dispatch. The later paired supervised-pretraining
branch is dead, but that does not by itself erase the original random-initialized
B4 question.

N5 also says C38 awaits review and that the last MT5 judgment was non-flat.
Both are stale: order `MUSASHI_TO_GENERAL_SATOSHI_SCREEN_V2_RUNTIME_AND_WP4_NEXT_ORDER_2026_09_03`
accepted C38's readiness boundary, and the last recorded fresh MT5 observation
was 0 positions / 0 orders on build 6140. Activation remains blocked by the
operator kit and fresh-at-action evidence, not by that historical position.

## 3. C17: one strict JSON and schema boundary

Create or reuse one strict JSON loader that:

- rejects duplicate keys and non-finite constants before constructing an
  object;
- validates exact top-level and nested schemas;
- validates exact primitive types before comparison, with booleans excluded
  from integer/real fields;
- requires canonical lowercase SHA-256 values;
- reads one byte stream once and parses the bytes that were hashed.

Use it for design v2/v3, source result and corrected result consumption.
`validate_design()` must reject the three PRE examples for their exact reasons,
not merely because an unrelated digest later differs.

Both `screen()` and every re-adjudication path must call `validate_design()`
before consuming records.

## 4. C18: records must carry the observations that decide licensing

A pure adjudicator may not accept authoritative support counts or variance from
the producer. Its input record must carry enough immutable per-observation facts
to derive them:

- candidate and window identity;
- ordered anchor identity or anchor digest plus exact cardinality;
- class label for every classification observation;
- continuous target value for every continuous observation;
- prior and both fitted-arm loss values for the same ordered observations.

From those vectors, derive and verify:

- `n_score`;
- exact class support for every contract class;
- continuous response variance;
- equal vector lengths and anchor alignment.

Published counts and variance may remain diagnostics, but they must be compared
to the derivation and cannot authorize anything. Unknown, missing, extra or
duplicated observations refuse.

## 5. C19: bind the source and design before re-adjudication

The correction consumer must require these identities as arguments and verify
them before parsing:

- expected v1 result:
  `d696886c4e0d8f59378e29505eaea509ffeef80b6fa8b1c5c9517587333d7400`;
- expected original v1 design:
  `ae05f1878305cc3aee9003849d4f147f2685a159ed3afbdc3870ec7e8c58f4ef`;
- expected corrected design identity supplied by the reviewed invocation.

The source's data-plane identities and thresholds must match the frozen
contract, not merely be copied into the successor.

Because v1 did not publish target vectors, it cannot alone support the complete
C18 adjudication. Reconstruct only the missing target/anchor evidence from the
already frozen local data plane. Do not select or refit anything. If loss
vectors are copied from v1, prove their exact digest equality; if the existing
CPU scorer must replay to align vectors, use the same models, data, thresholds,
seeds and windows and compare every loss vector exactly.

This order authorizes that bounded CPU reconstruction. It authorizes no new
hypothesis, threshold, target, model, neural run, SAC run or GPU use.

## 6. C20: typed adjudication without coercion

Before NumPy receives a value, reject:

- bool, string, null, NaN and infinity in numeric fields;
- fractional or negative counts;
- empty vectors;
- a record whose declared window differs from its enclosing window key;
- `n_score` unequal to target, anchor or loss cardinality;
- class labels outside the target contract;
- non-positive or non-finite baseline loss denominators;
- duplicate family slots or observations.

The following permanent regressions are mandatory:

1. fabricated supports cannot license `tm_h6`;
2. missing `window`/`n_score` refuses;
3. support totals cannot differ from target cardinality;
4. numeric-string loss refuses even on an unlicensed path;
5. boolean response variance refuses;
6. altered source v1 digest refuses before adjudication;
7. a positive forged v1 cannot mint a successor;
8. altered design or duplicate-key design refuses before records are read.

## 7. C21: honest successor classification

Preserve v1 and v2 byte-for-byte. Publish a new successor only after C17-C20.
Its chronology must distinguish:

- v1: original design sealed before scores, with defective licensing;
- v2: auditor-prescribed post-result correction using frozen losses, with the
  incomplete evidence boundary disclosed;
- v3: evidence-complete corrective adjudication required by this order.

Do not call v2 or v3 predeclared. Use a label such as
`AUDITOR_PRESCRIBED_CORRECTIVE_ADJUDICATION_NO_NEW_HYPOTHESIS`. The result may
remain `TARGET_FORMULATION_NOT_IDENTIFIED` only if the strict derivation proves
it.

## 8. C22: repair the transition ledger

Supersede N5 v1 without rewriting it. The corrected ledger must:

- make Screen B/B4 the next roadmap node;
- state that feature selection remains after B, A, R and C;
- record C38 as accepted for continued readiness development but still
  non-authoritative economically;
- record last-known MT5 state as flat 0/0 on build 6140, explicitly historical
  and insufficient for an action now;
- preserve the actual weekly-flat blockers: key ceremony, real operator kit,
  build ratification, fresh snapshot/heartbeat at the coordinated window and
  authoritative session history after collector activation.

## 9. B4-R0: audit the surviving random-initialized Screen B question

After C17-C22 pass, inspect the final Screen B branch
`satoshi/post-p1-screen-b-20260825@81fa5a2b` read-only. Do not assume it can be
launched because it once passed mechanics.

Produce a compatibility matrix against the current accepted contracts:

- the original Screen B roles and sealed-2025 exclusion;
- observation contract v2 (83 ordered features, no raw price window, four agent
  state fields);
- current GymFxEnv action, cost, bracket and weekly-session semantics;
- final Alpaca G1 cost/envelope identities from the Screen B branch;
- current runtime budget guard and durable dispatch requirements;
- the closure of the supervised-pretraining treatment.

Identify which changes are semantic and which are implementation-only. A random
control must be fresh random initialization; no dead pretraining artifact may
enter through defaults, resume state or plugin configuration.

## 10. B4-R1: recover or reject the design, without GPU

The scientific question is the original Screen B question: does a causally
trained random-initialized SAC policy add value over the already completed
B0-B3 rules under the same folds, costs and execution envelope?

Do not redesign the question around the negative pretraining branch. Reuse the
predeclared 2022-2024 development origins only if the old design and current
contracts can still be reconciled without result-informed parameter changes.
No claim may treat these reused development years as untouched confirmation.

Return exactly one:

- `B4_RANDOM_ONLY_DESIGN_RECOVERABLE`;
- `B4_RANDOM_ONLY_REQUIRES_SUPERSEDING_DESIGN`, naming every unavoidable
  semantic change; or
- `B4_RANDOM_ONLY_OBSOLETE`, with a concrete scientific reason.

## 11. B4-R2: bounded CPU mechanics package

If recoverable, materialize but do not launch the complete 3-origin x 4-seed B4
ledger and execute one tiny CPU mechanics cell through the current code path.
The mechanics cell must prove:

- random initialization and zero hidden pretrained keys;
- exact observation and data identities;
- causal role boundaries and sealed-2025 absence;
- same cost/envelope identity as its B0-B3 comparator;
- finite environment steps, actor/critic gradients and save/load;
- the executing budget guard can stop inside a learning segment;
- no artifact from the mechanics cell is promotable or economically scored.

If a superseding design is required, prepare the design and refusal tests only;
do not execute even the mechanics cell until it receives independent review.

## 12. B4-R3: return the next dispatch decision

Return one final operational disposition:

- `B4_BOUNDED_GPU_PREFLIGHT_READY_FOR_MUSASHI_REVIEW`, with one cell, one seed,
  explicit step/update/wall/RSS limits and an exact command, but do not launch;
- `B4_CORRECTION_REQUIRED`; or
- `B4_BLOCKED`, naming the smallest missing fact or owner decision.

Do not authorize feature selection, Screen A/R/C or a broad campaign from this
packet.

## 13. Owner-dependent facts

Keep these separate from Satoshi's implementation work:

1. owner ratification of the prospective 83-column observation contract;
2. owner ratification of MT5 terminal build 6140 for the collector kit;
3. operator Ed25519 key ceremony in restricted storage;
4. real backup/review/rollback kit and a fresh 0/0 MT5 window.

The first item gates B4. Items 2-4 gate only the read-only MT5 session collector
and do not block C17-C22 or B4 branch inspection.

## 14. Return packet

Return one packet with:

1. all PRE/POST counterexamples;
2. v1/v2/v3 design and result identities and truthful chronology;
3. complete target/anchor/loss evidence and strict adjudication output;
4. corrected 14-slot table and scoped verdict;
5. corrected N5 transition ledger;
6. B4 compatibility matrix and R1-R3 disposition;
7. focused and final-tip test counts from the final pushed tip;
8. explicit GPU, network, venue, service, key and live effects, all expected
   zero.

Run focused tests and the ordinary suite. Name known failures exactly and prove
them against the pristine parent. Do not touch running processes, GPU queues,
services, venues, positions, collectors, keys or checkpoints.
