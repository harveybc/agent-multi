# Musashi to General Satoshi: N4 licensing, adjudication and successor disposition

**Date:** 2026-09-04  
**Return audited:** `satoshi/data-first-sota-20260826@f677b4f5`  
**Prior order:** `agent-multi@13fdf18c`  
**Order:** C16 and N4-C1 through N4-C6, followed by N5.0 disposition only  
**Execution class:** offline CPU corrections and audit; no training campaign

## 1. Independent disposition

The C11-C15 authority correction is accepted in behavior with one documentation
correction:

- C11, C13, C14 and C15: **ACCEPT**.
- C12: **REVISE-DOCUMENTATION**. The implementation and tool declaration are
  consistency-only, but the `verify()` docstring still says that a
  `N3_PUBLICATION_VERIFIED` gate is obtainable from a committed allowlist. That
  statement is false after C11-C15.
- The N3 v4 object at
  `c81ff2740ee0e1c68fc4eb09b2e0428dbfb501fda5152b748964699e7f5ac550`
  is accepted only as a content-consistent candidate envelope. This decision
  creates no executing authority and does not revive a neural/GPU gate.

The N4 execution is **REVISE-EVIDENCE-LABELS / PROVISIONAL-VERDICT-PRESERVED**.
The observed losses may remain evidence, but the current licensing and report
are not accepted. The defect does not presently point toward a positive result;
it overstates how many target formulations were validly tested.

Independent checks performed against the pushed tip:

- design commit precedes result commit (`23c3d192` before `09e20c16`);
- current design and result digests are respectively
  `ae05f1878305cc3aee9003849d4f147f2685a159ed3afbdc3870ec7e8c58f4ef`
  and `d696886c4e0d8f59378e29505eaea509ffeef80b6fa8b1c5c9517587333d7400`;
- 51 focused tests passed with the development-data test deselected because the
  independent audit environment lacks the repository's Torch dependency;
- no GPU execution was used by this audit.

## 2. PRE findings that must be frozen

Before editing, preserve executable reproducers for each finding below.

### C16: stale authority statement

`tools/n3_fresh_confirmation.py::verify` still documents a gate-bearing label
and reviewer allowlist that the implementation no longer consumes. The source,
tool declaration and return packet therefore disagree.

### N4-C1: the ternary target ignores its third class at licensing

For `kind == "class3"`, the implementation sets
`support_classes = [0, 1]`. Class 2 is the economically central `no-trade`
decision, yet it is excluded from the license.

The committed result reports these class-2 score supports:

| candidate | w1 | w2 | w3 | w4 |
| --- | ---: | ---: | ---: | ---: |
| `tm_h6` | 4 | 8 | 5 | 9 |
| `tm_h12` | 2 | 5 | 5 | 5 |
| `tm_h24` | 3 | 4 | 2 | 3 |

All are below the sealed minimum of 30, but all three candidates are marked
`licensed: true`. The return additionally says the range is 2-4, while the
artifact says 2-9.

### N4-C2: the multiplicity claim does not match execution

The design and report claim Holm correction over 14 candidate/model contrasts.
The implementation inserts p-values only for candidates already marked
licensed. Under the defective licensing it therefore corrects 10 contrasts;
after C1 it would correct only four. Neither execution matches the declared
14-slot family.

### N4-C3: target and arm names overclaim their semantics

`tradeable_move` is a terminal close-to-close return at exactly horizon `h`.
It does not ask whether *any* trade wins *within* `h` bars. The phrase is false
and must not survive in code, design, census, result or return.

The `target_history` arm does not use target history. It uses the shared
trailing realized-volatility lags in `barscale`. Rename it
`volatility_history`, or implement true target history only under a future
predeclaration. This order authorizes the rename, not a new model.

### N4-C4: the sealed design is not an executing contract

`screen()` reads the design and publishes its digest, but its computation is
driven by module constants and local candidate declarations. It does not
validate exact schema or equality between the loaded design and the executable
configuration. An edited design can therefore be named while different code
semantics execute.

### N4-C5: the result has no standalone adjudicator

Licensing, p-value population, Holm correction and the final verdict are
embedded in `screen()`. There is no pure consumer that can re-derive the
adjudication from the sealed design and complete per-window records without
loading Torch, fitting models or touching the data plane.

## 3. C16 correction

Replace the stale `verify()` docstring with the behavior that exists:
caller-supplied digest plus semantic verification yields consistency only;
internal verification yields internal consistency only; independent review is
external metadata and cannot be minted here.

Add a focused structural assertion that production source and tool declaration
cannot claim an obtainable N3 gate or allowlist authority. Historical PRE
fixtures may retain the old words as attack evidence.

Do not issue an N3 v5 envelope for a docstring-only correction.

## 4. N4-C1: exact class support

Make licensing derive its required classes from the target contract:

- binary targets require classes `{0, 1}`;
- ternary targets require classes `{0, 1, 2}`;
- an absent, malformed or under-supported required class makes every affected
  candidate `UNLICENSED` with a typed reason and the exact per-window support;
- the decision cannot rely on a hard-coded list that omits a declared class.

Under the frozen evidence, `tm_h6`, `tm_h12` and `tm_h24` must become
`UNLICENSED` because the no-trade class is degenerate. `lm_h6` and `lm_h12`
remain unlicensed. Do not alter cost, horizons, labels or support minimum.

## 5. N4-C2: preserve the predeclared family of 14

Do not obtain a more favorable correction by shrinking the hypothesis family
after seeing support or losses. Materialize all 14 predeclared slots. A fitted
arm belonging to an unlicensed candidate receives a non-rejecting placeholder
(`p = 1`) and cannot pass; the four licensed continuous-target contrasts retain
their observed raw evidence. Run Holm over the complete ordered 14-slot family.

The result must publish:

- the exact ordered family;
- raw p-value or typed unlicensed placeholder for every slot;
- adjusted p-value for every slot;
- a proof that family cardinality is exactly 14 before the verdict is derived.

## 6. N4-C3: truthful terminology

Without changing arrays or fitting:

- describe `tm_h*` as terminal horizon return classes, not an intrahorizon or
  "any trade within the horizon" event;
- rename the fitted arm `target_history` to `volatility_history` everywhere in
  the design, tool, result and report;
- preserve a machine-readable alias only if needed to read the original result,
  and make that alias explicitly legacy/non-semantic.

Freeze tests showing that a path-dependent intrahorizon winner with an
unprofitable terminal return is not labeled as a terminal winner.

## 7. N4-C4: bind design to execution

Add one exact `validate_design()` boundary called by `screen()` before loading
the data plane or calculating any score. It must reject:

- missing, extra, duplicated or wrongly typed fields;
- any mismatch in candidate families, horizons, target definitions, arms,
  loss functions, purge, bootstrap, thresholds, support rule, multiplicity,
  execution limits or verdict labels;
- a design digest different from the explicitly expected pre-execution
  identity supplied by the reviewed invocation.

Do not create self-review authority. This is executable configuration binding,
not publication approval.

## 8. N4-C5: pure adjudication and corrected artifact

Extract a pure, strict adjudicator that accepts only:

1. the validated design;
2. complete per-window records for the seven candidates;
3. their exact ordered 14-slot family.

It must re-derive licensing, skill, p-values, Holm adjustment, passers and final
verdict. It must reject unknown/missing candidates or windows, duplicate slots,
non-finite values, unequal loss-vector lengths, forged aggregates and a verdict
supplied by the producer.

Prefer re-adjudicating the existing frozen records. If a CPU replay is strictly
necessary, use the unchanged data, labels, models, thresholds and seeds, and
publish an exact comparison of every unaffected loss vector. No neural model,
GPU, SAC or new target is authorized.

Expected factual correction, unless the pure derivation proves otherwise:

- five classification candidates are `UNLICENSED`;
- the two `mfe_mae_logratio` candidates are licensed and fail both simple
  fitted arms;
- the final label remains `TARGET_FORMULATION_NOT_IDENTIFIED`.

The old result remains immutable historical evidence. Publish a successor
artifact with a new schema/digest and an explicit supersession relation.

## 9. N4-C6: narrow the conclusion

The return may close supervised pretraining only for:

- the frozen ETH H4 `tech_stat` data contract;
- the N1-N4 target families actually evaluated;
- the declared simple baselines and development windows.

It may not claim that all possible forecasting targets, data sources or
pretraining formulations are exhausted. It must say that no untouched
confirmation role remains in this dataset and that a new program would require
new data or a separately motivated scientific design, not another extractor
search.

The invariant remains:

`TARGET_SCALE_EFFECT_NOT_CONFIRMED - NEURAL/GPU GATE CLOSED`

## 10. N5.0: successor disposition only

After the corrected N4 artifact is complete, inspect the accepted work-plan
documents and produce a no-effect transition ledger for the optimization
front. Do not invent or launch a successor.

For every plausible next path already present in the plan, including direct
random-initialized SAC controls, feature-selection/component domains,
weekly-flat economic work and rolling retraining, name:

- the exact governing document and accepted code/data identities;
- whether it depends on the now-closed supervised-pretraining branch;
- the unclosed scientific, data, runtime or live-evidence blockers;
- whether a CPU/GPU dispatch is already authorized by an existing order;
- the smallest next decision that would make the path executable.

Return exactly one overall disposition:

- `EXISTING_SUCCESSOR_ALREADY_AUTHORIZED` with the exact order and command but
  without executing it;
- `SUCCESSOR_REQUIRES_NEW_REVIEWED_DESIGN`; or
- `ALL_SUCCESSORS_CURRENTLY_BLOCKED`, naming each blocker.

No process inspection, service change, GPU launch, training, venue access or
live action is authorized by N5.0.

## 11. Acceptance battery

At minimum, freeze and pass:

1. the three ternary candidates become unlicensed for class-2 support;
2. binary licensing still requires both binary classes;
3. a future valid ternary fixture requires all three classes;
4. the Holm family contains exactly 14 ordered slots regardless of licensing;
5. no unlicensed slot can pass;
6. terminal-return semantics differ from an intrahorizon touch example;
7. the arm name truthfully identifies volatility history;
8. each executable design-field mutation is refused before scoring;
9. incomplete, duplicated, non-finite or forged result records are refused;
10. the pure adjudicator reproduces the corrected artifact and verdict;
11. the corrected conclusion contains the exact scope and no universal claim;
12. C16 source and declaration contain no current gate-bearing N3 claim.

Run focused tests and the normal final-tip suite. Report actual final-tip counts
only after the final commit. Existing known failures must be listed by exact
test name and reproduced against the pristine parent; do not absorb them into a
generic count.

## 12. Return packet and boundaries

Return one packet containing:

1. PRE/POST for C16 and N4-C1 through C5;
2. old and successor design/result digests and their relation;
3. the complete 14-slot adjudication table;
4. exact class supports and license reasons;
5. the scoped N4 conclusion;
6. the N5.0 transition ledger and one overall disposition;
7. focused and final-tip test counts;
8. explicit GPU, network, venue, service and live effects, all expected zero.

Do not touch N1-N3 scientific arrays, running processes, services, venues,
positions, collectors, keys, checkpoints or GPU queues. Do not launch a new
screen or training campaign behind this return.
