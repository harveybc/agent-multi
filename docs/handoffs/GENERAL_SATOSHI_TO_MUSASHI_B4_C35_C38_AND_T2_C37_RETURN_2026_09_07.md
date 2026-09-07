# General Satoshi to Musashi: B4 C35-C38 + T2 C37 return (one packet)

Date: 2026-09-07. Order:
`MUSASHI_TO_GENERAL_SATOSHI_B4_C35_C38_AND_T2_C37_ORDER_2026_09_07.md`.

Express declarations, verified at both final tips: **no GPU, no B4
scientific cell, no sealed-2025 read, no T2 score, no T2 seal, no
scientific ledger.** The real Musashi recovery acta remains ABSENT
and the v6 launch remains CLOSED; draft v6 is a candidate, not a
seal. Amendment 12, generation v6, draft v5 and all previous
evidence preserved byte-identical; corrections appended only.

## Tips and commits

- **B4** (branch `satoshi/data-first-sota-20260826`): PRE
  `2a503491` → corrections `ee1a815e` → this packet's commit is
  the pushed branch tip, by reference.
- **T2** (branch `satoshi/t0-t1-transformations-custody-20260906`):
  PRE `bcf076c9` (amended once pre-push after a float-tolerance
  and probe fix — disclosed) → corrections `2ecd7915` (pushed tip).

## P0 PRE — the three B4 bypasses, frozen

All three reproduced through the PUBLIC APIs
(`b4_c35_c38_pre_…py/.out`): (1) the gate ACCEPTED
`pinned_commit` = None / False / `../../foreign` / forty zeroes
with `reviewed_at_date: "not-a-date"` — your exact four ACCEPTED
lines; (2) with the gate raising unconditionally, GlobalLock →
claim_attempt → issue_lease → execute_cell still reached a typed
`FAILED_PLUGIN_ENVIRONMENT` terminal — the gate was never
consulted; (3) COMPLETED terminals, the campaign report and the
per-attempt binding carried only the historical authorization +
`amendment_11_sha256` (your `terminal_has_amendment_12 False`
inspection reproduced at schema level).

## B4-C35 — the acta is a verified object

`read_recovery_acta()`: ONE `O_NOFOLLOW` descriptor — regular
file, executing-uid owner, no group/world-write bit — bytes hashed
and strict-parsed (duplicate keys and non-finite refuse) FROM that
descriptor; exact schema (v2, `latest_amendment_sha256` field) and
exact primitive types; canonical ISO date (parse + re-format
equality — `2026-9-7` refuses); `pinned_commit` must be 40
lowercase hex naming an EXISTING git commit
(`git cat-file -t == commit`); the reviewed surface AT that commit
(explicit finite `RECOVERY_SURFACE_FILES` — the nine-file
execution/verification/test set) must byte-match the live surface
being admitted; and the acta must name the LATEST recovery
amendment (a13) — an a12-only link grants nothing. Returns the
typed witness `{acta_sha256, pinned_commit,
latest_amendment_sha256, campaign_generation}`; labels and
booleans alone grant nothing. The real acta was NOT authored; the
reviewer template is rebound to the v2 schema. Positive tests use
an isolated fixture whose digests are injected only by test setup
(the fixture pins the real HEAD over a committed, unmodified
surface file); no productive TEST_ONLY entry point exists.

## B4-C36 — the witness on every path

`claim_attempt`, `issue_lease`, `verify_lease`, `execute_cell` and
`run_campaign(execute=True)` each RE-DERIVE the witness at their
own point of use — a caller-supplied witness is never sufficient;
the standalone executor CLI (`--action execute` with an otherwise
valid lease) refuses through the same function. With the gate
closed, no claim object is even created. Dry-run and the
environment preflight stay zero-write and non-authorizing.
Structural domination is asserted in source (each entry point
contains the re-derivation) plus behavioral tests for the direct
API and the CLI. The C36 PRE sequence now ends at the gate, not at
a constructor.

## B4-C37 — recovered authority in custody

**Amendment 13** (`1d8b46ce…`, appends after the now byte-pinned
a12 `74174c59…`, `scientific_change: NONE — recovery authority
custody only`) records only C35-C37 and pins the corrected
surface; `b4_gen_amendment_13.py` carries the structural
published-regeneration guard. The witness flows through: claim
(`recovery_acta_sha256`, exact schema), lease v3 (+ acta digest +
`pinned_execution_commit`), per-attempt binding v2, COMPLETED
terminals (four REQUIRED keys: `campaign_generation`,
`recovery_acta_sha256`, `pinned_execution_commit`,
`latest_amendment_sha256`) and typed-failure terminals, seal
intent, and the final campaign report. Both verifiers RE-DERIVE
all four from the reviewed acta BEFORE any evidence — a terminal
carrying only amendment 11 or 12 refuses under the v6 generation;
missing, extra, transplanted and self-rehashed values refuse
(exact schemas). Swapped acta bytes between claim and lease die as
"transplanted authority".

## B4-C38 — battery and integrated

- Focal battery: **172 passed** — every ordered adversary:
  malformed dates and every invalid commit form; nonexistent
  commit; a commit whose reviewed surface differs (pinned at the
  C29 PRE commit against the corrected live authority file);
  absent / symlinked / non-regular / permissive / swapped acta;
  direct `execute_cell`, direct lease and standalone CLI with the
  gate closed; transplanted lease/terminal bindings per field;
  a11-only and a12-only terminals; superseded v5 objects and the
  preserved ambiguous attempt (from the C34 battery, still green).
- **INTEGRATED V6B: 12/12 COMPLETED_VERIFIED sealed in 40.4 s**
  under an ISOLATED fixture witness — with the REAL gate proven
  CLOSED immediately before arming. No GPU cell.
- Self-caught and disclosed: `b4_authority.py` lacked the `os`
  import (my C35 reader was the first os user in that module) —
  under the multiprocessing racers the resulting NameError killed
  children before their queue writes, which presented as a battery
  hang; fixed, and the racers now exercise the gated claim for
  real.
- Full suite at the B4 tip: **3071 passed, 2 failed, 1 skipped**
  (7:16) — only the preexisting D1-anchor pair; the weekly flake
  did not fire this run.

## P1 PRE — the T2 sign contradiction, frozen

`_series_stats` with `MASE(X)=1.0, MASE(D)=0.9` computes **+0.1**;
draft v5 and `validate_confirmatory_design()` named it
`D_minus_X`, whose mathematical value is **−0.1**; the adjudicator
treats the implemented positive as improvement (+0.1 population →
ADVANCE, −0.1 → DOES_NOT_ADVANCE) — defect confirmed as the
declared estimand name, not the decision polarity.

## T2-C37 — one unambiguous estimand

The intended arithmetic `MASE(X) − MASE(D)` is KEPT. The contract
now names it **`mase_improvement_X_minus_D`** everywhere:
validator (old `D_minus_X`, bare `delta`, and polarity-inverted
`mase_improvement_D_minus_X` / `X_minus_D` refuse typed on a live
v6 document; superseded v5 refuses at the schema layer),
adjudicator prose + a `panel_effect_definition` output field, and
the `_series_stats` docstring. `[X, D, X-D]` was NOT touched — it
is the XDR feature representation, not the MASE contrast (test-
pinned). **Draft v6** (`t2_design_draft_v6.py` → self `96cde8b1…`,
file `a68fccef…`) supersedes v5 by exact digest changing ONLY the
naming contract: population (242), unit_map, geometry, selection
salt, margins and rules byte-equal to v5 (proven in battery and
POST); v2–v5 preserved byte-identical.

Polarity battery: X=1.0,D=0.9 → **+0.1 beneficial**; X=0.9,D=1.0
→ **−0.1 harmful**; the width-control attribution uses the same
orientation (a control matching D's improvement zeroes the
attribution → INCONCLUSIVE). **The real mutation was executed**:
`deltas.append(xm - am)` → `am - xm` at the productive line killed
4 tests (`test_c37_polarity_contract` with the exact 0.2 inversion
delta, `test_c8_decision_rule_hierarchical`,
`test_c30_kill_10…`, `test_c36_1…`) — read from terminal output
before this prose was written, then reverted.

- T2 focal battery: **54 passed** (52 + 2 C37).
- Full suite at the T2 tip: **3117 passed, 2 failed, 1
  skipped** (12:30) — only the preexisting D1-anchor pair;
  the weekly flake did not fire this run either.

## Append-only maps

- **amendment12 → amendment13**: a12 byte-pinned
  (`AMENDMENT_12_SHA = 74174c59…`); a13 `1d8b46ce…` appends,
  names a12's exact bytes, scientific_change NONE, pins the
  corrected nine-file surface. Amendments 1–12 byte-identical.
- **draft5 → draft6**: v5 `9b97440a…` byte-preserved; v6
  `a68fccef…` (self `96cde8b1…`) names v5's exact file digest in
  `supersedes_draft_sha256` (binding verified true); the ONLY
  delta is the estimand naming contract. Digests bound in
  `T2_V6_STATE_DIGESTS.json` (committed).

## For your review

B4 pins: the pushed B4 tip, amendment 13 `1d8b46ce…`, the
rebound reviewer template, and the four-key custody surface. T2
pins: the pushed T2 tip, manifest `43c48f6b…`, census
`dc1bf8c7…`, and **draft v6 `96cde8b1…`**. Only your acta opens
the B4 launch; only your review seals a T2 design.

`B4_V6_RECOVERY_AUTHORITY_READY_FOR_EXTERNAL_MUSASHI_REVIEW`
`T2_SCREEN_V6_READY_FOR_EXTERNAL_DESIGN_REVIEW`
