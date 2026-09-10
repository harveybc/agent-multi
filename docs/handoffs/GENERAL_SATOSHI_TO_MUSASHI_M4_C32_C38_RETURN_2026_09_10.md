# General Satoshi to Musashi — M4 C32-C38 return: the CONFIRMATION protocol, frozen and closed (2026-09-10)

Order: agent-multi@889320ee P2 (M4 C32-C38), copied into the branch at the PRE
commit d8b45eb4. Branch `satoshi/model-capacity-m3-20260908`; built on the
reviewed tip 5e7a8fd4 (P0 and P1 were committed and pushed first, as the order
gates: B4 return 1172fac9 on am-data-first, T2 return cec58641 on am-t0t1).

Disposition: `M4_CONFIRMATION_PROTOCOL_READY_FOR_EXTERNAL_MUSASHI_REVIEW`

**No CONFIRMATION array, score or ledger exists anywhere.** The two-record
gate is the only door, both records are absent, and the templates grant
nothing.

## 1. PRE (committed d8b45eb4, before any edit)

At the reviewed tip, read-only: the four order-pinned identities verify by
recomputation (design self d7280a92…, numeric amendment self 43e0804e…,
governing adjudication self b35b6fd9…, HEAD == 5e7a8fd4); every order fact
re-derives from the adjudication STRUCTURES — 21/28 eligible + 7 typed
ineligible slots, exactly two incomplete generators
(state_space::clean::w64 g0/g6, six seed-units kept in the denominator), ZERO
calibration-incomplete cells at floor 13, M2 gain −0.41982887 exact, complete
generators 14–16/16 per cell; and the commanded protocol was wholly ABSENT
(no successor, runner, verifier, templates, battery; 1,008 CONFIRMATION
generators existed only as reservations). Zero writes (docs/ 2,120-file byte
inventory equal).

## 2. C32 — binding the accepted calibration evidence

`tools/m4_confirmation_protocol.py::bind_calibration_evidence()` consumes the
three documents through O_NOFOLLOW descriptors, recomputes each self-identity
against the order-pinned constants, verifies the amendment amends the pinned
design and the adjudication binds it, and re-derives the four order facts from
structures. One correction of my own reflex during construction: I first
expected an `authority` field value from memory; the real record declares
`CANDIDATE_FOR_MUSASHI_REVIEW_NO_CONFIRMATION_AUTHORITY` — which is the
honest value (the record claims no authority; what makes it governing is YOUR
order pinning its exact self-identity). The check now demands exactly that
declaration.

## 3. C33 — the confirmation successor (the "diff" from sealed v5)

`docs/research/model_capacity/M4_CONFIRMATION_SUCCESSOR_2026_09_10.json`
(self 6a50d97d…), append-only over the sealed v5 (supersedes by digest, v5
byte-untouched). What it adds/decides relative to v5 — each item CALIBRATION-
DERIVED, none predeclared:

| decision | value |
|---|---|
| selection rule label | `CALIBRATION_DERIVED_AND_REVIEWED` |
| classification | `SCIENTIFIC_ANALYSIS_FREEZE` (explicitly NOT `scientific_change: NONE`) |
| eligibility | ≥12/16 LEARNABLE_UNDER_FROZEN_BUDGET and zero NUMERICALLY_INVALID |
| population | the exact 21 eligible + 7 typed-ineligible slots, copied by identity from the governing adjudication |
| confirmation size | 48 generators per eligible slot × 3 nested seeds |
| attrition | allowance 0.20, floor max(3, ceil(48×0.8)) = **39**; below → `CONFIRMATION_INCOMPLETE`, never favorable |
| M2 | `DOES_NOT_ADVANCE_FROM_CALIBRATION` (gain −0.41982887); never fitted or scored on CONFIRMATION |
| multiplicity | **Holm over all 16 slots** — the v5 line "Bonferroni over the frozen confirmatory contrast family" is DECLARED superseded per your C34.7; both control FWER; frozen before any CONFIRMATION outcome. **This is the one point where the successor overrides a sealed-design sentence — named here for your review, not slipped through.** |

Regenerating a published successor refuses structurally (git-tracked check —
the a9/a15 lesson); `verify_confirmation_successor()` re-derives the ENTIRE
body live from the bound calibration evidence, so any scientific mutation
(threshold, slot, label, attrition, M2) dies against the re-derivation, not
against a stored copy.

## 4. C34 — the 16-contrast family, executable

`sixteen_contrasts()` is pure and total over the frozen family:

- 14 family/noise contrasts: generator-level paired effect (mean over the
  exactly-3 nested seeds), averaged EQUALLY across frozen-eligible widths; a
  generator missing a frozen width never averages.
- Real topology under the frozen population: 10 contrasts with both widths;
  **state_space::clean runs single-width (w16) and names that fact**;
  **parity4::clean, discontinuity::clean, discontinuity::white are
  NOT_EVALUABLE with non-rejecting p=1** (no eligible width).
- 15th: checkpoint_effect::primary_pair (paired within generator).
- 16th: incremental_prediction::M2_vs_M1 — non-rejecting p=1 placeholder.
- Per-contrast test frozen: two-sided one-sample t on generator-level
  effects, df = n−1; <2 complete generators → NOT_EVALUABLE p=1.
- Holm runs over ALL 16 including placeholders; a 15-slot analysis refuses.
- Width-specific effects and attrition are published only as
  `SECONDARY_HETEROGENEITY`; with asymmetric width effects the primary is
  provably the equal average (battery asserts ±0.0), never the better width.

## 5. C35 — runner and independent verifier

`tools/m4_confirmation_runner.py`:

- **Structural role isolation:** confirmation unit ids carry
  `::CONFIRMATION::` and generator identity derives from the role string
  (disjoint seed streams by construction); records with any other role are
  refused at the analysis boundary; at runtime every CONFIRMATION array
  digest must be absent from the prior-role digest census
  (`verify_role_disjointness`) or the run refuses.
- **Gate chain before ANY execution:** C32 bind → C33 verify →
  `require_both_records` (Musashi review + owner execution, CHAINED by
  digest — absent/forged/transplanted/placeholder records refuse) → clean
  checkout identity → exact census (3,024 units = 21×48×3; 4 checkpoint
  kinds; frozen update bounds 4,000 cadence 50 + 500 post) → complete O_EXCL
  pre-result ledger → sealed v5 CPU wall/RSS/nice/heartbeat/stop-file
  machinery. `execute` refuses at the two-record gate BEFORE creating any
  directory, array or ledger — verified live and by battery.
- `plan` (no records): reports 3,024 units / 21 slots / bounds /
  census digest, `execution_open: false`.
- **Verifier:** reconstructs paired effects from the ARM RECORDS
  (declared `paired_primary_difference` must re-derive from
  calibration_stop − initialization or the run refuses), attrition per slot
  against floor 39 (`CONFIRMATION_INCOMPLETE` never favorable — the width
  drops and the contrast degrades to NOT_EVALUABLE, never to a favorable
  subset), costs from `updates_done` per arm, and all 16 contrasts. Producer
  aggregates never determine a verdict.

## 6. C36 — external authority boundary

Two committed, still-uninstalled, non-authorizing templates:
`MUSASHI_M4_CONFIRMATION_DESIGN_REVIEW_TEMPLATE_2026_09_10.json` and
`OWNER_M4_CONFIRMATION_EXECUTION_TEMPLATE_2026_09_10.json` (extra key +
`<placeholders>`, both refused by the consuming APIs; owner record must chain
to the review record by digest). Candidate code never creates or installs a
real record; install paths are under `<state_root>/m4_confirmation_authority/`.

## 7. C37 — battery and mechanics evidence

`tests/test_m4_confirmation_protocol.py`: **28 passed** — the fifteen ordered
kills (substituted attempt-2; changed 12/16; changed slot; "predeclared"
label; width-as-extra-primary; better-width selection; Holm short of 16; M2
rehabilitation + always-placeholder; missing nested seed; attrition beyond
allowance; CAL/CONF byte overlap; forged producer aggregate; absent/forged/
transplanted/broken-chain/placeholder records; scientific change after
review; numerical failure leaving the denominator) plus guard-removal
mutants (16-check, re-derive, floor, disjointness — each admits exactly its
adversary when removed) and the closed-execution/plan tests.

DEVELOPMENT mechanics probe (real process machinery, CPU): 2 real DEVELOPMENT
units through the accepted v5 arm/limit path, terminal records written, and
**zero CONFIRMATION-named artifacts** in the output root and zero in the
state root.

## 8. POST (committed with the cycle)

`docs/audits/evidence/repro_runs/m4_c32_c38_post_2026_09_10.{py,out}`, sealed
run exit=0: Phase 1 re-asserts every live fact (bind, successor, plan 3,024,
execute refusal before artifacts, probe, zero artifacts). Phase 2 capital
mutants: **gate-off** → execute reaches census + pre-result ledger with NO
records installed (the two-record gate alone closes execution; the
dirty-checkout guard was stubbed in that child to isolate the gate, and that
stub is declared in the code); **diverge-off** → a successor with threshold
11 VERIFIES (the live re-derivation alone rejects scientific mutation).
Successor bytes restored byte-exact and re-verified after the mutants.

## 9. Suite at the final tip

Full suite at the cycle tip 89a3781f: **3256 passed / 2 failed / 5 skipped**
(31:00). The only failures are the inherited D1 pair
(`test_eth_sac_inner_curriculum_contract` — the private D1 evidence file is
absent from this host), reported unchanged in every prior packet and untouched
here. The 28-test C37 battery and the M3/M4 batteries are green, and the
engineering surface index accepts the two newly declared tools.

## 10. Confessions (mine, unprompted)

1. My first C32 check invented an `authority` field value from memory; the
   real bytes corrected me (§2). Verified-from-terminal before sealing.
2. The first battery run had the guard-removal mutants importing the REAL
   modules (my second `sys.path.insert(0, …)` put the real tools ahead of the
   mutant dir) — the mutants "refused" with live guards, which would have
   been a vacuous battery. The identity asserts I then added caught the real
   cause and the drivers now load mutants by explicit file path; a vacuous
   mutant can no longer pass.
3. The POST gate-off mutant initially collided with the legitimate
   dirty-checkout guard (this worktree carries the new files); the child now
   stubs only the status call, and the stub is named in the committed code.
4. Two mechanical probe fixes during construction (acct shape, missing
   `intervention/` subdir) — refusals of the real machinery met, not
   weakened.

## 11. What Musashi rules on

1. The confirmation successor (§3) — especially the **Holm-supersedes-
   Bonferroni declaration**, the 12/16 rule, floor 39, and the M2 exclusion.
2. The executable analysis (§4) including the frozen per-contrast t-test.
3. The runner gate chain and verifier reconstruction rules (§5).
4. Whether to fill and install the design-review record (template §6); the
   owner's execution record then chains to it. Only both together open
   execution.

Commits: PRE d8b45eb4 → cycle 89a3781f → packet (this file, committed on
top). No CONFIRMATION execution, no GPU, no DOIN integration, no financial
data, no live action, no production deployment; B4 and T2 fronts untouched by
this package; no pushed history rewritten.
