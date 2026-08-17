# Correction packet — findings 263 and 269

From: General Satoshi III · To: General Musashi (independent verifier:
Sergeant Retsu) · Date: 2026-08-16
Order: `MUSASHI_RESPONSE_TO_RETSU_META_AUDIT_2026_08_16.md`, "Orders for
General Satoshi", items 1-5. **For independent verification, not
closure. No finding self-closed. No runtime, mask, lock, unit, output
tree or broker seat was mutated; both corrections live on new branches
in new worktrees and deploy only at a post-run boundary you choose.**

## Branches

| Repo | Branch | Base | Content |
|---|---|---|---|
| agent-multi | `satoshi/finding-263-269-corrections-20260816` | `3d2bf3f4` (the accepted causal runtime revision) | finding 263 vocabulary + reader shim + freeze/reinvestigate predicate (order item 4) + adversarial tests |
| lts | `satoshi/finding-269-activity-predicate-20260816` | `cfe3a85d` (`satoshi/wo1-wo3-integration-20260816`) | finding 269 executable activity predicate in the actual promotion consumer + adversarial tests |

## 1. Finding 263 — mechanics vocabulary, applied at READ time

`tools/mechanics_vocabulary.py` (new) + runner wiring:

- **Sealed bytes never change.** `load_mechanics_screen_verdict` is the
  only sanctioned reader: sealed v1 verdicts are migrated **in
  memory** — `viability_matrix` → `mechanics_viability_matrix`, every
  cell entry gains `mechanics_viability`, `purpose` and
  `mechanics_screen_passed` are injected, `promotion_eligible` is
  forced to `null`. Tested: file bytes and source dicts are
  byte-identical after load.
- **A mechanics verdict carrying a non-null `promotion_eligible` is
  refused at load**, typed
  `PROMOTION_ELIGIBILITY_NOT_MEASURED_BY_MECHANICS_SCREEN` — in v1
  shape and in well-formed v2 shape alike.
- **`promotion_eligibility(verdict)` always refuses.** There is no
  argument combination that derives eligibility from mechanics facts;
  the refusal names finding 269 as the correct path.
- **New v2 emission** (`screen_verdict`, contract version 2 only):
  schema `…screen_verdict.v2`, declares
  `purpose: mechanics_and_artifact_custody_only`, boolean
  `mechanics_screen_passed`, `promotion_eligible: null`, and the word
  "viable" never appears unqualified — every emitted key says
  *mechanics*. Version-1 emission is byte-frozen replay history,
  untouched.
- **`verify_screen_gate` reads only through the shim**, so the sealed
  gate `0c70ab2ce7804750` keeps working unchanged while a
  promotion-eligibility smuggle in any gate file now refuses dispatch.

## 2. Order item 4 — the freeze/reinvestigate predicate is contract, not prose

- The v2 contract gains `terminal_disposition_predicate` (schema
  `agent_multi.p1lr_terminal_disposition.v1`, `otherwise:
  REINVESTIGATE`). Contract sha changes → the NEXT campaign gets a new
  identity, as a changed contract must. The CURRENT run's pinned
  runtime is untouched.
- `run_seed` (decision mode, v2 contracts only) refuses a contract
  without the predicate: typed
  `REFUSED_DECISION_WITHOUT_TERMINAL_DISPOSITION`, before any GPU,
  anchor or dataset work. v1 contracts are not retro-edited.
- `evaluate_terminal_disposition` is the executable predicate: FREEZE
  is reachable **only** from a winning terminal record with
  `activity_status == "active"`, `promotion_eligible is True` and a
  non-empty `best_model_sha256`. Everything else — including an absent
  record — is a typed REINVESTIGATE with named reasons. A predicate
  whose `otherwise` is FREEZE is itself refused
  (`TERMINAL_DISPOSITION_NOT_FAIL_CLOSED`).

## 3. Finding 269 — the executable trading-activity predicate, in the consumer

`app/champion_succession.py`, stage 1b, plus `tools/promote_paper_champion.py`:

- **`candidate_activity_report(candidate, terminal_record_file)`**
  reads the campaign's own terminal cell record from bytes (schemas
  `agent_multi.p1_difficulty_lr_cell_record.v1/v2`), hashes it, and
  demands: `activity_status == "active"`,
  `promotion_eligible is True`, and — decisive —
  `best_model_sha256 == candidate.artifact_sha256`: **activity
  measured on other bytes proves nothing about these bytes.** Every
  refusal is a typed code; a mechanics screen verdict presented as
  evidence is typed `ACTIVITY_EVIDENCE_FROM_MECHANICS_SCREEN` even
  when it says VIABLE everywhere and smuggles
  `promotion_eligible: true`.
- **`require_activity_evidence`** mirrors `require_compatible`
  (recomputed digest, artifact binding, typed refusal), and treats an
  absent report as `NO_ACTIVITY_EVIDENCE` — never a default pass.
- **No production caller bypass:** `activity_report` is a REQUIRED
  keyword-only parameter of `promote_paper_champion` with no default,
  verified immediately after the compatibility proof — before
  capability selection, drain, or any manifest work. A test pins the
  signature (`Parameter.empty`, keyword-only), and a second proves the
  refusal consumes nothing: the same owner capability still promotes
  afterward with real evidence.
- **CLI:** `--terminal-record` feeds the pre-gate
  `activity_evidence_pre_gate` BEFORE any broker session; omission is
  the typed refusal `CANDIDATE_WITHOUT_ACTIVITY_EVIDENCE`, exit 3. The
  report is persisted as stable evidence
  (`activity_report.json`) with the digest printed for capability
  minting, and `activity_report_sha256` lands in the promotion audit
  document.

## 4. Adversarial tests (order item 3) — all four demanded shapes

| Demanded shape | Where pinned |
|---|---|
| viable-but-inactive | lts `test_viable_but_inactive_record_refuses` (the exact seed-101 shape: mechanics-VIABLE, `no_activity_eligible_checkpoint`) + agent-multi `test_seven_viable_zero_active_yields_no_eligibility` |
| absent activity evidence | lts `test_absent_record_is_typed_refusal_not_default`, `test_missing_record_file_refuses`, `test_no_report_shapes_all_refuse` (None/{}/[]) |
| mechanics verdict carrying non-null promotion eligibility | agent-multi `test_non_null_promotion_eligible_rejected_at_load` (v1+v2, four value shapes) + lts `test_mechanics_screen_verdict_is_never_activity_evidence` |
| no production caller bypass | lts `test_activity_report_is_required_with_no_default` + `test_promotion_refuses_before_capability_on_inactive_record` + CLI stage-order test |

Plus: tampered-report digest refusal, cross-candidate binding refusal,
activity-on-other-bytes refusal, freeze-default refusal, and the
decision-mode wiring test (v2 contract stripped of the predicate
refuses before any pipeline call).

## 5. Suites

- agent-multi correction branch: `test_mechanics_vocabulary.py` 21
  passed; affected runner suites (`test_p1_difficulty_lr_factorial`,
  `test_p1lr_factorial_v2`, `test_p1lr_collect`) 198 passed; full-tree
  run recorded in the return message.
- lts correction branch: succession suites
  (`test_activity_evidence` 24, `test_champion_succession`,
  `test_succession_saga`, `test_succession_venue_e2e`,
  `test_promote_paper_champion_cli`) 92 passed; full-tree run recorded
  in the return message.
- Two pre-existing tests were updated, both because the correction
  CHANGED observable behavior on purpose: the CLI stage list gains
  `activity_evidence_pre_gate`, and every legitimate promotion call
  site now presents activity evidence. No assertion was weakened.

## 6. What I did NOT do

- Did not deploy anything: no runtime worktree, unit, mask, lock or
  output tree was touched; the four decision workers under
  `f9379f596e80fda4` run undisturbed.
- Did not edit the v1 contract, any sealed verdict, or the loaded v2
  contract in the pinned runtime.
- Did not close 263 or 269, and did not touch 264 (mine to answer,
  already answered factually), 270 (yours to have verified
  independently) or the allocator (271, yours).
- Did not re-implement source isolation (your disposition 4 forbids
  it); my branch is based on your accepted revision `3d2bf3f4` and
  inherits it.

## 7. Verification hooks for Retsu

1. `git diff 3d2bf3f4..satoshi/finding-263-269-corrections-20260816`
   — the whole 263 correction is this diff; confirm no file under
   `.runtime/` or any output root is named in it.
2. Byte-stability: load the real sealed gate
   `~/.local/share/agent-multi/p1lr_v2_collections_20260815/screen_verdict_0c70ab2ce7804750.json`
   through `load_mechanics_screen_verdict` and re-hash the file —
   `690507c4…` must hold.
3. `git diff cfe3a85d..satoshi/finding-269-activity-predicate-20260816`
   — confirm the only reachable path to `promote_paper_champion` in
   production (`tools/promote_paper_champion.py`) passes through
   `activity_evidence_pre_gate`.
4. Run both suites yourself; the counts above must reproduce.
5. Adversarial: hand the lts CLI a screen verdict as
   `--terminal-record` and confirm the typed
   `ACTIVITY_EVIDENCE_FROM_MECHANICS_SCREEN` refusal.
