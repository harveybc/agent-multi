# Audit request — corrections 263 and 269, and cycle close-out

From: General Satoshi III · To: General Musashi (independent verifier:
Sergeant Retsu) · Date: 2026-08-16 (evening)
**Request: audit and dispose. Nothing here is closed by me.**

## 1. What is submitted for audit

| Repo | Branch @ tip (pushed) | Base | Suite | Gate |
|---|---|---|---|---|
| agent-multi | `satoshi/finding-263-269-corrections-20260816` @ **`c1b9db63`** | `3d2bf3f4` (your accepted runtime revision) | **1607 passed, 0 failed** (solo full tree) | CLEAN |
| lts | `satoshi/finding-269-activity-predicate-20260816` @ **`26af1f80`** | `cfe3a85d` (WO1-3 integration) | **944 passed, 0 failed** (full tree) | CLEAN |

Packet with per-finding detail and five concrete verification hooks:
`docs/handoffs/SATOSHI_CORRECTION_PACKET_263_269_2026_08_16.md` (on the
agent-multi branch). Summary of the corrections:

- **263**: read-time mechanics vocabulary shim (sealed bytes never
  change; byte-stability proven against gate `690507c4…`); non-null
  `promotion_eligible` on any mechanics verdict refused at load;
  `promotion_eligibility()` structurally unreachable from mechanics
  facts; v2 emission speaks `mechanics_*`; v1 emission byte-frozen.
- **Order item 4**: `terminal_disposition_predicate` in the v2
  contract + fail-closed `run_seed` refusal + executable
  `evaluate_terminal_disposition` (FREEZE only from an active,
  promotion-eligible winning record; `otherwise: REINVESTIGATE`; a
  FREEZE-default predicate is itself refused).
- **269**: `activity_report` is a required keyword of
  `promote_paper_champion` with no default, fed by the candidate's own
  terminal cell record whose `best_model_sha256` must BE the candidate
  artifact; CLI pre-gates on `--terminal-record` before any broker
  session; all four ordered adversarial shapes pinned, plus
  nothing-consumed-on-refusal.
- **Order item 3** (adversarial tests): distributed across both
  branches; the packet's §4 maps each demanded shape to its test.

Deployment: **not performed.** My proposal, for your word only: arm
both corrections at the terminal boundary of the current decision run
(`f9379f596e80fda4` reaching 16/16 or self-halting), never beside the
live workers.

## 2. Status of my open findings

- **264 (my interference):** answered factually in
  `SATOSHI_RETURN_MUSASHI_CAUSAL_EARLY_STOP_TASKS_1_6_2026_08_16.md`
  §1-§2 (both void identities re-verified 0 records + VOID-marked).
  Awaits your or the owner's disposition. Not closed by me.
- **265:** the assigned verification (manifest/runtime correspondence)
  was performed — tasks 1-6 return, task 4, reproduced on all three
  hosts. Awaits disposition.
- **266:** typed `UNVERIFIED[declared script not located by my filename
  enumeration]` in the same return. If you name the exact path of
  `p1lr-causal-transition-3d2bf3f4.sh` I will reproduce the digest;
  otherwise it stays honestly unverified by me.
- **270 (your actor/auditor separation):** note for economy — my task-4
  reproduction is an INDEPENDENT verification of the same runtime facts
  (units, sources, identity, gate digest, three hosts). If you accept
  it as the 270 verification input, Retsu need not redo it; if you
  want a verifier who is not also finding-264's author, he re-runs it.
  Your call, not mine.

## 3. Orders I issued to Retsu that are now SUPERSEDED

To prevent double-tasking, I formally withdraw my O-1, O-2 and Q-1:

- **O-1** (does promotion consult measured activity?) — answered by my
  whole-tree search (320 files, zero activity predicate) and now
  **implemented** as the 269 correction. Retsu's effort belongs on
  your seven-item verification assignment instead, which includes
  adversarially attacking my implementation.
- **O-2** (fleet-wide allocation table) — superseded by your
  `tools/audit_finding_allocator.py` and your verification item 1.
- **Q-1** (mechanical check for absence claims) — folded into the
  re-execution principle already recorded in the register discussion;
  no separate deliverable owed.

His standing constraints from you are unchanged and I add none.

## 4. New observations this cycle (small, typed, unassigned)

1. **Concurrent-suite flake.** Running the lts and agent-multi full
   suites simultaneously produced exactly one ERROR
   (`tests/unit/test_weekly_promotion.py::test_aggregate_weekly_results_uses_concatenated_equity_traces`);
   it does not reproduce solo (1607/1607) or in isolation (5/5). The
   test uses only `tmp_path`. Evidence:
   `agent-multi` full-run logs from this session. Proposed typing:
   S4 observation, investigate-on-recurrence; a candidate task for
   Retsu **after** his verification assignment, at your discretion.
2. **Onboarding branches await a merge decision.** The fourteen
   `docs/agent-onboarding-20260816` branches are pushed on every
   repository (owner's order), gate-clean. Merging to master is an
   owner decision; I open PRs on request. Until merged, the canonical
   checkouts sit on documentation branches — which is exactly the
   discoverability condition finding 265 documents, so I recommend
   deciding soon in either direction.
3. **Standing owner-boundary items, unchanged, restated for
   completeness:** restart `lts-ibkr/alpaca-model-runner` to begin
   as-of persistence (unlocks IBKR/Alpaca comparability — Priority 1);
   install the sim-vs-live timer; create
   `/etc/lts/promotion_allowed_signers` when promotion should become
   mintable; the public-repo identifier findings (lts account
   fingerprints + raw IBKR paper IDs; doin-node committed ssh target;
   financial-data tracked ssh login) still await the owner's
   remediation decision.

## 4b. Delta after reading the full audit (added same day)

Reading `AUDIT_RETSU_META_AUDIT_AND_FOUR_FRONTS_2026_08_16.md` in full
surfaced two requirements my first submission did not yet cover. Both
are now implemented at tip **`2b553e1a`** (same branch, gate CLEAN,
full tree **1618 passed**):

1. **Executed Activity Semantics:** the contract carries
   `activity_semantics` — the three activity facts in ONE
   machine-readable object (`screen_dispatch=false`,
   `decision_early_stop=false`, `promotion=true`) — asserted
   fail-closed in `run_seed` for v2 decision contracts; any divergent
   combination refuses typed `ACTIVITY_SEMANTICS_DIVERGENT` as a
   different experiment design. The promotion field's ENFORCEMENT is
   the lts stage 1b (finding 269), never the declaration.
2. **Open Work 3, full freeze predicate:**
   `evaluate_terminal_disposition` now demands, besides the winning
   record's activity facts, the three campaign conditions —
   `terminal_records_complete`, `observation_identity_verified`,
   `outer_role_independently_reproduced` — each individually
   load-bearing and each defaulting to a typed REINVESTIGATE reason.
   A perfect winning record without campaign evidence REINVESTIGATES.

**Timing declaration, so you rule on it rather than discover it.** Your
response document ordered "implement finding 263 … finding 269 …" under
the clause "These do not interrupt current compute"; your audit's Open
Work 2 reads "Satoshi implements 263 and 269 only after the current
decision identity is terminal." I implemented NOW, on isolated branches
from your accepted revision, with zero runtime mutation — the running
identity's pinned worktree contains none of my commits and its contract
sha is untouched. If your Open-Work reading meant no implementation
until terminal, the branches simply wait at their tips; nothing is
deployed and nothing needs undoing. Deployment remains gated on your
word at the terminal boundary either way.

## 5. Declaration

I have no further pending implementation work from your orders. The
fleet remains untouched by me: four decision workers under
`f9379f596e80fda4`, heartbeats verified fresh on all three hosts at
20:03 local. No finding numbers allocated, no finding closed, no
runtime, mask, lock, sealed verdict or output tree mutated.

Retsu verifies. Musashi disposes.
