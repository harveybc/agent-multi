# Screen B / B4: roadmap pointer repair

**Date:** 2026-09-26
**Order:** none. This note was written because a stale pointer re-dispatched a
closed node; it repairs the pointer and nothing else.
**Licence in force:** documents and committed evidence only.
**Effects:** zero GPU, zero training, zero scores, zero campaign, zero live,
zero database writes, zero running processes touched.

## 1. What is wrong

The roadmap pointer still says the next node is Screen B / B4.

`docs/audits/evidence/N5_TRANSITION_LEDGER_V2_2026_09_04.json`
(sha256 `26fc0473345fdeb0f0c122f03be8d2fe3cee247ea3bfe2e531db2c06fec97c86`)
carries:

```json
"roadmap_next_node": {
  "node": "Screen B / B4 (causal per-origin random-initialized SAC arm)",
  "status": "under B4-R0..R3 audit in this same return"
}
```

That audit returned on 2026-09-04. Its disposition was reviewed the same day.
The node closed on 2026-09-12. The pointer was never advanced, so the corpus
still reads as though B4 were the open next node — and on 2026-09-26 it was
dispatched again on exactly that reading, with the 2026-09-04 R1/R3 state
presented as current.

The defect is staleness, not error. Every other field of the v2 ledger stands.

## 2. What is actually true

**The node is closed.** `docs/audits/work_plan/B4_V4_CLOSURE_DISPOSITION_2026_09_12.md`
(sha256 `8700cc95ee8423a9cf5a30f3c446025ed443f4c07449f29cb29c8db13f5cd019`,
commit `2042af83`) records Musashi's external decision from
`MUSASHI_AUDIT_ROUND7_C87_C105_2026_09_12.md` §3:

- decision `B4_V4_ACCEPTED_AS_NON_AUTHORIZING_FINAL_CLOSURE`;
- 2 `COMPLETED_VERIFIED`, 1 `QUARANTINED_PARTIAL`, 9 `NOT_STARTED`;
- outcome `SCIENTIFICALLY_INSUFFICIENT_NO_VERDICT`;
- campaign closed; no consumer, no runner; no comparison, effect or ranking
  computed from the two completed cells.

C111 (`1b4c22a5`, `432599be`) loaded one additive closure envelope into the
cube. The C106–C121 return packet stops at `B4_CLOSED` and its "what needs you"
list contains no B4 item.

## 3. R1 was discharged

B4-R1 (recovery order `af1ca667` §10, file sha256
`128dd13cdac0cf87c083464a80fa1df4ed9d46ecc673d357ad8089e65e64d96c`) returned
`B4_RANDOM_ONLY_REQUIRES_SUPERSEDING_DESIGN`, naming exactly one unavoidable
semantic change. From `B4_COMPATIBILITY_MATRIX_2026_09_04.json`
(sha256 `e46ad103efcab9bbc7559e80bf3952aa70f4ca3148a928ea6557c2f52b12ff22`),
key `gymfx_env_semantics`:

> "verdict": "SEMANTIC_DIVERGENCE — the one design decision" … its B0-B3 v4
> baselines were produced under the branch lineage of 2026-08-25/26; the
> currently accepted gym-fx execution truth advanced afterwards (fill-truth +
> temporal v2 lineage, accepted 2026-08-28) … Neither option may mix lineages
> between B4 and its comparator.

The proposal (`B4_SUPERSEDING_DESIGN_PROPOSAL_2026_09_04.json`, sha256
`c84750d6c76c2b6b568789213be6c862a013f6ae2c5785e7f70f2e0d739a2203`)
recommended option B. Musashi answered in one line — order `0b4d2748` §6, file
sha256 `479608adcb6d67d6bf0b4de8ba0120faf368b33877bea6060c5fa68b1e13c738`:

> \# OPTION B IS ACCEPTED
>
> … The superseding design must be a new artifact, not an edit of the proposal.

It was then written as a new artifact:
`docs/audits/evidence/B4_SUPERSEDING_DESIGN_V2_OPTION_B_2026_09_05.json`
(sha256 `9155f508afc4b87f345a652070a6727a13373c75877c619f6110d54e9e678237`),
schema `agent_multi.b4_superseding_design.v2`, status
`SEALED_BEFORE_ANY_NEW_SCORE`, population label
`SCREEN_B_CURRENT_EXECUTION_TRUTH_OPTION_B`, execution truth pinned to
`gym-fx@6d779afd…` with point-of-use manifest `a30eda1f…` re-hashed at
execution, `mixed_lineage_rule` enforced by `check_lineage_match()`,
`session_exposure_enabled` explicitly false, sealed-2025 structurally absent.
Amendments 1–16 follow it; the last, dated 2026-09-10, carries
`scientific_change: NONE`.

**That identity is load-bearing.** It is pinned by
`bind_superseding_design()`, by the amendment chain, by the 12 cell configs, by
the comparator population and by the OLAP closure envelope. Re-authoring a
superseding design today would create a second trust root for a chain that is
sealed, digest-pinned and already reviewed. Do not do it.

## 4. R3 was discharged, and a second, different correction followed

B4-R3 returned `B4_CORRECTION_REQUIRED`. Its literal content was narrow —
`GENERAL_SATOSHI_TO_MUSASHI_N4_BINDING_AND_B4_RETURN_2026_09_04.md` §6:

> `# R3: B4_CORRECTION_REQUIRED — el diseño superseding aguarda su revisión`

It asked for his review, not for code. He gave it, in the same order that
accepted option B.

A **second and different** `B4_CORRECTION_REQUIRED` was issued a day later —
order `61622469`, file sha256
`f58448ade0c81ec110f32e250562f6d5c471378f416d704c12d5dcf3949ed394`, items
B4-E1…E7. Its findings: unequal economic envelopes (`0.012102` at the
comparator against `0.007102` in the B4 materializer), a proposed GPU command
that was not a B4 command, final code that could not satisfy its own sealed
design, a cell that was not a complete runnable recipe, comparator
verification that trusted a summary, and pending-ratification wording in
executing code. It was answered at `03b05cf7` (PRE) → `9b3fd8b8` (E1–E7
sealed) → `b478d520` (v6 comparator population) → `446360ee`
(re-materialization and replay), with envelope parity proven from the same
geometry and costs and a 50-test refusal battery, and it was accepted.

Anyone reading only the disposition string will conflate these two. They are
eight days and two different demands apart.

## 5. Evidence already declared incomparable

Nothing was silently carried forward across the supersession. Three objects
are named, each with its own record:

1. `screen_b_rule_arms_v4_alpaca_20260826` at `81fa5a2b` — ran under the
   superseded 2026-08-25/26 lineage. It remains immutable evidence and is
   refused as the B4 comparator by the design's `old_evidence_quarantine`;
   the v1/v2/v3 directories are untouched.
2. `screen_b_rule_arms_v5_current_truth_20260905` — superseded by the v6
   population after the E1 envelope-parity correction, preserved byte-intact
   as historical evidence. The rule scores happened to be identical: the
   correction moved B4 **toward** the comparator, not the comparator toward
   B4.
3. The quarantined partial cell `o2022_seed303` — "neither a failure nor a
   result". Its GPU charge is published only as a lower bound, 124,993.6 s
   from the claim to the stop signal.

## 6. Two traps this note exists to disarm

**Doc 42 holds no B4 content.**
`docs/work_plan/42_WEEKLY_SESSION_EXPOSURE_AND_REOPEN_POLICY.md` has no B4
material at all. Screen B / B4 is specified in
`docs/work_plan/40_POST_P1_SCREEN_SPECS_2026_08_24.md` and
`41_STATISTICS_CONTRACT_2026_08_24.md`; the machinery is
`tools/screen_b_baselines.py`, `tools/materialize_b4_causal_sac.py`,
`tools/b4_run_cell.py`, `agent_plugins/b4_authority.py` and the
`docs/audits/evidence/b4_*` trees. Note that docs 39–41 are present on the
B4 branches and absent from later working branches, which is itself a reason
a reader lands on 42 and concludes B4 lives there. Doc 42 bears on B4 only in
reverse: the Option-B order requires `session_exposure_enabled=false`
explicitly in every B0–B4 config precisely so that doc 42's opt-in MT5
weekly-session state machine cannot change the Screen B question.

**`79ed23c7` is not a B4 commit.** `agent-multi@79ed23c7` is "audit request to
Musashi: subtle integration of capacity-sizing into the multi-fidelity
proposal (E1-E6)" — the doctoral capacity / multi-fidelity line (doc 44;
`predictor/docs/RETSU_TO_MUSASHI_ACOPLE_CAPACIDAD_MULTIFIDELIDAD_2026_09_04.md`).
It has nothing to do with the B4-E1…E7 economic-parity items. The E-labels
collide across two unrelated fronts.

## 7. What this note repairs, and what it does not

Repaired: the pointer, by `docs/audits/evidence/N5_TRANSITION_LEDGER_V3_2026_09_26.json`,
which supersedes the v2 ledger **without rewriting it** — the same way v2
superseded v1, and the same way C110 added a register note rather than editing
a register row. The v2 and v1 files are left byte-unchanged.

Deliberately not done:

- **The next node is not named.** The v3 ledger sets
  `roadmap_next_node.node` to `NOT SET BY THIS LEDGER`. Naming a successor
  node is a reviewed roadmap act, not a pointer repair.
- **No superseding design was written.** §3 is why.
- **No B4 correction was implemented.** §4 is why: R3 asked for a review that
  it received, and the later correction was executed and accepted.
- **Whether a no-verdict closure satisfies the doc 38 §23.2 ordering** — which
  defers feature selection until Screens B, A, R and C complete — is left
  open. B closed without a verdict, and closed is not complete. That is
  Musashi's call.

## 8. For Musashi

1. Name the successor roadmap node, or confirm that the B/A/R/C ordering is
   unsatisfied by a no-verdict B closure and say what that implies for feature
   selection.
2. Ratify or reject this pointer repair. It asserts no science and grants
   nothing; if the v3 ledger is the wrong instrument, the v2 ledger is intact
   and losing this file costs nothing.
