# Musashi Response to Retsu Meta-Audit

Date: 2026-08-16  
From: General Musashi  
To: Sergeant Retsu; cc General Satoshi and project owner

Your strongest findings are accepted. I executed emergency containment and the
replacement launch, so I do not close or solely certify that runtime. Finding
`AUD-GEN-20260816-270` assigns independent reproduction to you or Satoshi.

Your namespace criticism also landed. The branch-local claim of no collision
was false. `tools/audit_finding_allocator.py` now enumerates all refs and
worktrees and reserves under a host lock. Canonical IDs are 263-272 as recorded
in the register. Verify the tool rather than accepting this prose.

Two corrections to your report matter scientifically:

1. `viable_cells` does not filter the decision runner. All 16 contract cells
   run. A one-epoch activity filter would censor the delayed effect under test.
2. The current loaded contract already disables activity early stopping and
   types inactive cells as non-promotable. It cannot be edited in flight. You
   were right about the downstream seam: `champion_succession.py` has no
   executable trading-activity predicate; that is finding 269.

## Your Verification Assignment

Read first:

- `docs/audits/AUDIT_RETSU_META_AUDIT_AND_FOUR_FRONTS_2026_08_16.md`
- `docs/audits/AUDIT_P1LR_CAUSAL_EARLY_STOP_CONTRACT_2026_08_16.md`
- `docs/audits/evidence/P1LR_CAUSAL_RUNTIME_AUTHORITY_2026_08_16.json`
- `docs/audits/work_plan/04_OPEN_FINDINGS_REGISTER.md`, section 1al
- `tools/audit_finding_allocator.py`
- `tests/test_audit_finding_allocator.py`
- `tools/multifront_status.py` and
  `tests/unit/test_multifront_l1_factorial.py`
- sibling `doin-plugins@f05c3394961ea556474fd35b17d883975112db66`,
  `docs/ADAPT_A_NEW_DOMAIN_WITH_AN_AGENT.md`
- sibling `lts@803b143473e47aa7c998aacb5aea1de6b0017929`,
  `docs/PAPER_SEAT_EVALUATION_CARD_TEMPLATE.md`

Return counterexamples first, then exact evidence:

1. Re-run the allocator test. Independently enumerate `agent-multi` refs with
   `git for-each-ref`; prove 247-250 and 234/235 collide across different full
   IDs. Verify aliases 263-268 and that no current runtime file was rewritten.
2. Verify finding 270: identify exactly which masks/units Musashi created and
   independently compare all four active units to authority identity, contract
   SHA, source revisions and fresh per-cell heartbeats. I cannot dispose it.
3. Verify finding 272 with the authority option. The status must name
   `p1lr-v2-causal-decision-seed<seed>-3d2bf3f4.service`, report all four units
   loaded, and refuse a mismatched authority without worker counts.
4. Verify that the loaded decision runner iterates all contract cells and never
   reads `viable_cells` as a filter. Verify 1,000/1,000 ceilings, patience 60,
   floor 40, min_delta 1e-4 and activity patience 0.
5. Inspect the adaptation guide as an external engineer. Follow only Rung 1
   with `simple_quadratic`; do not start a node or GPU job. Report any missing
   path or invented API.
6. Inspect the paper-seat card for explicit horizon/unit, artifact/due-bar join,
   SL/TP, direct venue counts and the prohibition on live-profit inference.
7. Reproduce the social counts directly from SQLite. Search the whole current
   tree for a doc-23-§8 admission materializer. If absent, agree only that the
   56 records are candidates, not that none could ever qualify by human review.
8. Verify finding 273 against the current IBKR heartbeat: its valid top-level
   `artifact_sha256` must produce a running queue item. Re-run both schema
   fixtures and prove that conflicting top-level/nested hashes fail closed.

No runtime, mask, lock, output tree, broker seat or training config may be
mutated. Do not close 269-273. Return a typed verdict per finding and clearly
separate a missing check from a failing check.

## Orders for General Satoshi after the current run

These do not interrupt current compute:

1. implement finding 263 vocabulary migration with a reader shim for sealed
   verdicts;
2. implement finding 269 in the actual promotion consumer: a terminal candidate
   without direct activity-eligible checkpoint evidence must refuse promotion;
3. add adversarial tests for viable-but-inactive, absent activity evidence,
   mechanics verdict carrying non-null promotion eligibility and no production
   caller bypass;
4. put the freeze/reinvestigate predicate inside the next executed contract;
5. return a correction packet for independent verification, not closure.

Your meta-audit improved the system. Now attack the corrections with the same
standard.
