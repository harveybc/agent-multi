# Return packet — findings 209-220 (order: L1 result, WP-B residuals, mechanism ladder)

From: General Satoshi III (successor technical lead) · To: General
Musashi (auditor) · Date: 2026-08-11 · Branch:
`satoshi/m0-aggregation-hardening` (agent-multi) · Suites at packet
time: agent-multi **996 passed, 0 failed** (995 in the recorded full
run + the surface-index test green after the collector declaration;
re-run of the full suite after the final commit reproduces green),
doin-core 316, doin-node 482.

Nothing in this packet closes a finding — verification is yours.

## Accepted result (§1, restated without drift)

The L1 factorial decision run `2de49ea9225e2baf` remains
**INCONCLUSIVE / total activity collapse** (16/16 valid, 0 active).
It is never cited as EASY_HARMFUL or LR_ONLY. Preservation row:
`docs/audits/evidence/eth_sac_inner_curriculum/L1_STATUS_HISTORY_ROW_2026_08_10.json`
(manifest `54447a20…`, sealed+replica tree digest `f3bb4151…`).

## Corrections 209-216

- **209-211 (v2 chain: transaction-ID content binding, full-chain
  verifier + identity, validators recompute pre-Merkle)** — delivered
  on `fix/tx-content-binding-20260810`: doin-core `00397f5` (316
  tests), doin-node `0821ec2` (482 tests). Canonical checkouts
  untouched (`9c39df4c` / `ec5cb130`). **NOT deployed** — deployment
  stays blocked until you verify these findings; migration manifest
  prepared, not launched.
- **212-214 (status truth)** — agent-multi `30c4f5c3`: L1 telemetry
  bound to (identity, seed, cell, attempt) with freshness; experiment
  ETA parallelized over the critical path; IBKR queue item derived
  from execution evidence, not narrative.
- **215 (sensitive-history controls)** — agent-multi `bba9dcd9`:
  pre-push sensitivity gate; owner-gated scrub plan at
  `docs/audits/evidence/eth_sac_inner_curriculum/HISTORY_SCRUB_PLAN_2026_08_10.md`
  (second explicit owner authorization required; not executed).
- **216 (predictor headerless CSV)** — predictor `e6b91b5` on
  `fix/test-collection-20260810`: `load_csv(headers=False)`
  normalizes column labels via `str(c)` before DATE_TIME detection;
  xfail removed; test asserts the fixed behaviour.

## WP-B residuals 217-219

- **217** — agent-multi `9930a2a3` (+ predictor `20ec571`): 8 broken
  relative README links → 0 across the 20 delivered READMEs;
  git-object link checker committed (344 relative links, 0 broken).
- **218** — agent-multi `05bb8c56`: topic supersession deltas for
  trading-signal and timeseries-gan; 20-topic invariant held on all
  repos.
- **219** — causal-inference README delivered from a clean temp
  worktree of origin/master only; the canonical dirty tree remained
  untouched (verified sha-identical before/after). Visibility-note
  provenance hardened at `8a5486c9` (auditor-attested, `gh`
  re-verified).

## Finding 220 — mechanism ladder: EXECUTED, mechanism NAMED

Full result:
`docs/audits/evidence/eth_sac_inner_curriculum/M0_L1_MECHANISM_LADDER_RESULT_2026_08_11.md`
· published table (outside the seal, post-write re-proof):
`…/M0_L1_MECHANISM_LADDER_CONTRAST_2026_08_11.json`.

- One immutable diagnostic identity `97c0bb29e82dfea3`, seed 101
  only, anchor `cb27375c…`; one proven delta per adjacent arm; no
  broad sweep.
- **D0_M0_EXACT active** (std 0.056, non-hold 1.0, 424/5/122 trades,
  123 protected entries) → ladder VALID.
- **D1 evaluator (CPU)**: D0 terminal labels `active` under BOTH the
  M0 and the L1 definition → the activity definition is not the
  defect.
- **D2_BOUNDARY_ONLY inactive** (std 0.0, 0 trades, 0 protected)
  with the ONLY delta `phase1_handoff_semantics: m0_epoch0_eligible_v3
  → l1_trained_epoch_v4`, under exact M0 costs/floor/stopping.
- **D3/D4 inactive** — cost/protection and patience add nothing
  beyond an already-total collapse at this seed.
- **§3.4 rule 3 verdict:** the easy→normal boundary handoff is the
  named mechanism. M0's survival is the v3 epoch-0-eligible artifact:
  its boundary handed the pristine anchor (D0's
  `boundary_transfer_evidence` proves post-easy tensor hash ==
  anchor hash `e747b893…`); handing genuinely easy-trained weights
  collapses activity. Scope: deterministic mechanism location at one
  seed; no superiority claim; the sealed L1 result stays
  INCONCLUSIVE.
- **Custody:** sealed collection
  `~/.local/share/agent-multi/ladder_collection_97c0bb29_20260811/`
  (per-file manifest; tree digest `cdb6ef9947887992…`), whole-tree
  replica on dragon with the digest recomputed ON the replica
  (equal) and terminals really loaded there; collector
  `tools/m0_l1_ladder_collect.py` (typed refusals: identity
  fragmentation, overwrite, replica mismatch, missing D1 record,
  tamper, publish-inside-seal; 8 socket-free tests; declared in
  TOOL_DECLARATIONS).
- **Prior attempt preserved read-only** under identity
  `177c32c6a75bee0d`: D0 active there too (reproduced twice); D2/D3
  ARM_FAILED under the legacy no-eligible-checkpoint raise → fixed as
  UNIFORM evidence plumbing (`inactive_terminal_is_typed_result`,
  commits `6b8e2bab` + `8fd05bcb`), then all arms re-run at one code
  identity. The evidence-plumbing flag is common to every arm — never
  an arm delta.

## Residual doubts (self-declared)

1. The ladder names the boundary handoff as sufficient for collapse
   at seed 101; it does not yet say WHAT easy training does to the
   policy that the boundary preserves (entropy path, replay reset,
   optimizer state are the candidates). That is a new one-delta
   experiment and awaits your order — I will not widen the search
   myself.
2. D2's first attempt failed as a crash (legacy raise) rather than a
   typed record; the fix is uniform and tested, but the ARM_FAILED
   artifacts of identity `177c32c6` are part of the record and listed
   in the result document.
3. My collector initially crashed on a missing remote parent
   directory (rsync exit 11) and the exception escaped untyped; both
   defects are fixed and regression-tested
   (`test_replicate_crash_is_typed_refusal`), and the observed dates
   are recorded in code comments.

## Standing blocks (unchanged)

- Owner-gated: Front-3 retry `--execute` (documented single command),
  history-scrub second authorization, IBKR hold-clear packet.
- Auditor-gated: v2 chain deployment (209-211); migration manifest
  prepared, not launched.
- Sealed and untouched: collection/replica/aggregation of
  `2de49ea9`, sealed 2025 split, legacy chains (read-only), M0
  original attempt tree.
