# L1 Matched Factorial — Delivery Draft (Satoshi III, 2026-08-09)

STATUS: **DRAFT — NOT an audit request.** The 16 decision cells are
running; this document finalizes only after all records land and the
generic aggregator publishes its typed outcome. Slots marked PENDING
can only be filled by completed runtime evidence, never by hand.

## 1. Chain identity and deployment

- Owner launch order (2026-08-09, verbatim scope): deploy exactly
  `5322d42a` to the fleet, materialize and launch the four seed-101
  cells on a single chain identity, then dispatch the 16 L1 cells
  without further approval.
- Deployed lineage from `5322d42a` (delta disclosed at dispatch):
  `c867d46c` (contract v3 + runner) → `0620e32b`, `03f7badb`,
  `1c606631` (three smoke-found fixes). Fleet ran the 16 cells at
  **`1c606631`** on omega/dragon/gamma; canonical checkouts frozen for
  the whole experiment (the runner's code-motion guard aborts any cell
  whose repo revision moves mid-cell).
- Post-dispatch implementation continued on the worktree branch only:
  `3543ad5e` (generic aggregator + 39 mutation tests), `f17d6105`
  (dispatch hardening), `20bb9295` (clean-checkout reproducibility).
  These do NOT touch the running cells or canonical checkouts.
- Smoke acceptance (mechanics only, never aggregable): experiment
  `606348b82552510e`, 4/4 cells, both phase-1 modes through the same
  machinery, boundary tensor hash_match TRUE ×4,
  `evidence_class=mechanics_smoke`, `decision_eligible=false`.
- Decision experiment identity: PENDING (recorded in the 16 cell
  records; single identity required, mixed identities refuse
  aggregation).

## 2. Findings map (159–169, 170–177)

| Finding | Correction | Commit(s) | Tests / evidence |
|---|---|---|---|
| 159 (easy handoff epoch-0 anchor) | epoch-0 handoff structurally ineligible; paired selection among TRAINED epochs only; terminal-trained fallback | `96a23180` (WP2 era), re-verified in WP3 rework `1c606631` lineage | `tests/unit/test_solvency_curriculum_pipeline.py::test_epoch_zero_anchor_baseline_is_structurally_ineligible` |
| 160 (normal probe as gate) | normal probe telemetry-only (`normal_handoff_eligible_telemetry`), never a gate | same lineage | `::test_normal_probe_collapse_is_telemetry_not_a_gate` |
| 161–162 (evidence typing/immutability) | typed evidence classes on every record; smoke can never aggregate | `c867d46c`, `3543ad5e` | aggregator refuses smoke (mutation tests; real refusal exercised against `606348b82552510e`) |
| 163 (GPU identity) | GPU-UUID binding in contract + dispatch | `c867d46c`, `f17d6105` | contract `gpu_uuid` entries; dispatch script env pinning |
| 164 (completion transaction) | per-cell record written only after run completes; code-motion guard | `c867d46c` | `run_cell` raises on revision movement |
| 166 (consumer inventory) | typed per-source inventory (JSON/JSONL/SQLite known+generic, declared raw-scan) | WP0 `1c606631` lineage | `tests/test_successor_quarantine.py` (19) |
| 167 (idempotency) | single validator both paths; contain-first repair | same | same |
| 168 (missing evidence typed) | `QUARANTINED_EVIDENCE_INCOMPLETE` exit 3 | same | same |
| 169 (undeclared executables) | all new tools declared | `c867d46c`, `3543ad5e` | `tools/TOOL_DECLARATIONS.json` |
| 170 (nested chronological splits) | `pipeline_plugins/_nested_splits.py`, contract v1, materializer with refusals | WP1 commits | `tests/test_nested_splits.py` (21, incl. real-ETH derivation 11509/2190/2190/2196) |
| 171 (paired comparator) | `paired_generalization_weekly_v1`, gap penalty beta, ordinal keys never averaged | WP2 commits | `tests/test_paired_generalization.py` (12) |
| 172 (L1 stopping) | `PairedStoppingState` 2000/60/40, ineligible checkpoints never touch patience, resume refuses split-identity mismatch | WP2 commits | stopping tests in paired suite |
| 173 (L1 factorial) | contract v3, 4 cells × 4 seeds, matched boundary, mode-aware phase-1 | `c867d46c` | smoke `606348b82552510e`; decision run PENDING |
| 174 (budget ledger) | two-phase shared ledger (`total_max_passes`, `phase1_max_fraction`, `normal_phase_min_passes`) | WP3 commits | pipeline validation tests |
| 175 (aggregation contract) | generic aggregator: §7.1 direct facts (terminal probe + verification rollout), §7.2 exact ordered outcomes, append-only publication | `3543ad5e` | `tests/test_aggregate_l1_factorial.py` (39 mutation tests) |
| 176 (anti-idle orchestration) | dispatch script + monitor + CPU work sustained during GPU runs | `f17d6105` | this document's timeline |
| 177 (standing authorization) | no owner-phrase ceremonies; execution under doc 38 §9 | — | dispatch executed on the owner's order without further asks |

(WPs of doc 38 beyond L1 — L2 programs, FS0/FS1/FS2, conditional 2×2 —
remain sequenced AFTER this factorial completes and are not claimed
here.)

## 2b. INCIDENT 2026-08-09: first dispatch lost, corrected, relaunched

The first 16-cell dispatch (experiment identity of the `1c606631` run)
was LOST without records. Root cause chain, in order:

1. The activity stop (AUD-F1-127) RAISED when no checkpoint ever
   became activity-eligible — correct for D1/M0 screens, structurally
   wrong for a factorial whose N cells' full inactivity is the measured
   phenomenon. Seeds 202/303 died at their first cell's activity stop
   (~epoch 79, zero trades throughout) with no record; the raise
   precedes record writing, so the loss was total but STERILE (no
   partial evidence can ever be mistaken for records).
2. The original dispatch's hung ssh channels (nohup holding stdout)
   acted as time bombs: when the blocking remote process died, the
   channels unblocked and executed their REMAINING lines, launching a
   duplicate seed-404 (second writer into the same directory) and a
   fresh seed-303 on the doomed revision. Both zombie tasks were
   stopped; the dispatch hardening (`f17d6105`) removes the class.
3. The permission layer denied killing the doomed processes; omega's
   process ended without a crash signature in its log (external
   termination, unattributed), the gamma survivors were left to
   self-terminate at their activity stops.

Corrections in `9b6f0745` (fleet redeployed omega/dragon/gamma):
typed inactive-terminal result behind an explicit flag (legacy raise
preserved; REAL CPU proof: flag on → typed result with loadable
terminal, flag off → legacy raise, PROOF PASS), runner records
termination facts, `env_asset` binding (the aggregator's label-vs-id
comparison would have refused every real cell — caught pre-aggregation),
activity-stop budget made explicit in the frozen contract.

**Relaunch under the new single chain identity `16acf854c83b5051`**
(contract sha `a4cb963fac8c1e2b…`): omega seed-101 pid 2002936 and
dragon seed-202 pid 1026143 running; gamma seeds 303/404 queued behind
its doomed processes' self-termination. The lost first-dispatch delta
and this identity change are disclosed here, not absorbed.

## 3. Defects found and fixed during this campaign

1. `easy_min_trades` KeyError in normal-mode phase-1 → setdefault from
   plugin params.
2. Sham-handoff guard fired on smoke (SB3 `learning_starts` > smoke
   steps) — guard correct, budget wrong → smoke budget carries
   `learning_starts=100`.
3. `full_spread_rate` KeyError in meta → mode-safe `phase1_difficulty`
   block (`.get` fields) + `phase1_mode` recorded (meta v4).
4. Paired `_selection_value` demanded robust-only utility; summaries
   carry `mean_weekly_rap` → comparator's `_split_utility` resolver
   (robust-first, common fallback) — `1c606631`.
5. Aggregator probe compared `_n_updates` across boundary lineages; the
   boundary rebuild restarts the counter, so the terminal counter IS
   the phase-2 count → fixed in `3543ad5e` (caught by exercising the
   probe against a real smoke artifact).
6. Remote dispatch hung its ssh channel (nohup without stdin
   redirection) → full detachment + pid-file capture — `f17d6105`.

## 4. Runtime evidence (PENDING completion)

- 16 cell records under a single experiment identity: PENDING
- Per-cell §7.1 activity facts (probe + verification rollout): PENDING
- §7.2 typed outcome + rationale: PENDING
- Per-seed raw metrics with units (trades, mean weekly return, total
  return, max drawdown, Sharpe): PENDING
- Observed field signal at dispatch+2h (telemetry, not a conclusion):
  all four seeds' first cell (normal phase-1) showed sustained
  zero-activity; the no-activity stopper was counting. Whether easy
  phase-1 restores activity is exactly the factorial's question.

## 5. Clean-checkout reproducibility (audit requirement 6, non-blocking)

- Clean detached clone of `3543ad5e` on 2026-08-09 reproduced the
  audit's 13 failures EXACTLY: 829 passed, 13 failed — 10 from the
  gitignored pinned base contract (`ETH_BASE`), 3 from missing sibling
  `doin-node` campaign templates. Same classes Musashi named.
- Correction `20bb9295`: the pinned base contract is now TRACKED
  (whitelisted; its sha IS the decision pin `5df24c0d…` — a pinned
  contract belongs in Git), and `tools/bootstrap_test_fixtures.py`
  publishes the remaining fixture dependency explicitly with typed
  outcomes (`FIXTURES_READY | FIXTURES_INCOMPLETE | BOOTSTRAP_FAILED`,
  `--check-only` for CI).
- Proof from the clean clone: bootstrap reported FIXTURES_READY
  (doin-node cloned at `5bd6d396`), the 13 previously failing tests
  pass 112/112 in their five files. The intermediate full run at
  `20bb9295` failed exactly one NEW test — the engineering surface
  index flagging the not-yet-declared bootstrap executable (finding-169
  discipline catching its own author); declared in `ef42d110`.
  Final full suite from the clean detached clone at `ef42d110`:
  **842 passed, 0 failed** (31s). The suite is reproducible from
  `git clone` + `tools/bootstrap_test_fixtures.py`.
