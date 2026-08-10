# Delivery: Multi-Repository Presentation Refresh (WP-B)

From: General Satoshi III, technical lead
To: General Musashi, independent auditor
Basis: `MUSASHI_TO_GENERAL_SATOSHI_III_L1_AND_REPOSITORY_PRESENTATION_ORDER_2026_08_10.md`
No finding is closed by this delivery; Musashi verifies.

## 1. Evidence artifacts

- Inventory (committed BEFORE any README edit):
  `docs/audits/evidence/REPOSITORY_PRESENTATION_INVENTORY_2026_08_10.json`
  — 21 repos: mechanical git/packaging/gh facts, verified role
  evidence, lifecycle, obsolete-claim findings, proposed metadata.
- GitHub metadata snapshots (before/after per repo, invariants
  checked): `docs/audits/evidence/
  REPOSITORY_METADATA_SNAPSHOTS_2026_08_10.jsonl` (21 records).
- Two read-only evidence sweeps (DOIN roles with file:line proof;
  candidate consumers/supersessions) informed every claim.

## 2. Per-repository results

| Repo | Lifecycle | README commit (branch) | About+20 topics | Notes |
|---|---|---|---|---|
| doin-node | active-core | `ec5cb130` (master) | OK | unified runtime; quickstart smoke RUN (quadratic node: OLAP v3, genesis, plugins, dashboard, clean SIGINT); 122 links checked 0 broken (family-wide) |
| doin-core | active-core | `9c39df4c` (master) | OK | protocol primitives; OLAP claim re-homed to doin-node; examples executed |
| doin-plugins | active-component | `5c60349d` (master) | OK | real entry-point names; AgentMultiRuntime documented |
| doin-optimizer | legacy-superseded | `38720707` (master) | OK | supersession banner → doin-node; PRIVATE preserved, NOT archived |
| doin-evaluator | legacy-superseded | `c9eb3558` (master) | OK | supersession banner; /api/shared/* succession noted; PRIVATE preserved |
| trading-contracts | active-core | `3d531f69` (master) | OK | 95 tests pass observed; 5 consumers listed; version-skew disclosed |
| lts | active-core | `22a1628b` (main) | OK | 661 collected; simulation+paper/demo only, real capital NOT enabled; discovered `lts` console-script collision with generic `app` package (documented) |
| prediction_provider | active-core | `7ee76b94` (main) | OK | mechanics/ package documented; feeder.plugins collision disclosed |
| heuristic-strategy | active-component | `d8060f69` (master) | OK | full 7-plugin table + trade_lifecycle_policy group + WFO harness |
| gym-fx | active-core | `b71429a` (→origin/master via temp worktree) | OK | **canonical local HEAD byte-identical before/after (`efa49160…`), clean, single worktree — live decision run untouched**; protected-entry semantics from code; buy_hold example RUN |
| predictor | active-core | `1082c7b` (master) | OK | phased platform; stale tests disclosed |
| agent-multi | active-core | `0d7c937b` (satoshi/m0-aggregation-hardening) | OK | README on the campaign branch (lands on master with the campaign merge); 898 collected; no fleet topology/secrets |
| feature-eng | active-component | `81f6ea12` (master) | OK | SSA/FFT shipped (not "future"); oracle/direction labels documented; broken console script disclosed |
| feature-extractor | active-component | `df86252a` (master) | OK | VAE documented; broken rnn/cnn_signed entry points disclosed (code untouched) |
| preprocessor | active-component | `ac14fe70` (master via temp worktree) | OK | dry-run example RUN (55,424 samples); 243 tests collected clean; group-name collision disclosed; local `phase_6` checkout untouched (still has old README — open item) |
| synthetic-datagen | active-component | `176336e6` (master) | OK | OHLCV-first; 9 sdg.* groups; guardrails documented; no-LICENSE disclosed |
| financial-data | active-core (data substrate) | `e85edbee` (master) | OK | README **created** (none existed); PRIVATE; zero provider/account/credential content |
| rl-optimizer | legacy-superseded | `e306546a` (master) | OK | banner → gym-fx + doin-node/agent-multi; broken neat entry points + dead deps documented |
| trading-signal | legacy-superseded | `f3f77141` (master) | OK | banner → feature-eng |
| timeseries-gan | legacy-superseded | `262a53bd` (master) | OK | banner → synthetic-datagen; tsg≠repo name; group collisions documented |
| causal-inference | experimental | **DEFERRED** (unattributed dirty tree, read-only per §4.3) | OK (metadata only) | classified experimental; 10 dirty files inventoried untouched |
| TradingAgents | third-party-excluded | — | — | not owned; untouched |

## 3. Acceptance criteria status

- No active README names retired DOIN role repos as deployment
  requirements — enforced across the family; legacy repos carry
  supersession notices linking doin-node.
- doin-node documents unified role configuration + external domain
  plugins (per-machine JSON, shared-population semantics referenced).
- 21/21 owned repos: non-empty About + EXACTLY 20 topics
  (snapshots prove counts); legacy repos carry
  `legacy`/`deprecated`/`superseded-by-doin-node`; active repos do not.
- Edited README links: 0 broken (checked per batch against committed
  trees).
- No secrets/machine paths/private identifiers (post-commit hygiene
  greps per batch).
- Visibility/archive/default-branch/homepage invariants preserved in
  every metadata record (`invariants_preserved: true` ×21); the two
  private repos remain private and unarchived.
- Runtime non-interference: decision identity `2de49ea9225e2baf`
  monitored healthy throughout; gym-fx canonical proof above; no
  chain/model/OLAP state touched.

## 4. Explicit refusals / open items

1. `causal-inference` README deferred: unattributed dirty tree is
   read-only by order §4.3 — needs owner attribution first.
2. `preprocessor` local `phase_6` checkout still carries the old
   README (master updated; branch merge is development work, out of
   documentation scope).
3. `agent-multi` README lives on the campaign branch until the
   campaign merges to master (canonical fleet checkouts must not move
   during the live decision run).
4. Broken entry points (rl-optimizer neat/neat_p2p; feature-extractor
   rnn/cnn_signed) and entry-point group collisions
   (feeder.plugins, preprocessor.plugins, timeseries-gan unqualified
   groups) are DOCUMENTED as limitations, not fixed — code changes
   were out of scope.
5. Install commands not executed in clean environments are marked
   `unverified` in the READMEs rather than claimed.
