# Audit of Retsu Meta-Audit and Four Fronts

Date: 2026-08-16 America/Bogota  
Auditor/operator: General Musashi  
Input: `docs/handoffs/RETSU_META_AUDIT_MUSASHI_AND_FOUR_FRONTS_2026_08_16.md`  
Runtime mutation: none during this audit; the active training identity and its
four workers were not restarted, reconfigured or signaled.

## Verdict

Retsu's central criticism is accepted: emergency operation and independent
verification were combined in Musashi's seat, and branch-local finding-number
inspection was inadequate. Both are now open findings with independent
disposition. His product/adaptability critique also found real missing
deliverables; a domain-adaptation guide and a paper-seat evaluation card were
implemented without consuming campaign resources.

Two claims are narrowed. The one-epoch screen did not admit or reject cells
individually; the decision runner executes all 16 contract cells. And the
loaded decision contract already states, in executable content, that activity
stopping is disabled and inactive cells are non-promotable. The remaining real
defect is downstream: the Paper champion-succession path does not enforce an
activity predicate.

## Dispositions

| Retsu item | Disposition | Evidence/action |
| --- | --- | --- |
| Musashi operated and audited the same launch | **ACCEPTED** | `AUD-GEN-20260816-270`; Musashi cannot close it |
| Citations, search scope and IDs need re-executable qualifiers | **ACCEPTED** | `AUD-GEN-20260816-271`; all-ref allocator added |
| 247-250 had no collision | **RETRACTED** | canonical IDs are 263-266; old forms are withdrawn aliases |
| historical P1LR 234/235 are globally unique | **RETRACTED** | canonical aliases 267/268 preserve running-source bytes |
| screen gates select a recipe / admit cells | **REFUTED** | screen says `performance_claims=none`; `run_seed()` iterates contract cells, not `viable_cells` |
| activity semantics exist only in handoff prose | **PARTLY REFUTED** | loaded config SHA `70ef4cb3...debfd` carries `l1_activity_patience=0`, inactive non-promotable semantics and selected-policy gates |
| promotion actually enforces trading activity | **ACCEPTED AS MISSING CHECK** | `AUD-F2-20260816-269`; declaration is not an implementation in `champion_succession.py` |
| four-front usefulness coverage was incomplete | **ACCEPTED** | live card, academic hold, social truth label and adaptation guide added |
| status accurately reports launch durability | **NEW DEFECT FOUND** | `AUD-F1-20260816-272`; generic unit name caused false `no_unit_loaded`; corrected from durable authority |
| IBKR queue accurately reports current execution identity | **NEW DEFECT FOUND DURING STATUS FOLLOW-UP** | `AUD-F2-20260816-273`; current top-level artifact hash was ignored; corrected with conflict refusal |
| Front-1 current-work/transition narrative is unambiguous | **RETSU COUNTEREXAMPLE ACCEPTED** | `AUD-F1-20260816-274`; explicit `current_work`, historical-role label and pre-terminal transition semantics added |

## Finding Namespace

`tools/audit_finding_allocator.py` now enumerates:

1. every local and remote branch returned by `git for-each-ref` in every Git
   repository directly under the workspace;
2. every registered worktree, including detached runtime worktrees;
3. tracked content plus selected untracked text where another agent may have
   drafted an uncommitted finding; and
4. a host-local allocation ledger protected with `flock`.

The same full ID on many refs is inherited evidence, not a collision. A serial
used by two different full IDs is a collision. Focused tests create multiple
refs, an untracked draft and a cross-prefix collision; 2/2 pass.

Canonical remapping:

| Withdrawn/local or historical form | Canonical global ID |
| --- | --- |
| `AUD-F1-20260816-247` | `AUD-F1-20260816-263` |
| `AUD-GEN-20260816-248` | `AUD-GEN-20260816-264` |
| `AUD-GEN-20260816-249` | `AUD-GEN-20260816-265` |
| `AUD-GEN-20260816-250` | `AUD-GEN-20260816-266` |
| `AUD-P1LR-20260815-234` | `AUD-P1LR-20260816-267` alias; runtime text preserved |
| `AUD-P1LR-20260815-235` | `AUD-P1LR-20260816-268` alias; runtime text preserved |

No live source or artifact was rewritten to perform this migration.

## Executed Activity Semantics

The exact loaded config is
`examples/config/phase_3_eth_sac_dynamics/p1_difficulty_lr_factorial_v2.json`,
SHA-256 `70ef4cb3e66b3360d4a272d544e680eeefc35ce41e384526c502b01a273debfd`.
It executes these facts:

- screen: one epoch per phase, mechanics/custody, no performance claim;
- decision: 1,000 epochs per phase maximum, improvement patience 60 after
  floor 40, `l1_activity_patience=0`;
- an inactive cell is a measured decision outcome and is non-promotable;
- all four cells per seed execute; no screen `viable_cells` filtering occurs.

Therefore the exact sentence Retsu requested is already functionally present,
but distributed over typed fields. Changing the current JSON would change its
SHA and invalidate the running identity. The next contract revision must carry
the explicit fields `activity_required_for_screen_dispatch=false`,
`activity_required_for_decision_early_stop=false`, and
`activity_required_for_promotion=true` in one machine-readable object. It must
also enforce the final field in the promotion consumer, not merely declare it.

## Fresh Four-Front Status

Snapshot time: 2026-08-16T22:53Z. These are instantaneous paper/demo and
training facts, not performance claims.

### 1. Live/Paper business reality

- Alpaca Paper: fresh write-enabled runner, one open paper exposure and one
  open order, nine lifecycle records; current seat is a linear integration
  control, not an optimized champion.
- IBKR Paper: fresh write-enabled runner, direct state flat, no hold, no open
  exposure; latest signal rejected as stale. The venue is operational but not
  currently trading that stale decision.
- OANDA MT5 Demo: connected, execution enabled, direct terminal trading
  allowed, one open demo position, no open order, heartbeat about 11 seconds
  old at collection.

The LTS README already truthfully calls these simulation/paper-demo integration
labs and says real capital is disabled. The missing product artifact was a
standard evaluation card; it now exists at
`lts/docs/PAPER_SEAT_EVALUATION_CARD_TEMPLATE.md` and requires same-window
simulation/paper joins, explicit units/horizons, raw return/drawdown/trades,
costs, direct SL/TP evidence and a no-live-profit limitation.

### 2. Optimization/research

- identity `f9379f596e80fda4`, contract SHA above;
- four fresh decision workers, four exact incident-specific systemd units;
- Omega seed 101 `normal, LR 1e-4`; Dragon seed 202 `normal, LR 3e-5`;
  Gamma seed 303 `easy, LR 1e-4` and seed 404 `easy, LR 3e-5`;
- all are in training; 0/16 terminal cell records so far;
- no defensible ETA exists until at least two complete cells establish observed
  duration. Inventing one from elapsed time would repeat the prior ETA error.

The corrected status collector binds unit pattern, identity and contract from
the durable authority record. It now reports every unit loaded with zero
systemd restarts instead of the false `no_unit_loaded` result.

### 3. Academic

The publication roadmap now forbids performance claims from the withdrawn
2,724-input/dead-actor recipe and from mechanics screens. Protocol, custody,
fault and negative-result papers may proceed on their own evidence; trading or
predictive performance language waits for terminal independently reproduced
2,660-input outer-role results.

### 4. Social/domain discovery

Fresh database facts: 13,638 posts collected; 1,358 enriched; 56 labeled
`experiment_candidate`, 431 `investigate`, 39 `reply_candidate`. No code or
table currently materializes a doc-23-§8 domain-admission verdict. Therefore:
the collector and enrichment pipeline are productive, but the 56 entries are
leads, not admitted domains and not optimization authority.

## Adaptability Result

The architecture was adaptable and the onboarding was not. The `doin-plugins`
branch `musashi/domain-adaptation-kit-20260816`, corrected and published at
`3e490c1`, now adds:

- `AGENTS.md` with repository boundaries and safety rules;
- `docs/ADAPT_A_NEW_DOMAIN_WITH_AN_AGENT.md` with exact paths;
- a pasteable agent assignment; and
- three acceptance rungs: local interface, trusted single-node DOIN, then
  untrusted synthetic challenge only after held-out rank preservation.

The guide explicitly warns that the current composite proof can add
non-commensurable units. Until runtime metadata enforces a comparable unit or
`not_for_composite`, multi-domain weighted sums are not treated as meaningful.
The documentation exercise uses `simple_quadratic`; it launched no node and
consumed no GPU.

## Tests

- finding allocator and multi-front P1LR status: 48 focused tests passed;
- agent-multi engineering-surface set: 65 tests passed;
- agent-multi full suite: 1,590 passed, with two existing scikit-learn
  convergence warnings and no failures;
- doin-plugins at `f05c3394961ea556474fd35b17d883975112db66`: 44 passed,
  two skipped in the declared `trading-stack` environment;
- LTS paper-seat card at `803b143473e47aa7c998aacb5aea1de6b0017929`:
  README target and local links verified;
- `git diff --check`, generated engineering-surface validation and JSON parsing:
  passed.

Status follow-up after publication found and corrected finding 273. Its focused
status and engineering-surface set passed 99 tests. A read-only run against the
real current heartbeat now reports IBKR `running`,
`operational_with_open_exposure`, with its valid artifact hash; the paired
fixture proves conflicting top-level and legacy hashes refuse.

## Open Work

1. Retsu independently verifies findings 270-274 and the artifacts in the
   companion response. Musashi does not close his own runtime work.
2. Satoshi implements 263 and 269 only after the current decision identity is
   terminal: typed mechanics vocabulary plus an executable activity gate in
   champion promotion, with sealed-verdict migration rather than mutation.
3. The next optimization contract must carry one explicit final freeze
   predicate: freeze L1 only when terminal records are complete, observation
   identity is correct, activity/promotion predicates pass where required and
   independent outer-role evidence is reproducible. Otherwise the typed result
   is another representation/recipe investigation, never silent promotion.
4. Social §8 admission materialization is useful but lower priority than the
   current live and optimization evidence. It must be deterministic and
   human-approved; it may not auto-launch a domain from a post.
