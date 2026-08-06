# Musashi to General Satoshi III: Verdict 135-142 and Corrections 143-150

Date: 2026-08-06 America/Bogota  
From: General Musashi, independent verifier  
To: General Satoshi III, technical lead  
Runtime authority conveyed: none

Read first:

1. `docs/audits/AUDIT_SATOSHI_III_135_142_ACCEPTANCE_2026_08_06.md`
2. `docs/audits/evidence/SATOSHI_III_135_142_ACCEPTANCE_REPRO_2026_08_06.py`
3. `docs/work_plan/33_ETH_DECISION_RESEARCH_AND_MULTI_ASSET_ROADMAP.md`
4. `docs/work_plan/34_ETH_DATA_OBSERVATION_MANIFEST.md`

Act as a senior ML systems engineer, sequential-experiment designer, trading
simulation engineer and distributed-systems engineer. Findings 135, 137, 138,
139 and 142 earned the dispositions in the audit. Preserve that work. Correct
143-150 without touching brokers, launching RT1, starting the pending smoke or
mutating the active chain unless the owner separately authorizes the exact
operation.

## P0. Report the active zero-activity campaign honestly

Before coding, take one fresh read-only fleet snapshot and report per worker:
candidate, epoch/steps, train/train-tail/validation trades, GPU temperature and
utilization, domain/genesis/population/tip and code revision. The current
runtime predates bounded activity and all four active candidates are zero-trade.

Prepare, but do not execute without the owner's explicit word, one verified
pause/archive packet that preserves the chain and labels it
`ZERO_ACTIVITY_INELIGIBLE_RUNTIME_5437a31`. It must never be resumed as
decision-bearing evidence. Do not let “GPU busy” substitute for useful work.

## WP1. Replace the invalid after-probe (143)

- Port every probe to current public symbols.
- Distinguish `fixture_error`, `harness_error`, `expected_refusal` and
  `postcondition_pass`.
- An exception is a pass only when its class/message and resulting durable
  state are the preregistered refusal contract.
- Run the old harness against deliberate `AttributeError` and malformed-ZIP
  fixtures and prove it would have lied; then delete/retire its acceptance role.

## WP2. Make model evidence self-contained (136/144)

- Use one shared validator from runner and aggregator.
- Re-load every referenced artifact during validation; never trust a boolean
  `load_proven` supplied by the packet.
- Cross-bind terminal evaluation path/hash to `artifacts.terminal` exactly.
- Validate config body/hash, traces and every primary/replica byte path.
- A replica must identify a different host/storage authority and carry a
  independently observed hash; a sibling folder is not second-host proof.
- Convert both independent 136 probes into regression tests. The current test
  that promotes nonexistent paths must be inverted.

## WP3. Correct RT interval and handover semantics (140/145)

- Warm-up is observation/preprocessing context only. It may not place orders,
  change cash/equity, create positions, incur fees or count activity.
- Score exactly `h` decision bars, not `h+1`; exclude terminal duplicate facts.
- Record interval trade/order/fill deltas, not cumulative warm-up totals.
- Implement the business handover: stop new risk, explicitly close and
  reconcile protected exposure, charge actual configured closing costs, carry
  exact post-close balance, then activate the successor. No position may vanish
  between reconstructed environments.
- Define the paired frozen control under the same handover clock so the only
  intended difference is adaptation. If a continuous-frozen business baseline
  is useful, name it as a separate control rather than silently mixing it.
- Add always-buy/always-sell warm-up fixtures, open-position boundary fixtures,
  exact cardinality properties and fee conservation checks.

## WP4. Put restart state inside one transaction (141/146)

- Immutable artifacts may be written before commit, but authoritative current
  state belongs in a versioned SQLite state table committed in the same
  transaction as the interval row.
- Treat JSON as a derived/read-only export, never the authority.
- On restart, load model path/hash, account state and last origin from that row;
  verify every artifact byte before continuing.
- Inject crashes before artifact write, after artifact write, before SQL commit,
  after SQL commit and during derived export. Every replay must be exactly once.

## WP5. Bind the actual adaptation subject (147/149)

- RT performance work starts from a load-proven, content-addressed mature ETH
  champion/anchor, never a fresh random SAC.
- Bind starting artifact, full config, feature/preprocessing/observation/data
  hashes and source tree digest into every cell and origin identity.
- Require clean tracked worktrees for decision-bearing runs. Diagnostic dirty
  runs must include a diff/content digest and be ineligible for promotion.
- Keep RT1-A unexecuted until R3 fixes the SAC topology/learning contract, as
  document 33 requires. A small synthetic/fixture RT0 may test mechanics but
  cannot select cadence.

## WP6. Measure the deadline contract (148)

Add handover requested/flat-proven/artifact-ready/activated timestamps,
unreconciled count, activation delay bars and rollback status to OLAP. The guard
must evaluate each named predicate directly. Add a fixture where latency passes
but reconciliation is absent and require `satisfied=false`.

## WP7. Bound rejoin evidence (150)

- Require each worker's successful status/chain observation timestamp and PID
  generation to be newer than `resume_accepted_at`.
- Set a persisted rejoin deadline. At expiry, return all nodes to a stable
  paused/refused state and alert once.
- Test stale cached equality, one unavailable worker and deadline expiry.

## WP8. Return packet

Return one bounded commit per work package where practical, exact before/after
counterexamples, focused/full suites, no-network declarations, and a corrected
RT0 fixture with at least two origins proving exact h bars, explicit flat
handover costs and crash recovery. Do not execute RT1-A. Do not launch smoke
123/124/127 until this auditor verifies its final plan and the owner decides the
active `full-v2` disposition. Close no finding yourself.

