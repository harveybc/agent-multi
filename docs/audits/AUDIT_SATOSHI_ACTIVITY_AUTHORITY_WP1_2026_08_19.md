# Audit: Satoshi Activity Authority WP1

Date: 2026-08-19 America/Bogota  
Auditor: General Musashi  
Commit under review: `agent-multi@3069d564`  
Runtime mutation: none

## Verdict

Satoshi demonstrably knows the 2026-08-18 execution order. Commit `3069d564`
names WP1, preserves the running P1LR boundary and implements a useful shared
activity object. The branch is clean and pushed. The declared focused suite
passes, and the complete suite independently reproduces as 1,671 passed.

WP1 is **not accepted yet**. The implementation removes the `-1e6` early-stop
sentinel, but another numeric ineligible sentinel remains; malformed evidence
can become eligible; absent evidence becomes zero or crashes before typing; and several
consumers listed in Satoshi's own BEFORE map still do not consume the shared
authority. R1 smoke and decision remain blocked by WP1 correctness, while the
current P1LR decision identity continues unchanged.

## Findings

### HARR-WP1-01 - S2 - Ineligible selection remains numerically rankable

`pipeline_plugins/_lexicographic_selection.py:58,167` retains
`INELIGIBLE_ORDER_KEY = 0.0` and returns it as `transport_scalar` when the
candidate is ineligible. This is precisely a fixed numeric sentinel, despite
the WP1 requirement that an ineligible candidate have no comparable score.
The new tests inspect only `_early_stop_composite`; none rejects this second
sentinel.

Required correction: make the ineligible transport value `None` or a typed
non-orderable variant and update every scalar consumer to refuse before
comparison. Two ineligible candidates must not be tie-broken into a winner.

### HARR-WP1-02 - S2 - Malformed counts and floors become valid or crash

`pipeline_plugins/_activity_authority.py:71-108` uses `int(value)`. Therefore
`1.5`, `True` and `"1"` become valid integer one; infinity raises an uncaught
`OverflowError`. The same defect affects threshold floors. This contradicts
the module's own claim that strings and malformed measurements are unavailable.

Required correction: define and enforce one exact integer contract, excluding
booleans and fractional values. A malformed measurement returns typed
unavailable; a malformed requested floor raises `ActivityAuthorityError`.

### HARR-WP1-03 - S2 - The authority is not yet the single authority

The commit's structural test merely searches for the string
`_activity_authority` in four files. It does not prove semantic use. In
particular:

- `_paired_generalization.py:44-49` retains an independent activity predicate;
- `optimizer_plugins/l2_curriculum_optimizer.py` and
  `tools/l2_curriculum_arms.py` retain independent promotion/activity logic;
- `tools/p1_difficulty_lr_factorial.py:1813-1824` attaches a static authority
  label inferred from checkpoint existence instead of persisting and consuming
  the typed result;
- Satoshi's BEFORE map also names `m0_l1_mechanism_ladder`, weekly promotion
  and LTS champion succession, none changed by `3069d564`.

Required correction: produce an AFTER consumer map and behavioral tests proving
that stopping, checkpointing, paired selection, handoff, aggregation, L2 and
champion succession either consume the same typed result or explicitly carry a
documented non-decision diagnostic exemption.

### HARR-WP1-04 - S3 - Missing evidence is zero-filled, crashes or remains eligible

`rl_pipeline_with_validation._trade_count({})` raises `ValueError` before the
authority can type the missing fact. Lexicographic selection instead renders a
missing trade count as zero. Both destroy the required unavailable semantics.
Conversely, `evaluate_activity(1, 1)` permits selection with empty
`evidence_refs`, unavailable `active_weeks` and unavailable
`exposure_fraction`.

Required correction: pass raw facts into the authority, preserve `None` in
artifacts, and require source references for the two trade facts. WP2 may add
activity-week and exposure thresholds later, but their availability and role in
eligibility must be explicit in the threshold contract.

### HARR-WP1-05 - S3 - Threshold identity does not identify the threshold

Floor 1 and floor 12 both emit
`agent_multi.activity_floor.strict_nonzero.v1`. Several consumers also use
`max(configured_floor, 1)`, silently accepting an explicit forbidden zero rather
than calling `resolve_floor` and refusing it.

Required correction: floor 1 keeps the strict-nonzero ID. Any calibrated floor
requires a distinct explicit contract ID bound to that value and its evidence.
All consumers resolve through one function; none silently clamps zero to one.

## Independent Evidence

```text
focused tests: 40 passed
complete suite: 1671 passed, 2 sklearn convergence warnings
counterexample packet: all 11 counterexamples reproduced
```

Reproducer:
`docs/audits/evidence/WP1_ACTIVITY_AUTHORITY_COUNTEREXAMPLES_2026_08_19.py`

At 2026-08-19 13:59 America/Bogota, P1LR identity
`ac0941e7bdb1a163` remained active with 4/4 fresh workers, zero service
restarts and 0/16 terminal records. Runtime processes and immutable worktree
were not modified by this audit.

## Scope Disposition

- WP0 provenance, WP1 BEFORE map and WP4 initial calibration at `c6c88f5b`:
  retained as useful evidence.
- WP1 implementation at `3069d564`: correction required; not accepted.
- Hierarchical WP2, WP3 and WP5: not delivered in this commit.
- Explicit-close track: EC-13 exists on the separate `3a02cf8c` branch; WP0
  live-control reproduction, WP1 parity, WP3 publisher and WP4 lifecycle
  evidence remain pending.
- No owner decision is required for these corrections.
