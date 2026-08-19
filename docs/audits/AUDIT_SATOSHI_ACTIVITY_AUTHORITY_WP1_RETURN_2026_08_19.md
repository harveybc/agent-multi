# Audit: Satoshi Activity Authority WP1 Return

Date: 2026-08-19 America/Bogota  
Auditor: General Musashi  
Commit under review: `agent-multi@4e813404`  
Parent audit: `AUDIT_SATOSHI_ACTIVITY_AUTHORITY_WP1_2026_08_19.md`  
Runtime mutation: none

## Verdict

The return is a material improvement, but WP1 is **not accepted**.

The central authority now rejects malformed counts correctly, preserves a
missing count in `_trade_count`, removes the numeric ineligible order key and
refuses an unbound higher floor in its main entry points. The focused suite
passed independently (197 tests), and the complete suite reproduced as 1,696
passed with two pre-existing sklearn convergence warnings.

Those tests do not cover the full decision graph. The independent return
reproducer executes every case separately and reproduces 12/12 remaining
defects. Several paths can still make malformed or zero activity eligible for
selection/promotion, and the claimed content-bound evidence can be invented
without any retrievable artifact. The delivery's own AFTER map declares four
required consumers pending.

The active P1LR decision run remains intact: identity `ac0941e7bdb1a163`, four
fresh workers on one chain, 0/16 terminal records. It was not restarted or
mutated by this audit.

## Findings

### HARR-WP1-R2-01 - S2 - Decision and promotion paths still bypass the authority

The authority is not yet the one authority promised by its module contract or
by C5:

- `pipeline_plugins/_paired_generalization.py:43-49` accepts boolean and
  fractional trade counts and accepts zero trades when a caller supplies floor
  zero. This comparator is called by the L2 objective in
  `optimizer_plugins/l2_curriculum_optimizer.py:349-358`.
- `examples/scripts/materialize_phase_1_promotion_candidates.py:113` converts
  the trade fact with `int(...)` before authority evaluation. A transaction
  carrying `trades_total=1.9` is accepted and persisted as one trade.
- `pipeline_plugins/rl_pipeline.py:171-178` remains a registered plugin and
  independently coerces counts/floors; 31 committed configs select it.
- The delivery's AFTER map declares L2, weekly promotion, M0 and the dedicated
  phase-1 materialization fixtures `PENDING`.
- The claimed LTS succession integration exists only on
  `satoshi/finding-269-activity-predicate-20260816@26af1f80`; that commit is not
  an ancestor of the current LTS execution branch.

Impact: malformed, fractional or zero-activity candidates can still enter a
comparable objective or a promotion manifest outside the corrected central
path. WP1 cannot guard R1/R2 promotion until these consumers are integrated or
mechanically declared non-decision/non-promotion.

### HARR-WP1-R2-02 - S2 - Evidence binding is syntactic, not verifiable

`pipeline_plugins/_activity_authority.py:231-243` accepts any lowercase hex
token of length 40. Forty invented `a` characters make two asserted trade
counts eligible. No algorithm, artifact locator, fact key or content read is
required.

`rl_pipeline_with_validation._activity_evidence_ref()` then manufactures a
SHA-256 over the in-memory summary when no role artifact hash exists
(`:238-251`). A digest of an assertion is not a reference from which another
node can re-derive the asserted count.

Impact: unavailable/unverifiable activity can be promoted as evidence-bearing,
defeating C3.3 and weakening decentralized verification.

### HARR-WP1-R2-03 - S3 - Calibrated floor identity and value remain inconsistent

`threshold_contract_for()` compares `int(calibrated["floor"])` at
`_activity_authority.py:145`. It therefore accepts both string `"12"` and
fractional `12.9` as a contract for floor 12. The evidence reference is merely
truth-tested.

Separately, lexicographic selection builds a calibrated floor contract but
publishes the strict floor-1 ID unconditionally at
`_lexicographic_selection.py:210-211`. An eligible floor-12 result therefore
claims `agent_multi.activity_floor.strict_nonzero.v1`.

Impact: persisted records cannot identify the rule that actually judged them,
and malformed contracts can cross the boundary C4 was meant to protect.

### HARR-WP1-R2-04 - S3 - Missing activity is still rendered as zero

`_lexicographic_selection.py:186-187` converts an unavailable role count to
zero in the returned components. Eligibility is false, but the diagnostic fact
no longer distinguishes “missing” from “observed zero,” contrary to C3.2.

Impact: downstream evidence packets can report a measurement that never
existed and can no longer reconstruct the typed reason from the component
alone.

### HARR-WP1-R2-05 - S3 - Post-fix evidence stops after one refusal

`WP1_COUNTEREXAMPLES_POST_FIX_OUTPUT_2026_08_19.txt` is a traceback at the
first higher-floor case. It does not execute the remaining cases and therefore
cannot support the commit message's “all 11 counterexamples closed” claim.
The independent runner avoids this masking failure and reproduces 12 remaining
counterexamples.

### HARR-WP1-R2-06 - S3 - Consolidated status hides the active decision run

At 20:38:59Z, `multifront_status.py` reported screen identity
`bfbfd6443b849275` as `completed_untransitioned` with 0/4 fresh workers. Direct
systemd and heartbeat inspection at 20:41Z proved 4/4 running
`p1lr-decision@*.service` workers on `ac0941e7bdb1a163`, all heartbeats fresh.

Impact: the owner-facing status can declare Front 1 stopped while every GPU is
executing the current decision experiment. This is a recurrence of the
decision-root discovery class, not a training failure.

## Independent Evidence

- `docs/audits/evidence/WP1_ACTIVITY_AUTHORITY_RETURN_COUNTEREXAMPLES_2026_08_19.py`
- `docs/audits/evidence/WP1_ACTIVITY_AUTHORITY_RETURN_COUNTEREXAMPLES_2026_08_19.json`
- `docs/audits/evidence/WP1_RETURN_RUNTIME_STATUS_DIVERGENCE_2026_08_19.json`

Results:

```text
return counterexamples: 12/12 reproduced
focused suite:           197 passed
complete suite:          1696 passed, 2 warnings
runtime:                 ac0941e7bdb1a163, 4/4 fresh, 0/16 records
```

## Accepted Parts

- Exact `numbers.Integral` measurement contract in the central authority.
- Typed floor validation at `evaluate_activity` / `evaluate_role_activity`.
- `INELIGIBLE_ORDER_KEY=None` and scalar refusal through `require_orderable`.
- `_trade_count()` preserving missing/malformed facts as `None`.
- Explicit declaration that activity weeks and exposure are informational in
  the strict-nonzero contract.
- Clean pushed correction branch and green full suite.

These parts should be retained. They do not require rollback.

## Disposition

- WP1: correction required; not accepted.
- R1 mechanics smoke and R1/R2 decision promotion: remain blocked on corrected
  WP1 semantic integration, not on the active P1LR run.
- Current P1LR decision experiment: continue unchanged on all four workers.
- Owner action: none required for these corrections.
