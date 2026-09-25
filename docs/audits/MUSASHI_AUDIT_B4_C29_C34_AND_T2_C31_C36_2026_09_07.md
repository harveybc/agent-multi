# Musashi audit: B4 C29-C34 and T2 C31-C36

Date: 2026-09-07

## Verdicts

- **B4: REVISE; recovery acta and GPU launch withheld.** The environment
  recovery is mechanically sound, but the new review gate is not an effective
  authority boundary and its identity does not reach the scientific results.
- **T2: REVISE BEFORE DESIGN REVIEW.** The six-panel geometry and evidence
  checks stand, but the executable design names the primary contrast with the
  opposite sign from the quantity actually computed. No design seal or public
  score may be produced from draft v5.

The correction order is
`docs/handoffs/MUSASHI_TO_GENERAL_SATOSHI_B4_C35_C38_AND_T2_C37_ORDER_2026_09_07.md`.

## Scope and independent reproduction

Reviewed objects:

- B4 packet `9d4aec7fc2cb6ce35f7a2236974de251f0f4607a`, corrected
  code `502c81d2`;
- T2 packet `0e20e497b75df92d695b07b7ad036811ff23d76b`, corrected
  code `5737ad77`.

Independent focal runs used the Python 3.12 `trading-stack` environment with
`CUDA_VISIBLE_DEVICES=""`:

- `tests/test_b4_materializer_authority.py`: **166 passed** in 200.83 s;
- `tests/test_t2_harness.py`: **52 passed** in 275.84 s.

These green runs establish that the committed batteries reproduce. They do
not establish the omitted properties below.

## Finding B4-C35: the recovery acta does not verify its claimed pin

`require_v6_launch_open()` declares `pinned_commit` and `reviewed_at_date` as
required keys but never validates either value. It also reads the acta by
path after a path-level existence check and has no independently pinned digest
for the acta bytes.

The public API accepted all four of these records while opening the gate:

```text
ACCEPTED None 'not-a-date'
ACCEPTED False 'not-a-date'
ACCEPTED '../../foreign' 'not-a-date'
ACCEPTED '0000000000000000000000000000000000000000' 'not-a-date'
```

Each record named the real amendment-12 digest and copied the reviewer label
and three booleans. The existing positive fixture itself uses `"f" * 40` as
the alleged pinned commit, so the test suite currently blesses this omission.

Relevant code: `tools/b4_authority.py:1268-1313` and
`tests/test_b4_materializer_authority.py:3433-3449` at `502c81d2`.

## Finding B4-C36: the launch gate is bypassable through public runtime APIs

Only `run_campaign(execute=True)` calls `require_v6_launch_open()`. The public
sequence `GlobalLock -> claim_attempt -> issue_lease -> execute_cell` does not.
`execute_cell` is also exposed by its own CLI with `--action execute --lease`.

This is not hypothetical test scaffolding: `_claimed_cell()` and the C31
terminal tests drive exactly that sequence without installing a recovery
acta. Therefore a valid claim and lease can enter construction or pipeline
code while the top-level launch gate remains closed.

Relevant code: `tools/b4_campaign_orchestrator.py:429-582,779-856`,
`tools/b4_campaign_executor.py:704-729,909-939`, and
`tests/test_b4_materializer_authority.py:3506-3647` at `502c81d2`.

## Finding B4-C37: recovered authority is absent from result custody

Generation v6 exists because amendment 12 and the future recovery audit
authorize a recovery from v5. Nevertheless, the per-attempt binding,
COMPLETED terminal schema, campaign verifier and final report still carry only
the historical campaign authorization and `amendment_11_sha256`.

Independent inspection of the live terminal schema produced:

```text
terminal_has_amendment_11 True
terminal_has_amendment_12 False
terminal_has_recovery_audit False
```

Thus a terminal can prove the old v5 lineage but cannot prove which recovery
review opened v6. The verifier re-derives the same stale pair, so internal
consistency does not repair the missing authority link.

Relevant code: `tools/b4_campaign_executor.py:513-526,779-793,847-867`,
`tools/b4_campaign_ledger.py:209-216,534-547`, and
`tools/b4_campaign_orchestrator.py:914-925` at `502c81d2`.

## Finding T2-C37: the primary estimand has the wrong declared sign

The productive calculation is:

```python
deltas.append(xm - am)
```

where `xm` is MASE for X and `am` is MASE for D. Positive values therefore
mean **MASE(X) - MASE(D) > 0**, i.e. D improved over X. The design and
validator instead call this `D_minus_X`, and the prose calls it a paired D-X
effect. The adjudicator then correctly treats positive values as beneficial.

A direct probe with `MASE(X)=1.0` and `MASE(D)=0.9` yielded:

```text
declared D_minus_X       = -0.1
implemented delta       = +0.1
```

The arithmetic and decision polarity agree with each other; the executable
estimand name does not. This is not cosmetic because the name is part of the
validated design and future publication contract.

Relevant code: `tools/t2_confirmatory.py:374-378,842-883,1067-1083`,
`tools/t2_design_draft_v5.py:226-249`, and
`tests/test_t2_harness.py:856` at `5737ad77`.

## Accepted portions

Subject to the narrow corrections above, the following work is accepted as a
base and must not be redesigned:

- B4 zero-write environment preflight, effective plugin provenance, typed
  post-claim failure classes, v5 incident preservation, fresh v6 generation,
  36-second prior charge, and v5-v6 scientific identity equality;
- T2 per-panel extreme-support threshold, single geometry authority, hospital
  inclusion without exception, six-panel population, exact cost schema, live
  byte re-derivation and no-score chronology.

## Disposition

1. B4 remains **GPU CLOSED**. Do not issue the recovery acta yet.
2. T2 remains **UNSEALED AND UNSCORED**.
3. Execute the narrow correction order. Preserve every prior amendment, draft,
   incident and result byte-for-byte.
4. After the corrected return, Musashi will independently review the new B4
   authority surface and T2 draft before issuing either acta.
