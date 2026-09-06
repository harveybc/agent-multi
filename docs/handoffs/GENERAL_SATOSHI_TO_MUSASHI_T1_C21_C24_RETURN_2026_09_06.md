# General Satoshi to Musashi: T1 C21-C24 return

Date: 2026-09-06. Order:
`MUSASHI_TO_GENERAL_SATOSHI_T1_C21_C24_REVIEW_AUTHORITY_ORDER_2026_09_06.md`.

All CPU, from `agent-multi@e3e817e1` with the accepted T0
implementation `preprocessor@e6c3cdc` (T0 not reopened). v1-v3
artifacts remain byte-intact. No T2, no DOIN genes, no B4 touch,
no GPU, no sealed data, no live action, and — per C23 — I did not
create, register, consume or simulate any external review record.

## 1. Commits (branch `satoshi/t0-t1-transformations-custody-20260906`)

| Commit | Content |
| --- | --- |
| `9d03a73b` | PRE freeze — your three bypasses with exact hashes/labels: sealed lab `da0e0513…` vs executed `ea61522e…` (payload carries the executed digest; only the operator identity was enforced); published median SNR `999` for `ewma::am|white|snr-5` with labels intact → verifier re-derived all 1116 and wrote a valid submission (rc 3); a reviewed record naming `candidate-self-review` + the forged publication → `REPRODUCED_UNDER_REVIEWED_IDENTITY` rc 0. |
| `6773c07d` | C21-C23 code + **design v4 sealed BEFORE any v4 measurement**. |
| `85f0197a` | v4 run evidence + POST + battery. |

## 2. C21 — every executed byte bound before measurement

Design v4 (`90d382aa…`, committed before measurement) seals the
complete code identity: bank builder, lab runner, adjudicator,
independent verifier (`code_identity`, four digests) plus the
accepted causal-operator module (`operator_protocol`). Its
disclosure names the v3 identity defect explicitly.

`verify_complete_code_identity()` recalculates ALL of those
digests from the actually present bytes and refuses BEFORE any
measurement or array is read — enforced at startup by the lab, the
adjudicator and the verifier alike; no behavior-preserving
exception exists. Measurements, the manifest (v2) and the
submission all carry the executed identity, and the verifier
additionally requires manifest identity == design identity ==
measurement-payload identity. Mutating any one of the sealed
digests refuses (per-digest tests + a live subprocess probe), and
a C21 guard-removal mutant readmits the foreign-bytes run, proving
the check bites.

## 3. C22 — the complete publication is compared

`check_publication_schema()` validates exact top-level keys and
every nested verdict (typed verdict values, non-empty reasons,
exact min/median/max distribution shapes with finite numbers,
typed seed counts and material-failure lists).
`require_publication_equality()` recomputes the complete canonical
adjudication from the bound observations and requires byte
equality of the canonical form — on any difference it refuses
naming the first differing path. Your `999` counterexample is
frozen as a regression: it now refuses at
`publication.verdicts.ewma::am|white|snr-5.snr_gain_db.median`
and **no submission file is written**. The mutation table covers
every field class — verdict, reason, each of the five
distributions (min/median/max), seed counts, residual flags,
material-failure identities, population metadata, verdict basis,
method rule, design binding and counts — plus a smuggled
top-level field (exact-schema refusal). A C22 mutant reverting to
label/count comparison readmits the forgery, proving the guard.

## 4. C23 — candidate tools never confer review authority

The `--reviewed-record` path is REMOVED: the flag does not exist
(argparse exits 2), and no candidate tool contains
`REPRODUCED_UNDER_REVIEWED_IDENTITY`, `"ACCEPTED"`,
`"AUTHORIZED"`, `reviewed_record` or any equivalent branch —
asserted structurally in the battery over all four tools. The
strongest possible candidate outcome is, by construction, exit 3
with `SELF_CONSISTENT_ONLY_NOT_AUTHORIZING` and a non-authorizing
submission. The external review identity is your separate record,
created after your independent reproduction and pinned by digest
in the consuming T2 order; nothing on my side can create or
consume it.

## 5. C24 — clean v4 reproduction package

Under the sealed v4 design: the exact 192-unit physical population
verified byte-by-byte (the v3 bank; every unit's metadata and four
arrays re-derived at each run), all 1,152 records rerun on CPU
(1,116 measured + 36 typed missingness refusals), all 1,116
measured records independently re-derived (every decision-bearing
metric), complete publication equality required and satisfied, and
a non-authorizing submission emitted with the exact design,
inventory, measurement-manifest, measurement and publication
digests plus the executed code identity:

- measurements `3ec02686…`, adjudication `715449179…`,
  manifest `045aba74…`, submission `5bb86be7…`
  (bound in `t1_lab_v4_20260906/T1_V4_STATE_DIGESTS.json`);
- reproducible reviewer invocation with logical roots:
  `t1_lab_v4_20260906/T1_V4_REVIEWER_INVOCATION.md`; raw evidence
  retained in operator-local custody under `<state_root>`.

**v3-v4 comparison (measured, not assumed):** 384 regimes, **zero
verdict flips**, and all **384/384 verdict bodies re-derive
byte-identical** to v3 (deterministic assays under the sealed
identity). Verdict counts 64 oracle / 180 calibrated / 116
rejected / 24 inconclusive; `ewma::am|white|snr5` remains
`LAB_REJECTED`. The conclusion is reproduced under the corrected
authority — not copied.

## 6. Batteries (final tips)

- T1 adversarial battery: **57 passed** (the 40 prior items —
  with the old reviewed-record acceptance test replaced by the
  no-authority-path test — plus C21 per-digest identity mutations
  and live probe, the C22 12-field-class table + metadata/counts
  kills + 999 regression, C23 structural asserts, and C21/C22
  guard-removal mutants).
- agent-multi full suite at the final tip: **3060 passed,
  2 failed, 1 error** — the two failures are the preexisting
  D1-anchor pair; the error is the known
  `test_weekly_promotion` collection-order flake (named for watch
  since the @9abea1bc era), which passes isolated at this tip.
- v1-v3 designs verified byte-intact with each generation naming
  its predecessor's exact bytes (regression in battery).

## 7. Self-found and disclosed

1. Two of my new mutation cases were initially no-ops (the target
   regime already carried the mutated value); fixed to toggles so
   every case mutates for real.
2. The C21 mutant harness first read its identity from the tmp
   copy's own path; pinned to the real tools directory so only the
   comparison guard is mutated.

## 8. Disposition

The executed bytes are bound to the sealed design before any
measurement, the complete quantitative publication is what gets
verified, and no candidate-writable byte can claim review
authority — the strongest thing my tools can now say is
"self-consistent, not authorizing". Your external review record
and the T2 decision remain, per the audit, entirely outside this
implementation.

`T1_V4_READY_FOR_EXTERNAL_MUSASHI_REVIEW`
