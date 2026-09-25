# Musashi to General Satoshi: B4 C35-C38 and T2 C37 order

Date: 2026-09-07

## Authority and priority

Execute two narrow CPU-only corrections. B4 is P0 and T2 is P1. The owner's
campaign intent remains recorded, but this order does **not** authorize GPU,
scientific B4 cells, sealed-2025 access, T2 scores or a T2 scientific ledger.

Inputs:

- B4 packet `9d4aec7fc2cb6ce35f7a2236974de251f0f4607a`;
- T2 packet `0e20e497b75df92d695b07b7ad036811ff23d76b`;
- audit
  `docs/audits/MUSASHI_AUDIT_B4_C29_C34_AND_T2_C31_C36_2026_09_07.md`.

Preserve amendment 12, generation v6, draft v5 and all previous evidence as
immutable history. Corrections append; they do not rewrite.

## P0 PRE: freeze the three B4 bypasses

Before editing, add executable regressions that reproduce all three facts:

1. `require_v6_launch_open()` accepts `pinned_commit` equal to `None`, `False`,
   a traversal-like string and forty zeroes, together with a malformed date.
2. Make `require_v6_launch_open()` raise unconditionally, then use the public
   claim/lease/direct-executor sequence. Prove that current code still reaches
   a post-claim typed construction or plugin terminal instead of refusing at
   the recovery gate.
3. Show that a v6 COMPLETED terminal and campaign report can pass their current
   schemas with `amendment_11_sha256` while omitting amendment 12, recovery-acta
   digest, pinned execution commit and v6 generation.

Each PRE must exercise the real public API or exact productive schema. Keep
the outputs and test identities in the return packet.

## B4-C35: make the recovery record a verified object

Create one descriptor-bound reader for the recovery acta:

- open once with `O_NOFOLLOW`;
- verify regular file and owner from that descriptor;
- reject group/world-writable mode;
- read and hash the complete bytes from the same descriptor;
- parse those exact bytes with duplicate-key and non-finite rejection;
- require the exact schema and exact primitive types;
- require a canonical ISO date;
- require `pinned_commit` to be 40 lowercase hexadecimal characters naming an
  existing Git commit;
- verify that the reviewed execution surface at that commit matches the live
  surface being admitted, under an explicitly declared finite file set;
- require the latest recovery amendment digest, not merely an older link.

The function must return a typed witness containing at least the acta's
SHA-256, pinned commit, latest amendment SHA-256 and campaign generation. A
reviewer label or a set of booleans alone grants nothing.

Do **not** author, copy, synthesize or install the real Musashi recovery acta.
Production remains closed. Positive tests may use an isolated fixture whose
authority digest is injected only by test setup; no productive `TEST_ONLY`
entry point may ship.

## B4-C36: consume the witness on every execution path

There must be no public sequence that reaches a claim, lease, constructor,
pipeline or CUDA while the recovery gate is closed.

- `run_campaign(execute=True)`, `claim_attempt`, `issue_lease`, `verify_lease`
  and `execute_cell` must all require or re-derive the same recovery witness.
- The standalone executor CLI must refuse `--action execute` without it even
  when handed an otherwise valid lease.
- A witness supplied by a caller is never sufficient by itself; productive
  code re-derives it from the reviewed object at the last point of use.
- Dry-run and environment diagnosis may remain zero-write and non-authorizing.

Add a call-graph or structural regression proving that every model/CUDA entry
point is dominated by this check, plus behavioral tests for the direct API and
standalone CLI.

## B4-C37: bind recovered authority into custody

Append a new amendment after amendment 12 with `scientific_change: NONE` that
records only C35-C37 and pins the final corrected authority/execution/test
surface. Earlier amendments remain byte-identical.

The verified witness must flow through:

- claim and execution lease;
- per-attempt authority binding;
- COMPLETED and typed-failure terminal schemas where applicable;
- seal intent/completion or their verified parent binding;
- final campaign report;
- single-cell and full-campaign verification.

At minimum every successful scientific result must bind and the verifier must
re-derive: v6 generation, recovery-acta SHA-256, pinned execution commit and
the latest amendment SHA-256. A terminal carrying only amendment 11 or 12 must
refuse under the new generation. Exact schemas must reject missing, extra,
transplanted and self-rehashed values.

## B4-C38: acceptance battery and stop point

Required adversaries:

- malformed date and every non-string/invalid commit form;
- nonexistent commit and commit whose reviewed surface differs;
- absent, symlinked, non-regular, permissive or swapped acta;
- acta with correct labels but altered bytes;
- direct `execute_cell`, direct lease issue and standalone CLI with gate closed;
- lease/terminal/report missing or transplanting each recovery binding;
- old amendment-11-only and amendment-12-only terminals;
- superseded v5 objects and the preserved ambiguous attempt.

Re-run the 166-test B4 battery and the integrated CPU-double campaign under a
fixture witness. Return a candidate correction and reviewer template only.
Stop with the real recovery acta absent and **zero GPU execution**.

## P1 PRE: freeze the T2 sign contradiction

Through `_series_stats`, set `MASE(X)=1.0`, `MASE(D)=0.9`. Freeze that current
code computes `+0.1` while draft v5 and `validate_confirmatory_design()` call
the quantity `D_minus_X`, whose mathematical value is `-0.1`.

Also freeze that the adjudicator treats the implemented positive value as
improvement. This proves the defect is the declared estimand, not the decision
polarity.

## T2-C37: one unambiguous estimand

Keep the intended arithmetic `MASE(X) - MASE(D)`, because positive then means
that D reduces error. Rename it everywhere in the productive contract to an
unambiguous term such as:

`mase_improvement_X_minus_D`

Update the validator, adjudicator fields, current prose, current tests and
state-digest record. Do not globally replace `[X,D,X-D]`: that is a feature
representation, not the MASE contrast. Preserve drafts v2-v5 byte-identical
as historical records and append draft v6 superseding v5 by exact digest.

Required polarity tests:

- X=1.0, D=0.9 -> `+0.1`, beneficial;
- X=0.9, D=1.0 -> `-0.1`, harmful;
- width-control attribution uses the same orientation;
- old `D_minus_X`, ambiguous `delta`, and opposite-polarity designs refuse;
- mutation from `xm - am` to `am - xm` breaks the battery.

Re-run the 52-test focal battery after adding the new tests. Do not seal draft
v6, run confirmatory scores or create a scientific ledger.

## Return

Return one packet with:

- PRE and POST outputs for every bypass;
- exact append-only maps amendment12 -> next amendment and draft5 -> draft6;
- mutation results read from terminal output before prose is written;
- final focal and full-suite counts from the final pushed tips;
- explicit declarations: no GPU, no B4 scientific cell, no sealed-2025 read,
  no T2 score, no T2 seal and no scientific ledger.

Final dispositions must remain:

- `B4_V6_RECOVERY_AUTHORITY_READY_FOR_EXTERNAL_MUSASHI_REVIEW`;
- `T2_SCREEN_V6_READY_FOR_EXTERNAL_DESIGN_REVIEW`.
