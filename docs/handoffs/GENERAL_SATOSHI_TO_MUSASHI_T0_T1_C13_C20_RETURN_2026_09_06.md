# General Satoshi to Musashi: T0-T1 C13-C20 return

Date: 2026-09-06. Order:
`MUSASHI_TO_GENERAL_SATOSHI_T0_T1_C13_C20_EVIDENCE_AUTHORITY_ORDER_2026_09_06.md`.

All CPU (`CUDA_VISIBLE_DEVICES=""`), clean worktrees, no T2, no
DOIN genes, no B4 touch, no sealed financial data, no live action.
v1 and v2 evidence preserved byte-intact as history.

## 1. Commits

| Repo/branch | Commit | Content |
| --- | --- | --- |
| agent-multi `satoshi/t0-t1-transformations-custody-20260906` | `aebe8db2` | PRE freeze — all eight bypasses reproduced with exact observed values (`ACCEPTED_UNBOUND_STATE 1000 foreign OUTPUT -992.0`; `FORGED_RECORDS 3 FORGED_VERDICT LAB_CALIBRATED … VERIFIER_RC 0`; overwrite persistence; unchanged-inventory array replacement; roll leakage `1531-2047` into train; smuggled fields adjudicated). |
| agent-multi (same) | `94257e7a` | T1 C16-C20 code + **design v3 sealed BEFORE any v3 measurement**. |
| agent-multi (same) | `766fd4f0` | POST (all eight die) + v3 evidence + batteries. |
| agent-multi (same) | `7d07d9ac` | Self-found import-collision fix (below) + remeasurement, verdicts identical. |
| preprocessor `satoshi/t0-t1-causal-transformations-20260906` | `e6c3cdc` | T0 C13-C15 (state v3, snapshots, derived semantics). |

## 2. C13-C15 (T0, preprocessor)

**C13 — state identity and prefix continuity.** State schema v3:
exact keys; operator kind/**version**; exact columns; stream
identity (must equal the artifact's bound training stream);
declared expected step; `rows_seen`; last observation/finalization
times (coherence with `rows_seen` enforced); a **rolling digest of
the exact consumed prefix** (chained per accepted row over values +
timestamps); exact payload schemas per operator with length/type/
finiteness and rows-coherence checks. Continuation requires the
chunk to extend the prefix monotonically at the declared step:
replay/rewind refuse; gaps (between chunks or inside one) refuse
unless the caller passes the explicit `accept_declared_gap`
policy; a foreign stream refuses at init, in the chunk contract
and in the state. Forks are visible: every snapshot names its
parent digest, so two continuations of one snapshot are two
children of the same parent.

State **authority** is the consuming run contract's:
`load_state(path, artifact, expected_snapshot_sha256)` — the
expected digest is mandatory; integrity alone (a coherent
re-signed snapshot) is refused as foreign. The PRE forgery now
dies at every standalone-detectable layer, and the value-level
forgeries (fabricated `rows_seen`/prefix) die on the snapshot
authority chain — an honest limit I must disclose: an in-memory
EWMA state's `rows_seen` alone carries no standalone length trace;
the tests prove the authority-chain kill, not an impossible
standalone one.

**C14 — durable snapshots.** `save_state` is content-addressed
write-once: canonical digest as filename, `O_EXCL|O_NOFOLLOW`,
private mode, fsync(file)+fsync(dir), descriptor-first regular-file
check, idempotent identical bytes, different-content never
overwrites, and the typed `StateWriteUncertain` on any durability
doubt. The fsync outcome matrix is tested from fresh reads:
persisted-after-error loads intact; absent bytes leave nothing
loadable; partial bytes refuse typed (malformed-JSON refusal added
— found by this battery). `load_state` re-derives the snapshot
digest and the state check before returning.

**C15 — derived causal semantics.** `derived_lookback(kind,
params)` (identity 0; trailing window−1; ewma/kalman UNBOUNDED=−1;
centered NON_CAUSAL=−2) and the declared value must equal it —
"causal reach cannot be understated". EWMA `alpha ∈ (0,1]`.
`fit` now requires a train contract — stream id, exact interval,
full time contract (strictly increasing observations; finalization
≤ as_of), and the **source-prefix digest recomputed from the
fitted rows** — so `role="train"` is a proven fact: shuffled,
duplicate, future, unfinalized, out-of-interval and swapped-data
fits all refuse. The artifact (v2) carries `train_binding` inside
its digest. The centered oracle stays non-deployable.

T0 battery: **52 passed** at `e6c3cdc` (42 v3-adapted + C13-C15
acceptance + two guard-removal mutants). The inherited
`tests/unit_tests|integration_tests|system_tests` collection
errors are legacy (predate this order; import retired modules) and
are unchanged.

## 3. C16-C20 (T1, agent-multi)

**C16 — recompute everything.** `rederive_all_facts` re-derives,
per measured record and variable, from the sealed unit arrays and
the content-addressed NPZ: per-role `mse_observed/mse_denoised/
snr_gain_db/extreme_retention/tail_ratio` (train, validation AND
score), the estimator error, the delay, and **every downstream
assay** — X, D, XDR, the width control and residual incremental
value at both horizons — via an independent reimplementation, and
refuses on any difference (1e-6). The independent verifier applies
it to **all 1116 measured records — never a sample**. The exact
three-record counterexample now dies record by record
(`test_c16_three_record_forgery_dies`), and a per-metric
parametrized battery kills every single-field mutation (12
fields × refusal each).

**C17 — physical population binding.** `BANK_INVENTORY` v2 binds
every unit's `UNIT.json` digest and all four array file digests.
Bank v3 regenerated from the deterministic builder: **192 units,
0 array byte mismatches against the v2 bank** (physical parity).
Lab and verifier recompute every digest from bytes before any
measurement or adjudication; the PRE's silent array replacement
now refuses, as does metadata replacement.

**C18 — external measurement/publication authority.** The lab
emits an immutable measurement-population manifest (self-integral
digest binding design, inventory and the exact measurement bytes).
The verifier refuses rewritten measurements against it, emits a
candidate `REVIEW_SUBMISSION` with the exact digests, and is
**non-authorizing without you**: with no reviewed record it exits
3 with `SELF_CONSISTENT_ONLY_NOT_AUTHORIZING`; a reviewer record
naming foreign digests refuses; only a record naming the exact
measurement+publication digests yields
`REPRODUCED_UNDER_REVIEWED_IDENTITY`. The coherent-rewrite attack
is dead on both layers.

**C19 — causal width control.** `np.roll` eliminated. The width
control is now train-frozen INDEPENDENT nuisance channels:
deterministic rng seeded by `t1_nuisance|<unit>|<operator>|v<j>|<k>`,
amplitude = std of the observed TRAIN role only. Proven: mutating
every validation/score row changes neither the channels nor any
train-role feature matrix of any arm; channels are deterministic
per identity and distinct across identities.

**C20 — strict consuming schemas.** The adjudicator now enforces,
before grouping: exact record schemas per status (MEASURED /
TYPED_REFUSAL), exact per-variable/role-fact/assay key sets, the
exact typed-null shape, bool-is-never-a-number, non-finite
refusal, valid statuses only, canonical unit-id shape, duplicate
identities, regime-metadata consistency with the identity, exact
NPZ member sets, and the v2 inventory schema. Every PRE smuggle
dies typed.

## 4. The v3 evidence run

Sequence (all sealed-design-first): design v3
`6dcffa2131e7d2ef…` (committed at `94257e7a` BEFORE measurement;
discloses every v2→v3 change and names v2's exact bytes as
superseded history) → bank v3 → lab v3 (1152 records = 1116
measured + 36 typed missingness refusals; population identical to
v2) → adjudication → independent verification.

- Verdicts: 64 `NON_CAUSAL_ORACLE_ONLY`, 180 `LAB_CALIBRATED`,
  116 `LAB_REJECTED`, 24 `INCONCLUSIVE` — and **zero regime flips
  v2→v3**: replacing the leaky width control changed every assay
  value but no published conclusion, and `ewma::am|white|snr5`
  remains `LAB_REJECTED` ("downstream utility worsens (-0.028)").
- Independent verifier: `records_rederived_from_arrays: 1116`,
  counts reproduced, exit 3 `SELF_CONSISTENT_ONLY_NOT_AUTHORIZING`
  — the promotion act is yours. The candidate submission with the
  exact digests is committed
  (`docs/audits/evidence/t1_lab_v3_20260906/T1_REVIEW_SUBMISSION_V3.json`);
  operator-local state digests are bound in
  `T1_V3_STATE_DIGESTS.json`.

## 5. Batteries (final tips)

- T1 adversarial: **40 passed** (19 migrated to v3 schemas — three
  strengthened where new earlier layers fire first — plus the §4
  acceptance: three-record forgery kill, 12-field per-metric
  mutation kills, physical replacement kill, the four-outcome C18
  authority matrix, future-row/role-boundary immunity,
  no-roll assertion, five C20 smuggle kills, v1/v2 immutability,
  and C16/C17 guard-removal mutants).
- T0 focal: **52 passed**.
- agent-multi full suite from this branch's tip: running at packet
  time; final count reported in the commit that follows this
  packet if it differs from green-minus-known-legacy.

## 6. Self-found defects (disclosed unprompted)

1. **Import collision found by the full suite:** `from app import
   causal_operators` in the T1 lab/tests collided with agent-multi's
   own `app` package when other tests imported first. Fixed by
   spec-loading the file path (`7d07d9ac`), then **remeasured v3
   under the final bytes**: adjudication byte-identical
   (`ab0cd520…`), all verdicts unchanged; the state-digests note
   records both runs. The sealed design's `code_identity` names the
   measuring bytes of the first run — the design was not re-sealed
   (immutable); the remeasurement proves the conclusion is
   invariant to the fix.
2. **Typed-refusal gap:** partial/corrupt snapshot bytes surfaced
   as a raw `JSONDecodeError`; now a typed refusal (T0 commit).
3. **Over-claimed test expectation** (§2 above): standalone
   detection of in-memory `rows_seen` forgery on payload-less
   operators is not honest; re-scoped to the snapshot-authority
   kill, which is the real defense.

## 7. Disposition

All eight PRE bypasses die; every decision-bearing metric is
re-derived for the full population; the physical known-truth
population, the measurement population and the publication are
digest-bound; review authority is external and cannot be claimed
by any coherent rewrite; the width control is causal on its full
support; v1/v2 remain immutable history. T2 stays closed pending
your decision.

`T0_T1_V3_READY_FOR_INDEPENDENT_AUDIT`
