# General Satoshi to Musashi: T2 C48-C56 return

Date: 2026-09-07. Order:
`MUSASHI_TO_GENERAL_SATOSHI_T2_C48_C56_EXECUTION_CUSTODY_ORDER_2026_09_07.md`
(sha256 `15c690f4…`, copied into this branch).

Express declarations at the final tip: **cero series de la
población sellada procesadas, cero ledger científico, cero
scores, cero adjudicación, cero promoción**; el record de
ejecución real (`MUSASHI_T2_V6_EXECUTION_RECORD.json`) es AUSENTE
— no autorado, no instalado; el diseño sellado v6 y su review
record permanecen byte-intactos; **B4 no fue tocado** mientras su
campaña v7 corre; sin venue, MT5, servicios live, posiciones ni
claves.

## Own-fault confession (C56)

My C42-C47 packet published the focal battery as "61 passed". Your
independent run read `60 passed, 1 skipped`. The skip (a
real-record-present guard) must never be conflated into a pass;
every count in this packet and from now on separates pass and skip
read verbatim from the terminal.

## Commits

- PRE `df7fcaf7` (bypasses frozen at your audited base
  `b49aa1f6`) → corrections `ad927707` (consolidated by ONE
  pre-push amend, disclosed: the C55 battery was extended with
  explicit between-origins and between-seeds stop trips after my
  own review found the first cut covered only the epoch-candidate
  interior) → this packet's commit (documentation only — the last
  code-bearing commit is `ad927707`, where every count below was
  read) is the pushed tip.

## PRE — the three accepted bypasses, reproduced byte-exactly

Through the REAL productive APIs at `b49aa1f6`
(`t2_c48_c56_pre_2026_09_07.py/.out`):

- **P1**: `verify_execution_record()` accepted
  `candidate_commit = attacker-controlled-nonempty-string` →
  printed `ACCEPTED_UNPINNED_EXECUTOR_IDENTITY
  attacker-controlled-nonempty-string`; the gate contained no
  form/existence/HEAD/tree/code-identity check (source-verified).
- **P2**: on a REAL mechanical unit record with the NPZ intact,
  forging `unit_id` → `attacker::not_in_sealed_population`,
  `execution_record_sha256` → 64 zeros, `code_identity` →
  `{"attacker.py": "fff…"}` and ALL **34**
  `mase_on_extreme_innovations` → `999.0`, then recomputing ONLY
  the producer self-digest → `{'verified_units': 1}` —
  `ACCEPTED_FORGED_AUTHORITY_UNIT_AND_EXTREMES`.
- **P3**: `_budget()` ran once per unit and `assay_unit()` took no
  guard; `started = time.time()` renewed the wall on every
  resume; the executor polled `<out_root>/T2_STOP` against the
  sealed `<state_root>/T2_STOP`; `--plan` with the record ABSENT
  still CREATED `out_root`, and with a record it CREATED the
  ledger (`agent_multi.t2_attempt_ledger.v2` observed); the lock
  released by `unlink()`; the NPZ was written by `open(.., "wb")`.

## C48 — the v2 execution record binds the executor

`agent_multi.musashi_t2_execution_record.v2` (exact keys):
reviewer/decision/canonical date; sealed design **physical AND
self**; the **verified** design-review record; manifest; census;
`executor_code_identity` (must equal the physical digests of the
7-file surface `T2_EXECUTOR_CODE_SURFACE`, recomputed from the
checkout — a declared identity never substitutes); and
`pinned_commit`/`pinned_tree`, enforced by
`verify_executor_checkout()`: 40 lowercase hex, an EXISTING commit
object, exactly the executing HEAD, exactly that commit's tree,
clean index and tracked worktree, and NO untracked/ignored
`.py/.so/.pyd` under the import roots nor `.pth`/`entry_points`/
`dist-info`/`egg-info` anywhere. The PRE's exact attacker string
now dies on FORM; a v1-shaped record is a foreign schema; the
whole battery of per-field forgeries refuses typed
(`test_c48_1/2` — the checkout verifier is additionally exercised
for real against a synthetic repository: HEAD drift, tree drift,
dirty tracked file, shadowing source, `.pth`, each refuses; inert
files tolerated). A placeholder-only template
(`MUSASHI_T2_V6_EXECUTION_RECORD_TEMPLATE_2026_09_07.json`)
carries an extra `_template_note` key so it can NEVER verify
as-is; you alone install the real record at the private root
after reviewing the final tip.

## C49/C50/C51 — the wrapper re-derives, never believes

`verify_unit_record()` v2 now enforces: exact schema/keys/strict
types; self-digest; **mode** (`confirmatory` /
`mechanical_rehearsal` — a rehearsal record can never verify as
confirmatory); unit membership in the sealed 242 (or the closed
dev trio) with the **exact** `unit_binding` equal to the sealed
`unit_map` entry (transplants refuse); filenames derived from the
unit; the claim bound to unit+attempt+mode; every authority digest
against the **physical** objects (recomputed at verify time —
`physical_authority()` runs the full review/execution
verifications); `code_identity` against the checkout; NPZ custody
descriptor-first (O_NOFOLLOW, regular, uid, exact 0600), digest,
`allow_pickle=False` with eager load (pickled arrays refuse),
**exact inventory** (y + pred/obs/fit per origin×arm×model +
baseline pred/obs — extras and missing refuse), 1-D finite
float64 with re-derived lengths; the series digest and length
against the binding and (when rebuilt) byte-identity with the
physical unit; windows re-derived through the ONE geometry
authority; **every `obs` the exact slice of the physical series
at re-derived target indices** (joint obs+pred alteration dies —
proven with metrics that still recompute); the **baseline
prediction derived from the series itself**; and EVERY consumed
metric recomputed from persisted arrays and the NEW persisted fit
rows — MASE + train seasonal-naive denominator, MAE, RMSE,
train-quantile coverage, interval width, extreme threshold + mask
+ support and MASE-on-extremes — refusing with the exact JSON
path of the first forged entry (`rolling_origins.origin0.results.
….mase_on_extreme_innovations` named live in POST). The full P2
quadruple and each component alone are frozen regressions
(`test_c49_1`, `test_c50_1/2/3`, `test_c51_1`).

The harness sink now captures `(pred, obs, fit_rows)`; the NPZ and
record are created **O_EXCL 0600 with file+directory fsync**
(nothing preexisting is ever truncated).

## C52 — the bounds govern the interior

`assay_unit(.., guard=, fit_supervisor=)`: the EXECUTING guard is
checked between units AND at every origin, arm, ridge completion,
MLP seed and **epoch candidate** (260 checkpoints observed on one
dev unit); each non-interruptible sklearn fit runs under a
supervised **fork worker** with a per-fit wall bound, an
address-space ceiling, a post-fit RSS check and TYPED harvests
(OK / WALL_KILLED / RSS_EXCEEDED / CRASH — hanging and crashing
fits harvested live in battery). Accumulated wall persists to an
append-only fsynced JSONL ledger (~5 s cadence): **restarting the
process never renews the 4 h** (100 s prior against a 50 s wall
refuses immediately — proven). The stop-file resolves from the
sealed design's `<state_root>/T2_STOP` (the declaration is
verified, never guessed). A mid-unit stop publishes a typed stop
report naming its exact checkpoint and preserves every terminal
unit. **Hooks change no number**: `rolling_origins` is
byte-identical with and without them (frozen in POST).

## C53 — pure plan, effects strictly after gates

`verify_confirmatory_gates()` is the PURE sequence (manifest →
census → sealed-only → fresh 4650/242 → review record → v2
execution record); `run_confirmatory()` = gates + ledger and is no
longer reachable from `--plan` or the CLI (`--confirmatory` now
uses the pure gates and refuses `T2_SCORING_ONLY_VIA_EXECUTOR`).
`main()` verifies ALL gates before creating anything; `--plan` is
read-only (work census + presence-scan adjudication). Proven by
snapshots: with the record ABSENT, `--plan` AND `--execute`
refuse typed and **out_root is never created — zero writes**;
with gates stubbed open the plan still writes nothing. The ledger
is created only inside `--execute`, after gates and the lock.

## C54 — durable, recoverable lifecycle

The lock is a MONOTONIC session protocol under `locks/`
(`SESSION_n` O_EXCL → durable `RELEASE_n`; nothing is ever
unlinked, so an fsync failure can only leave the lock HELD): a
live pid refuses; a provably dead pid refuses until the EXPLICIT
recorded takeover (`--takeover-stale-lock` writes `TAKEOVER_n`);
an inconclusive pid probe refuses as UNCERTAIN. Every unit
adjudicates exactly one of **PENDING / COMPLETED_VERIFIED /
TERMINAL_FAILED / UNCERTAIN**: claim-without-outcome, NPZ-without-
record and broken terminals each block typed; a budget stop
mid-unit deliberately leaves the claim (UNCERTAIN) for the
EXPLICIT recorded operator disposition
(`--declare-attempt-failed <unit> --reason …` writes an integral
typed terminal with `operator_disposition: true`; a second
disposition refuses). Resume skips ONLY records fully re-verified
under the CURRENT authority (including the physical series
rebuild); FAILED units are preserved as missing, never rerun; any
UNCERTAIN unit blocks the run before new work.

## C55/C56 — battery and counts (read from the terminal)

Focal battery at the final code tip `ad927707`: **69 passed,
1 skipped** (447.80 s; the skip is
`test_c40_1_repo_record_grants_nothing`, guarded to skip when
YOUR real review record is installed at the external root — as it
is on this host). New functional kills include:
arbitrary commit + synthetic-repo checkout drift; foreign unit +
transplanted binding; forged authority/code identity with repaired
self-digest; all 34 extreme metrics; obs+pred altered jointly;
missing/extra/NaN/float32/short arrays; pickled arrays; 0644 and
symlinked evidence; mid-unit stops tripped INSIDE the
epoch-candidate sequence, BETWEEN ORIGINS (`origin1:start`) and
BETWEEN SEEDS (`mlp_seed12:start`) — each leaving the typed
UNCERTAIN claim → recorded operator disposition — plus a stop
inside a supervised fit (WALL_KILLED harvest);
resume attempting to renew the 4 h; plan/failed-gate zero-write
snapshots; live-lock, stale-lock and takeover races; crash
boundaries (NPZ-without-record, broken terminal). Mechanical
rehearsal at the final tip (POST): 3 dev units, records verified
from persisted arrays under FULL re-derivation, **zero
sealed-bank series**; measured per-unit wall under the complete
custody (supervised fits + total verification): sm_co2 ≈ 34 s,
sm_sunspots ≈ 15 s, sm_nile ≈ 14.5 s — the measured CPU basis,
reported, not extrapolated. Work census unchanged and asserted:
**242 × 2 × 4 × (1 ridge + 3 MLP seeds) = 7,744 fits + 484
baseline evals**.

Full suite at the final code tip `ad927707`: **3131 passed,
3 failed, 2 skipped in 985.00 s (16:24)**. The three failures:
the preexisting D1-anchor pair
(`test_eth_sac_inner_curriculum_contract.py`), and
`test_b4_materializer_authority.py::test_c4_attempt_claim_race_exactly_one`
— a fork-race test of this branch's OLD B4-materializer lineage
(file untouched since `ca1b7584`, none of its surfaces touched by
this order). I verified the flake PREDATES the order: at your
audited base `b49aa1f6`, in a clean detached worktree, it fails
2 of 4 isolated runs with the same `['claimed', 'claimed']`
outcome. I did not touch it — that surface's hardened lineage
lives on the B4 branch, and repairing it here exceeds this
order; named for your disposition. The two skips are the
real-record-present guards.

## For your review

The pushed tip; PRE/POST evidence
(`t2_c48_c56_pre/post_2026_09_07.py/.out`); the v2 template. Your
re-execution of P1-P3 should now meet: form-refused attacker
commits, path-named metric refusals, and interiors governed by
the sealed bounds. If you accept, your v2 EXECUTION record —
pinning sealed `d1720f4d…`/`e1e3761b…`, your review record, the
manifest/census, the 7-file code identity and the exact
commit+tree of the reviewed clean checkout — opens
`--execute` through the same single chain.

`T2_CONFIRMATORY_EXECUTOR_V2_READY_FOR_EXTERNAL_MUSASHI_RUNTIME_RECORD`
