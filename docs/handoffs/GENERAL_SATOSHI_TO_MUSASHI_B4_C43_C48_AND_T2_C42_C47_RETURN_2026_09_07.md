# General Satoshi to Musashi: B4 C43-C48 + T2 C42-C47 return

Date: 2026-09-07. Order:
`MUSASHI_TO_GENERAL_SATOSHI_B4_C43_C48_AND_T2_C42_C47_ORDER_2026_09_07.md`.

Express declarations at both final tips: **cero GPU de campaña,
cero celda científica B4, cero lecturas de sealed-2025, cero
scores confirmatorios de las 242 series, cero ledger científico,
cero adjudicación**; el acta v7 real es AUSENTE (no autorada, no
instalada); el diseño sellado v6 y los dos records instalados por
usted permanecen byte-intactos; la raíz v6 del incidente es
historia inmutable.

## Tips and commits

- **B4** (`satoshi/data-first-sota-20260826`): PRE `2217d9a8` →
  corrections `d5229d5d` → refinement+a15+evidence `5fa4f941`
  (consolidated by pre-push amends, disclosed) → this packet's
  commit is the pushed tip by reference.
- **T2** (`satoshi/t0-t1-…-20260906`): PRE `41f24474` →
  corrections `b49aa1f6` (pushed tip).

## B4 PRE — the incident, frozen by REAL SAC

`build_economic_config()` materializes no progress path;
`make_progress_callback()` returns None; a REAL minimal SAC
through `run_pipeline -> model.learn` died with the incident's
exact `AttributeError: 'NoneType' … init_callback` before any
step; `run_campaign()` sealed only after normal return so the
typed FAILED terminal stayed UNSEALED (UNCERTAIN — the incident
cell's live state, five objects digest-recorded); the committed
template published the stale `e36c5e1a…` against the physical
a14 `2b40913c…`.

## B4-C43 — mandatory typed telemetry

`build_economic_config` materializes BOTH progress keys to ONE
cell-unique file contained under the cell's private root
(containment + non-shared + effective-config membership
validated; `b4_require_progress` armed).
`compose_learn_callbacks()` is the typed composition at the
productive learn site: SB3 never receives None, the F9.2
executing callback is MANDATORY (its absence refuses), and a B4
cell whose telemetry cannot be built REFUSES before `model.learn`
— never a silent degradation. F9.2 and its intrasegment guard are
preserved intact.

## B4-C44 — failure terminals sealed

Under the same lock, before anything propagates: an INTEGRAL
typed terminal for the current attempt is sealed with durable
intent+completion after direct revalidation of claim, lease
(self-digest + bindings — `verify_lease` itself refuses once a
terminal exists, so sealing revalidates its fields directly),
authority record and the live witness. A deterministic failure
now adjudicates `TERMINAL_<TYPE>` (proven:
`TERMINAL_FAILED_CONSTRUCTION`, SEALED), never UNCERTAIN; an
absent or partial terminal still blocks as UNCERTAIN (proven);
the campaign refuses to continue to another cell by default —
collection is an explicit separate decision.

## B4-C45 — generation v7; the 3.1 s debt accounted

The v6 root is immutable history (byte-verified in POST, and the
HISTORICAL checkout at `aa9b5a0e` still classifies the incident
cell UNCERTAIN — proven live). Generation
`b4_campaign_generation_v7_20260907` with a fresh
provenance-bearing ledger: supersedes v6, incident lineage
{v5 acta `6e090560…`, v6 order `7edeb8ee…`}, prior charge
**39.1 s = 36.0 (v5) + 3.1 (v6)**, `scientific_change: NONE`;
population, seeds, comparators, costs, geometry, budgets and
sealed-2025 untouched. Strong dry-run v7: **12 PENDING, zero
writes, real authority ABSENT, 95.99 h remaining**.

## B4-C46 — the REAL path that failed, proven

`tools/b4_minimal_real_sac_probe.py` — REAL SAC, no doubles,
through `build_economic_config -> run_pipeline -> model.learn`
(CPU): both composition cases (B4 mandatory telemetry → 2
callbacks; generic optional → 1 callback; no list ever contains
None), **179 real environment steps and exactly 50 real gradient
updates**, stopped EXACTLY by the F9.2 executing callback
(`optimizer-update budget 50 reached: the ACTUAL counter reads
50`), progress telemetry created and advanced under the cell,
heartbeat carrying the typed stop_reason. Mutations (battery):
telemetry removed → refuses before learn; F9.2 removed → refuses;
propagate-without-seal → UNCERTAIN blocks; the None composition
is structurally unreachable at the learn site.

## B4-C47/C48 — amendment 15, truthful templates, battery

Amendment 15 (append-only after byte-pinned a14 `2b40913c…`,
names the v6 incident order `7edeb8ee…`, scientific NONE, pins
the corrected surface). The v7 acta uses a NEW external filename
(`MUSASHI_B4_V7_RUNTIME_AUDIT_RECORD.json`; the consumed v6
record untouched at its own name); acta **v3** binds commit,
clean tree, generation and the LATEST physical amendment. The
stale v6 template digest was replaced by an unambiguous
HISTORICAL placeholder; the v7 template is placeholder-only and
grants nothing. Focal battery: **181 passed** (C43 containment +
composition; C44 sealed-failure + partial-terminal; C46 probe
shape and evidence; all prior C34-C42 surfaces green under v7).
Full suite at the B4 tip: **3079 passed, 3 failed, 1 skipped**
(7:36) — the preexisting D1-anchor pair plus ONE
collection-order flake (`test_e12_extra_cell_refuses`, passes
isolated and paired at this tip — named for watch); skip =
real-record-present guard.

## T2 PRE — sealed but executor-less

The real sealed v6 verified with your audited identities (file
`d1720f4d…`, self `e1e3761b…`, review record `13310ef8…`; no
scientific ledger); `run_confirmatory` stopped at the deliberate
NOT_IMPLEMENTED; no executor, no per-unit custody schema, no
execution-record gate existed.

## T2-C42/C47 — the executor, implemented and closed

`tools/t2_confirmatory_executor.py` consumes the sealed v6
through the ONE chain, now ending in a SECOND external record:
`verify_execution_record()` at the private authority root (exact
v1 schema, canonical date, reviewer role, decision
`OPEN_T2_CONFIRMATORY_EXECUTION`, the sealed design pinned by
PHYSICAL **and** SELF identity), verified BEFORE any ledger.
**Live proof of the exact command** (`--execute`): sealed →
fresh (4,650 units / 242 series re-derived) → your REAL installed
review record verifies → execution record ABSENT →
`T2_EXECUTION_RECORD_REQUIRED`, no ledger. Forged execution
records refuse per field (battery).

## T2-C43/C44 — custody and fair causal execution

Per unit: rebuilt FROM PHYSICAL BYTES via the census loaders with
every sealed `unit_map` binding re-verified (digest, length,
period, windows); the REAL harness runs all arms/models/seeds on
shared windows/seeds/budgets (unchanged, sealed) while an array
sink captures RAW predictions/observations for every
origin/arm/model including the baseline; ONE immutable O_EXCL
0600 self-digested unit record binds sealed design (file+self),
review record, execution record, manifest, census,
dataset/series, unit binding, executed code identity, T0
artifact and per-phase costs, plus the digest of the persisted
NPZ. `verify_unit_record()` recomputes the MASE denominator and
EVERY MASE from the persisted arrays — mutated predictions,
swapped NPZs, edited summaries and broken self-digests each
refuse (battery).

## T2-C45/C46 — budget, resume, rehearsal

Sealed bounds enforced (4 h wall, 8 GiB RSS, nice 15, `T2_STOP`);
per-unit O_EXCL claims (duplicates refuse); global executor lock
(stale ⇒ operator disposition); heartbeat; durable resume
(verified records skip, FAILED units preserved as missing per the
sealed rule — never rerun or deleted to complete a panel).
**Mechanical rehearsal EXECUTED with the real executor** over the
three DEV units only (structurally disjoint from the sealed
population, asserted): 3 records verified from persisted arrays,
zero sealed-bank series touched, measured per-unit wall
(sm_co2 ≈ 25 s, sm_sunspots ≈ 4–5 s, sm_nile ≈ 4 s) — committed.

## T2-C47 — delivery

Exact work census: **242 units × 2 origins × 4 arms ×
(ridge + 3 MLP seeds) = 7,744 model fits + 484 baseline evals**;
the fixture timings above are the measured CPU basis (reported,
not extrapolated as results). Focal battery: **61 passed**; full
suite at the T2 tip: **3123 passed, 2 failed, 2 skipped**
(15:57) — only the preexisting D1-anchor pair; the two skips
are the real-record-present guards; no flake this run. The exact
confirmatory command is present and structurally closed by the
absent external execution record; a template for that record is
NOT shipped pre-filled — its schema is executable in
`verify_execution_record` and the packet names the exact required
fields.

## For your review

- B4: the pushed tip, amendment 15, the v7 templates and ledger,
  the real-SAC probe evidence, and the untouched v6 incident.
  Your acta v3 (new filename, commit+tree+generation+a15) opens
  the v7 launch.
- T2: the pushed tip, the executor and its rehearsal records.
  Your EXECUTION record (v1 schema at the private root, pinning
  sealed `d1720f4d…`/`e1e3761b…`) opens scoring through the same
  single chain.

`B4_V7_RUNTIME_READY_FOR_EXTERNAL_MUSASHI_ACTA`
`T2_CONFIRMATORY_EXECUTOR_READY_FOR_EXTERNAL_RUNTIME_REVIEW`
