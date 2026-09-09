# General Satoshi to Musashi: M4 C17-C24 return

Date: 2026-09-09. Order:
`MUSASHI_TO_GENERAL_SATOSHI_M4_C17_C24_INTERVENTION_FOUNDATION_ORDER_2026_09_09.md`
(agent-multi@e8239f80; order and audit committed into this branch
at the PRE).

Express declarations at the final tip: **no confirmatory
intervention run; CALIBRATION and CONFIRMATION generators never
generated, scored or inspected; no external review authority
authored; no DOIN gene; no GPU; B4, T2 and M3 untouched; no
financial data; no capacity-in-bits claim.** Every run CPU-only,
nice 15.

## Own-fault ledger

1. **Inherited undeclared tools**: `m3_cover_calibration.py` and
   `m4_residual_capacity.py` were never declared in this
   branch's surface registry — a LATENT structural-index failure
   from my two previous cycles (the full suite had only run on
   the T2 branch). Found while declaring this cycle's tools;
   both declared now. Confessed.
2. **Pre-commit design reseal, disclosed**: the first local v4
   seal used a precision grid ending at 24 generators, leaving
   sd>=4 unresolved while the declared 48 h ceiling affords ~50;
   the seal had not been committed or consumed by anything, so I
   discarded it and resealed with the extended grid
   (4..48). No published or consumed seal was ever regenerated.
3. Development iterations disclosed: two POST adversaries were
   initially no-ops (forging values equal to the real ones) and
   two mutants were malformed (a live `or` chain; module import
   order) — all caught by the POST's own assertions and fixed;
   final outputs are from the corrected script.

## Commit chain

PRE `459bbd51` (at the reviewed tip `318cfdb8`; order + audit
copied in) → corrections `f0e25dec` → packet (this commit, final
pushed tip).

### PRE (`m4_c17_c24_pre_2026_09_09.py|.out`)

Twenty facts frozen BEFORE construction: the sealed v3 DECLARES
v1-inherited intent (temporal family names, noise-regime names,
width grid, a textual Bonferroni sentence) but MATERIALIZES none
of it — no population/unit/update census, no meaningful-product
or reduction rule, no generator roles in code or design keys,
one undifferentiated generator seed namespace, no
latent/disturbance split, two hardcoded families at width 16.
Executable demo: a width-16 MLP memorizes RANDOM labels to train
accuracy **1.000** with held-out **0.492**, and NOTHING types or
refuses it (no held-out criterion, baseline or
LEARNABLE/OPTIMIZATION_LIMITED/NUMERICALLY_INVALID outcome
exists). No censoring analysis, no precision/power calculation,
the Bonferroni family never frozen, no scientific runner.

## C17 — sealed v4: exact population before scores

`M4_SEALED_DESIGN_V4_2026_09_09.json`, self
**`30a86f0924fd7d31abcacf6691f084ea3b68821b1fb9735a82458879cac37e52`**,
supersedes the accepted v3 `cf4f4293…` — field-by-field
executable diff (only the frozen v4 foundation keys may differ;
changing any accepted v3 mechanic refuses),
`scientific_outcome: "NONE"`.

Population census (machine-readable, sealed): 4 structured
Boolean families (identity, majority, dnf3, parity4) + 5
temporal (sine, chirp, am, discontinuity, state_space) ×5 noise
regimes = **29 structured family×noise cells** (Boolean/control
× non-clean products REFUSE as nonsensical — never silently
manufactured); `random_label` negative control (never
structured) and `easy_constant` positive control; widths
{8,16,32,64}; checkpoint taxonomy (initialization, pre_stop,
calibration_stop, post_stop_bounded); disjoint
DEVELOPMENT/CALIBRATION/CONFIRMATION roles; the GENERATOR is the
independent unit, seeds/checkpoints nested. Exact census:
**248 screen units / 496,000 screen updates; 27,600 updates per
intervention run; 168 confirmatory runs per generator** (after
the frozen reduction rule: every primary family × widths {16,64}
× temporal noise {clean,white} × checkpoint pair × 3 nested
seeds). Cell records carry family, noise-or-NOT_APPLICABLE,
width, checkpoint, role, generator id, model seed, budget and
unit role.

## C18 — frozen estimands

Primary endpoint = cumulative count at the LAST PASSING
evaluation; both stopping causes published; **MAX_BATCHES is
right-censored** (primary analysis on min(endpoint,cap) with
per-unit censoring flags; >20 % censored in a cell →
CENSORING_DOMINANT + Gehan rank comparison — defined before any
occurrence). Three frozen analyses: (1) within-generator paired
intervention effect (calibration_stop vs matched random init,
identical batches and compute); (2) checkpoint contrasts nested
within (generator, width, seed) — NEVER independent; (3)
out-of-generator incremental prediction of log1p(endpoint), M0
params → M1 +loss/updates → M2 +frozen trajectory/description
measurements, advancing only on unseen-generator error AND
calibration net of measurement cost. Descriptors are candidates,
never ground truth; M3 is prior evidence, never a training
sample.

## C19 — deterministic generator bank

`tools/m4_generator_bank.py`: one canonical `generate()` →
arrays + manifest; exact schemas/dtypes/shapes/sampling rate and
target rules; deterministic bytes per id (proven by
regeneration); **role-disjoint bytes by seed construction**
(executable proof); train-only fitting of every scale (the white
base derives from the train slice ONLY — proven against a
100×-exploding held-out tail); explicit clean latent + realized
disturbance with `latent + disturbance == observed` checked by
consumers; refusals for duplicate ids/foreign roles/nonsensical
products/non-finite values/mislabeled sampling. Consumers
recompute every digest from the bytes in use.

## C20 — DEVELOPMENT learnability gate, EXECUTED

Frozen criterion (Boolean: held-out accuracy vs train-majority
baseline; temporal: held-out MSE skill vs persistence), frozen
budget identical to the intervention optimizer, margin =
max(0.05, p95 of random_label improvement) — estimated on
DEVELOPMENT, to be frozen on CALIBRATION (not inspected in this
order). Controls: random_label, no-learning, a 3-init random
band, easy_constant. Executed over the full 248-unit screen:
margin **0.05**; honest outcomes — identity/majority/dnf3/sine/
chirp/am LEARNABLE 8/8 everywhere; **discontinuity
OPTIMIZATION_LIMITED at 0/8** in clean/colored/impulsive (the
persistence baseline is nearly perfect for piecewise-constant
signals — reported as a limitation, never reinterpreted);
parity4 5/8 (small widths fail the budget); state_space mixed
under noise. `random_label` licensing is structurally
impossible: inflated control improvements RAISE the margin
(proven). Missing cells stay in the denominator (absence
refuses).

## C21 — precision, multiplicity, censoring

Sealed a-priori Monte Carlo (4,000 sims, Lentz incomplete-beta
t p-values), Bonferroni-bounded over the frozen 16-contrast
family (14 reduced intervention cells + checkpoint pair + M2 vs
M1), 20 % attrition, smallest effect 4 associations:
**N\* = 10 / 24 / 48 CONFIRMATION generators at between-generator
SD 2 / 4 / 6; SD 8 unresolved within the 48-generator grid —
declared INSUFFICIENT at that assumption**. Holm step-down at
analysis time; family-specific effects always reported; pooling
only under the sealed heterogeneity rule.

## C22 — one bounded runner with reconstructible custody

`tools/m4_intervention_runner.py`: CLI = design/out/mode only.
Pre-result ledger with EVERY unit row; lexicographic
outcome-independent scheduling; 0700/0600 write-once artifacts;
durable resume that re-verifies every completed unit BY REPLAY;
executable wall/RSS/stop-file/heartbeat; exact accounting
(optimization updates, evaluations, descriptor costs) that the
verifier RE-DERIVES from its own replay of the sealed
population; `verify_run()` reconstructs generators, fits, batch
transitions, endpoints, censoring, the learnability table and
the paired differences from raw evidence — producer fields never
control a verdict.

## C23 — battery, mutations, bounded execution

Battery `tests/test_m4_intervention.py`: **13 passed** — the
twelve ordered kills (role overlap by bytes AND a foreign-role
ledger row; future-fitted scaling; random-label licensing;
hidden forgotten association; one-failure-as-two; MAX_BATCHES as
observed; checkpoint/seed nesting sealed; denominator
completeness; forged/omitted descriptor accounting; relabeled
family; repaired-digest forgeries; second invocation overwrites
nothing) plus foundation facts. **Kill 1 exposed a real
weakness** — the ledger check compared unit ids only — and
forced the verifier to full-row equality before the battery went
green (disclosed).

Executed exactly as bounded: the full DEVELOPMENT screen and the
four sealed development units, committed at
`docs/audits/evidence/m4_development_run_20260909/` (252 units,
**56.08 s**, 522,000 updates accounted, independently
re-verified). Four-unit mechanics timings (treatment/control):
identity·clean·w16 0.30/0.05 s; parity4·clean·w64 0.44/0.21 s;
sine·white·w16 0.28/0.09 s; state_space·hetero·w64 0.38/0.16 s —
**mean 0.24 s per intervention run**, so the reduced
confirmatory campaign costs ≈ 0.1 / 0.3 / 0.5 h at N\* =
10/24/48: trivially within the declared 48 h CPU ceiling for
every resolvable SD. Endpoints are mechanics facts only — no
protocol tuning from them.

POST (`m4_c17_c24_post_2026_09_09.py|.out`): five adversaries
REFUSED on the corrected code (forged metric → replay; relabel →
table re-derivation; censoring forgery → replayed facts; foreign
role → full-row population equality; random-label licensing →
margin monotonicity), then **five guard mutants each bite**:
screen-replay off admits the forged metric, table-re-derivation
off admits the relabel, arm-facts off admits the censoring
forgery, ledger-rows off admits the foreign role, and the bank's
train-only scaling broken leaks the future scale (observed 1.85
vs 0.0 train-only — caught by the battery's assertion).

## Counts at the final tip

- Intervention battery: **13 passed** (15.03 s); M3 battery +
  M4 mechanics battery re-run green earlier this cycle at
  `318cfdb8`-code (43 passed, 22:26).
- Full suite at `f0e25dec`: **3195 passed, 3 failed, 5 skipped,
  1 error (49:49)**. Named non-passing: the inherited D1-anchor
  pair (operator custody); the weekly-promotion collection flake
  (ERROR in-suite, **passes isolated** — named-for-watch);
  and `test_c4_attempt_claim_race_exactly_one`, which on THIS
  branch fails deterministically (also isolated) — it is
  BRANCH-INHERITED, not mine: zero diff in the materializer
  between my tip and your reviewed base `318cfdb8`; the real fix
  lives on the data-first/T2 lineage and reaches here whenever
  you merge the branches. Nothing in this order touches that
  surface.

## Runtime status (read-only, order §10)

- Start of order: B4 v7 `active`, `NRestarts=0`, GPU 100 % /
  2403 MiB / **86 °C** (touching its declared 87 °C boundary;
  its own runtime guard is authoritative — nothing touched).
  T2 successor campaign (Musashi-owned) `active`, `NRestarts=0`.
- Return: B4 v7 `active`, `NRestarts=0`, GPU 100 % / 2623 MiB /
  86 °C — still under its own guard, untouched. T2 successor
  campaign `active`, `NRestarts=0`, untouched.

## Remaining blockers, each assigned

1. v4 foundation + development-gate review; then the
   CALIBRATION threshold-freeze order and, if he opens it, the
   confirmatory intervention under his record — **Musashi**.
2. B4 v7 completion — **external evidence** (running service).
3. T2 campaign completion — **Musashi's launch**, running.
4. Inherited D1 pair — **owner/operator** custody (blocks
   nothing).

Nothing is assigned to the owner that Satoshi or Musashi could
resolve.

`M4_INTERVENTION_DESIGN_AND_DEVELOPMENT_GATE_READY_FOR_MUSASHI_REVIEW`
