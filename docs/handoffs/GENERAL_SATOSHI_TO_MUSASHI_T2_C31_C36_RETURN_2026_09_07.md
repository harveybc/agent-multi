# General Satoshi to Musashi: T2 C31-C36 return (final screen design)

Date: 2026-09-07. Order:
`MUSASHI_TO_GENERAL_SATOSHI_T2_C31_C36_FINAL_SCREEN_DESIGN_ORDER_2026_09_06.md`
(executed AFTER B4 C29-C34 per its priority; the B4 return is
`GENERAL_SATOSHI_TO_MUSASHI_B4_C29_C34_RETURN_2026_09_07.md`).

Express declarations at the final tip: **zero downloads, zero
confirmatory scores, zero scientific ledger**; v2, v3 and v4
preserved byte-identical as history; B4 untouched on its own
branch. Output: **draft v5, not a seal**.

## Commits

- PRE `488d408f` — support-1/N licenses a panel; the 120-floor
  blocks length 84 under any origin count; the ordered 2×17
  arithmetic frozen with the real model minimums; v4 state and sim
  digests recorded.
- Corrections `5737ad77` — everything below.
- This packet — the commit carrying this file.

## C31 — per-panel extreme support

`extreme_support_rule` fixed in the design before results:
`min_evaluable_series_absolute: 8` AND
`min_evaluable_fraction: 0.25` (per panel:
`need = max(absolute, ceil(fraction × n_series))`; 40-series
panels → 10). The adjudicator counts EVALUABLE series (EVALUATED
or HARM_INFINITE — evaluable evidence of damage is never absence);
below the minimum the panel is **INCONCLUSIVE, never favorable**.
The PRE positive (one evaluable series licensing a panel) is a
frozen kill; `X=0∧D>0` stays HARM_INFINITE and `X=0∧D=0` stays
ratio 1.0; a design lacking the rule refuses typed in BOTH the
validator and the adjudicator.

## C32 — one geometry authority

`t2_bank.origin_windows_for` is now the SINGLE authority and
derives feasibility from the models' REAL minimums — score window
≥ lags+4 with ≥1 scored row after lags+horizon; the MLP epoch rule
keeps ≥8 fit rows after its own validation tail (`max(8, 20%)`);
period-aware seasonal-naive denominator (`train > period+1`). The
arbitrary 120 floor is retired from the productive surface. The
harness **delegates** `unit_origins` to it (live equality proven
across the real length grid) — harness, generator, validator and
fresh verifier can no longer drift. The validator and the fresh
verifier RECOMPUTE every unit's windows period-aware from length +
contract: **generator-emitted windows are never the authority** —
a one-row design-window forgery dies in the live byte
re-derivation, and a one-row record-window shift dies before
metrics (both frozen).

## C33 — exact costs in the productive adjudicator

Five INDEPENDENT omissions frozen against `adjudicate_screen`:
target construction, seasonal baseline, one arm's lag features,
one arm's ridge, one MLP seed — each refuses via its own path
(`cost phases are not the exact schema` /
`arm cost phases are not the exact schema`); extra keys refuse at
both levels. The schema is exact, never an open minimum.

## C34 — fresh verifier as executing precondition

Eight per-field forgeries — family, panel, digest, period,
horizon, length, temporal identity, each window — mutated
INDEPENDENTLY with the rest consistent: all die in
`fresh_verify()`'s live re-derivation from physical bytes
(4650 units rebuilt; the 242-series v5 population reproduced).
`run_confirmatory` calls the same function immediately before the
ledger (live: the honest v5 reaches `DESIGN_REVIEW_REQUIRED` with
no ledger artifact; forged semantics die BEFORE the review gate;
source-order assertion fresh < review < ledger). The bank root is
a real directory opened descriptor-first; a symlink root — even
pointing at the correct directory — refuses before the manifest is
consumed (frozen kill).

## C35 — the common two-origin geometry; hospital resolved

Draft v5 (`tools/t2_design_draft_v5.py` →
`t2_screen_design_DRAFT_V5_20260907.json`, self `2a4c51d0…`, file
`9b97440a…`) supersedes v4 **by digest** with, for ALL panels:
initial fit fraction **0.60**, rolling origins **2**, lags **8**,
horizon **1**, consecutive score windows over the final 40%. No
hospital exception: its 767 length-84 series mechanically produce
**two consecutive 17-observation windows** (`train [0,50]/[0,67]`,
`score [50,67]/[67,84]`), verified per unit — 8 scored rows per
origin, 33 MLP fit rows after the validation tail at origin0, all
model minimums intact (`model_minimums_kept: true`; any real MLP
requirement failure raises `GEOMETRY_INFEASIBLE` and returns the
draft — wired and never tripped). **Population: 242 series**
(40×6 primary panels — hospital now a full member with 40 selected
of 767 — + 2 sensitivity singletons), selection salt
`t2_design_v5` over period-aware geometry-admissible units,
per-panel `geometry_feasibility` block recording the minimum
window/scored/fit facts. The screen rules are unchanged (t df=5 +
exact 6/6 signs + LOPO + non-inferiority + attribution + observed
precision + preservation incl. C31 + costs). Both simulations
were RE-EXECUTED under the new geometry: byte-identical outputs
(`e7fae781…`, `76581fbb…`) — they operate on synthetic panel
effects and are geometry-independent, as now declared in the
draft.

## C36 — battery

Focal battery: **52 passed** — the ten ordered cases:
(1) extreme support 1/N → INCONCLUSIVE, never ADVANCE
(`test_c36_1`); (2) every `unit_map` field forged → dies live
(`test_c36_2`, 8 fields); (3) window shifted one row
(`test_c30_kill_4`); (4) each cost phase omitted (`test_c36_4`,
5 phases + extra keys); (5) symlink root (`test_c30_kill_7`);
(6) fresh verifier disconnection (`test_c30_kill_8`, live +
source order); (7) hospital produces 2×17 with the retired floor
proven gone (`test_c36_7`); (8) every selected unit of the six
panels geometry-admissible, population 242, v4 superseded by
digest (`test_c36_8`); (9) damaged panel → DOES_NOT_ADVANCE
(`test_c30_kill_10`); (10) dominant panel fails LOPO
(`test_c30_kill_9`). All CPU.

## v4 → v5 map

- Geometry: 3 origins → **2 common origins** (0.60/8/1 kept);
  feasibility from real model minimums (120 floor retired); the
  harness delegates to the single bank rule.
- Hospital: `GEOMETRY_LIMITED`, 0 admissible → **full member**,
  40/767 selected, 2×17 windows; population 202 → **242**;
  selection salt v4 → v5.
- New: `extreme_support_rule` (8 / 0.25) consumed by the
  adjudicator; `geometry_feasibility` per panel;
  `GEOMETRY_INFEASIBLE` refusal path.
- Unchanged: estimand, outputs, margins, seed tape, arms, models
  (never weakened), multiplicity, sims (byte-identical).

## Digests (bound in `T2_V5_STATE_DIGESTS.json`, committed)

manifest `43c48f6b…`, census `dc1bf8c7…`, v4 `b483f16a…`, **v5
`9b97440a…` (self `2a4c51d0…`, supersedes-v4 verified true)**,
coverage sim `e7fae781…`, screen sim `76581fbb…`.

## Suites

- T2 focal battery at the corrected tip: **52 passed** (~5 min;
  three live 4650-unit byte re-derivations included).
- Full suite at the final tip (trading-stack env): **3115
  passed, 2 failed, 1 skipped, 1 error** in 12:44 — the two
  failures are the preexisting D1-anchor pair and the error is
  the known `test_weekly_promotion` collection-order flake
  (passes isolated at this tip: 5 passed). None touch T2.

## For your review (the pin)

The external review pins: this branch tip, manifest `43c48f6b…`,
the re-derived census `dc1bf8c7…`, and **draft v5 `2a4c51d0…`**.
Minimum windows: hospital 17 (8 scored rows/origin); the smallest
non-hospital score window is electricity_weekly at 31. Any later
code/data change voids the pin; no candidate-written artifact
confers authority.

`T2_SCREEN_V5_READY_FOR_EXTERNAL_DESIGN_REVIEW`
