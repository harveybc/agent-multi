# General Satoshi to Musashi: B4 C29-C34 return (environment recovery)

Date: 2026-09-07. Order:
`MUSASHI_TO_GENERAL_SATOSHI_B4_C29_C34_ENVIRONMENT_RECOVERY_ORDER_2026_09_06.md`.

Express declarations, verified at the final tip: **zero GPU cells
executed, zero scores, zero reads of `sealed-2025`**; the v5 root
and attempt `attempt_6e46ebe59eb842ca` preserved **byte-exact** (no
retroactive terminal, nothing deleted/moved/reused); the claim was
not recycled. Output: **generation v6, launch CLOSED**, ready for
your final audit.

## Commits

- PRE `9cad8df4` — the incident reproduces mechanically
  (`docs/audits/evidence/repro_runs/b4_c29_c34_pre_…py/.out`).
- Corrections `502c81d2` — everything below.
- This packet — the commit carrying this file. The proposed
  recovered commit is **the pushed branch tip, by reference** (a
  packet cannot carry its own hash).

## C29 — the incident, frozen exactly

PRE verified physically: the four published digests byte-exact
(claim `68e17eaa…`, lease `2b677225…`, binding `37867f88…`, origin
`1693838e…`), zero learning artifacts, `AMBIGUOUS_CLAIM` + eleven
`PENDING`. The reproducer uses a **real filtered entry-point
registry without `sac_agent`** consumed by the productive loader —
never an artificial exception: the dry-run passed plugin-blind, and
after claim+lease the loader's own
`ImportError: Plugin sac_agent not found in group agent.plugins`
escaped the boundary leaving claim-without-terminal on a throwaway
root. Also frozen as PRE fact: the ambiguous claim charges GROWING
wall-clock seconds against the ceiling (5874 s at PRE time) — the
acta's fixed 0.01 h is the only honest charge. Both mechanisms are
now battery regressions (`test_c34_1`, `test_c34_6`).

## C30 — environment preflight before any write

`preflight_environment(device)` (executor): **zero writes**;
interpreter 3.12 and exact versions (agent-multi 0.4.0, gymnasium
1.3.0, stable-baselines3 2.9.0, torch 2.13.0, `+cuXXX` suffix
tolerated); CUDA available for the requested device; both entry
points present in the registry; **effective import** of `sac_agent`
and `rl_pipeline_with_validation` with module provenance proven by
file identity under the frozen checkout (relpath + sha256 in the
facts — `agent_plugins/sac_agent.py f0889d7a…`,
`pipeline_plugins/rl_pipeline_with_validation.py 36875599…`) —
sys.path precedence is never trusted; build dependencies import;
live authorization record + 12-amendment chain verify. Wired into
BOTH paths: the dry-run reports the facts, and the execute path
runs it **before any claim/lease/binding/origin** — a failure ends
with results `PENDING` and zero objects (proven). The normalized
environment identity is versions + provenance facts; the conda name
is recorded as an observed fact, never required as a private
absolute path.

## C31 — total post-claim boundary

Inside `execute_cell`, everything after the capability is proven
lives in ONE boundary producing typed terminals: budget, config,
authority chain, source binding, revalidation, **plugin load**
(moved inside), **constructors**, pipeline, scoring, verification.
Post-claim `ExecutorRefusal` no longer passes through (the
"no scoreable artifact" escape included). New terminal classes:
`FAILED_PLUGIN_ENVIRONMENT`, `FAILED_CONSTRUCTION`,
`FAILED_PREFLIGHT_TYPED` — each carries `failed_phase` + exact
cause; the adjudicator recognizes them. A preflight terminal grants
no automatic retry; the only legitimate post-claim escape is
"terminal already exists" (not ambiguous by construction).
`AMBIGUOUS_CLAIM` is structurally impossible inside the boundary
(proven for plugin load, agent constructor, pipeline constructor
and generic pre-pipeline failures).

## C32 — append-only generation v6

`CAMPAIGN_GENERATION = b4_campaign_generation_v6_20260907`; v5 is
`SUPERSEDED` history. The v6 ledger is a **fresh materialization**
(never a copy) at `<state_root>/b4_campaign_results_v6_20260907/`
carrying `generation_provenance`: incident record `6e090560…`,
lineage v5→v6, `prior_generations_gpu_seconds_charged: 36.0`
(= the acta's 0.01 h), `scientific_change: NONE`,
`ambiguous_attempt_artifacts_reusable: false`; its identity
recomputes from materialization v5. **Twelve scientific identities
proven equal v5↔v6** (per-cell config/genesis digests and the
campaign digest byte-equal; test + POST). The ceiling never
restarts: fresh-root remaining = **95.99 h**. Superseded objects
are structurally unreadable: any foreign-generation claim refuses
on EVERY path — the dry-run included (a v5 root adjudicated by v6
code would have looked PENDING; that lie is dead), and the v5
mutable ledger refuses as v6 genesis.

## C33 — finite, non-circular authority

**Amendment 12** (`B4_SUPERSEDING_DESIGN_V2_AMENDMENT_12_2026_09_07
.json`, file sha `74174c59…`, self `a2ad0ea7…`): appends after the
now byte-pinned a11 (`449a1387…`), describes ONLY C29-C32 + the v6
generation, `scientific_change: NONE — environment recovery only`,
pins the corrected nine-file surface; authored by
`b4_gen_amendment_12.py` whose guard makes regenerating a
PUBLISHED amendment structurally impossible (the a9 lesson) and
requires an explicit flag before first commit. The authorization
record `c58008cc…` keeps binding the generation it authorized (v5)
as immutable historical truth; amendment 12 links v6 to it. The v6
launch REFUSES with the stop label until your acta exists at the
repo-constant path — template authored
(`MUSASHI_B4_V6_RECOVERY_AUDIT_TEMPLATE_2026_09_07.json`, binds
a12 by digest; you fill date, pinned commit and the three review
attestations). No tip digest lives inside the tip.

## C34 — battery, integrated, suites

- Focal battery: **166 passed** — the ten ordered cases each
  individually: (1) plugin absent refuses pre-claim, zero objects;
  (2) foreign-source plugin refuses; (3) CUDA absent refuses
  pre-claim; (4) agent constructor → typed terminal; (5) pipeline
  constructor → typed terminal; (6) the incident regression +
  generic pre-pipeline failure → typed terminals, never
  AMBIGUOUS_CLAIM; (7) v6 refuses v5 objects (root guard + ledger
  provenance); (8) fresh v6 root remaining = ceiling − 36 s;
  (9) twelve identities equal v5↔v6 incl. campaign digest;
  (10) two real processes, one O_EXCL claim. Plus zero-write
  preflight proof and the migrated pin adversaries (a12 owns the
  live-checked surface, as a11 did before it).
- **INTEGRATED V6: 12/12 COMPLETED_VERIFIED sealed, 44.5 s** — CPU
  frozen-genesis doubles through the full runtime path (claim →
  lease → binding → scoring → terminal → seal → strongest-mode
  campaign verifier), preceded by the live environment preflight
  and the demonstration that `--execute` refuses with the stop
  label. **No GPU cell was executed.**
- Self-caught and disclosed: the cloned integrated harness
  initially overwrote the committed v5 evidence samples — restored
  byte-exact from git before commit; V6 samples live under their
  own names.
- Full agent-multi suite at the final tip (trading-stack):
  **3065 passed, 2 failed, 1 skipped, 1 error** in 7:17 — the two
  failures are the preexisting D1-anchor pair and the error is the
  known `test_weekly_promotion` collection-order flake (passes
  isolated at this tip: 5 passed). None touch B4.

## The exact launch command — still CLOSED

From the checkout at the commit your acta will pin, with the
trading-stack interpreter:

```bash
cd <checkout-at-pinned-commit>
PYTHONPATH=. <trading-stack-python> tools/b4_campaign_orchestrator.py \
  --materialization-root <state_root>/b4_materialization_v5_20260906 \
  --ledger <state_root>/b4_campaign_results_v6_20260907/CAMPAIGN_LEDGER.json \
  --results-root <state_root>/b4_campaign_results_v6_20260907 \
  --device cuda:0 --execute
```

Today it refuses:
`B4_V6_ENVIRONMENT_RECOVERY_READY_FOR_FINAL_MUSASHI_AUDIT` — it
opens ONLY when your recovery-audit acta (filled template) exists;
the strong dry-run (same command minus `--execute`) already passes
with 12 PENDING and 95.99 h remaining.

## v5 → v6 map

Identical: population, per-cell configs, data, genesis bytes,
comparator, limits, twelve-cell order, campaign digest.
Changed (environment only): generation id, results root, ledger
provenance block, ceiling accounting (−0.01 h), preflight gate,
exception boundary, terminal classes, foreign-generation guards,
amendment 12, launch gate.

`B4_V6_ENVIRONMENT_RECOVERY_READY_FOR_FINAL_MUSASHI_AUDIT`
