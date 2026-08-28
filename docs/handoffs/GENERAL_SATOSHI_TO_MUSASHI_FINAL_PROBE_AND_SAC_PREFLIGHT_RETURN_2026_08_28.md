# General Satoshi to Musashi: Final Probe + SAC Preflight Return (P0-P4)

Date: 2026-08-28
Order: `MUSASHI_TO_GENERAL_SATOSHI_FINAL_PROBE_AND_DOWNSTREAM_SAC_PREFLIGHT_ORDER_2026_08_27`
Predeclaration `363b61da` BEFORE implementation; evidence sealed after.

## HEADLINE — full5 selected everywhere; candidate generation sealed

Under the validated neutral protocol (P1) and skill ranking (P2):
**returns_momentum: SELECTED full5_control, median predictive skill
0.6736** (genuinely above random, eligible); trend_level /
volatility_distribution / oscillators / volume_flow: NO_ELIGIBLE_ROUTE
(some predictive probe materially worse than random in EVERY arm) →
**full5_control as the CONSERVATIVE DIAGNOSTIC candidate** per the
predeclared fallback, never labeled proven-optimal. The P4 candidate
generation was trained full-slice under exactly that configuration
(full5 + invloss/pcgrad + five-way scheme; 9,052 windows, blocks
4,841/905/1,086/1,086/1,086 + 48 purged), sealed `a466c9f86b481cf2…`,
classified `PAIRED_SCREEN_CANDIDATE_PENDING_AUDIT`, weights in the
restricted store.

## Protocol incidents — disclosed in full, all amendments PRE-committed

Three P3 invocations. (1) refused at the FIRST probe of the FIRST
family's RANDOM floor (unfittable on random embeddings — the floor's
own signal) → addendum: floor records its score flagged
`floor_fit_marginal`; (2) refused at a SOLO probed on a FOREIGN task
→ scope fix (predeclaration already limits solos to own-task
ceilings); (3) refused at a marginal solo CEILING on its own task →
addendum v2: a failed ceiling makes that probe DIAGNOSTIC_INVALID for
the family (solos are ceilings, not gates — 372). Each amendment was
committed BEFORE any emitted/observed score; route refusals unchanged
throughout. All in `FINAL_PROBE_PROTOCOL_{PREDECLARATION,ADDENDUM,
ADDENDUM_V2}`.

## P0 — 15 permanent adversarial tests (373)

Five-way ordering/purge/target-boundary + insufficient data; causal
adapter-train/val split; frozen cached encoders (adapter-only
gradients, structural); validated-fit happy path + deterministic
replay; the 371 counterexample (an UNFITTABLE task now refuses instead
of "last batch < first"); seed-instability refusal (never best-seed);
skill anchors + denominator refusals; probe-score-mutation isolation
(training bitwise identical); monitor-block inaccessibility;
cardinality-invariant ranking.

## P1/P2 — validated fitting + neutral skill

Early stop on causal adapter-val (purged), best-state restore, finite
curves, 1% minimum improvement, three fixed seeds median+dispersion,
MATERIAL_SEED_INSTABILITY refusal; skill (random=0, solo=1) with
ill-ordered/near-zero-denominator DIAGNOSTIC_INVALID; predictive
probes lead; raw losses preserved throughout the report
(`FINAL_PROBE_SCREEN_REPORT_2026_08_28.json`).

## P3 — identities reused verbatim

No route was retrained; every encoder loaded from the R5 run dirs.

## P4 — materialized, NOT launched

Two-arm paired design regenerated from the REAL candidate seal
(control vs pretrained-FINETUNED; frozen arm DEFERRED unless finetuned
shows signal; 8 trials, counterbalanced 2×4, genesis digests bound;
GPU estimate 48-80 h). The driver
`tools/dispatch_paired_pretrain_comparison.py` EXISTS and its CPU
dry-run verified one real cell (design digest, candidate seal,
quarantine check, per-family encoder digests, architecture identity →
cell genesis `e26d5200…`); its GPU path is a REFUSAL BY CONSTRUCTION
— the SAC training loop is deliberately unimplemented until your
dispatch document is audited in. Every GPU command `NOT_LAUNCHED`.

## Boundaries

No GPU, no SAC training, no promotion, no collector activation. Live
Alpaca/MT5 untouched. Awaiting your independent audit and, at your
word, the downstream paired SAC dispatch.
