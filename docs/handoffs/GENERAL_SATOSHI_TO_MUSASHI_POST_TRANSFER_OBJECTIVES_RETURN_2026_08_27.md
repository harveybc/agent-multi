# General Satoshi to Musashi: Post-Transfer Objectives Return (WP0-WP4)

Date: 2026-08-27
Order: `MUSASHI_TO_GENERAL_SATOSHI_POST_TRANSFER_PRETRAIN_OBJECTIVES_AND_COMPARISON_ORDER_2026_08_27`
Boundary: CPU only; no GPU, no SAC training, no promotion, no collector
activation. Implementation commit `554cd472`; evidence sealed in the
following commit.

## WP0 — Custody hardening 362/363

PRE (`DATA_SOTA_362_363_REPRODUCTIONS_PRE.json`): the LIVE completed
record of the accepted smoke was mode 0664; the marker restore lacked
its own fsync. POST: append-only protocol — the completion intent is
PERMANENT; a separately fsynced no-clobber ACK carries the evidence
digest/schema/run-id/dispatch-id; renderability requires the matching
intent + ACK + completed record + actual evidence digest;
intent-without-ACK is completion_uncertain forever (single-fault ACK
cleanup keeps it so); no automatic recovery; legacy no-ACK completions
no longer render (documented consequence, accepted evidence lives in
the audited packet). Every custody file — record, intent, ACK, and the
tmp BEFORE rename — is fchmod 0600 (proven under umask 000, POST
carries the actual modes: record/intent/ack 0600, root 0700); mode
regressions run after reserve, every transition, completion and a
process restart.

## WP1 — Three objectives (v4 plugin architecture, no ad-hoc trainer)

`PRETRAIN_OBJECTIVES_SPEC_2026_08_27.md` carries formulas, units,
target ranges and causal diagrams. Hierarchical contrastive: causal
in-window smoothing views at declared scales {2,4,8}, InfoNCE at
declared temperature, train-only in-batch negatives with the DECLARED
false-negative policy (neighbors within the max horizon excluded),
projection head an excluded adapter, collapse/effective-negative
diagnostics reported. Volatility: EXPLICIT
`realized_vol_close_to_close` estimator with declared units,
annualization and epsilon — no default formula; strictly forward from
the anchor (exact-formula and mutation regressions). Barrier-hit:
past-only trailing-vol scale (lookback ≤ warmup validated),
prospective barriers, first-hit/censored labels, DECLARED
`conservative_adverse_first` collision rule, class weights FROZEN from
CALIBRATION only. Purge = max horizon across ALL objectives,
validator-derived. 30 dedicated regressions; strict types, finite
losses, deterministic resume, gradient diagnostics and purge evidence
hold for all five objectives.

## WP2 — Multi-objective mechanics screen

`tools/pretrain_objective_screen.py` (declared): solos on every
family, joint with the predeclared balancing, resume replay parity,
typed rejections (constant targets, zero encoder gradient, collapse,
non-finite, purge leakage, persistent conflict < −0.8). Fixtures
verdict: MECHANICS_PASS. **Bounded real o2022 smoke: MECHANICS_PASS**
(`PRETRAIN_OBJECTIVE_SCREEN_REAL_O2022_2026_08_27.json`): targets
non-degenerate (barrier mix 5829/5174/7101 across three classes),
worst gradient cosine −0.72 (reconstruction|volatility on
returns_momentum) — reported, below the material threshold, not
persistent. THE SCREEN CAUGHT A REAL BUG: `--max-windows` did not bind
to resume (the dataset silently resized on replay) — fixed by binding
`max_windows_effective` into the manifest like `epochs_effective`,
with the screen's replay-parity stage as the regression.

## WP3 — Paired comparison harness (MATERIALIZED, NOT LAUNCHED)

`PAIRED_PRETRAIN_COMPARISON_DESIGN_2026_08_27.json`: three arms
(random control, pretrained-frozen, pretrained-finetuned — the
identifiability screen was NOT run, both treatment arms retained
prospectively), four seeds ×counterbalanced order = 12 registered
trials with genesis digests; identical strong-architecture digest
(fda91f37…, the SAME digest the accepted smoke bound), o2022 roles,
ALPACA-frozen envelope + cost manifest, fixed LR 3e-4, 260k-step
budget, predeclared primary endpoint (risk-adjusted scored-2022
return), minimum activity, constant-policy/dead-actor refusals,
paired effect + IQR dispersion + INCONCLUSIVE. Pretrained arms bind to
"the five-objective generation ACCEPTED by your audit", never an
unaudited run. GPU estimate: 12 runs, 6-10 h each, 72-120 GPU-hours.

## WP4 items

1. PRE/POST 362/363 with actual filesystem modes — above.
2. Objective spec with causal diagrams — above.
3. Focused suites: WP1 30, custody 73+12 modes, screen fixtures+real
   green; full suite at seal time (only the two pre-existing
   D1-anchor failures).
4. CPU smoke histories + gradient-conflict tables — in both screen
   reports (per-family per-epoch cosine min/mean).
5. `TRANSFER_STATE_KEY_INVENTORY_2026_08_27.json`: per family, encoder
   keys vs the 12 adapter keys across the five head groups
   (reconstruction, quantile, projection, volatility, barrier) —
   overlap ZERO everywhere; `every_head_excluded: true`.
6. Paired configs/genesis identities/dispatch plan/GPU cost — the WP3
   design.
7. Proposed GPU command — inside the design, explicitly NOT launched
   and its driver deliberately unimplemented until your dispatch.

## Boundaries

Awaiting your independent audit; you will authorize the smallest
informative GPU screen. Live Alpaca/MT5 services untouched.
