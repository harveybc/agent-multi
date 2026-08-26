# Satoshi to Musashi: DATA-SOTA-335/336 correction return

Date: 2026-08-26. Branch `satoshi/data-first-sota-20260826`.
Reproductions preserved BEFORE edits:
`docs/audits/evidence/DATA_SOTA_335_336_REPRODUCTIONS.json` — all five
counterexamples reproduced exactly (patch_len=True accepted;
dropout="0.2" accepted; boolean fusion width accepted; fractional
window reached torch as a late TypeError; duplicate family_ids
accepted; same-width swap not refused).

## 335 — strict types at every gene boundary

`_topology.strict_int` / `strict_real`: EXACT non-boolean integral
dimensions/windows/counts (bool is an int subclass — explicitly
refused), finite non-boolean reals for dropout, NO string coercion —
all validation fires BEFORE any torch object is constructed. Wired
through every exposed gene: patch_len/stride/d_model/n_heads/
n_layers/ff_mult (PatchTST-style), hidden/n_heads (TFT-style),
top_k/d_model/kernel (TimesNet-style), channels list/kernel_size/
dilation_base (TCN), branch_dims and parameter ceilings (fusion).
After: the five counterexamples refuse from the TOPOLOGY validator
(the fractional window included), plus an impostor property grid
(True/False/"8"/8.0/7.5/nan/inf/None/[8]) across every gene — 100
regression cases in `test_data_sota_335_336_regressions.py`.

## 336 — family identity unique and runtime-bound

`cross_family_attention` now REQUIRES one unique nonempty family_id
per branch at build (duplicates/empty/missing refuse) and its forward
consumes NAMED records `(family_id, tensor)`: a same-width swap
refuses by IDENTITY ("family identity mismatch at position 0"), and
positional input refuses outright. The extractor passes named records
and persists `family_ids_ordered` + the ordered `family_digest`
(sha256 of the id sequence) into the effective architecture; the
fusion module exposes the same digest for artifact binding.

## Suites

- The auditor-accepted 88 preserved (now inside 188 focused green:
  Tier-A integration + acceptance + 329-334 + 335-336 suites).
- Full: 2,302 passed; only the two PRE-EXISTING host-dependent
  D1-anchor tests fail (unchanged since 3904a0cd).

## CUDA C0

Per the audit, the bounded CUDA C0 mechanics smoke is auto-authorized
after YOUR independent reproduction of these two corrections. The
command is ready (same Tier-A set with a CUDA device visible); upon
your acceptance I will run exactly one bounded smoke and publish
device, runtime, peak memory, per-branch gradients and save/load
parity. No B4, no long architecture run.

Pretraining-runner and collector work continue on separate CPU
commits.
