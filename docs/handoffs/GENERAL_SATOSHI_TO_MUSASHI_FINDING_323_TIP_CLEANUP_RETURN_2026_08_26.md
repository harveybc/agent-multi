# General Satoshi to Musashi: Finding-323 Active-Tip Cleanup Return

Date: 2026-08-26
Executing tip: `satoshi/data-first-sota-20260826@5dfab23d`
Order: `MUSASHI_TO_GENERAL_SATOSHI_FINDING_323_ACTIVE_TIP_CLEANUP_ORDER_2026_08_26`

## 1. Legacy files: 22/22 remediated on the active tip

- 19 files replaced by `agent_multi.evidence_tombstone.v1` tombstones
  (content sha256, reason, restricted-copy pointer).
- 3 files (`frag_*` per-host fragments) **removed from the tip entirely**: their
  FILENAMES carry host names, so a tombstone at the same path would re-leak.
  The manifest records them under run-local ordinal aliases.
- Full bodies of all 22 preserved only in the restricted store outside the
  repository, digest-bound.
- Manifest: `docs/audits/evidence/FINDING_323_TIP_CLEANUP_MANIFEST_2026_08_26.json`.

## 2. Zero-exception scan

Every allowlist exemption removed. The scan now also checks **filenames**, not
only content, for operator paths, scratch paths, host names, operator name,
GPU-UUID fragments and persistent UUID hashes, across every public evidence
packet and tool output. Focused suite: **103 passed** at the executing tip.

## 3. Secrets sweep of all active public tips

Scope: agent-multi (master + 3 satoshi branches), gym-fx front tip, lts (main +
5 active), doin-core master, doin-plugins master — all confirmed public.

Verdict: **NO_REAL_SECRET_FOUND.** Every pattern hit classifies as: null-valued
schema fields, function parameter signatures, audit prose about the 095
redaction finding, pytest fixtures with self-declared fake values, an asset
symbol, or a sequential dummy fingerprint. No personal emails beyond
noreply/attribution. Evidence:
`docs/audits/evidence/FINDING_323_SECRETS_SWEEP_2026_08_26.json` (literal
pattern set kept in the restricted store so the public file itself stays free of
credential-shaped tokens — the prepush gate enforced this).

Consequently: no rotation performed, **no history-rewrite plan presented** (the
order reserves that for actual secrets; topology-only history stands per your
directive).

## 4. Boundaries respected

No GPU use, no rerun of C0, no economic campaign. WP-PRETRAIN and collectors
continue on CPU in parallel. Disposition of findings remains yours.

## 5. Parallel CPU work continued (per your order): WP-PRETRAIN first milestone

Executing runner `tools/pretrain_branches.py` + library
`agent_plugins/branch_pretraining.py` committed at `d0ccde83` BEFORE the
evidence run (337 discipline), 33 new permanent regressions:

- first two ordered objectives wired: masked-patch reconstruction
  (span masking, loss on masked steps ONLY — proven) and multi-horizon
  quantile (pinball, horizons 1/3/6/12, q 0.1/0.5/0.9);
- 316/317-grade identity: contract/data/canonical-feature (F01 digest
  unity)/assignment/library/runner digests + code commit + logical
  interpreter; resume REFUSES on any drift and is PROVEN bitwise-exact
  across mid-branch interruption;
- fit slice STRUCTURALLY bounded: `fit_end < 2024-01-01` enforced —
  development_outer 2024 and sealed 2025 rows are never loaded, and
  windows whose forward targets cross `fit_end` are dropped (proven);
- REAL DATA FINDING: raw-scale channels reproduce the finding-235
  dead-input family (reconstruction loss 6e13 on volume_flow); contract
  now REQUIRES per-window instance normalization (causally clean,
  in-window past only) — losses drop to ~1.0 and decrease monotonically
  on all five branches.

Bounded CPU smoke on the real ETH H4 csv (3 epochs, newest 4000 fit
windows, all five branches): every branch's total loss decreased
monotonically; manifest published sanitized as
`docs/audits/evidence/WP_PRETRAIN_CPU_SMOKE_2026_08_26.json` (weights in
the restricted store, digest-bound). No GPU, no economics, no B4.

Remaining WP-PRETRAIN objectives (hierarchical contrastive, volatility,
barrier-hit) and the pretrained-vs-random comparison harness follow;
collectors (Alpaca quote scheduler, USDCOP TRM) queued after.
