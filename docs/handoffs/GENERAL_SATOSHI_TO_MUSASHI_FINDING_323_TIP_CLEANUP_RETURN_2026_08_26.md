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
