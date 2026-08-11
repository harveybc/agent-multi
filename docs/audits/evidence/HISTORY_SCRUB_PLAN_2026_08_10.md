# History Scrub Plan — agent-multi (AUD-SEC-20260810-215)

Date: 2026-08-10 America/Bogota
Author: General Satoshi III, technical lead
Status: **PLAN ONLY — NOT EXECUTED.** Execution requires the owner's
SECOND explicit authorization (the first authorization covered private
visibility only) and General Musashi's independent verification. This
document changes nothing; it binds how the rewrite will be done when —
and only when — it is authorized.

## 1. Problem

`harveybc/agent-multi` was public while sensitive evidence was committed.
The packets were later removed from HEAD (`42578f70` third-party social
content, `047892b5` Front-2 operational evidence), but removal commits do
not unpublish history: every earlier commit remains retrievable. One
topology sample and one enrichment dry-run additionally remain reachable
at HEAD. Immediate containment is already in force: the repository is
private by owner authorization. The standing contract is pointer-only
public evidence.

## 2. Exact affected commits and paths

| Commit | Introduced | Sensitivity |
|---|---|---|
| `662e1dde5bcbcd6462a6929a9834cd8b2ef2d3f3` | `docs/audits/evidence/MULTIFRONT_F1_L1_SAMPLE_2026_08_10.json` (blob `4d936572841d9c9b8d4ca829c1f6ea07e704f6c5`) | fleet topology sample: hostnames, GPU UUIDs, absolute operator paths. **Still present at HEAD (same blob).** |
| `95cb74c0776477465c6c5cf1fb471a3eb4988004` | `docs/audits/evidence/SOCIAL_REVIEW_PACKET_2026_08_10.json` (blob `e92fce859948ba5201e127ad76b1dbea42436f69`), `docs/audits/evidence/SOCIAL_REVIEW_PACKET_2026_08_10.md` (blob `757c6c839f2f0a79e41bc261869761f8e76975ba`), `docs/audits/evidence/SOCIAL_ENRICHMENT_RETRY_DRYRUN_2026_08_10.json` (blob `9ba709e93ffc75879a78974259785a437ac900f4`) | third-party social content and operational enrichment detail. The two packet files were removed from HEAD at `42578f70`; **the dry-run file is still present at HEAD (same blob).** |
| `c8bcb7a80035915e37a2e9a357496cc905dc5551` | `docs/audits/evidence/FRONT2_LIVE_VS_SIM_ROLLING_2026_08_10.json` (blob `6b5c314110a181c34732db75a7a8f182b44ae71a`), `docs/audits/evidence/FRONT2_VENUE_PROTECTION_FACTS_2026_08_10.json` (blob `454d440dd1e503605e810bc7d02a7578fbf403f7`), `docs/audits/evidence/IBKR_PAPER_RECONCILIATION_PACKET_2026_08_10.md` (blob `1a6d29ea2f11b3b3e75b0df1983c0ae3d0417d71`) | Front-2 operational/account metadata and live protection levels. Removed from HEAD at `047892b5`; retrievable from history. |

The removal commits `42578f70bc89bcb20ed7952332d2264f5c85727c` and
`047892b5dc8d6d6b101ed2ad6f118cb968ef5268` will be rewritten
automatically (becoming no-ops for the scrubbed paths) by the method
below; they need no separate handling.

## 3. Preconditions (all mandatory before any execution)

1. **Owner second authorization**: an explicit, authenticated owner
   instruction naming this plan and authorizing the history rewrite and
   force-push. Private visibility (already granted) is NOT that
   authorization.
2. **Musashi verification commitment**: General Musashi agrees to verify
   the rewritten mirror BEFORE the force-push and again after it.
3. **Repository stays private** for the entire window — from now, through
   execution, until Musashi's post-push verification passes. Public
   visibility may only be restored after the scrub is verified.
4. **Push freeze**: no agent or automation pushes to `agent-multi`
   between mirror creation and force-push completion; the pre-push
   sensitivity gate (`tools/prepush_sensitivity_gate.py`) is installed in
   every writing clone before the freeze lifts.
5. **Fork/clone census**: confirm via the hosting provider that no forks
   exist; any fork or external clone is recorded and disposed of by the
   owner before execution (a rewrite does not unpublish forks).
6. **Private relocation verified**: the sensitive payloads exist in
   private operator storage with recorded SHA-256 digests (pointer files
   `SOCIAL_REVIEW_PACKET_2026_08_10.POINTER.md` and
   `FRONT2_EVIDENCE_2026_08_10.POINTER.md` already reference them), so no
   evidence is lost by the scrub.

## 4. Method

Step 0 — HEAD hygiene (normal commits, not history rewrite): replace
`docs/audits/evidence/MULTIFRONT_F1_L1_SAMPLE_2026_08_10.json` and
`docs/audits/evidence/SOCIAL_ENRICHMENT_RETRY_DRYRUN_2026_08_10.json`
at HEAD with pointer files to private operator storage, so the rewritten
history has no tip reintroducing the content.

1. **Mirror**: `git clone --mirror` the canonical remote to an isolated
   private working area. Record `git rev-parse` of every ref and
   `git rev-list --count --all`.
2. **Rollback bundle**: `git bundle create agent-multi-prescrub.bundle
   --all` from the untouched mirror; store it in private operator storage
   with its SHA-256. This is the sole rollback artifact.
3. **Rewrite**: on a copy of the mirror, run `git filter-repo
   --invert-paths` with one `--path` argument per affected file listed in
   section 2 (all seven paths). No other paths, messages or authors are
   altered.
4. **Verification (Satoshi, then Musashi independently)** on the
   rewritten mirror:
   - `git log --all --full-history -- <path>` is EMPTY for all seven
     paths;
   - `git cat-file -e <blob>` FAILS for all seven blob IDs in section 2
     (after `git reflog expire --expire=now --all && git gc --prune=now
     --aggressive` on the rewritten mirror);
   - ref-for-ref tree comparison against the pre-scrub mirror shows the
     ONLY differences are the seven removed paths;
   - the full test suite passes on a checkout of the rewritten master;
   - the audit register and handoff corpus remain intact.
5. **Force-push**: `git push --mirror` from the verified rewritten mirror
   to the canonical remote during the freeze window.
6. **Server-side residue**: request provider-side garbage collection /
   cached-view purge (GitHub support ticket) so unreachable objects and
   cached PR/commit views stop serving the old blobs; the repository
   remains private until this and Musashi's post-push verification
   complete.
7. **Re-clone discipline**: every operator clone is re-cloned (not
   pulled) after the force-push; stale clones must never push old refs.
   The installed pre-push gate blocks re-introduction pushes regardless.

## 5. Rollback

If verification fails at any step before the force-push: discard the
rewritten mirror; nothing has changed remotely. If a defect is found
AFTER the force-push: restore the canonical remote by force-pushing the
preserved `agent-multi-prescrub.bundle` (verified by its recorded
SHA-256), reinstate the push freeze, and return to step 3 with the
defect documented. The repository stays private in every rollback state,
so restoration never re-publishes the sensitive history.

## 6. Explicitly out of scope

- No rewrite of any other repository.
- No mutation of the sealed L1 collection, its replica, or any run
  artifact.
- No change to the incident/audit register content beyond what the seven
  path removals imply.
- This plan does not authorize itself: without the owner's second
  authorization it stays a document.
