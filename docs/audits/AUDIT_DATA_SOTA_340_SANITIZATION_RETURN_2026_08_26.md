# Audit: DATA-SOTA-340 Sanitization Return

Date: 2026-08-26
Audited tip: `satoshi/data-first-sota-20260826@2c70c27c`
Auditor: General Musashi
Disposition: **340 ACCEPTED; FINDING-323 TIP CLEANUP ORDERED**

## Verification

- Public v3 contains logical interpreter/environment identities and no absolute
  operator paths or persistent GPU identifier.
- Public v1 and v2 are tombstones bound to restricted copies by digest.
- v2-to-v3 mapping is explicit.
- Focused sanitization/topology suite independently reproduced: **103 passed**.

DATA-SOTA-340 is accepted. CUDA C0 remains accepted; no rerun is needed.

## Legacy disposition

The registered allowlist correctly exposes, but does not remediate, 22 legacy
files still carrying public topology. Under the owner's standing instruction to
clean public repositories:

1. sanitize/tombstone those 22 files on the active public tip now, preserving
   restricted copies and content digests;
2. remove their allowlist exceptions so the scan is zero-exception;
3. scan all active public branch tips for credentials, broker/account IDs,
   tokens, private keys and personal identifiers in addition to topology;
4. do **not** rewrite shared Git history solely for host paths/GPU UUIDs: the
   disruption exceeds the security benefit once current tips are clean;
5. if any actual secret/account identifier is found, report it immediately,
   rotate/revoke it first, and present a scoped history-rewrite plan for owner
   approval.

This is finding-323 continuation, not a new architecture blocker. WP-PRETRAIN
and data collectors continue in parallel.
