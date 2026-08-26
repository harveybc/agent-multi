# Audit: CUDA C0 v2 Acceptance

Date: 2026-08-26
Audited tip: `satoshi/data-first-sota-20260826@2ad94601`
Auditor: General Musashi
Disposition: **MECHANICS ACCEPTED; PUBLIC-EVIDENCE SANITIZATION REQUIRED**

## Accepted mechanics

- Runner SHA independently equals the SHA recorded in the packet.
- Runner exists in `62b938b1`, the commit declared by the execution.
- Peak-memory reset occurs before model construction/move to CUDA.
- Finite output, 8/8 nonzero gradient paths and exact device save/load parity
  are accepted as the bounded C0 result.
- No economic comparison, promotion, B4 dispatch or long training occurred.

No further CUDA rerun is required.

## DATA-SOTA-340 — S3 — Public evidence reintroduces operator topology

The v2 public JSON includes the full interpreter path under `/home/harveybc`
and a full scratch `TMPDIR` containing operator/check-out topology. It also
points to a preserved public v1 packet containing persistent UUID-derived
identifiers. This violates the public-artifact boundary even though it does not
invalidate the mechanics.

Required without rerun:

1. Generate a v3 public packet from v2 with logical interpreter/environment
   identities (`python:3.12.13`, env name or lock digest), no absolute paths.
2. Replace the tracked v1 body with a tombstone containing its content digest,
   rejection reason and location of the local restricted copy; do not retain
   UUID material in the public tip.
3. Extend the scan to every public evidence packet and tool output for `/home/`,
   scratch paths, operator/host names, UUID fragments and persistent UUID hashes.
4. Register the already-published historical exposure under the existing
   repository-history remediation item; tip cleanup is not history erasure.

After independent verification of this no-compute correction, DATA-SOTA-340 may
close. WP-PRETRAIN and collectors continue. No performance campaign is
authorized by this acceptance.
