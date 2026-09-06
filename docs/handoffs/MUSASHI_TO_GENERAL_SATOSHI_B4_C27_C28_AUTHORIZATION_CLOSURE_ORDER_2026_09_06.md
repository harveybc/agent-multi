# Musashi to General Satoshi: B4 C27-C28 Authorization Closure Order

Date: 2026-09-06

## Authority and Scope

The owner authorizes the B4 twelve-cell campaign. Consume these exact
reviewer-authored records:

- `docs/audits/evidence/OWNER_B4_CAMPAIGN_AUTHORIZATION_RATIFIED_2026_09_06.json`
  SHA-256
  `540fb175f0203338aa08a21bc91bba1dddf942008e011ab7b34c6695af776c63`;
- `docs/audits/evidence/MUSASHI_B4_CAMPAIGN_AUTHORIZATION_RECORD.json`
  SHA-256
  `c58008cc5285365b4c64e2827a9b9d1a329e3b64f7c72a37b62c1c6e702ae55d`.

This order authorizes only the CPU implementation and tests required to make
that record consumable without contradicting the append-only chain. It does
not yet authorize the twelve GPU cells. Return for one final review after the
activation closure is complete.

## PRE

Freeze the exact contradiction before editing:

1. Amendment 10 pins `tools/b4_campaign_executor.py` at
   `c852c3cc5dd8469507f1238d07db60cf77c72bf25966e843d4911e6af93c6cfc`.
2. The executor contains `CAMPAIGN_AUTH_SHA = None`.
3. Replacing that value with the reviewed digest changes the file hash and
   makes the live amendment-10 code-pin check refuse.
4. The campaign is therefore not executable even though the owner decision is
   valid.

Keep this as a permanent regression. A test that monkeypatches the value in
memory is not sufficient; exercise the physical file/code identity boundary.

## C27: finite append-only activation

1. Copy the two reviewer records byte-for-byte. Hash before parsing and require
   exact equality to the digests above. A candidate-authored substitute grants
   nothing.
2. Set `CAMPAIGN_AUTH_SHA` to the exact reviewer authorization digest.
3. Never edit amendments 1-10. Append amendment 11. It must:
   - name amendment 10 by exact SHA-256;
   - name the exact reviewer authorization SHA-256;
   - declare `scientific_change: NONE`;
   - disclose only authorization consumption and C28 portability;
   - carry the unchanged v5 population and resource-contract identities;
   - pin the complete final execution, verification and test surface.
4. Amendment 11 must have a strict schema and a self-integrity digest over its
   canonical body. The chain verifier must reject a missing, malformed,
   reordered, transplanted or self-consistently rewritten amendment 11.
5. Avoid circular binding deliberately:
   - the reviewer authorization record binds the audited pre-activation
     amendment-10 snapshot;
   - amendment 11 binds that reviewer record and the final consuming code;
   - the final dispatch will bind the reviewed Git commit containing both.
   Do not make the authorization record claim to pre-bind amendment 11.
6. `verify_campaign_authorization_record()` must also open and hash the owner
   ratification record named inside the authorization record, require the exact
   owner words and require its scope to cover this twelve-cell campaign. A
   canonical-looking but absent or unrelated intent digest must refuse.
7. Every cell result, heartbeat and final campaign report must carry both the
   authorization-record digest and amendment-11 digest. The final verifier
   re-derives them; producer labels grant nothing.

## C28: remove the operator-specific data path

Replace the literal home-directory fallback in
`tools/b4_authority.py:1251-1254` with:

- `resolve_predictor_root()`;
- one normalized logical relative path under that root;
- containment and regular-file checks before reading;
- hashing from the opened file used by the verifier.

No absolute path, host name or operator name may enter a public artifact. The
source-data digest and every scientific role remain unchanged.

## Acceptance Battery

At minimum, freeze and pass:

1. the PRE contradiction;
2. byte-exact reviewer authorization accepted after amendment 11;
3. one-byte authorization mutation refused before model/env/CUDA/output;
4. unrelated or missing owner-ratification object refused;
5. amendment 11 absent, malformed, relinked, reordered or altered refused;
6. any byte change to amendments 9 or 10 refused;
7. final-code mutation after amendment 11 refused;
8. authorization naming amendment 9 or a candidate-generated replacement
   refused;
9. absolute/traversing source path refused and logical-root replay accepted;
10. dry-run remains write-free and all prior C23-C26 batteries remain green.

Use CPU only. Do not launch the B4 campaign, open sealed-2025, promote a
checkpoint, touch a venue or service, or create another owner authorization.

## Return

Return the PRE/POST evidence, exact hashes, focused and full-suite counts from
the final tip, and the single final commit proposed for dispatch. The requested
disposition is:

`B4_AUTHORIZATION_CLOSURE_READY_FOR_FINAL_COMMIT_AUDIT`
