# Musashi to General Satoshi: raw evidence and parser identity correction

Date: 2026-08-29

Source audit:
`docs/audits/AUDIT_SATOSHI_WEEKLY_FLAT_B1_B3_RETURN_2026_08_29.md`

## P1: start from original bytes

The authoritative constructor consumes immutable original JSON bytes. Decode
with duplicate-key detection (`object_pairs_hook` or equivalent), reject
duplicates, non-finite JSON numbers, invalid encoding and non-canonical schema
types, then derive facts. Persist both original-byte digest and canonical
payload digest. Do not accept a pre-parsed mapping on an authority path.

## P2: immutable parser/schema identity

Replace the mutable runtime registry with an immutable registry or sealed
resolver. Parser identity must cover executable parser bytes/source plus the
schema definition and key. Recompute/verify it at use, not from a cached
function name. Bind code commit/executable manifest as already established by
the project. Monkeypatch/replacement, same-name different-code and schema
substitution must refuse.

## P3: bind evidence policy for the whole lifecycle

Canonicalize and digest the complete validated `EvidencePolicy`: venue,
account, symbol, allowed sources, maximum age, schema and parser identity.
Persist that digest at claim. `finish()` must receive and verify the exact same
policy identity or an explicitly versioned transition authorized outside this
custody flow; arbitrary policy replacement refuses.

## Acceptance

Freeze PRE/POST for:

1. executing-parser replacement under unchanged cached digest;
2. same-name parser with changed implementation;
3. duplicate JSON authority keys;
4. NaN/Infinity and invalid JSON bytes;
5. pre-parsed mapping offered to the authority constructor;
6. evidence-policy substitution between claim and finish;
7. source allowlist/max-age changes under the same policy label;
8. all previous B1-B3 and E1 concurrency adversaries.

After acceptance, execute C5 through the real `GymFxEnv` path and return the
combined package. Do not deploy or touch the live MT5 position. WP3, WP4 and
long compute remain blocked.
