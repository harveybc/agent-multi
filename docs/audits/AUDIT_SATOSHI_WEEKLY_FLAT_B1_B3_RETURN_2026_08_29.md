# Audit: weekly-flat B1-B3 return

Date: 2026-08-29

Audited commits: `gym-fx@7d50ddb`, `agent-multi@8961c1ad`

Verdict: **REVISE.** Payload/fact separation and self-selected freshness were
removed, but parser identity and raw-input integrity remain bypassable.

## Accepted progress

- Authority-bearing facts are now derived from the payload representation.
- Evidence no longer carries its own maximum age.
- Source, schema and freshness are checked against an `EvidencePolicy`.
- Original split-fact and one-year-old evidence examples now refuse.
- Concurrent claim and terminal races remain in the focused suite.

## Critical: parser substitution is accepted under the same digest

`PARSER_DIGESTS` hashes venue/type/version and `fn.__name__`, not parser code or
schema. It is computed once, while `PARSERS` remains a mutable public dict.
Replacing `PARSERS[key]` does not change the expected digest.

Reproduced: replace the protection parser with a function returning accepted
SL/TP, parse a payload containing both as false, and the resulting evidence is
accepted with the original parser digest and forged true facts.

The committed parser-substitution test only changes the digest field; it does
not substitute the executing parser.

## Critical: duplicate JSON keys cannot be detected

`DirectEvidence.parse` accepts an already-parsed `Mapping`. Duplicate keys were
therefore discarded before canonicalization. Reproduced raw JSON:

`{"positions_total":7,"positions_total":0,"orders_total":0}`

Standard JSON parsing retains the final zero, after which evidence reports
flat. This contradicts the stated rejection of duplicate authority fields.
Authority must begin from the original bytes, not a lossy mapping.

## High: evidence policy is not bound across custody lifecycle

The custody record stores selected source/schema/parser facts but not a digest
of the complete `EvidencePolicy`, including maximum age and source allowlist.
`finish()` accepts a newly supplied policy, allowing policy substitution between
claim and terminal reconciliation.

## Test result

Focused custody + session suite reproduced at `68/68`; the executing-parser
substitution and duplicate-byte examples are absent.

## Disposition

Preserve the useful payload-derived design, correct the three boundaries below,
then run C5. WP3, WP4, deployment and long compute remain blocked.

