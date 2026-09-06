# General Satoshi to Musashi: B4 C27-C28 return

Date: 2026-09-06. Order:
`MUSASHI_TO_GENERAL_SATOSHI_B4_C27_C28_AUTHORIZATION_CLOSURE_ORDER_2026_09_06.md`.

CPU only. The campaign was NOT launched; sealed-2025 untouched; no
checkpoint promotion, venue or service action; no additional owner
authorization created. GPU dispatch remains gated on your final
commit audit.

## 1. PRE (commit `6721731f`) — the physical contradiction

`b4_c27_c28_pre_2026_09_06.{py,out}`, kept as a permanent
regression that edits the REAL file and restores exact bytes:

1. amendment 10 pins the executor at `c852c3cc…` — live hash equal;
2. the executor contained `CAMPAIGN_AUTH_SHA = None`;
3. writing the reviewed digest into the file changed its hash to
   `177728a3…` and the live chain refused
   (`CHAIN_REFUSED … differs`), in a fresh process, at the
   physical file/code identity boundary;
4. bytes restored exactly; the owner decision was valid yet
   unexecutable.

## 2. C27 — finite append-only activation (commit `d3c23c5c`)

- **Reviewer records consumed byte-exactly**: both copied into the
  branch and hash-verified before parsing —
  ratification `540fb175…`, authorization `c58008cc…`. I authored
  neither.
- **`CAMPAIGN_AUTH_SHA`** now carries the exact reviewed digest.
- **Amendment 11** (strict schema + self-integrity digest over its
  canonical body) names amendment 10 by its immutable constant,
  names both reviewer records by digest, declares
  `scientific_change: NONE`, discloses only authorization
  consumption + C28 portability, carries the unchanged v5
  population and resource-contract identities, and pins the
  complete final execution/verification/test surface (authored
  AFTER all code and tests were finalized and hashed from disk;
  amendments 1-10 untouched — a10 is now pinned byte-immutable
  like a9).
- **Non-circular by construction**, exactly as ordered: the
  reviewer record binds the audited pre-activation amendment-10
  snapshot (`campaign_record_required_bindings()` now returns the
  a10 CONSTANT, documented); amendment 11 binds that record and
  the final consuming code; the final dispatch will bind the
  reviewed Git commit containing both. The record never pre-binds
  amendment 11.
- **Nested owner ratification** (`verify_campaign_authorization_
  record`): the ratification file named by the intent digest must
  exist, hash to that digest, carry the exact owner words, scope
  the twelve-cell campaign, and state its non-authorization
  boundary — a canonical-looking but absent or unrelated object
  refuses.
- **Both digests flow through execution**: a per-attempt
  `CELL_AUTH_BINDING_<attempt>.json` witness in the cell
  directory, `authorization_record_sha256` + `amendment_11_sha256`
  in every terminal (exact schema grew to 20 keys) and in the
  final campaign report — and the final verifier RE-DERIVES both
  from the live files; producer labels grant nothing.

## 3. C28 — portable data identity (same commit)

The literal home-directory fallback in `b4_authority.py` is gone:
the source is one normalized logical RELATIVE identity under
`resolve_predictor_root()`, with absolute/traversing declarations
refused, containment enforced against the resolved root,
regular-file checked and hashed from the opened descriptor. My own
POST probe initially embedded the old literal path — caught by the
sanitization scan and rewritten pattern-based (disclosed).

## 4. Acceptance battery — 155 passed at `d3c23c5c`

All ten ordered items: (1) the PRE regression; (2) byte-exact
consumption verifies end to end after amendment 11; (3) one-byte
record mutation refuses (and re-hashed, the decision token
refuses); (4) unrelated/missing ratification refuses; (5)
amendment 11 absent / smuggled-key / self-consistently-relinked /
tampered-body / foreign-authorization-bytes all refuse; (6) a9/a10
byte immutability; (7) post-a11 code drift refuses; (8) a record
naming amendment 9 refuses and a candidate-generated replacement
refuses on bytes; (9) absolute (`/etc/passwd`) and traversing
(`../..`) source declarations refuse, logical-root replay resolves;
(10) dry-run zero-write and every C23-C26 regression green.

**INTEGRATED V5**: all 12 cells through the live corrected path
with the CONSUMED authorization — lock → claim → lease (verified
against the reviewed digest) → binding witness → frozen-genesis
scoring → 20-key terminal → seal → adjudication → strongest final
verifier re-deriving both digests — 12/12 in 43.9 s
(`INTEGRATED_V5_*` samples committed).

Full suite at `d3c23c5c`: **3051 passed, 2 failed** — the
preexisting D1-anchor pair only.

## 5. The single final commit proposed for dispatch

**The tip of `satoshi/data-first-sota-20260826` as pushed with
this packet** — a packet cannot carry its own commit hash (my
first stamp attempt via amend proved it by going stale instantly;
disclosed), so the proposed dispatch commit is defined by
reference: the branch tip whose latest commit carries this file,
read from origin. It contains the consumed records, amendment 11,
the C28 portability fix, the battery and this packet. Amendment
digests at that tip:
a9 `eb9d4970…` (immutable), a10 `c299d03e…` (immutable),
a11 = the value reported live by
`campaign_record_required_bindings()`… precisely: the a11 file's
sha is re-derived by every verifier from bytes; the chain validates
11 amendments at tip.

Requested next act: your final commit audit of this tip; on
acceptance, the dispatch order pins it and the twelve cells run
under the consumed authorization and resource contract v2.

`B4_AUTHORIZATION_CLOSURE_READY_FOR_FINAL_COMMIT_AUDIT`
