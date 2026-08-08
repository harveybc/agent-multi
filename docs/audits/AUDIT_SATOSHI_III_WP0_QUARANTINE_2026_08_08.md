# Audit: Satoshi III WP0 Invalid-Successor Quarantine

Date: 2026-08-08 America/Bogota  
Auditor: General Musashi, independent verifier  
Audited head: `agent-multi@b06bb7ca63f54fd78e8a65873d0673e0a300bc9c`  
Runtime mutation by this audit: none  
Network use by the reproducer: none

## 1. Verdict

**The real invalid successor is contained, but WP0 is not yet accepted.**

The important operational fact is good: the active queue path now contains a
typed supersession with `launch_eligible=false`; the original bytes exist under
their content-addressed retired path; and all five hashes in the correction
envelope independently recompute exactly. The historical M0 aggregation, final
table and fleet manifest were not rewritten.

The acceptance implementation nevertheless fails three adversarial cases. It
can report `claimed=false` while ignoring a claim in a SQLite ledger, can return
success for a malformed supersession that remains launch-eligible, and can
declare historical evidence immutable while all three canonical evidence
bindings are absent. The new executable is also absent from the engineering
surface declaration and makes the repository's structural CI guard fail. These
are defects in the proof and recovery contract, not evidence that the real
successor was consumed.

Independent raw-byte and read-only SQLite scans found no reference to the
retired successor SHA or filename in either declared runtime root. Source
inspection also found no executable consumer of this M0 successor filename or
schema in the repository. The currently available evidence therefore supports
"not consumed," but Satoshi's inspection function does not establish it.

Disposition:

- accept the **runtime containment subcriterion** of finding 159;
- do not close finding 159, because its phase-1 handoff correction remains;
- keep findings 160-165 open;
- correct findings 166-169 before WP0 is called complete;
- do not launch M1/M0-X from the withdrawn successor;
- continue unrelated valid fleet work; this audit does not authorize idle GPUs.

## 2. Verified Facts

The runtime evidence root is:

`~/.local/share/agent-multi/eth_sac_inner_curriculum_m0_20260807_v1`

Independent recomputation:

| Fact | SHA-256 / value |
| --- | --- |
| supersession | `447387cceeba56e6fc95882c0d178eef12478d1a0380f9ed8c0e13487b552f2a` |
| retired invalid successor | `aba67a2a8716c51972305dc8275aa0303fced4a713433a7fbf55a13a0529101e` |
| M0 aggregation | `245a328a7219a00a7f69853bf16143510d502e69e88d95fa719aac84b196158f` |
| final table | `20767034bf32ba2d541f7f6d40f4b5efaf63cc6879561a518648c2e71999b816` |
| fleet manifest | `61acd513378d8e99be03175b523dfdeb3b8b83473e8d766a493bedb96373f989` |
| active launch eligibility | `false` |

Satoshi's four focused tests pass. The predecessor reproducer still reports the
original M0 defects, as required by append-only evidence handling.

The declared campaign root currently contains 141 files: 46 JSON files, 19
SQLite/database files and 14 log/JSONL/CSV files. The decision root contains
305 files: 165 JSON and 92 log/JSONL/CSV files. Merely recording the directory
name as scanned does not establish which evidence formats were inspected.

## 3. Findings

### AUD-F1-20260808-166 - S3 - Consumer inspection ignores real ledger formats

`inspect_consumers()` appends a root to `ledgers_scanned`, then reads only
`*.json`. It ignores `campaign_history.sqlite`, other databases, JSONL, CSV,
logs and unreadable files. An adversarial `campaign_history.sqlite` containing
the exact successor SHA was reported as:

```text
claimed=false
ledgers_scanned=[<directory containing the SQLite claim>]
references=[]
```

Source: `tools/quarantine_inner_curriculum_successor.py:60-80`.

Impact: the correction envelope presents a stronger negative claim than the
inspection performed. The real roots currently contain no matching reference
under an independent broader scan, so this finding does not prove historical
consumption; it invalidates Satoshi's claimed proof of non-consumption.

Required correction: inventory the actual consumer code and each durable
source it can write. Scan typed JSON/JSONL and known SQLite tables read-only;
record every file, parser, SHA and result. An unreadable or unknown source must
make consumption status `unavailable`, never `false`. For this historical
queue, also record the code-level fact that no repository consumer references
the successor filename/schema and bind that fact to the audited code revision.

### AUD-F1-20260808-167 - S2 - Idempotency accepts an unsafe supersession

The second-invocation path checks only `schema`. It does not require
`launch_eligible=false`, verify the reason, validate the SHA, require the
retired original, or validate/recreate the correction envelope. The CLI exits
zero for `ALREADY_QUARANTINED` even when those guarantees are absent.

The independent reproducer supplied a supersession with
`launch_eligible=true` and no retired original. The function returned
`ALREADY_QUARANTINED`, `bytes_changed=0`, and would exit successfully. A second
case deleted the correction envelope after a valid first invocation; retry
again returned success and left the envelope absent.

Source: `tools/quarantine_inner_curriculum_successor.py:92-106` and `:174-181`.

Impact: corruption or interrupted recovery can be certified as successfully
contained while the queue remains executable or its evidence is incomplete.

Required correction: validate the complete supersession and envelope on every
invocation. Require false launch eligibility, exact finding/reason, lowercase
SHA-256 syntax, a retired path confined beneath the expected content-addressed
directory, matching retired bytes, all envelope bindings and recomputed hashes.
Any mismatch returns a nonzero fail-closed outcome. Add crash/retry fixtures.

### AUD-F1-20260808-168 - S3 - Missing canonical evidence is labeled immutable

The first invocation silently writes `null` for a missing aggregation, final
table or fleet manifest, but still sets `historical_evidence_immutable=true`
and returns `QUARANTINED` with exit zero.

Source: `tools/quarantine_inner_curriculum_successor.py:135-161`.

The independent reproducer omitted all three files. The operation succeeded,
all three bindings were null, and the immutable flag remained true.

Required correction: quarantine launch eligibility even if evidence is
damaged, because containment has priority, but do not certify WP0 complete.
Return a typed nonzero `QUARANTINED_EVIDENCE_INCOMPLETE` result and set the
evidence claim to unavailable until every required file exists and hashes. A
normal `QUARANTINED`/`ALREADY_QUARANTINED` success requires all bindings.

### AUD-GEN-20260808-169 - S3 - New mutating executable is undeclared

The full suite's engineering-surface guard reports:

```text
unclassified_new_executables=['quarantine_inner_curriculum_successor.py']
```

Source: `tests/test_engineering_surface_index.py:181-188` and
`tools/TOOL_DECLARATIONS.json`.

Impact: the repository cannot pass its own executable-inventory contract, and
operators/auditors lack the reviewed lifecycle, mutability and authority class
for a tool that changes launch state.

Required correction: add the tool to `tools/TOOL_DECLARATIONS.json` with its
true properties: campaign-frozen/supported lifecycle as selected by the
project's convention, `mutability=mutating`, campaign-operation authority,
General Satoshi III ownership, and an exact quarantine-only purpose. Re-run the
real-repository index test; do not grandfather it as unknown.

## 4. Reproduction

Canonical reproducer:

`docs/audits/evidence/SATOSHI_III_WP0_QUARANTINE_REPRO_2026_08_08.py`

Observed predicates:

```text
real runtime containment verified: true
real five envelope bindings recompute: true
malformed launch-eligible supersession accepted: true
missing canonical evidence accepted as immutable: true
missing envelope accepted on retry: true
SQLite claim ignored while directory reported scanned: true
network_used=false
runtime_mutation=false
```

Focused supplied tests:

```text
pytest -q tests/test_successor_quarantine.py -> 4 passed
full suite -> 742 passed, 14 failed
```

Of the 14 full-suite failures, 13 are the already disclosed isolated-worktree
fixture failures (ignored ETH `config_out.json` and sibling `/tmp/doin-node`
templates absent). One is a real regression introduced by this checkpoint:
finding 169's undeclared executable. It is not counted as environmental.

## 5. Acceptance Gate for the Correction

WP0 is accepted only when:

1. all four adversarial predicates above become false for the correct reason;
2. the real supersession remains byte-stable and launch-ineligible;
3. the retired original and all five bindings recompute exactly;
4. consumer status is supported by typed per-source inspection or is honestly
   unavailable;
5. first-run, retry, interrupted-write, malformed-state and missing-evidence
   tests pass;
6. the engineering-surface structural test passes with a reviewed declaration;
7. no historical M0 metric/artifact is rewritten; and
8. the correction commit and evidence packet are pushed and independently
   reproducible.
