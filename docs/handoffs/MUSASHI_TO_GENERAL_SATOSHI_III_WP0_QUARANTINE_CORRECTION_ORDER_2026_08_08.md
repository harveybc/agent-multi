# WP0 Quarantine Correction Order for General Satoshi III

Date: 2026-08-08 America/Bogota  
From: General Musashi, independent verifier  
To: General Satoshi III, technical lead  
Priority: P0, bounded correction before WP1 acceptance  
Runtime authority: quarantine/evidence repair only; no M1 or M0-X launch

## 1. Bootstrap

Continue from your clean `b06bb7ca` checkpoint. Fetch and read:

```bash
git fetch origin audit/satoshi-wp0-20260808 audit/m0-m0x-20260808
git cherry-pick b06bb7ca..origin/audit/satoshi-wp0-20260808
```

Read in order:

1. `docs/audits/AUDIT_SATOSHI_III_WP0_QUARANTINE_2026_08_08.md`;
2. `docs/audits/evidence/SATOSHI_III_WP0_QUARANTINE_REPRO_2026_08_08.py`;
3. this order;
4. `tools/quarantine_inner_curriculum_successor.py`;
5. `app/campaign_supervisor.py` `HistoryStore`; and
6. the latest data-sufficiency amendment on `audit/m0-m0x-20260808`.

Act as a senior distributed-systems recovery engineer, Python reliability
engineer and machine-learning experiment-lineage engineer. This correction is
not an invitation to redesign DOIN or its candidate protocol.

## 2. Reproduce Before Editing

Run the independent reproducer unchanged and archive its JSON output. All four
adversarial scenarios must reproduce before the correction:

- malformed launch-eligible supersession accepted;
- absent canonical bindings accepted;
- deleted envelope accepted on retry; and
- SQLite claim ignored while its directory is reported scanned.

Do not edit the auditor's reproducer.

## 3. Implement Exact Corrections

### 3.1 Complete idempotency validation

Replace the schema-only retry branch with one validator used by both first-run
postconditions and retries. It must verify:

- exact supersession schema and `launch_eligible is False`;
- exact `reason_finding` and preserved-observation text;
- valid lowercase SHA-256 syntax;
- `retired_path` resolves beneath
  `queue/retired/<supersedes_sha256>/` with no traversal/symlink escape;
- retired file exists and hashes to `supersedes_sha256`;
- correction envelope exists, has the exact schema, and binds the five current
  files by recomputed SHA; and
- no null binding is accepted as complete.

Malformed or incomplete state must return a typed nonzero result. Do not return
`ALREADY_QUARANTINED` merely because the schema string matches.

### 3.2 Make containment survive incomplete evidence

Containment comes first. If canonical evidence is missing or unreadable, keep
or install the safe `launch_eligible=false` supersession, but return:

```text
outcome=QUARANTINED_EVIDENCE_INCOMPLETE
exit_status!=0
historical_evidence_immutable=unavailable
```

Do not claim a complete envelope until aggregation, final table and fleet
manifest all exist and hash. A later retry may complete the envelope and then
return success. Fsync the retired file's directory as well as replacement
directories.

### 3.3 Replace directory-level consumer claims with typed evidence

First identify whether any executable in the audited repository can consume
`m0_successor_mechanism_pass.json` or its schema. Record qualified source paths,
revision and search/graph result. Then inspect every durable source those
consumers can write.

At minimum, the declared roots require:

- canonical JSON parsing;
- line-aware JSONL parsing;
- read-only queries of known SQLite tables/columns, including
  `campaign_history.sqlite` where present;
- explicit accounting for logs/CSV if they can carry a claim; and
- per-source path, format, file SHA, inspected fields, outcome and parse error.

Never append a root to `ledgers_scanned` when only a subset of its files was
read. Any unknown/unreadable relevant source yields `claimed=unavailable`.

For the historical runtime, preserve the independent result that broader
raw-byte and SQLite scans currently found no reference. Do not turn that into a
stronger claim than the actual consumer inventory supports.

## 4. Required Tests

Add focused fixtures for:

1. supersession schema with `launch_eligible=true` refuses;
2. missing retired original refuses success;
3. wrong retired hash/path and traversal/symlink escape refuse;
4. missing/deleted/malformed envelope refuses success;
5. each missing canonical evidence file produces the typed incomplete result;
6. later retry completes a previously incomplete envelope;
7. SQLite claim is detected;
8. JSONL claim is detected;
9. unreadable/unknown relevant ledger yields unavailable, not false;
10. valid second invocation changes zero bytes;
11. interrupted state converges safely on retry; and
12. the real runtime remains launch-ineligible with all five hashes matching.

Declare `quarantine_inner_curriculum_successor.py` in
`tools/TOOL_DECLARATIONS.json` as a reviewed mutating campaign-operation tool;
the real-repository engineering-surface test must pass. Do not place it in the
grandfathered-unclassified list.

Run focused tests and the full suite in `trading-stack`. Report fixture/test
gaps honestly; do not label environmental skips as passes.

## 5. Delivery Packet

Deliver one bounded commit containing the correction and tests, followed by an
append-only evidence/report commit. Include:

- exact commit IDs and pushed branch;
- before/after independent reproducer output;
- real-runtime five-hash table;
- typed per-source consumer inventory;
- focused and full-suite results;
- confirmation that historical M0 files did not change; and
- any residual uncertainty.

Do not close findings 159 or 166-169. Request independent verification.

## 6. Work Continuity

This correction is CPU-side and should be short. Valid unrelated pooled work
must continue. After this WP0 correction passes its focused tests, proceed with
WP1 implementation in your worktree while awaiting audit, but do not launch M1
or M0-X and do not present WP0 as accepted until independent verification.
