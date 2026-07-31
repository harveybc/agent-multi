# Satoshi Post-Fix Verification Task

Version: 1.1.0
Date: 2026-07-31
Role: independent auditor
Technical lead: Musashi

## 1. Required Posture

Act as a senior machine-learning systems auditor, distributed-systems
engineer, software security reviewer and trading-platform reliability
reviewer. Be rigorous, pragmatic and evidence-first. Do not infer missing
behavior from directory names or test taxonomy. Read the implementation and
execute the smallest bounded verification that can prove or refute a claim.

Your job is verification, not architecture ownership. Preserve reported
history, accept corrections when evidence contradicts an earlier breadth-first
inference, and open a finding only for a reproducible defect with a concrete
impact.

## 2. Safety and Authority

You may read tracked repositories, redacted snapshots, SQLite schemas and
bounded test output. You may run CPU-only existing tests.

You must not:

- edit code, configs, services, jobs, broker state or blockchain state;
- use broker, Telegram or Moltbook credentials;
- place or simulate a live order;
- stop, restart, migrate or repair a DOIN worker;
- mutate chain files, choose a canonical tip or force synchronization;
- run training, GPU workloads, package installs or broad network scans;
- expose secrets, raw account IDs or personal paths in the report.

If a required fact needs authority outside this scope, record the smallest
missing evidence and stop that check.

## 3. Required Inputs

Read in this order:

1. `docs/audits/CODEX_AUDIT_TRIAGE_2026_07_31.md`
2. `docs/audits/work_plan/04_OPEN_FINDINGS_REGISTER.md`
3. `docs/audits/AUDIT_FULL_CROSS_FRONT_2026_07_30.md`
4. `docs/audits/AUDIT_QUALITY_SECURITY_TESTING_2026_07_31.md`
5. `docs/work_plan/23_SOCIAL_INTELLIGENCE_AND_OPERATIONAL_CONTINUITY.md`
6. newest `~/.local/state/agent-multi/audit-snapshots/latest.json`
7. `~/.local/state/agent-multi/audit-test-evidence/latest.json`

Use the repository revisions recorded by the newest snapshot. If the
agent-multi working tree is dirty or differs from `origin/master`, report that
and do not claim closure.

## 4. Verification Tasks

### A. Social findings 006, 007, 008 and 013

Verify from code and tests that:

1. multilingual and encoded malicious fixtures are flagged;
2. flagged rows are excluded before SQL `LIMIT`, while the withheld count
   covers the entire period;
3. a draft without an existing OLAP source is rejected;
4. duplicate retrieval replaces stale relevance values and complete-corpus
   re-scoring occurs after collection;
5. ranking distinguishes a specific fresh phrase from generic or old content;
6. triage and review have separate Hermes wrapper identities;
7. a model call records provider, model, tier, config/prompt/packet hashes and
   conservative token reservation;
8. exceeding input, daily or monthly caps sets `wakeAgent=false`;
9. no code claims actual token usage or dollar cost when the provider does not
   expose it.

Run only:

```bash
pytest -q tests/unit/test_social_intelligence.py tests/unit/test_audit_test_evidence.py
```

### B. Audit test evidence

Verify the latest packet schema, all suite exit codes, pass counts, durations,
 command/output hashes and repository revisions. Confirm the timer is enabled
and that the runner skips when Omega owns a candidate or lacks declared
resource headroom.

### C. Corrected quality findings

Reissue corrected state changes:

- 009: S3 unless a current unsafe deployment/regression is demonstrated;
- 010: inventory the ten named invariants and mark existing versus missing;
- 011: withdraw the repository-wide claim; name only specific missing
  operational scenarios;
- 012: acknowledge the canonical 143-pin lock and reconstruction procedure;
  retain per-repository lock/SBOM work as S4 release hardening.

Do not use test-directory layout as a proxy for behavioral absence.

### D. Equal-height fork

Run `AT-F1-011` only after generation advances beyond 2 or finalization
advances beyond height 2. Compare accepted transactions, fork-choice behavior
and finalized anchors read-only. If neither boundary has advanced, report
`deferred_no_new_boundary`; do not repeat hours-of-persistence as new evidence
and do not recommend chain mutation.

### E. Academic audit gap

The preceding cross-front audit omitted academic preservation and
reproducibility as an explicit dimension. Verify that both files now exist and
are internally consistent:

- `docs/work_plan/25_ACADEMIC_PUBLICATION_AND_REPRODUCIBILITY.md`
- `docs/handoffs/SATOSHI_ACADEMIC_PUBLICATION_AUDIT_TASK_2026_07_31.md`

Do not claim that the academic program is audited from this existence check.
After this post-fix report, execute the separate academic task as the next
heavy audit task.

## 5. Acceptance and Output

Write one report:

`docs/audits/AUDIT_POST_FIX_VERIFICATION_2026_07_31.md`

For each finding provide:

- `verified_closed`, `partially_verified`, `rejected_as_written` or `open`;
- exact file/line or packet path;
- command and result summary;
- residual risk and smallest next action.

Update the register only after writing the report. Do not edit Musashi's
triage document or either originating audit report. End with a concise list of
facts that require owner action; an empty list is valid. Also name
`SATOSHI_ACADEMIC_PUBLICATION_AUDIT_TASK_2026_07_31.md` as the next scheduled
task unless an `S0` or `S1` finding preempts it.
