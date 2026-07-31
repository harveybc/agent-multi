# 02. Hermes Leverage and Token Economy

Version: 1.0.0
Date: 2026-07-30
Owner: Satoshi
Reviewer: Musashi
Constraint source: role spec section "HERMES AND CONTINUOUS OPERATION" and
document 24 section 9. Satoshi proposes here; only Musashi-issued task packets
(document 12) implement anything.

## 1. Cost Model and Standing Order

Satoshi (Claude, Mythos-class) is the most expensive unit of work in the audit
loop by a wide margin. Hermes local/cheap models and deterministic scripts are
orders of magnitude cheaper. Therefore:

| Tier | Who | Work | Cost rule |
| --- | --- | --- | --- |
| 0 | Deterministic scripts (cron/systemd) | collect, hash, diff, redact, run bounded test suites, package snapshots | no LLM at all |
| 1 | Hermes local/cheap model, isolated budget | summarize snapshot deltas, classify "changed / unchanged / anomalous", draft Telegram digests | never escalates to paid tiers on its own |
| 2 | Satoshi | audit reasoning over pre-collected evidence: contradiction hunting, contract verification, finding writing | invoked on change, event or schedule slot only |
| 3 | Satoshi deep session | weekly full cross-front audit, incident forensics | the only tier allowed to re-read work-plan documents broadly |

Standing order for every Satoshi session:

1. Never re-derive what tier 0/1 already delivered; read the snapshot first.
2. Never read a full log, database or repository tree when a hash, diff or
   extract answers the question.
3. Never poll. If evidence is not ready, state the missing input and stop.
4. Keep Layer A/B re-reads out of delta sessions; the recovery prompt carries
   the distilled context.
5. Prefer `rg` with tight patterns and bounded `head` over open-ended reads.
6. One heavy task per session; depth over breadth.

## 2. What Hermes May and May Not Do (from the spec)

Hermes agents may: run deterministic collectors, deliver redacted snapshots
and digests to Telegram, and run bounded local-model summaries with an
isolated provider/model budget.

Hermes agents may not: hold an unrestricted shell for audit purposes, touch
brokers/campaigns/secrets/publication, mark findings closed, or be commanded
free-form by Satoshi. Current production Hermes jobs must not be repurposed;
the audit gets its own identity and budget.

## 3. Delegation Map (token-heavy work pushed down)

| Work Satoshi would otherwise do | Delegated to | Tier |
| --- | --- | --- |
| Git provenance sweep across 11 repos | collector script | 0 |
| Supervisor/node/network API polling and extraction | collector script | 0 |
| Broker-observer freshness, GPU/RAM/temp, process identity | existing watchdogs + collector | 0 |
| Running focused test suites and capturing pass counts | scheduled bounded test runner | 0 |
| Dataset/artifact hash verification listings | collector script | 0 |
| "What changed since last audit" triage | Hermes local model over two snapshots | 1 |
| Daily Telegram audit digest ("no change" suppression) | Hermes audit identity | 1 |
| Deciding whether a change warrants invoking Satoshi | Hermes digest + user/Musashi judgment | 1 |
| Contradiction analysis, contract verification, findings | Satoshi | 2 |
| Weekly cross-front reasoning, incident root cause | Satoshi | 3 |

## 4. Draft Task-Packet Requests for Musashi

These are requests, not implementations. Each needs a document-12 packet with
base commit, allowed files and tests before anyone builds it.

### 4.1 `AUDIT-SNAPSHOT-COLLECTOR-001` (tier 0)

A deterministic script plus systemd user timer that materializes the snapshot
defined in file 03 to a local, redacted, hash-stamped JSON/markdown pair
(target under `~/.local/state/agent-multi/audit-snapshots/`), on a 6-hour
cadence and on demand. Read-only: git plumbing, existing HTTP GET APIs,
`nvidia-smi`, `ps`, bounded `journalctl`. Redaction: account fingerprints
only, no tokens, no raw IDs, no personal paths beyond the workspace. Retention:
last 28 snapshots. No LLM involved.

### 4.2 `AUDIT-TEST-EVIDENCE-002` (tier 0)

A scheduled bounded runner that executes the declared focused suites
(`agent-multi` safety/campaign, `gym-fx` full, `doin-node` focused) at most
once daily off-peak, capturing exit codes and pass counts into the snapshot.
Must not train, must respect a CPU/GPU guard so it never competes with a DOIN
candidate, and must record suite duration.

### 4.3 `HERMES-AUDIT-DIGEST-003` (tier 1)

A dedicated read-only Hermes audit identity, per spec constraints: input
limited to the two most recent redacted snapshots; output limited to a
Telegram digest ("changed/unchanged/anomalous" per section, with hashes);
isolated cheap/local model budget with hard caps; no shell, no broker, no
campaign, no secrets, no closure authority. Digest explicitly states "Satoshi
invocation suggested: yes/no + reason" as advice to humans, never as a command.

### 4.4 Sequencing

4.1 first (it alone removes most Satoshi collection cost), then 4.2, then 4.3.
Until 4.1 exists, the interim procedure in section 5 applies.

## 5. Interim Procedure (before packets are implemented)

- Satoshi self-collects at session start, but only via the fixed bounded
  command set in file 03 section 4 (a few hundred tokens), never by
  re-exploring repositories or re-reading the work-plan corpus.
- The user may paste a Hermes/watchdog Telegram summary into the session as a
  cheap snapshot substitute; Satoshi treats it as tier-1 input (untrusted,
  verify anomalies only).
- Delta sessions cap context loading at: recovery prompt, files 01/04, latest
  report, snapshot. Anything more requires an explicit reason recorded in the
  session's report.

## 6. Budget Accounting

Every audit report's "Commands and Queries" section notes which evidence came
pre-collected (tier 0/1) versus Satoshi-collected, so the user can see the
delegation ratio improve. Target steady state: Satoshi consumes snapshots and
writes findings; it collects raw evidence only when verifying an anomaly the
lower tiers surfaced.
