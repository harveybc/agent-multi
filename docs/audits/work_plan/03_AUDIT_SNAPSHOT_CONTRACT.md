# 03. Audit Snapshot Contract

Version: 1.0.0
Date: 2026-07-30
Owner: Satoshi
Reviewer: Musashi

## 1. Purpose

One compact, redacted, hash-stamped evidence packet replaces open-ended
exploration at the start of every Satoshi session. It is produced by the
tier-0 collector (file 02 section 4.1) or, until that exists, by the interim
command set in section 4 below.

## 2. Required Sections

Target size: under 32 KB total. Every section carries a collection timestamp.

```text
audit_snapshot.v1
  meta:        generated_at, host, collector version, snapshot sha256
  provenance:  per repo (11): branch, HEAD, dirty count, upstream ahead/behind
  runtime:
    supervisor: plan_id, plan_hash, phase, job_id/index, per-worker
                {status, gen, population_fingerprint, chain_height,
                 finalized_hash, component_versions, owned_candidates,
                 api/optimization errors}, alerts
    node:       domains, uptime, peer summary (counts only)
    eta:        per-worker candidate duration stats, fleet cand/h
  machines:    per host: reachable, GPU util/mem/temp, RAM/swap/disk headroom,
               OOM-kill count since last snapshot
  brokers:     per venue observer: mode (read-only?), last heartbeat age,
               unexpected-exposure flag, account fingerprint (never raw ID)
  tests:       per declared suite: last run time, exit code, pass/fail counts
  watchdogs:   active alerts, dedup-suppressed count, last Telegram delivery
  hashes:      active dataset sha256, campaign plan hash, latest champion
               artifact sha256 (if archived)
  delta:       list of sections whose hash changed vs previous snapshot
```

## 3. Rules

- Redaction: fingerprints only; no tokens, credentials, raw account IDs,
  customer data or private personal paths.
- The snapshot is evidence input, not truth: Satoshi verifies anomalies
  against primary sources before writing a finding on them.
- A missing section is recorded as missing, never silently filled.
- Snapshots stay out of Git; reports reference their sha256 and local path.

## 4. Interim Manual Collection (bounded, until the collector exists)

The fixed command set, all read-only:

```bash
# provenance (11 repos)
for r in agent-multi trading-contracts gym-fx heuristic-strategy doin-core \
         doin-node doin-plugins prediction_provider lts financial-data predictor; do
  git -C ~/Documents/GitHub/$r rev-parse --abbrev-ref HEAD HEAD 2>/dev/null | tr '\n' ' '; \
  git -C ~/Documents/GitHub/$r status --porcelain | wc -l; done

# runtime
curl -s -m 8 http://127.0.0.1:8795/api/status
curl -s -m 8 http://127.0.0.1:8795/api/network
# machine
uname -n; uptime; free -h | head -2; df -h ~ | tail -1
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
ps aux | grep -E 'doin|campaign_supervisor|ibkr|alpaca|mt5|lts' | grep -v grep
```

Satoshi extracts what it needs from the JSON with a short Python one-liner
rather than reading the raw dump into context.

## 5. Consumption Order in a Session

1. `delta` section first: unchanged sections are skipped entirely.
2. Alerts and errors second.
3. Only then the sections relevant to the scheduled task.
