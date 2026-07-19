# DOIN Swarm Telegram Watchdog

Status: implemented
Date: 2026-07-16

The watchdog is intentionally separate from DOIN consensus and the campaign
supervisor:

```text
tools/swarm_telegram_watchdog.py
```

It reads the replicated campaign supervisor API and durable campaign history.
It never starts, stops, repairs or modifies a DOIN process.

## Notification policy

- execute every five minutes from each physical host;
- use the first fresh online participant in campaign-plan order as the
  notification owner: Omega, then Dragon, then Gamma;
- notify each completed optimization once using job ID, exact fitness metric,
  available return/RAP/drawdown/Sharpe/trade metrics, champion peer and
  content-addressed model artifact;
- deduplicate completion notifications by `job_id + artifact_sha256`, which is
  identical across replicas even when their local completion timestamps differ;
- alert after a five-minute grace period for offline or stale machines,
  unhealthy workers, supervisor alerts, mismatched jobs/domains/plans,
  divergent genesis/population lineages or generations;
- inspect local `/proc` on every host and alert when an extra, missing or
  duplicated DOIN configuration reveals a parallel local swarm;
- compare the configured profile plan with the live supervisor plan before
  interpreting processes, fleet state or completion history; a stale profile
  produces one explicit configuration alert and suppresses misleading
  parallel-swarm diagnostics;
- alert when there is no observable fleet progress for 120 minutes;
- repeat unresolved alerts every 60 minutes and send one recovery message.
- retry transient Telegram transport failures three times with bounded
  backoff; permanent HTTP errors still fail immediately and remain visible in
  the cron log.

The primary/failover owner avoids ordinary duplicate global notifications.
During a network partition, duplicate alerts are preferred over silently
missing a dangerous condition.

State is retained under:

```text
~/.local/state/agent-multi/swarm-telegram-watchdog/
```

Installation preserves unrelated crontab entries:

```bash
python3 examples/scripts/install_swarm_telegram_watchdog.py \
  --profile examples/campaigns/<active-campaign>/omega_profile.json \
  --apply
```

Use the matching Dragon or Gamma profile on those hosts.
The installer must be rerun whenever the campaign supervisor changes to a new
profile. The runtime plan-ID guard is a fail-closed diagnostic, not a substitute
for updating the cron entry.
