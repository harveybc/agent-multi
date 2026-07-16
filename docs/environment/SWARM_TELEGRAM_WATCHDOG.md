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
- alert when there is no observable fleet progress for 120 minutes;
- repeat unresolved alerts every 60 minutes and send one recovery message.

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
  --profile examples/campaigns/phase_1_asset_policy_fleet_v3/omega_profile.json \
  --apply
```

Use the matching Dragon or Gamma profile on those hosts.
