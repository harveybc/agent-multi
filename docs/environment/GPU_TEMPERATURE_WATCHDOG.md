# GPU Temperature Watchdog

Status: implemented
Date: 2026-07-16

Every compute host runs the standard-library watchdog:

```text
tools/gpu_temperature_watchdog.py
```

The user crontab executes it every five minutes. System cron is used because
Dragon and Gamma do not currently have user linger enabled.

Operational policy:

- temperature alert: `>= 78 C`;
- recovery notification: `<= 72 C`;
- repeat an unresolved alert every 60 minutes;
- expected GPU count: Omega `1`, Dragon `1`, Gamma `2`;
- alert when `nvidia-smi` fails or the expected eGPU disappears;
- send a one-time recovery notification when NVIDIA monitoring returns;
- send through the Telegram bot and group already configured by Hermes 2;
- retain state and logs under
  `~/.local/state/agent-multi/gpu-temperature-watchdog/`.

The watchdog never changes power, clocks or fan controls. It detects and
reports conditions requiring physical inspection. Notification failures leave
the event unsent so cron retries it on the next five-minute check.

Installation is idempotent and preserves unrelated crontab entries:

```bash
python3 examples/scripts/install_gpu_temperature_watchdog.py \
  --expected-gpus 1 --apply
```

Use `--expected-gpus 2` on Gamma.
