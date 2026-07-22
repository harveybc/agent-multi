# Memory and Kernel Recovery

Status: implemented on Omega
Date: 2026-07-21

Omega's 2026-07-21 freeze was not an OOM event. The previous boot contains no
kernel or `systemd-oomd` kill; its final record is a kernel `NULL pointer
dereference`. Earlier in that boot the NVIDIA driver emitted `Xid 56` events.

The operational mitigation has three layers:

1. `doin-campaign-supervisor.service` uses `MemoryHigh=20G`, `MemoryMax=24G`
   and `MemorySwapMax=6G`. A userspace leak is contained inside the DOIN cgroup
   instead of exhausting the host.
2. `tools/memory_pressure_watchdog.py` runs every two minutes, records host,
   swap, cgroup and `memory.events` evidence, and sends Telegram alerts through
   the existing Hermes configuration.
3. Root-level recovery sets `kernel.panic_on_oops=1`, reboots 30 seconds after
   a kernel oops, probes the AMD `sp5100_tco` hardware watchdog and installs the
   latest Ubuntu kernel patch.

Install the user watchdog idempotently:

```bash
python3 examples/scripts/install_memory_pressure_watchdog.py --apply
```

Install root recovery and the current kernel patch:

```bash
sudo python3 examples/scripts/install_omega_crash_recovery.py --apply
```

Reboot at a swarm generation boundary after the kernel package finishes. The
campaign supervisor then verifies plan, domain, seed, genesis, population,
dataset and component lineage before claiming new work.
