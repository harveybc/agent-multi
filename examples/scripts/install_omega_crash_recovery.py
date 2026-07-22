#!/usr/bin/env python3
"""Install Omega kernel-oops recovery and the latest Ubuntu kernel patch."""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


SYSCTL_PATH = Path("/etc/sysctl.d/90-agent-multi-crash-recovery.conf")
MODULE_PATH = Path("/etc/modules-load.d/agent-multi-watchdog.conf")
SYSTEMD_PATH = Path("/etc/systemd/system.conf.d/90-agent-multi-watchdog.conf")


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--skip-kernel-update", action="store_true")
    args = parser.parse_args()
    if not args.apply:
        print("Would enable panic-on-oops recovery, probe the AMD watchdog and install the latest kernel patch.")
        return 0
    if os.geteuid() != 0:
        raise PermissionError("run this installer with sudo")

    write(SYSCTL_PATH, "kernel.panic_on_oops = 1\nkernel.panic = 30\n")
    subprocess.run(["sysctl", "--system"], check=True)
    subprocess.run(["modprobe", "sp5100_tco"], check=False)
    if Path("/dev/watchdog0").exists():
        write(MODULE_PATH, "sp5100_tco\n")
        write(
            SYSTEMD_PATH,
            "[Manager]\nRuntimeWatchdogSec=60s\nRebootWatchdogSec=10min\n",
        )
        print("AMD hardware watchdog detected; systemd watchdog configuration installed for next boot")
    else:
        print("No hardware watchdog device exposed; panic-on-oops recovery remains enabled")
    if not args.skip_kernel_update:
        subprocess.run(
            ["apt-get", "install", "-y", "linux-image-generic", "linux-headers-generic"],
            check=True,
        )
    print("Crash recovery installed. Reboot at a swarm generation boundary to load the new kernel.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
