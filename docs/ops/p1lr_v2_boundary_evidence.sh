#!/usr/bin/env bash
# READ-ONLY P1LR v2 boundary evidence. Usage: SEED=101 PHASE=before bash this
set -uo pipefail
"$HOME/anaconda3/envs/trading-stack/bin/python" - "${SEED:?}" "${PHASE:?}" <<'PY'
import hashlib, json, os, re, socket, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
seed, phase = sys.argv[1], sys.argv[2]
home = Path.home()
runtime = home / "Documents/GitHub/.runtime/agent-multi-p1lr-v2-924910fe"
contract = runtime / ("examples/config/phase_3_eth_sac_dynamics/"
                      "p1_difficulty_lr_factorial_v2.json")
envfile = home / f".config/agent-multi/p1lr-v2/seed{seed}.env.conf"
pat = re.compile(rf"python\s+\S*p1_difficulty_lr_factorial\.py\s+--seed\s+{seed}\s+--mode\s+decision")
pid = None
for entry in sorted(Path("/proc").glob("[0-9]*")):
    try:
        cmd = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode()
    except OSError:
        continue
    if pat.search(cmd) and (entry / "comm").read_text().strip().startswith("python"):
        pid = int(entry.name)
        break
def proc(field, default=None):
    if pid is None:
        return default
    try:
        if field == "cwd":
            return os.readlink(f"/proc/{pid}/cwd")
        if field == "cuda":
            env = Path(f"/proc/{pid}/environ").read_bytes().decode().split("\0")
            return next((v.split("=", 1)[1] for v in env
                         if v.startswith("CUDA_VISIBLE_DEVICES=")), None)
        if field == "start":
            return subprocess.run(["ps", "-o", "lstart=", "-p", str(pid)],
                                  capture_output=True, text=True).stdout.strip()
    except OSError:
        return default
    return default
def envval(key):
    try:
        for line in envfile.read_text().splitlines():
            if line.startswith(key + "="):
                return line.split("=", 1)[1]
    except OSError:
        pass
    return None
def unit(prop):
    r = subprocess.run(["systemctl", "--user", "show", "-p", prop, "--value",
                        f"p1lr-decision@{seed}.service"],
                       capture_output=True, text=True)
    return (r.stdout or "").strip() or None
mainpid = unit("MainPID")
control = None
tool = home / "Documents/GitHub/agent-multi/tools/p1lr_identity_supervision.py"
if tool.is_file():
    r = subprocess.run([sys.executable, str(tool), "verify-control"],
                       capture_output=True, text=True)
    try:
        control = json.loads(r.stdout)
    except ValueError:
        control = {"verdict": "UNREADABLE", "stderr": r.stderr[-200:]}
print(json.dumps({
    "schema": "agent_multi.p1lr_v2_boundary_evidence.v1",
    "phase": phase, "host": socket.gethostname(), "seed": int(seed),
    "captured_utc": datetime.now(timezone.utc).isoformat(),
    "pid": pid, "pid_started": proc("start"), "proc_cwd": proc("cwd"),
    "launcher": (None if pid is None else
                 ("systemd" if mainpid == str(pid) else "nohup")),
    "unit_active_state": unit("ActiveState"),
    "unit_main_pid": mainpid,
    "unit_enabled": unit("UnitFileState"),
    "chain_id": envval("P1LR_EXPECTED_CHAIN_ID"),
    "experiment": envval("P1LR_EXPERIMENT"),
    "contract_path": str(contract),
    "contract_sha256": (hashlib.sha256(contract.read_bytes()).hexdigest()
                        if contract.is_file() else None),
    "contract_sha256_env": envval("P1LR_CONTRACT_SHA256"),
    "output_root": envval("P1LR_OUTPUT_ROOT"),
    "cuda_uuid_declared": envval("CUDA_VISIBLE_DEVICES"),
    "cuda_uuid_process": proc("cuda"),
    "control_bundle": control,
}, indent=1, sort_keys=True))
PY
