"""B4 C51: the production cell CHILD.

Runs exactly one cell attempt inside its own process group under
the supervisor's capability: validates the capability binding,
starts the heartbeat thread (monotonic sequence + child pid —
liveness identity only, NEVER progress), then calls the existing
accepted `execute_cell` unchanged. Scientific behavior is
byte-identical to the in-process path; only the process boundary
and telemetry are new.
"""
import argparse
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

HEARTBEAT_PERIOD_S = 5.0


def _self_sha(doc, key):
    return hashlib.sha256(json.dumps(
        {k: doc[k] for k in sorted(doc) if k != key},
        sort_keys=True).encode()).hexdigest()


def load_capability(path: Path) -> dict:
    cap = json.loads(Path(path).read_text())
    if cap.get("schema") != "agent_multi.b4_cell_capability.v1":
        raise SystemExit("REFUSED: foreign capability schema")
    if _self_sha(cap, "capability_sha256") != \
            cap.get("capability_sha256"):
        raise SystemExit("REFUSED: capability self-digest does "
                         "not re-derive")
    # the child must be running under the parent that issued it
    if cap["parent_pid"] != os.getppid():
        raise SystemExit(
            "REFUSED: capability parent PID does not match the "
            "actual parent — a transplanted capability never "
            "authorizes an attempt")
    return cap


def start_heartbeat(cell_dir: Path):
    def beat():
        seq = 0
        while True:
            doc = {"schema":
                   "agent_multi.b4_supervisor_heartbeat.v1",
                   "seq": seq, "child_pid": os.getpid(),
                   "monotonic": time.monotonic()}
            tmp = cell_dir / f".hb_{os.getpid()}_{seq}"
            try:
                tmp.write_text(json.dumps(doc, sort_keys=True))
                os.chmod(tmp, 0o600)
                os.replace(tmp,
                           cell_dir / "SUPERVISOR_HEARTBEAT.json")
            except OSError:
                pass
            seq += 1
            time.sleep(HEARTBEAT_PERIOD_S)
    t = threading.Thread(target=beat, daemon=True,
                         name="b4-supervisor-heartbeat")
    t.start()
    return t


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capability", type=Path, required=True)
    ap.add_argument("--cell", type=str, required=True)
    ap.add_argument("--mat-root", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--device", type=str, required=True)
    ap.add_argument("--lease", type=Path, required=True)
    a = ap.parse_args(argv)
    cap = load_capability(a.capability)
    if cap["cell"] != a.cell:
        raise SystemExit("REFUSED: capability names a different "
                         "cell")
    cell_dir = Path(a.out_root) / a.cell
    start_heartbeat(cell_dir)
    import b4_campaign_executor as executor
    return executor.execute_cell(
        a.cell, a.mat_root, a.out_root, a.device,
        lease_path=a.lease,
        global_wall_remaining_seconds=cap[
            "global_remaining_seconds"])


if __name__ == "__main__":
    raise SystemExit(main())
