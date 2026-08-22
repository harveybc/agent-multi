"""Reproducer for AUD-F1-20260821-REC-01..04.

PRE (at 4640bb27, from the audit's independent execution): `supervise`
was argparse `invalid choice`; `{}` classified completed; short commit
prefixes accepted and no launch binding existed; no directory fsync.

POST: run from the repo root; exit 0 with {"reproduced": false} when
every finding no longer reproduces.
"""
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TOOL = REPO / "tools" / "screen_recovery_controller.py"
spec = importlib.util.spec_from_file_location("rc", TOOL)
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)

surviving = {}
FULL = "a" * 40

with tempfile.TemporaryDirectory() as td:
    root = Path(td)

    def mk(**over):
        base = dict(seed=101, arm="plateau", frozen_commit=FULL,
                    config_sha256="c" * 64, gpu_mask="GPU-X",
                    output_dir=str(root / "out"),
                    report_path=str(root / "rep.json"),
                    log_path=str(root / "log"),
                    contract={}, argv=["/bin/true"], cwd=str(root),
                    clock=lambda: 1.0)
        base.update(over)
        return rc.write_attempt_manifest(root / "a", **base)

    # REC-01: supervise exists in the CLI and executes.
    helped = subprocess.run([sys.executable, str(TOOL), "supervise",
                             "--help"], capture_output=True, text=True)
    if helped.returncode != 0 or "invalid choice" in helped.stderr:
        surviving["REC-01"] = "supervise subcommand still absent"

    # REC-02: {} and typed negatives never complete.
    m = mk()
    (root / "rep.json").write_text("{}")
    if rc.classify_attempt(m)["state"] == rc.COMPLETED:
        surviving["REC-02"] = "empty JSON still classifies completed"

    # REC-03: short commit refused; artifact substitution refused.
    try:
        mk(seed=202, frozen_commit="93880beb")
        surviving["REC-03-short"] = "short commit accepted"
    except rc.RecoveryError:
        pass
    m3 = mk(seed=303, report_path=str(root / "rep3.json"),
            output_dir=str(root / "out3"),
            log_path=str(root / "log3"))
    art = Path(json.loads(m3.read_text())["launch_artifact"])
    art.write_text(art.read_text() + " ")
    try:
        rc.verify_launch_preconditions(
            m3, git_head=lambda: FULL, git_dirty=lambda: False,
            gpu_masks_present=lambda: ["GPU-X"],
            expected_config_sha256="c" * 64)
        surviving["REC-03-subst"] = "mutated launch artifact accepted"
    except rc.RecoveryError:
        pass

    # REC-04: a failing directory fsync must be loud.
    def boom(_p):
        raise OSError("fsync failed")
    try:
        mk(seed=404, fsync_dir=boom,
           report_path=str(root / "rep4.json"),
           output_dir=str(root / "out4"), log_path=str(root / "log4"))
        surviving["REC-04"] = "directory fsync failure was silent"
    except OSError:
        pass

print(json.dumps({"reproduced": bool(surviving),
                  "surviving": surviving}, indent=1))
sys.exit(1 if surviving else 0)
