"""Reproducer for order 2026-08-22 corrections C1..C4.

PRE (at 9433ebb3, from the audit): foreign commit/config with matching
composites exited 0 from the diagnostic; schema prefix matching
classified `agent_multi.wp4_smoke_malicious.v999` completed; the
launch artifact's temp file was never fsynced; no preflight existed.

POST: exit 0 with {"reproduced": false} when all four are closed.
"""
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, REPO / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

surviving = {}

# C1: the audit's counterexample — copy seed101 plateau report, change
# commit + config hash, keep seed/data/composites: must REFUSE.
diag = _load("plateau_post_intervention_diagnostic")
E = REPO / "docs/audits/evidence/plateau_screen_20260821"
with tempfile.TemporaryDirectory() as td:
    d = Path(td)
    for s in (101,):
        for arm in ("fixed", "plateau"):
            src = (E / f"seed{s}_{arm}_report.json").read_text()
            (d / f"seed{s}_{arm}_report.json").write_text(src)
    doc = json.loads((d / "seed101_plateau_report.json").read_text())
    doc["commit"] = "deadbeef" + "0" * 32
    doc["config_sha256"] = "e" * 64
    (d / "seed101_plateau_report.json").write_text(json.dumps(doc))
    try:
        diag.diagnose_seed(d / "seed101_fixed_report.json",
                           d / "seed101_plateau_report.json", 101)
        surviving["C1"] = ("foreign commit/config still enters the "
                           "diagnostic")
    except diag.DiagnosticError:
        pass

# C2: exact allowlist — malicious suffix schema never completes.
rc = _load("screen_recovery_controller")
src = (REPO / "tools" / "screen_recovery_controller.py").read_text()
if "startswith(REPORT_SCHEMA" in src:
    surviving["C2"] = "schema prefix matching still present"
if "REPORT_SCHEMA_ALLOWLIST" not in src:
    surviving["C2-allowlist"] = "no exact allowlist"

# C3: launch artifact durable write — file fsync failure must refuse
# and leave no manifest.
with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    def boom(_fd):
        raise OSError("file fsync failed")
    try:
        rc.write_attempt_manifest(
            root / "a", seed=1, arm="fixed", frozen_commit="a" * 40,
            config_sha256="c" * 64, gpu_mask="G",
            output_dir=str(root / "o"), report_path=str(root / "r"),
            log_path=str(root / "l"), contract={}, argv=["/bin/true"],
            cwd=str(root), fsync_file=boom)
        surviving["C3"] = "file-fsync failure acknowledged a manifest"
    except OSError:
        if list((root / "a").glob("attempt_*.json")):
            surviving["C3-manifest"] = "manifest exists after failure"

# C4: preflight artifact exists, uses the proposed contract, and its
# mechanical rule was applied.
art = E / "PREFLIGHT_EARLY_INTERVENTION_2026_08_22.json"
if not art.is_file():
    surviving["C4"] = "no preflight artifact"
else:
    doc = json.loads(art.read_text())
    if doc.get("proposed_contract", {}).get("lr_patience") != 8:
        surviving["C4-contract"] = "wrong proposed contract"
    if "seeds_with_treatment_window" not in doc:
        surviving["C4-rule"] = "no mechanical window rule"
    if "not plateau scheduling as a universal" not in doc.get(
            "scope_statement", ""):
        surviving["C4-scope"] = "scope statement missing"

print(json.dumps({"reproduced": bool(surviving),
                  "surviving": surviving}, indent=1))
sys.exit(1 if surviving else 0)
