"""POST for order B4 C43-C48: telemetry mandatory and typed,
failure terminals sealed, generation v7, template truthful. Zero
campaign GPU, zero sealed-2025, zero promotion; the real v7 acta
is ABSENT and the launch CLOSED."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
import b4_authority as b4a  # noqa: E402


def _load(name, rel):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


executor = _load("b4exec_post43", "tools/b4_campaign_executor.py")
STATE = Path.home() / ".local/share/agent-multi"
V6 = STATE / "b4_campaign_results_v6_20260907"
MAT = STATE / "b4_materialization_v5_20260906"
CELL = "o2022_seed101"

print("== 1. C43: telemetry materialized; composition typed ==")
TR = Path.home() / ".cache/b4_c43_post_root"
if TR.exists():
    shutil.rmtree(TR)
os.makedirs(TR, mode=0o700)
built = executor.build_economic_config(CELL, MAT, TR, "cpu")
cfg = built["config"]
print("progress file (cell-contained, cell-unique):",
      Path(cfg["training_progress_file"]).name)
assert cfg["training_progress_file"] == cfg["progress_file"]
Path(cfg["training_progress_file"]).relative_to(TR / CELL)
assert cfg["b4_require_progress"] is True
import pipeline_plugins.rl_pipeline_with_validation as rp  # noqa
bc = rp.make_executing_budget_callback(cfg, 0.0)
cbs = rp.compose_learn_callbacks(cfg, 1000, bc)
assert len(cbs) == 2 and all(c is not None for c in cbs)
print("composed:", [type(c).__name__ for c in cbs])
c2 = dict(cfg)
c2.pop("training_progress_file")
c2.pop("progress_file")
try:
    rp.compose_learn_callbacks(c2, 1000, bc)
    raise AssertionError("no refusal without telemetry")
except RuntimeError as exc:
    print("B4 without telemetry:", str(exc)[:70])
shutil.rmtree(TR)
print("REAL SAC evidence: docs/audits/evidence/repro_runs/"
      "b4_c46_real_sac_probe_2026_09_07.out (steps 179, updates "
      "50, F9.2 exact stop, progress advanced, heartbeat under "
      "the cell)")

print("\n== 2. C45/C48: v6 incident byte-identical; historical "
      "dry-run still UNCERTAIN ==")
want = {"B4_CELL_TERMINAL.json": "836dcc0ba87f",
        "CELL_AUTH_BINDING_attempt_da479182eb8e4197.json":
            "ac983084c869",
        "CLAIM_b4_campaign_generation_v6_20260907.json":
            "b980b697e3f4",
        "LEASE_attempt_da479182eb8e4197.json": "7d7abe393279",
        "resolved_origin_contract.json": "1693838e0ae5"}
for n, w in want.items():
    got = hashlib.sha256(
        (V6 / CELL / n).read_bytes()).hexdigest()[:12]
    assert got == w, (n, got)
print("five incident objects byte-identical")
SC = Path.home() / ".cache/b4_c48_hist_checkout"
subprocess.run(["git", "-C", str(REPO), "worktree", "remove",
                "--force", str(SC)], capture_output=True)
r = subprocess.run(["git", "-C", str(REPO), "worktree", "add",
                    "--detach", str(SC),
                    "aa9b5a0e84e9e406375acf5acd571712cfef7aa0"],
                   capture_output=True, text=True)
assert r.returncode == 0, r.stderr[-200:]
rr = subprocess.run(
    [sys.executable, str(SC / "tools/b4_campaign_orchestrator.py"),
     "--materialization-root", str(MAT),
     "--ledger", str(V6 / "CAMPAIGN_LEDGER.json"),
     "--results-root", str(V6), "--device", "cpu"],
    capture_output=True, text=True, cwd=str(SC))
out = rr.stdout + rr.stderr
import re
m = re.search(r'"o2022_seed101": "([A-Z_]+)"', out)
print("historical (aa9b5a0e) dry-run classifies the incident "
      "cell:", m.group(1) if m else out[-160:])
assert m and m.group(1) == "UNCERTAIN"
subprocess.run(["git", "-C", str(REPO), "worktree", "remove",
                "--force", str(SC)], capture_output=True)

print("\n== 3. C45: generation v7; 3.1 s debt accounted ==")
assert b4a.CAMPAIGN_GENERATION == \
    "b4_campaign_generation_v7_20260907"
assert b4a.PRIOR_GENERATIONS_GPU_SECONDS == 39.1
v7led = json.loads(
    (STATE / "b4_campaign_results_v7_20260907" /
     "CAMPAIGN_LEDGER.json").read_text())
gp = v7led["generation_provenance"]
assert gp["supersedes_generation"] == b4a.V6_GENERATION
assert gp["incident_lineage"][
    "v6_runtime_incident_order_sha256"] == \
    b4a.V6_INCIDENT_ORDER_SHA
print("v7 ledger provenance: supersedes v6, lineage v5+v6, "
      "prior charge 39.1 s (36.0 + 3.1)")

print("\n== 4. C47: templates truthful; v7 acta path distinct; "
      "gate closed ==")
t6 = json.loads((REPO / "docs/audits/evidence/"
                 "MUSASHI_B4_V6_RECOVERY_AUDIT_TEMPLATE_"
                 "2026_09_07.json").read_text())
assert t6["latest_amendment_sha256"].startswith("<HISTORICAL")
t7 = json.loads((REPO / "docs/audits/evidence/"
                 "MUSASHI_B4_V7_RUNTIME_AUDIT_TEMPLATE_"
                 "2026_09_07.json").read_text())
assert t7["latest_amendment_sha256"].startswith("<DERIVE")
assert t7["schema"].endswith("v3")
print("stale v6 template -> unambiguous placeholder; v7 template "
      "is placeholder-only (grants nothing)")
assert b4a.RECOVERY_AUDIT_RECORD_PATH.name == \
    "MUSASHI_B4_V7_RUNTIME_AUDIT_RECORD.json"
assert b4a.CONSUMED_V6_ACTA_PATH.name != \
    b4a.RECOVERY_AUDIT_RECORD_PATH.name
try:
    b4a.require_v6_launch_open()
    raise AssertionError("gate open without the v7 acta")
except SystemExit as exc:
    print("v7 gate:", str(exc)[:86])
    assert "READY_FOR_EXTERNAL_MUSASHI_ACTA" in str(exc)

print("\nPOST CONFIRMED: C43-C47 executable; v6 history intact; "
      "v7 closed pending the Musashi acta")
