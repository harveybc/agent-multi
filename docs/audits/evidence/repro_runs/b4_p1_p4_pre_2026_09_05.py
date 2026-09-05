"""PRE freeze for order @9fb017e3 (B4-P1..P4): no positive
authorization path, caller-rebindable science, presence-only
complete-envelope verification, and a campaign-sized GPU mode."""
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
SCRATCH = Path.home() / ".cache" / "b4_p_pre"
if SCRATCH.exists():
    shutil.rmtree(SCRATCH)
SCRATCH.mkdir(parents=True)


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


import b4_authority as a  # noqa: E402
runner = _load("b4run", "tools/b4_run_cell.py")
MAT = Path.home() / ".local/share/agent-multi/b4_materialization_v2_20260905"

print("== P1: no positive authorization path exists ==")
src = (REPO / "tools/b4_run_cell.py").read_text()
print("runner reads an authorization artifact:",
      "OWNER_AUTHORIZATION" in src)
print("unconditional non-CPU refusal:",
      'if args.device != "cpu":' in src)
print('always assigns run_cfg["device"] = "cpu":',
      'run_cfg["device"] = "cpu"' in src)
assert "OWNER_AUTHORIZATION" not in src
assert 'if args.device != "cpu":' in src

print("\n== P2: a caller can self-rebind a different scientific "
      "cell ==")
fake = SCRATCH / "mat"
fake.mkdir()
(fake / "genesis").mkdir()
cells = json.loads((MAT / "B4_CELL_CONFIGS.json").read_text())
cfg = cells["o2024_seed101"]["effective_config"]
cfg["learning_rate"] = 0.123
new_digest = hashlib.sha256(json.dumps(
    cfg, sort_keys=True, default=str).encode()).hexdigest()
cells["o2024_seed101"]["config_sha256"] = new_digest
(fake / "B4_CELL_CONFIGS.json").write_text(json.dumps(cells))
binding = json.loads(
    (MAT / "genesis" / "GENESIS_BINDING.json").read_text())
binding["binding"]["o2024_seed101"] = new_digest
(fake / "genesis" / "GENESIS_BINDING.json").write_text(
    json.dumps(binding))
try:
    cell = runner.load_cell(fake, "o2024_seed101")
    print("ACCEPTED_SELF_REBOUND_CELL",
          cell["effective_config"]["learning_rate"])
    accepted = True
except SystemExit as exc:
    print("refused:", exc)
    accepted = False
assert accepted, "expected the self-rebound cell to be accepted"

print("\n== P3: complete-envelope verification is presence-only ==")
V6 = (REPO / "docs/audits/evidence/"
      "screen_b_rule_arms_v6_e_corrected_20260905")
copy = SCRATCH / "v6"
copy.mkdir()
for f in ("RUN_MANIFEST.json", "trial_ledger.jsonl",
          "ENVELOPE_CALIBRATION_o2022.json",
          "ENVELOPE_CALIBRATION_o2023.json",
          "ENVELOPE_CALIBRATION_o2024.json"):
    shutil.copy(V6 / f, copy / f)
packet = json.loads((V6 / "SCREEN_B_RESULTS.json").read_text())
packet["results"][0]["complete_envelope_digest"] = "0" * 64
(copy / "SCREEN_B_RESULTS.json").write_text(json.dumps(packet))
design = json.loads(a.DESIGN_PATH.read_text())
try:
    facts = a.verify_comparator_population(copy, design)
    print("ACCEPTED_FORGED_COMPLETE_ENVELOPE_DIGEST",
          facts["n_results"], facts["n_ledger"])
    p3 = True
except SystemExit as exc:
    print("refused:", exc)
    p3 = False
assert p3, "expected the forged digest to be accepted"

print("\n== P4: the materialized GPU mode is a campaign budget ==")
econ = cells["o2024_seed101"]["effective_config"][
    "execution_modes"]["gpu_economic"]
print("env steps:", econ["budget_max_env_steps"],
      "| updates:", econ["budget_max_updates"],
      "| wall:", econ["budget_max_wall_seconds"],
      "| thermal:", econ["thermal_cap_celsius"],
      "| cuda limit present:",
      any("cuda" in k for k in econ))
assert econ["budget_max_env_steps"] == 40_020_000
assert econ["thermal_cap_celsius"] == 95
assert not any("cuda" in k for k in econ)

shutil.rmtree(SCRATCH)
print("\nPRE CONFIRMED: all four findings reproduce")
