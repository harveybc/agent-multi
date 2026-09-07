"""PRE freeze for order B4 C43-C48: the v6 runtime incident
reproduces from aa9b5a0e through the REAL APIs and a REAL SAC
learn call.

1. C43: build_economic_config() materializes NO progress path;
   make_progress_callback() returns None for that config; the
   productive learn call composes [None, budget_cb] — a REAL
   minimal SAC through build_economic_config -> run_pipeline ->
   model.learn dies with the incident's exact
   AttributeError: 'NoneType' object has no attribute
   'init_callback', before any training step.
2. C44: run_campaign() calls seal_attempt() only after a normal
   return — a deterministic in-boundary failure leaves a typed
   terminal UNSEALED and the cell adjudicates UNCERTAIN (the
   incident cell's live state).
3. C45/C48: the v6 incident objects stand byte-identical; the
   template's latest_amendment digest is STALE vs the physical
   amendment 14 (the productive verifier derives the physical
   value and was not fooled).

Zero GPU, zero campaign cells, zero writes under the v6 root
(probe writes go to a throwaway root)."""
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


orch = _load("b4orch_pre43", "tools/b4_campaign_orchestrator.py")
executor = _load("b4exec_pre43", "tools/b4_campaign_executor.py")

STATE = Path.home() / ".local/share/agent-multi"
V6 = STATE / "b4_campaign_results_v6_20260907"
MAT = STATE / "b4_materialization_v5_20260906"
CELL = "o2022_seed101"
ATTEMPT = "attempt_da479182eb8e4197"

print("== 1. C43: the effective config carries NO progress path "
      "==")
TR = Path.home() / ".cache/b4_c43_pre_root"
if TR.exists():
    shutil.rmtree(TR)
os.makedirs(TR, mode=0o700)
built = executor.build_economic_config(CELL, MAT, TR, "cpu")
cfg = built["config"]
print("training_progress_file:",
      cfg.get("training_progress_file"),
      "| progress_file:", cfg.get("progress_file"))
assert not cfg.get("training_progress_file")
assert not cfg.get("progress_file")
from agent_plugins._progress_callback import \
    make_progress_callback  # noqa: E402
pc = make_progress_callback(cfg, 1000)
print("make_progress_callback(effective_config) ->", pc)
assert pc is None
line = next(ln for ln in
            (REPO / "pipeline_plugins/rl_pipeline_with_validation"
                    ".py").read_text().splitlines()
            if "callback=[make_progress_callback" in ln)
print("productive learn call:", line.strip()[:76])
print("=> SB3 receives [None, budget_cb]")

print("\n== 2. C43: REAL minimal SAC dies with the incident's "
      "exact error ==")
# budgets stay wide enough that F9 does not stop BEFORE learn —
# the composition failure fires on entering model.learn of epoch
# 1, before any training step (the incident died at 3.1 s)
cfg["budget_max_wall_seconds"] = 600.0
from app.plugin_loader import load_plugin  # noqa: E402
agent_cls, _ = load_plugin("agent.plugins", cfg["agent_plugin"])
pipeline_cls, _ = load_plugin("pipeline.plugins",
                              cfg["pipeline_plugin"])
agent_plugin = agent_cls(cfg)
pipeline = pipeline_cls(cfg)
try:
    pipeline.run_pipeline(config=cfg, env_plugin=None,
                          agent_plugin=agent_plugin, mode="train")
    raise AssertionError("run_pipeline did not fail")
except AttributeError as exc:
    print("REAL SAC learn:", f"{type(exc).__name__}: {exc}")
    assert "init_callback" in str(exc)
except SystemExit as exc:
    # some wrappers convert; the cause must still be the callback
    print("SystemExit:", str(exc)[:90])
    assert "init_callback" in str(exc)
print("=> the incident's exact failure, before any training "
      "step")
shutil.rmtree(TR)

print("\n== 3. C44: a deterministic failure ends UNSEALED -> "
      "UNCERTAIN ==")
src = (REPO / "tools/b4_campaign_orchestrator.py").read_text()
seg = src[src.index("def run_campaign"):]
seg = seg[:seg.index("def main")]
exec_idx = seg.index("executor.execute_cell")
seal_idx = seg.index("seal_attempt")
has_try = "try:" in seg[max(0, exec_idx - 400):exec_idx]
print("execute_cell wrapped for failure-sealing:", has_try,
      "| seal only after normal return:", seal_idx > exec_idx)
assert not has_try
term = json.loads(
    (V6 / CELL / "B4_CELL_TERMINAL.json").read_text())
print("incident terminal:", term["terminal"], "| phase:",
      term["failed_phase"], "| wall:", term["wall_seconds"])
assert term["terminal"] == "FAILED" and \
    term["failed_phase"] == "pipeline"
assert term["attempt_id"] == ATTEMPT
seals = list((V6 / CELL).glob("SEAL_*"))
print("seal witnesses for the attempt:", seals)
assert seals == []
st = orch.seal_state(V6, CELL)
print("physical seal state:", st, "-> the incident cell "
      "adjudicates UNCERTAIN under its generation's code")
assert st == "UNSEALED"

print("\n== 4. C45/C47: incident preserved; template digest "
      "STALE ==")
objs = sorted(p.name for p in (V6 / CELL).iterdir()
              if p.is_file())
print("v6 incident objects:", objs)
digests = {p: hashlib.sha256(
    (V6 / CELL / p).read_bytes()).hexdigest()[:12] for p in objs}
print("digests:", digests)
tmpl = json.loads(
    (REPO / "docs/audits/evidence/"
            "MUSASHI_B4_V6_RECOVERY_AUDIT_TEMPLATE_2026_09_07"
            ".json").read_text())
phys = hashlib.sha256(
    (REPO / "docs/audits/evidence/"
            "B4_SUPERSEDING_DESIGN_V2_AMENDMENT_14_2026_09_07"
            ".json").read_bytes()).hexdigest()
print("template latest_amendment:",
      tmpl["latest_amendment_sha256"][:12],
      "| physical a14:", phys[:12])
assert tmpl["latest_amendment_sha256"] != phys
assert tmpl["latest_amendment_sha256"].startswith("e36c5e1a")
assert phys.startswith("2b40913c")
print("=> the committed template publishes a stale digest (the "
      "productive verifier derives the physical value)")

print("\nPRE CONFIRMED: C43 callback composition, C44 unsealed "
      "failure, stale template — all frozen at aa9b5a0e")
