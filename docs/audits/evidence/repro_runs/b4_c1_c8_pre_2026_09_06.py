"""PRE freeze for order @0ce52740 (C1-C8): the seven independent
runtime findings F1-F7 reproduced against the audited implementation."""
import hashlib
import importlib.util
import json
import multiprocessing
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
SCRATCH = Path.home() / ".cache" / "b4_c_pre"
import shutil
if SCRATCH.exists():
    shutil.rmtree(SCRATCH)
SCRATCH.mkdir(parents=True)


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


executor = _load("b4exec", "tools/b4_campaign_executor.py")
ledger_mod = _load("b4led", "tools/b4_campaign_ledger.py")
adj = _load("b4adj", "tools/b4_adjudicator.py")
MAT = Path.home() / ".local/share/agent-multi/b4_materialization_v3_20260905"

print("== F1: the scoring path is dead and the terminal is "
      "inadmissible ==")
src = (REPO / "tools/b4_campaign_executor.py").read_text()
body_exec = src[src.index("def execute_cell"):
                src.index("def main(")]
print("execute_cell calls score_frozen_checkpoint:",
      "score_frozen_checkpoint(" in body_exec)
import re
required_by_ledger = ("attempt_id", "per_bar_csv", "per_bar_sha256",
                      "sealed_2025_used")
missing = [k for k in required_by_ledger if k not in body_exec]
print("terminal fields missing from execute_cell COMPLETED:",
      missing)
assert "score_frozen_checkpoint(" not in body_exec
assert len(missing) == 4
print("-> the executor's normal COMPLETED output is REJECTED by "
      "verify_campaign_results (its own authoritative consumer); "
      "the return packet's published call graph claimed otherwise")

print("\n== F2: approved limits do not govern the process ==")
built = executor.build_economic_config(
    "o2024_seed101", MAT, SCRATCH / "out", "cpu")
cfg = built["config"]
print("effective wall_seconds =", cfg.get("budget_max_wall_seconds"))
print("effective rss_cap      =", cfg.get("rss_cap_bytes"))
print("effective cuda_cap     =", cfg.get("cuda_cap_bytes"))
print("effective thermal_cap  =", cfg.get("thermal_cap_celsius"))
contract = json.loads(
    (REPO / "docs/audits/evidence/"
     "B4_CAMPAIGN_RESOURCE_CONTRACT_PROPOSAL_2026_09_05.json"
     ).read_text())
print("authorized:", contract["per_cell_limits"][
    "wall_seconds_max"], contract["per_cell_limits"][
    "host_rss_bytes_max"], contract["per_cell_limits"][
    "cuda_allocated_bytes_max"], contract["per_cell_limits"][
    "gpu_temperature_celsius_max"])
assert cfg.get("budget_max_wall_seconds") == 57600.0
assert "rss_cap_bytes" not in cfg or cfg.get("rss_cap_bytes") is None
print("-> the builder copies the OLD gpu_economic mode; the "
      "resource contract and the future record govern nothing; no "
      "global 96 GPU-h consumer exists")

print("\n== F3: no per-cell isolation or observability ==")
for k in ("save_model", "checkpoint_bundle_dir", "cell_runtime_dir"):
    print(f"cell/builder sets {k}:", k in cfg)
assert all(k not in cfg for k in
           ("save_model", "checkpoint_bundle_dir",
            "cell_runtime_dir"))
print("-> the pipeline falls back to ./agent_model.zip shared "
      "between sequential cells; CellRuntime never activates")

print("\n== F4: two terminal writers can both win ==")


def _writer(tag, barrier, out_root, results):
    barrier.wait()
    try:
        executor.write_terminal(out_root, "o2022_seed101", "FAILED",
                                {"reason": tag})
        results.put(("success", tag))
    except SystemExit:
        results.put(("refused", tag))


mgr_barrier = multiprocessing.Barrier(2)
results_q = multiprocessing.Queue()
procs = [multiprocessing.Process(
    target=_writer, args=(t, mgr_barrier, SCRATCH / "race",
                          results_q)) for t in ("A", "B")]
[p.start() for p in procs]
[p.join() for p in procs]
outcomes = sorted([results_q.get(), results_q.get()])
print("writer_outcomes =", outcomes)
durable = json.loads((SCRATCH / "race" / "o2022_seed101" /
                      "B4_CELL_TERMINAL.json").read_text())
print("durable_reason  =", durable["reason"])
n_success = sum(1 for o, _ in outcomes if o == "success")
assert n_success == 2, "expected BOTH writers to claim success"
print("-> exists()+write_text() has no O_EXCL/fsync/CAS; no "
      "durable attempt claim before CUDA")

print("\n== F5: 'exact pairing' accepts different bars ==")
a = SCRATCH / "cand.csv"
b = SCRATCH / "ctrl.csv"
a.write_text("bar_index,net_return\n1,0.001\n2,0.002\n")
b.write_text("bar_index,net_return\n0,0.001\n1,0.002\n")
va = adj._per_bar_net(a, hashlib.sha256(a.read_bytes()).hexdigest(),
                      "cand")
vb = adj._per_bar_net(b, hashlib.sha256(b.read_bytes()).hexdigest(),
                      "ctrl")
import numpy as np
print("candidate indices [1,2] vs control [0,1]; vectors equal:",
      bool(np.array_equal(va, vb)))
assert np.array_equal(va, vb)
print("-> bar_index/datetime discarded; a shifted series enters "
      "G1/SPA/IQM as if paired")

print("\n== F6: the scoring helper does not satisfy the economic "
      "contract ==")
helper = src[src.index("def score_frozen_checkpoint"):
             src.index("def execute_cell")]
for field, present in (
        ("datetime identity", "DATE_TIME" in helper
         or "datetime" in helper),
        ("gross per-bar return", "gross_return" in helper),
        ("cost DELTA conversion", "delta" in helper.lower()),
        ("gross-costs=net reconciliation",
         "reconcil" in helper.lower())):
    print(f"helper publishes {field}:", present)
print("commission_paid copied raw from info:",
      'info.get("commission_paid"' in helper)
assert "DATE_TIME" not in helper
assert 'info.get("commission_paid"' in helper

print("\n== F7: public evidence is not portable ==")
ev = REPO / "docs/audits/evidence/b4_campaign_preparation_20260905"
hits = 0
for f in sorted(ev.rglob("*.json")):
    n = f.read_text().count("/home/")
    if n:
        hits += 1
print("committed evidence files containing absolute user paths:",
      hits)
assert hits >= 10
print("-> local topology published; evidence irreproducible from "
      "another checkout")

shutil.rmtree(SCRATCH)
print("\nPRE CONFIRMED: F1-F7 all reproduce")
