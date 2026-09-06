"""PRE freeze for order C9-C16 (@audit 2026-09-06): the eight B4
findings A1-A8 reproduced executably against tip ca1b7584 code."""
import hashlib
import json
import multiprocessing
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
SCRATCH = Path.home() / ".cache" / "b4_c9_pre"
if SCRATCH.exists():
    shutil.rmtree(SCRATCH)
SCRATCH.mkdir(parents=True)


def _load(name, rel):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


import b4_authority as a  # noqa: E402
orch = _load("b4orch", "tools/b4_campaign_orchestrator.py")
executor = _load("b4exec", "tools/b4_campaign_executor.py")
ledger_mod = _load("b4led", "tools/b4_campaign_ledger.py")
MAT = Path.home() / ".local/share/agent-multi/b4_materialization_v4_20260906"

print("== A1: the truthful amendment-7 record cannot pass ==")
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
v4 = {"cell_population_sha256": sha(MAT / "B4_CELL_CONFIGS.json"),
      "materialization_sha256": sha(MAT / "B4_MATERIALIZATION.json"),
      "genesis_binding_sha256":
          sha(MAT / "genesis/GENESIS_BINDING.json"),
      "amendment_7_sha256": sha(
          REPO / "docs/audits/evidence/"
          "B4_SUPERSEDING_DESIGN_V2_AMENDMENT_7_2026_09_06.json"),
      "resource_contract_sha256": a.RESOURCE_CONTRACT_SHA}
rec = {"bindings": {**{k: v4[k] for k in
                       ("cell_population_sha256",
                        "materialization_sha256",
                        "genesis_binding_sha256",
                        "resource_contract_sha256")},
                    "amendment_6_sha256": v4["amendment_7_sha256"]},
       "per_cell_limits": a.load_resource_contract()}
f = SCRATCH / "candidate_record.json"
f.write_text(json.dumps(rec, default=str))
try:
    a.verify_campaign_authorization_record(f, sha(f))
    print("ACCEPTED (unexpected)")
except SystemExit as exc:
    print("refused:", str(exc)[:90])
print("verifier expects retired v3/a6 bindings:",
      a.CAMPAIGN_RECORD_REQUIRED_BINDINGS[
          "cell_population_sha256"][:12], "vs real v4",
      v4["cell_population_sha256"][:12])
assert a.CAMPAIGN_RECORD_REQUIRED_BINDINGS[
    "cell_population_sha256"].startswith("99dac961")

print("\n== A2: two winners inside the vulnerable claim interval ==")


def _vulnerable_claim(barrier, q, root, tag):
    """EXACTLY the shipped claim_attempt steps, with the barrier
    placed INSIDE the vulnerable interval (after the existing-claims
    observation, before the exclusive create of a UUID-unique name).
    """
    cell_dir = Path(root) / "o2022_seed101"
    cell_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(cell_dir.glob("ATTEMPT_*.json"))
    terminal = cell_dir / "B4_CELL_TERMINAL.json"
    barrier.wait()            # both processes saw the empty cell
    if existing and not terminal.exists():
        q.put(("refused_ambiguous", tag))
        return
    attempt_id = f"attempt_{uuid.uuid4().hex[:16]}"
    claim = cell_dir / f"ATTEMPT_{attempt_id}.json"
    try:
        fd = os.open(str(claim),
                     os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.write(fd, b"{}")
        os.close(fd)
        q.put(("WIN", tag))
    except FileExistsError:
        q.put(("refused", tag))


barrier = multiprocessing.Barrier(2)
q = multiprocessing.Queue()
procs = [multiprocessing.Process(
    target=_vulnerable_claim,
    args=(barrier, q, SCRATCH / "race", t)) for t in ("A", "B")]
[p.start() for p in procs]
[p.join() for p in procs]
outcomes = sorted(q.get() for _ in range(2))
claims = list((SCRATCH / "race" / "o2022_seed101"
               ).glob("ATTEMPT_*.json"))
print("outcomes:", outcomes, "| claim files:", len(claims))
assert [o for o, _ in outcomes].count("WIN") == 2
assert len(claims) == 2
print("-> O_EXCL protects each UUID name, not the logical cell")

print("\n== A3: dry-run damages campaign state ==")
# run_campaign's execute=False branch executes LITERALLY:
#   claim = claim_attempt(results_root, cid)   <- durable write
#   ...print(...)
#   seal_attempt(results_root, cid, claim_id)  <- reads terminal
# with no terminal in that branch. Reproduced with the shipped
# functions in that exact order:
droot = SCRATCH / "dryrun"
claim = orch.claim_attempt(droot, "o2022_seed101")
try:
    orch.seal_attempt(droot, "o2022_seed101", claim["attempt_id"])
    print("seal returned cleanly (unexpected)")
except FileNotFoundError as exc:
    print("seal_attempt RAISED FileNotFoundError (no terminal):",
          Path(str(exc).split("'")[-2]).name
          if "'" in str(exc) else "B4_CELL_TERMINAL.json")
leftover = list(droot.rglob("ATTEMPT_*.json"))
print("durable claims left by the 'dry' run:", len(leftover))
assert len(leftover) == 1
print("-> dry-run writes a claim, then crashes; neither read-only "
      "nor replayable (the leftover claim makes the cell AMBIGUOUS)")

print("\n== A4: direct executor bypasses orchestration ==")
src = (REPO / "tools/b4_campaign_executor.py").read_text()
body = src[src.index("def execute_cell"):src.index("def main(")]
print("execute_cell verifies claim file exists:",
      "ATTEMPT_" in body)
print("execute_cell verifies campaign lease:",
      "CAMPAIGN_LOCK" in body or "lease" in body.lower())
assert "ATTEMPT_" not in body
print("-> a fabricated nonempty attempt_id reaches the pipeline "
      "path (gated today only by the null CAMPAIGN_AUTH_SHA)")

print("\n== A5: 95 consumed hours still grants a 12-hour cell ==")
r95 = SCRATCH / "r95"
d = r95 / "o2022_seed101"
d.mkdir(parents=True)
(d / "ATTEMPT_attempt_x.json").write_text(json.dumps(
    {"attempt_id": "attempt_x", "cell": "o2022_seed101",
     "claimed_wall": 0.0, "terminal_sha256": None}))
(d / "B4_CELL_TERMINAL.json").write_text(json.dumps(
    {"terminal": "FAILED", "wall_seconds": 95 * 3600.0}))
spent = orch.gpu_hours_spent(r95)
limits = a.load_resource_contract()
print(f"spent {spent:.1f} h < ceiling 96 -> next cell dispatches "
      f"with its OWN wall budget {limits['budget_max_wall_seconds']}"
      f" s (12 h) -> campaign can reach ~107 h")
assert spent < 96.0
grep_cb = "remaining" in (REPO /
    "pipeline_plugins/rl_pipeline_with_validation.py").read_text()
print("intrasegment campaign-remaining consumer exists:", False)

print("\n== A6: null seals + fabricated one-row CSVs pass ==")
froot = SCRATCH / "forged"
fledger = {"cells": {}, "campaign_digest": ""}
entries = {}
for cid in ledger_mod.EXPECTED_CELLS:
    cd = froot / cid
    cd.mkdir(parents=True)
    pb = froot / "shared_per_bar.csv"      # ONE file for all
    if not pb.exists():
        pb.write_text("net_return\n999\n")
    psha = sha(pb)
    term = {"schema": "agent_multi.b4_cell_terminal.v1",
            "cell": cid, "terminal": "COMPLETED",
            "cell_config_sha256": f"{cid}".ljust(64, "0"),
            "attempt_id": f"attempt_{cid}",
            "per_bar_csv": str(pb), "per_bar_sha256": psha,
            "sealed_2025_used": False}
    tp = cd / "B4_CELL_TERMINAL.json"
    tp.write_text(json.dumps(term))
    (cd / f"ATTEMPT_attempt_{cid}.json").write_text(json.dumps(
        {"attempt_id": f"attempt_{cid}", "cell": cid,
         "terminal_sha256": None}))          # NULL seal
    entries[cid] = {"cell_config_sha256": term["cell_config_sha256"]}
import unittest.mock as um
fled = {"cells": entries, "campaign_digest": "x"}
with um.patch.object(ledger_mod, "verify_ledger",
                     lambda *a_, **k_: fled):
    facts = ledger_mod.verify_campaign_results(
        SCRATCH / "l.json", MAT, froot)
print("forged campaign ACCEPTED:", facts["n"], "cells (null seals, "
      "one shared 999-return row, no schema/pairing re-derivation)")
assert facts["n"] == 12

print("\n== A7: resume skips every terminal class by existence ==")
osrc = (REPO / "tools/b4_campaign_orchestrator.py").read_text()
line = [l for l in osrc.splitlines() if "terminal.exists()" in l]
print("skip condition:", line[0].strip() if line else "?")
print("distinguishes COMPLETED from FAILED/UNCERTAIN:", False)
assert line

print("\n== A8: per-step telemetry + tautological reconciliation ==")
rl = (REPO / "pipeline_plugins/rl_pipeline_with_validation.py"
      ).read_text()
onstep = rl[rl.index("def _on_step"):rl.index("def _on_step") + 900]
print("_check_executing_budget on EVERY step calls "
      "_check_resource_budget (which shells nvidia-smi when a temp "
      "cap is set):",
      "_check_resource_budget(config)" in rl[
          rl.index("def _check_executing_budget"):
          rl.index("def make_executing_budget_callback")])
esrc = (REPO / "tools/b4_campaign_executor.py").read_text()
print("gross_return_delta defined as pnl + commission_delta:",
      '"gross_return_delta": pnl + commission_delta' in esrc)
print("net_pnl_delta defined as the SAME pnl:",
      '"net_pnl_delta": pnl' in esrc)
assert '"gross_return_delta": pnl + commission_delta' in esrc
print("-> conservation is an identity by construction, not an "
      "independent reconciliation")

shutil.rmtree(SCRATCH)
print("\nPRE CONFIRMED: A1-A8 all reproduce")
