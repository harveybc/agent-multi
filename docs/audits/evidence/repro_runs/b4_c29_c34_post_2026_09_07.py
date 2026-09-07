"""POST for order B4 C29-C34: every incident mechanism is dead and
the append-only v6 recovery stands. Zero GPU, zero score, zero
sealed-2025 reads; the v5 root and its ambiguous attempt preserved
byte-exact."""
import contextlib
import hashlib
import io
import json
import os
import shutil
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


orch = _load("b4orch_post", "tools/b4_campaign_orchestrator.py")
executor = _load("b4exec_post", "tools/b4_campaign_executor.py")
ledger_mod = _load("b4led_post", "tools/b4_campaign_ledger.py")

STATE = Path.home() / ".local/share/agent-multi"
V5 = STATE / "b4_campaign_results_20260906"
V6 = STATE / "b4_campaign_results_v6_20260907"
MAT = STATE / "b4_materialization_v5_20260906"
CELL = "o2022_seed101"
ATTEMPT = "attempt_6e46ebe59eb842ca"

print("== 1. v5 incident objects preserved byte-exact ==")
want = {
    "CLAIM_b4_campaign_generation_v5_20260906.json":
        "68e17eaaabe261b89636eb90bc19c61f240cd6c5a4152e64ca0d00"
        "b2c7799b23",
    f"LEASE_{ATTEMPT}.json":
        "2b67722512c9cc0931da4aaf132d6486f59521cabc79e8649ccdc0"
        "3541218300",
    f"CELL_AUTH_BINDING_{ATTEMPT}.json":
        "37867f886352bd5b79bd004d0606dd894c450fd6e7eb3a7420b138"
        "3bc45b1692",
    "resolved_origin_contract.json":
        "1693838e0ae530898926cb4c85bdddf9c598ba64af16e96aa2ea13"
        "97ef4e0e44"}
for n, w in want.items():
    got = hashlib.sha256((V5 / CELL / n).read_bytes()).hexdigest()
    assert got == w, n
print("four digests match; v5 objects preserved byte-exact")
# v6 code never re-adjudicates the superseded root — it REFUSES it
try:
    orch.run_campaign(MAT, V5 / "CAMPAIGN_LEDGER.json", V5, "cpu",
                      execute=False)
    raise AssertionError("v6 code interpreted the v5 root")
except SystemExit as exc:
    assert "foreign-generation object" in str(exc)
    print("v6 over the v5 root:", str(exc)[:80])

print("\n== 2. dry-run now refuses the missing plugin BEFORE any "
      "claim ==")
import importlib.metadata as md  # noqa: E402
import app.plugin_loader as apl  # noqa: E402
_real_md, _real_apl = md.entry_points, apl.entry_points


def _view():
    class _V:
        def select(self, group):
            eps = _real_md().select(group=group)
            if group == "agent.plugins":
                return [e for e in eps if e.name != "sac_agent"]
            return eps
    return _V()


md.entry_points = _view
apl.entry_points = _view
try:
    try:
        orch.run_campaign(MAT, V6 / "CAMPAIGN_LEDGER.json", V6,
                          "cpu", execute=False)
        raise AssertionError("dry-run stayed plugin-blind")
    except SystemExit as exc:
        print("dry-run:", str(exc)[:80])
        assert "entry point 'sac_agent' absent" in str(exc)

    print("\n== 3. the escape is dead: typed terminal, never "
          "AMBIGUOUS_CLAIM ==")
    TR = Path.home() / ".cache/b4_c29_post_20260907"
    if TR.exists():
        shutil.rmtree(TR)
    os.makedirs(TR, mode=0o700)
    with orch.GlobalLock(TR):
        claim = orch.claim_attempt(TR, CELL)
        lease = orch.issue_lease(TR, CELL, claim,
                                 executor.CAMPAIGN_AUTH_SHA, MAT)
        try:
            executor.execute_cell(CELL, MAT, TR, "cpu",
                                  lease_path=lease)
            raise AssertionError("execute_cell did not fail")
        except ImportError:
            pass
    term = json.loads(
        (TR / CELL / "B4_CELL_TERMINAL.json").read_text())
    st = orch.adjudicate_cell_state(TR, CELL)
    print(f"terminal: {term['terminal']} | phase: "
          f"{term['failed_phase']} | state: {st}")
    assert term["terminal"] == "FAILED_PLUGIN_ENVIRONMENT"
    assert term["failed_phase"] == "plugin_load"
    assert "sac_agent not found" in term["reason"]
    assert st != "AMBIGUOUS_CLAIM"
    shutil.rmtree(TR)
finally:
    md.entry_points = _real_md
    apl.entry_points = _real_apl

print("\n== 4. v6 accounting: 0.01h charged, budget never "
      "restarted ==")
limits = b4a.load_resource_contract()
import tempfile  # noqa: E402
with tempfile.TemporaryDirectory() as td:
    remaining = orch.remaining_global_seconds(Path(td), limits)
ceiling = float(limits["global_gpu_hours_ceiling"]) * 3600.0
print(f"fresh-root remaining = ceiling - 36.0s -> "
      f"{remaining / 3600.0:.2f} h")
assert remaining == ceiling - 36.0

print("\n== 5. the honest v6 dry-run passes; the LAUNCH stays "
      "closed ==")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    rc = orch.run_campaign(MAT, V6 / "CAMPAIGN_LEDGER.json", V6,
                           "cpu", execute=False)
out_txt = buf.getvalue()
rep = json.loads(out_txt[out_txt.rindex("{\n \"dry_run\""):])
print("dry-run rc:", rc, "| remaining:",
      rep["gpu_hours_remaining"], "h | preflight chain:",
      rep["environment_preflight"]["amendment_chain_length"])
assert rc == 0 and rep["gpu_hours_remaining"] == 95.99
try:
    orch.run_campaign(MAT, V6 / "CAMPAIGN_LEDGER.json", V6,
                      "cpu", execute=True)
    raise AssertionError("launch opened without the acta")
except SystemExit as exc:
    print("launch gate:", str(exc)[:90])
    assert "READY_FOR_FINAL_MUSASHI_AUDIT" in str(exc)

print("\n== 6. v6 never consumes v5 objects ==")
try:
    ledger_mod.verify_ledger(V5 / "CAMPAIGN_LEDGER.json", MAT)
    raise AssertionError("v5 mutable ledger accepted")
except SystemExit as exc:
    assert "explicit generation provenance" in str(exc)
    print("v5 ledger as v6 genesis:", str(exc)[:70])

print("\n== 7. twelve scientific identities equal v5<->v6 ==")
v5led = json.loads((V5 / "CAMPAIGN_LEDGER.json").read_text())
v6led = json.loads((V6 / "CAMPAIGN_LEDGER.json").read_text())
for cid in v5led["cells"]:
    for k in ("cell_config_sha256", "genesis_binding_sha256",
              "genesis_container_sha256", "genesis_tensor_sha256"):
        assert v5led["cells"][cid][k] == v6led["cells"][cid][k]
assert v5led["campaign_digest"] == v6led["campaign_digest"]
gp = v6led["generation_provenance"]
print("campaign_digest equal; provenance names incident",
      gp["incident_record_sha256"][:12], "| scientific_change",
      gp["scientific_change"])
assert gp["scientific_change"] == "NONE"

print("\n== 8. chain: 12 amendments; published a12 can never be "
      "regenerated ==")
chain = b4a.verify_amendment_chain()
print("amendments:", len(chain["amendment_shas"]))
assert len(chain["amendment_shas"]) == 12
src = (REPO / "tools/b4_gen_amendment_12.py").read_text()
assert "PUBLISHED history" in src and "ls-files" in src

print("\nPOST CONFIRMED: incident mechanisms dead; v6 recovery "
      "append-only; launch closed pending the Musashi acta")
