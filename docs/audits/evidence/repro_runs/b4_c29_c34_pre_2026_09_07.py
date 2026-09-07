"""PRE freeze for order B4 C29-C34 (environment recovery): the
dispatch-environment incident reproduces mechanically against the
code at the accepted commit.

Reproduced here, all against PHYSICAL state:
1. the four incident objects exist byte-exact (claim/lease/binding/
   origin digests from MUSASHI_B4_DISPATCH_ENVIRONMENT_INCIDENT);
2. the failed cell has NO heartbeat, checkpoint or terminal and
   adjudicates AMBIGUOUS_CLAIM (eleven cells PENDING);
3. the orchestrator dry-run is PLUGIN-BLIND: with a real entry-point
   registry lacking sac_agent it still passes with zero findings;
4. the escape: claim+lease created, then execute_cell loads plugins
   OUTSIDE the typed-terminal boundary — the productive loader's own
   ImportError (from the filtered REAL registry, no artificial
   exception) escapes, leaving claim without terminal =>
   AMBIGUOUS_CLAIM on a throwaway root;
5. the ambiguous claim keeps charging wall-clock seconds against the
   ceiling (the acta fixes the charge at 0.01 h instead).

Zero writes into the v5 root; the throwaway root is removed."""
import contextlib
import hashlib
import io
import json
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


orch = _load("b4orch_pre", "tools/b4_campaign_orchestrator.py")
executor = _load("b4exec_pre", "tools/b4_campaign_executor.py")

STATE = Path.home() / ".local/share/agent-multi"
V5 = STATE / "b4_campaign_results_20260906"
MAT = STATE / "b4_materialization_v5_20260906"
CELL = "o2022_seed101"
ATTEMPT = "attempt_6e46ebe59eb842ca"

INCIDENT_DIGESTS = {
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
        "97ef4e0e44",
}

print("== 1. incident objects byte-exact ==")
cdir = V5 / CELL
for name, want in INCIDENT_DIGESTS.items():
    got = hashlib.sha256((cdir / name).read_bytes()).hexdigest()
    print(f"{name[:44]:44s} match={got == want}")
    assert got == want, name

print("\n== 2. no learning artifacts; state AMBIGUOUS_CLAIM ==")
leftovers = [p.name for p in cdir.iterdir()
             if p.name not in INCIDENT_DIGESTS]
print("other objects in the cell dir:", leftovers)
assert leftovers == []
assert not (cdir / "B4_CELL_TERMINAL.json").exists()
states = {cid: orch.adjudicate_cell_state(V5, cid)
          for cid in _load("b4led_pre",
                           "tools/b4_campaign_ledger.py"
                           ).EXPECTED_CELLS}
amb = [c for c, s in states.items() if s == "AMBIGUOUS_CLAIM"]
pend = [c for c, s in states.items() if s == "PENDING"]
print(f"AMBIGUOUS_CLAIM={amb} PENDING={len(pend)}")
assert amb == [CELL] and len(pend) == 11

print("\n== 3. REAL registry without sac_agent ==")
import app.plugin_loader as apl  # noqa: E402
_real_eps = apl.entry_points


def _filtered_eps():
    class _View:
        def select(self, group):
            eps = _real_eps().select(group=group)
            if group == "agent.plugins":
                return [e for e in eps if e.name != "sac_agent"]
            return eps
    return _View()


apl.entry_points = _filtered_eps
try:
    got = [e.name for e in
           _filtered_eps().select(group="agent.plugins")]
    print("agent.plugins (filtered, REAL registry):",
          "sac_agent" in got and "STILL PRESENT — BUG" or
          f"{len(got)} plugins, sac_agent absent")
    assert "sac_agent" not in got and got

    print("\n== 3a. the orchestrator dry-run is PLUGIN-BLIND ==")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = orch.run_campaign(MAT, V5 / "CAMPAIGN_LEDGER.json",
                               V5, "cuda:0", execute=False)
    rep = json.loads(buf.getvalue())
    print("dry-run rc:", rc, "| writes:", rep["writes"],
          "| plugin checks in report:",
          "plugin" in buf.getvalue().lower())
    assert rc == 0 and rep["writes"] == 0
    assert "plugin" not in buf.getvalue().lower()
    print("DRY_RUN_BLIND_TO_MISSING_PLUGIN: True")

    print("\n== 4. the escape: claim exists, ImportError crosses "
          "the boundary, NO terminal ==")
    TR = Path.home() / ".cache/b4_c29_pre_20260907"
    if TR.exists():
        shutil.rmtree(TR)
    import os
    os.makedirs(TR, mode=0o700)
    with orch.GlobalLock(TR):
        claim = orch.claim_attempt(TR, CELL)
        lease = orch.issue_lease(TR, CELL, claim,
                                 executor.CAMPAIGN_AUTH_SHA, MAT)
        escaped = None
        try:
            executor.execute_cell(CELL, MAT, TR, "cpu",
                                  lease_path=lease)
        except ImportError as exc:
            escaped = f"{type(exc).__name__}: {exc}"
        except SystemExit as exc:
            escaped = f"TYPED (SystemExit): {exc}"
    print("escaped exception:", str(escaped)[:90])
    assert escaped and escaped.startswith("ImportError"), escaped
    assert "sac_agent not found in group agent.plugins" in escaped
    has_terminal = (TR / CELL / "B4_CELL_TERMINAL.json").exists()
    st = orch.adjudicate_cell_state(TR, CELL)
    print(f"terminal written: {has_terminal} | state: {st}")
    assert not has_terminal and st == "AMBIGUOUS_CLAIM"
    print("POST_CLAIM_IMPORT_ESCAPES_WITHOUT_TERMINAL: True")
    shutil.rmtree(TR)
finally:
    apl.entry_points = _real_eps

print("\n== 5. the ambiguous claim charges wall-clock time ==")
spent = orch.gpu_seconds_spent(V5)
print(f"gpu_seconds_spent(v5) = {spent:.0f}s and GROWING with "
      "wall clock (no terminal caps it); the incident acta fixes "
      "the v5 charge at 0.01 h = 36 s for the v6 ceiling")
assert spent > 36.0

print("\nPRE CONFIRMED: C29-C31 findings reproduce; v5 untouched")
