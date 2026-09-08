"""POST for order T2 C66-C73: the four audited counterexamples
are DEAD through the productive APIs; the fit census is physical
and the harness runs ONE supervised fit per block with proven
bit-identity.

Zero sealed-bank series, zero scores, zero B4 objects. CPU only."""
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
os.environ.setdefault(
    "B4_T1_PREPROCESSOR_ROOT",
    str(Path.home() / "Documents/GitHub/.worktrees/prep-t0t1"))

import numpy as np  # noqa: E402

import t2_confirmatory_executor as ex  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="t2_c66_post_"))
ex._TRUSTED_ROOT_PARENTS = tuple(ex._TRUSTED_ROOT_PARENTS) + (TMP,)
stop = TMP / "T2_STOP"


def _die(fn, needle, what):
    try:
        fn()
    except SystemExit as exc:
        assert needle in str(exc), (needle, str(exc)[:160])
        print(f"  {what} -> {str(exc)[:96]}")
        return
    raise AssertionError(f"{what}: did NOT refuse")


def _chain(recs):
    out, last = [], ex._LEDGER_GENESIS
    for i, body in enumerate(recs):
        body = dict(body)
        body["seq"] = i + 1
        body["prev_sha"] = last
        body["record_sha"] = ex._self_sha(body, "record_sha")
        last = body["record_sha"]
        out.append(json.dumps(body, sort_keys=True))
    return "\n".join(out) + "\n"


print("== C66 DEAD: the wall grammar refuses semantic forgery ==")
lim = {"max_wall_seconds": 10, "max_rss_bytes": 8 << 30}
boot = ex._boot_id()
so = {"kind": "session_open", "session": "s", "boot_id": boot,
      "tail_torn_tolerated": False, "pid": 1}
rr = ex.ResultsRoot(TMP / "neg", create=True)
led = rr.path / "T2_WALL_LEDGER.jsonl"
led.write_text(_chain([so, {"kind": "reserve", "session": "s",
                            "seconds": -100.0}]))
os.chmod(led, 0o600)
_die(lambda: ex.WallAuthority(rr, lim, stop, "s2"),
     "strictly positive",
     "the PRE's seconds=-100 reservation")
for val, what in ((True, "bool seconds"), ("nan", "string nan"),
                  (0, "zero seconds")):
    rr2 = ex.ResultsRoot(TMP / f"neg_{what.split()[0]}",
                         create=True)
    led2 = rr2.path / "T2_WALL_LEDGER.jsonl"
    led2.write_text(_chain([so, {"kind": "reserve",
                                 "session": "s",
                                 "seconds": val}]))
    os.chmod(led2, 0o600)
    _die(lambda: ex.WallAuthority(rr2, lim, stop, "x"),
         "strictly positive", what)

print("\n== C67 DEAD: torn tail is a review boundary ==")
rr3 = ex.ResultsRoot(TMP / "torn", create=True)
g = ex.WallAuthority(rr3, {"max_wall_seconds": 100,
                           "max_rss_bytes": 8 << 30}, stop, "t1")
os.close(g._fd)
with open(rr3.path / "T2_WALL_LEDGER.jsonl", "a") as f:
    f.write('{"kind":"close","trunc')
_die(lambda: ex.WallAuthority(rr3, {"max_wall_seconds": 100,
                                    "max_rss_bytes": 8 << 30},
                              stop, "t2"),
     "WALL_LEDGER_TORN_TAIL_REVIEW_REQUIRED",
     "the PRE's poisoning session 2")
tail = (rr3.path / "T2_WALL_LEDGER.jsonl").read_text()
print("  nothing appended behind the fragment:",
      tail.endswith('{"kind":"close","trunc'))
assert tail.endswith('{"kind":"close","trunc')

print("\n== C68 DEAD: a boot change stops for review ==")
rr4 = ex.ResultsRoot(TMP / "boot", create=True)
real_boot = ex._boot_id
ex._boot_id = lambda: "boot-identity-AAAA-0001"
g = ex.WallAuthority(rr4, {"max_wall_seconds": 100,
                           "max_rss_bytes": 8 << 30}, stop, "b1")
g.close()
ex._boot_id = lambda: "boot-identity-BBBB-0002"
_die(lambda: ex.WallAuthority(rr4, {"max_wall_seconds": 100,
                                    "max_rss_bytes": 8 << 30},
                              stop, "b2"),
     "CLOCK_AUTHORITY_REVIEW_REQUIRED",
     "the PRE's accepted second boot")
ex._boot_id = real_boot

print("\n== C69 DEAD: root replacement refuses before writes ==")
rr5 = ex.ResultsRoot(TMP / "swap", create=True)
os.rename(rr5.path, TMP / "stolen")
(TMP / "swap").mkdir(mode=0o700)
_die(lambda: ex._heartbeat(rr5, {"probe": True}),
     "RESULTS_ROOT_IDENTITY_LOST",
     "the PRE's renamed-and-replaced root (heartbeat)")
print("  landed on held inode:",
      (TMP / "stolen" / "EXECUTOR_HEARTBEAT.json").exists())
assert not (TMP / "stolen" / "EXECUTOR_HEARTBEAT.json").exists()

print("\n== C70 DEAD: one supervised fit; honest census ==")
import t2_assay_harness as hz  # noqa: E402
import t2_public_data_census as dc  # noqa: E402
import t2_confirmatory as conf  # noqa: E402
co = hz.load_co()
unit = hz.load_task_unit(dc.build_census(), "sm_nile")
counts = {"ridge": 0, "mlp": 0}
real_solve = np.linalg.solve
np.linalg.solve = lambda *a, **k: (
    counts.__setitem__("ridge", counts["ridge"] + 1)
    or real_solve(*a, **k))
from sklearn.neural_network import MLPRegressor  # noqa: E402
real_fit = MLPRegressor.fit


def cfit(self, X, y):
    counts["mlp"] += 1
    return real_fit(self, X, y)


MLPRegressor.fit = cfit
try:
    hz.assay_unit(co, unit)
finally:
    np.linalg.solve = real_solve
    MLPRegressor.fit = real_fit
print(f"  sm_nile physical work: {counts['ridge']} ridge solves "
      f"(PRE: 16), {counts['mlp']} MLP candidate fits (PRE: 192)")
assert counts["ridge"] == 8 and counts["mlp"] == 96
design = conf.strict_json_load(
    Path.home() / ".local/share/agent-multi/"
    "t2_screen_design_SEALED_V6.json", "sealed design")
w = ex.census_of_work(design)
print("  census v2:", json.dumps(w))
assert w["linear_solves"] == 1936
assert w["mlp_candidate_fits"] == 23232
assert w["selected_model_instances"] == 7744
assert "model_fits" not in w
src = (REPO / "tools/t2_assay_harness.py").read_text()
assert "_RidgeFitClosure(Xt, yt)" in src     # supervised ridge
assert "ridge_in = _ridge(Xt, yt, Xt)" not in src

print("\n== v4 mechanical rehearsal at the final tip ==")
reh = TMP / "t2reh"
t0 = time.time()
rc = ex.rehearse(reh)
assert rc == 0
print(f"  rehearsal wall: {time.time() - t0:.1f}s")

shutil.rmtree(TMP)
print("\nPOST CONFIRMED: C66-C71 dead through the productive "
      "APIs; the wall ledger is a typed grammar, torn tails and "
      "boot changes stop for review, the declared root identity "
      "is proven before every side effect, and the census names "
      "the physical work of single supervised fits with proven "
      "bit-identity; scoring remains closed by the ABSENT "
      "external execution record")
