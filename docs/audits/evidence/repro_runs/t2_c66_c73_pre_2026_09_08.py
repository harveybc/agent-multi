"""PRE freeze for order T2 C66-C73: the four audited public-path
counterexamples (plus the census facts) reproduce at 81cfc253
through the productive APIs.

P0-a (C66): a self-consistent wall record with seconds=-100.0 is
      accepted; under a sealed 10 s limit, replay yields
      prior=-100.0 and remaining=110.0 — a direct budget-renewal
      path (hash integrity without semantic validity).
P0-b (C67): a tolerated torn tail is FOLLOWED in the same file —
      session 2 runs and closes normally, and session 3 then
      refuses because the former tail became a malformed interior
      line: the ledger is poisoned by design.
P0-c (C69): after ResultsRoot opens its descriptors, the declared
      root is renamed and replaced; the heartbeat write is
      accepted, lands on the old held inode, and is invisible at
      the declared path.
P1-d (C68): two sessions under DIFFERENT boot identities are
      accepted — the ordered 'stop for review' was silently
      weakened to 'does not renew' (my own weakening, confessed).
P1-e (C70): the physical fitting work is counted live on one dev
      unit: per origin/arm TWO ridge solves and per seed TWO _mlp
      calls of FOUR candidate fits each — 16 ridge solves and 192
      MLP candidate fits on sm_nile, projecting to 3,872 + 46,464
      = 50,336 fitting operations for the sealed population, not
      the published 7,744; ridge runs in the parent, outside the
      fit supervisor (source).

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

TMP = Path(tempfile.mkdtemp(prefix="t2_c66_pre_"))
ex._TRUSTED_ROOT_PARENTS = tuple(ex._TRUSTED_ROOT_PARENTS) + (TMP,)
stop = TMP / "T2_STOP"

print("== P0-a (C66): a negative reservation renews the budget ==")
rr = ex.ResultsRoot(TMP / "neg", create=True)
led = rr.path / "T2_WALL_LEDGER.jsonl"
docs = []
last = ex._LEDGER_GENESIS
for i, body in enumerate((
        {"kind": "session_open", "session": "evil",
         "boot_id": ex._boot_id(),
         "tail_torn_tolerated": False, "pid": 1},
        {"kind": "reserve", "session": "evil",
         "seconds": -100.0})):
    body["seq"] = i + 1
    body["prev_sha"] = last
    body["record_sha"] = ex._self_sha(body, "record_sha")
    last = body["record_sha"]
    docs.append(json.dumps(body, sort_keys=True))
led.write_text("\n".join(docs) + "\n")
os.chmod(led, 0o600)
w = ex.WallAuthority(rr, {"max_wall_seconds": 10,
                          "max_rss_bytes": 8 << 30}, stop, "s2")
print(f"prior_charged={w.prior} remaining={w.remaining():.1f} "
      f"(sealed limit 10 s) -> ACCEPTED")
assert w.prior == -100.0 and w.remaining() > 100
w.close()

print("\n== P0-b (C67): the torn tail poisons the NEXT replay ==")
rr2 = ex.ResultsRoot(TMP / "torn", create=True)
g = ex.WallAuthority(rr2, {"max_wall_seconds": 100,
                           "max_rss_bytes": 8 << 30}, stop, "t1")
os.close(g._fd)
with open(rr2.path / "T2_WALL_LEDGER.jsonl", "a") as f:
    f.write('{"kind":"close","trunc')          # torn tail
g2 = ex.WallAuthority(rr2, {"max_wall_seconds": 100,
                            "max_rss_bytes": 8 << 30}, stop, "t2")
g2.close()
print("session 2: accepted_and_closed (records appended BEHIND "
      "the torn fragment)")
try:
    ex.WallAuthority(rr2, {"max_wall_seconds": 100,
                           "max_rss_bytes": 8 << 30}, stop, "t3")
    raise AssertionError("session 3 replayed")
except SystemExit as exc:
    print(f"session 3: {str(exc)[:88]}")
    assert "malformed" in str(exc)

print("\n== P0-c (C69): root replacement is not detected ==")
rr3 = ex.ResultsRoot(TMP / "swap", create=True)
os.rename(rr3.path, TMP / "stolen")
(TMP / "swap").mkdir(mode=0o700)
ex._heartbeat(rr3, {"probe": True})
facts = {"write_accepted": True,
         "landed_on_held_inode":
             (TMP / "stolen" / "EXECUTOR_HEARTBEAT.json").exists(),
         "visible_at_declared_path":
             (TMP / "swap" / "EXECUTOR_HEARTBEAT.json").exists()}
print(json.dumps(facts, indent=1))
assert facts["landed_on_held_inode"] and \
    not facts["visible_at_declared_path"]

print("\n== P1-d (C68): a boot change does not stop for review ==")
rr4 = ex.ResultsRoot(TMP / "boot", create=True)
real_boot = ex._boot_id
ex._boot_id = lambda: "boot-identity-AAAA-0001"
g = ex.WallAuthority(rr4, {"max_wall_seconds": 100,
                           "max_rss_bytes": 8 << 30}, stop, "b1")
g.close()
ex._boot_id = lambda: "boot-identity-BBBB-0002"
g2 = ex.WallAuthority(rr4, {"max_wall_seconds": 100,
                            "max_rss_bytes": 8 << 30}, stop, "b2")
accepted = True
g2.close()
ex._boot_id = real_boot
print("second session under a DIFFERENT boot id: accepted =",
      accepted, "(ordered semantics: stop for review — silently "
      "weakened in C57 to 'does not renew'; my fault, confessed)")
src = (REPO / "tools/t2_confirmatory_executor.py").read_text()
assert "CLOCK_AUTHORITY_REVIEW_REQUIRED" not in src

print("\n== P1-e (C70): the physical fit census, counted live ==")
import t2_assay_harness as hz  # noqa: E402
import t2_public_data_census as dc  # noqa: E402
co = hz.load_co()
unit = hz.load_task_unit(dc.build_census(), "sm_nile")
counts = {"ridge_solves": 0, "mlp_fits": 0}
real_solve = np.linalg.solve
np.linalg.solve = lambda *a, **k: (
    counts.__setitem__("ridge_solves", counts["ridge_solves"] + 1)
    or real_solve(*a, **k))
from sklearn.neural_network import MLPRegressor  # noqa: E402
real_fit = MLPRegressor.fit


def counting_fit(self, X, y):
    counts["mlp_fits"] += 1
    return real_fit(self, X, y)


MLPRegressor.fit = counting_fit
try:
    hz.assay_unit(co, unit)
finally:
    np.linalg.solve = real_solve
    MLPRegressor.fit = real_fit
print(f"sm_nile (2 origins x 4 arms x 3 seeds): "
      f"{counts['ridge_solves']} ridge solves, "
      f"{counts['mlp_fits']} MLP candidate fits")
assert counts["ridge_solves"] == 2 * 4 * 2      # score + insample
assert counts["mlp_fits"] == 2 * 4 * 3 * 8      # 2 _mlp calls x 4
proj_ridge = 242 * 2 * 4 * 2
proj_mlp = 242 * 2 * 4 * 3 * 8
print(f"projected sealed population: {proj_ridge} ridge + "
      f"{proj_mlp} MLP candidate fits = "
      f"{proj_ridge + proj_mlp} fitting operations "
      f"(published census: 7744)")
assert proj_ridge == 3872 and proj_mlp == 46464
hsrc = (REPO / "tools/t2_assay_harness.py").read_text()
seg = hsrc[hsrc.index("def assay_unit"):]
assert "ridge_pred = _ridge(Xt, yt, Xs)" in seg
assert "ridge_in = _ridge(Xt, yt, Xt)" in seg   # second solve
assert "fit_supervisor" not in seg.split("ridge_in")[0].split(
    "ridge_pred")[1]                            # ridge unsupervised
print("=> ridge runs in the parent OUTSIDE the fit supervisor "
      "(source); one stalled solve can evade the per-fit "
      "contract")

shutil.rmtree(TMP)
print("\nPRE CONFIRMED at 81cfc253: negative-charge renewal, "
      "torn-tail poisoning, undetected root replacement, "
      "accepted boot change, and a 7,744-count that is "
      "physically 50,336 — all frozen")
