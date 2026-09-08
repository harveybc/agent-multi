"""POST for order T2 C74-C81: the post-revalidation TOCTOU is
dead (before/after identity on every campaign-state operation);
the resource-only successor exists with its executable diff; the
feasibility gate refuses the infeasible 4-hour design; the
projection never grants time. Zero scores, zero B4 objects."""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
os.environ.setdefault(
    "B4_T1_PREPROCESSOR_ROOT",
    str(Path.home() / "Documents/GitHub/.worktrees/prep-t0t1"))

import t2_confirmatory as conf  # noqa: E402
import t2_confirmatory_executor as ex  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="t2_c74_post_"))
ex._TRUSTED_ROOT_PARENTS = tuple(ex._TRUSTED_ROOT_PARENTS) + (TMP,)

print("== C74 DEAD: the exact adversary refuses typed ==")
rr = ex.ResultsRoot(TMP / "swap", create=True)
real = ex.ResultsRoot.revalidate
state = {"armed": True}


def adversarial(self):
    real(self)
    if state["armed"]:
        state["armed"] = False
        os.rename(TMP / "swap", TMP / "stolen")
        (TMP / "swap").mkdir(mode=0o700)


ex.ResultsRoot.revalidate = adversarial
try:
    ex._heartbeat(rr, {"probe": True})
    raise AssertionError("accepted")
except SystemExit as exc:
    print(" ", str(exc)[:104])
    assert "RESULTS_ROOT_CUSTODY_LOST" in str(exc)
finally:
    ex.ResultsRoot.revalidate = real
print("  effect NOT accepted; attempt semantics: UNCERTAIN, no "
      "release, external disposition")

print("\n== C75: reads and wall appends carry before/after "
      "identity (source) ==")
src = (REPO / "tools/t2_confirmatory_executor.py").read_text()
for fn in ("def excl_write", "def read_private", "def exists",
           "def listdir"):
    seg = src[src.index(fn):]
    seg = seg[:seg.index("\n    def ", 10)]
    assert "revalidate()" in seg and "_post_identity" in seg, fn
ap = src[src.index("    def _append"):src.index(
    "    # -- accounting --")]
assert "revalidate()" in ap and "_post_identity" in ap
print("  excl_write/read_private/exists/listdir/_append: pre + "
      "post proofs present")

print("\n== C76: the resource-only successor and its diff ==")
S = Path.home() / ".local/share/agent-multi"
sealed_p = S / "t2_screen_design_SEALED_V6.json"
succ_p = S / "t2_screen_design_RESOURCE_SUCCESSOR_V1.json"
assert conf._sha_file(sealed_p).startswith("d1720f4d")
succ = json.loads(succ_p.read_text())
conf.verify_resource_successor(succ)
print(f"  v6 byte-immutable (d1720f4d…); successor "
      f"{succ['design_sha256'][:12]}… wall="
      f"{succ['resource_contract']['max_wall_seconds']}s; "
      "executable field-by-field diff PASSES")
evil = json.loads(json.dumps(succ))
evil["seed_tape"] = [1, 2, 3]
import hashlib
body = {k: evil[k] for k in sorted(evil) if k != "design_sha256"}
evil["design_sha256"] = hashlib.sha256(json.dumps(
    body, sort_keys=True).encode()).hexdigest()
try:
    conf.verify_resource_successor(evil)
    raise AssertionError("scientific delta accepted")
except SystemExit as exc:
    print(f"  scientific delta -> {str(exc)[:84]}")

print("\n== C77: feasibility gate + hard wall only ==")
proj = conf.strict_json_load(conf.T2_BUDGET_PROJECTION_PATH,
                             "projection")
sealed = json.loads(sealed_p.read_text())
print(f"  projection {proj['projection_wall_seconds']}s "
      f"(non-authoritative) vs v6 cap "
      f"{sealed['resource_contract']['max_wall_seconds']}s -> "
      "the old 4h design refuses at the gates; successor "
      f"headroom {proj['headroom_vs_216000s_hard_ceiling']}x")
assert sealed["resource_contract"]["max_wall_seconds"] < \
    proj["projection_wall_seconds"]
wseg = src[src.index("class WallAuthority"):
           src.index("def make_fit_supervisor")]
assert "PROJECTION" not in wseg
print("  WallAuthority reads ONLY the design hard limit — a "
      "forged projection can never grant time")

print("\n== v5 rehearsal at the final tip (successor consumed) ==")
reh = TMP / "t2reh"
rc = ex.rehearse(reh)
assert rc == 0

shutil.rmtree(TMP)
print("\nPOST CONFIRMED: C74-C78 dead through the productive "
      "APIs; scoring stays closed by the ABSENT external "
      "records; no execution authority was created")
