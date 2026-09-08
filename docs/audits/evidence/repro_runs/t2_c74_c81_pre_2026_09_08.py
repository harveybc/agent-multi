"""PRE freeze for order T2 C74-C81: the audited post-revalidation
TOCTOU reproduces at a8fe2be6 through the productive APIs.

C74: the EXACT deterministic adversary — run the REAL root
     revalidation, replace the root immediately before it
     returns, then invoke the productive heartbeat: the call is
     accepted, the bytes land on the old held inode, and nothing
     is visible at the declared path. There is no post-write
     identity proof.
C75: `read_private()` and `WallAuthority._append()` carry no
     before/after root-identity proof at all (source facts).
C76/C77: the sealed v6 cap (14,400 s) is physically incompatible
     with the corrected census (~34 h projected); no successor
     design or persisted projection basis exists; the gates
     accept the infeasible design.

Zero sealed-bank series, zero scores, zero B4 objects. CPU only."""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

import t2_confirmatory_executor as ex  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="t2_c74_pre_"))
ex._TRUSTED_ROOT_PARENTS = tuple(ex._TRUSTED_ROOT_PARENTS) + (TMP,)

print("== C74: replacement AFTER pre-write revalidation is "
      "accepted ==")
rr = ex.ResultsRoot(TMP / "swap", create=True)
real_revalidate = ex.ResultsRoot.revalidate


def adversarial_revalidate(self):
    real_revalidate(self)                 # the REAL precheck
    if getattr(self, "_armed", False):    # replace right before
        self._armed = False               # it returns
        os.rename(self.path, TMP / "stolen")
        (TMP / "swap").mkdir(mode=0o700)


ex.ResultsRoot.revalidate = adversarial_revalidate
rr._armed = True
try:
    ex._heartbeat(rr, {"probe": True})
    accepted = True
except SystemExit:
    accepted = False
finally:
    ex.ResultsRoot.revalidate = real_revalidate
facts = {"heartbeat_call_accepted": accepted,
         "landed_on_old_held_inode":
             (TMP / "stolen" / "EXECUTOR_HEARTBEAT.json").exists(),
         "visible_at_declared_path":
             (TMP / "swap" / "EXECUTOR_HEARTBEAT.json").exists()}
print(json.dumps(facts, indent=1))
assert facts == {"heartbeat_call_accepted": True,
                 "landed_on_old_held_inode": True,
                 "visible_at_declared_path": False}

print("\n== C74/C75: no post-operation identity proof exists "
      "(source) ==")
src = (REPO / "tools/t2_confirmatory_executor.py").read_text()
hb = src[src.index("def _heartbeat"):src.index(
    "def _adjudication_counts")]
assert hb.count("revalidate()") == 1          # pre only, no post
ew = src[src.index("    def excl_write"):src.index(
    "    def read_private")]
assert ew.count("revalidate()") == 1          # pre only, no post
rp = src[src.index("    def read_private"):src.index(
    "    def exists")]
assert "revalidate" not in rp                 # reads: none at all
ap = src[src.index("    def _append"):src.index(
    "    # -- accounting --")]
assert "revalidate" not in ap and "assert_identity" not in ap
print("heartbeat/excl_write: pre-check only; read_private and "
      "WallAuthority._append: no identity proof at all")

print("\n== C76/C77: the infeasible 4-hour design is accepted "
      "==")
import t2_confirmatory as conf  # noqa: E402
STATE = Path.home() / ".local/share/agent-multi"
design = conf.strict_json_load(
    STATE / "t2_screen_design_SEALED_V6.json", "sealed design")
cap = design["resource_contract"]["max_wall_seconds"]
print(f"sealed cap: {cap}s = {cap / 3600:.1f}h; recorded "
      "projection: ~123,282s (~34.2h) from the corrected "
      "rehearsal — physically incompatible")
assert cap == 14400
assert not (STATE / "t2_screen_design_RESOURCE_SUCCESSOR_V1"
            ".json").exists()
assert "SUCCESSOR" not in src
assert "feasibility" not in (
    REPO / "tools/t2_confirmatory.py").read_text()
print("=> no successor design, no persisted projection basis, "
      "no feasibility gate — the gates would consume the "
      "infeasible design")

shutil.rmtree(TMP)
print("\nPRE CONFIRMED at a8fe2be6: the post-revalidation "
      "replacement window, the unproven reads/appends, and the "
      "absent resource successor/feasibility gate — all frozen")
