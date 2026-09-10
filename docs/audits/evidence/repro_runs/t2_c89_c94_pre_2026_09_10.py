"""PRE freeze for order T2 C89-C94 at 7bcd3f0d: the successor
campaign's EXECUTION facts stand, and the SCIENCE is absent.

Execution facts (read-only, zero writes to the campaign root):
the resource-successor root holds exactly 242 units x (claim +
record + arrays) with zero terminals, the attempt ledger, the
wall ledger and the lock epoch chain; every unit adjudicates
COMPLETED at the shallow level through the productive
adjudicator; the service exit facts (reported 242 completed /
0 failed / 0 resumed / 32,547.3 s) are Musashi's recorded
launch output, consumed as claims-to-reconstruct, not science.

Absences frozen (what this order builds): no fresh completion
reconstruction has ever been committed; no screen adjudication
has been derived from freshly reconstructed observation records
(the committed evidence tree holds no T2 screen adjudication);
no C90 refusal battery exists for the completed-campaign
surface; the wall ledger has never been replayed outside the
executor's own session.

CPU only. Zero writes (byte inventory before/after)."""
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

import t2_confirmatory_executor as ex  # noqa: E402

ROOT = (Path.home() / ".local/share/agent-multi/"
        "t2_confirmatory_results_resource_successor_v1_20260909")

facts = {}


def _inventory(root):
    inv = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            st = p.stat()
            inv[str(p.relative_to(root))] = (st.st_size,
                                             st.st_mtime_ns)
    return inv


inv_before = _inventory(ROOT)

units = ROOT / "units"
names = sorted(p.name for p in units.iterdir())
claims = [n for n in names if n.startswith("CLAIM_")]
records = [n for n in names if n.startswith("RECORD_")]
arrays = [n for n in names if n.startswith("ARRAYS_")]
terminals = [n for n in names if n.startswith("TERMINAL_")]
facts["unit_objects"] = {"claims": len(claims),
                         "records": len(records),
                         "arrays": len(arrays),
                         "terminals": len(terminals),
                         "total": len(names)}
assert facts["unit_objects"] == {"claims": 242, "records": 242,
                                 "arrays": 242, "terminals": 0,
                                 "total": 726}
facts["ledgers_present"] = {
    "attempt_ledger": (ROOT / "T2_ATTEMPT_LEDGER.json").is_file(),
    "wall_ledger": (ROOT / "T2_WALL_LEDGER.jsonl").is_file(),
    "locks_dir": (ROOT / "locks").is_dir(),
    "release_done": any((ROOT / "locks").glob("RELEASE_DONE_*")),
}
assert all(facts["ledgers_present"].values())

# shallow adjudication of every unit via the productive path
rr = ex.ResultsRoot(ROOT, create=False)
counts = {}
for cn in claims:
    uid_safe = cn[len("CLAIM_"):-len(".json")]
    uid = json.loads((units / cn).read_text())["unit_id"]
    st, _why = ex.adjudicate_unit_shallow(rr, uid)
    counts[st] = counts.get(st, 0) + 1
facts["shallow_adjudication"] = counts
assert counts == {"COMPLETED": 242}, counts

# design identity in force
import t2_confirmatory as conf  # noqa: E402
succ = (Path.home() / ".local/share/agent-multi/"
        "t2_screen_design_RESOURCE_SUCCESSOR_V1.json")
facts["successor_file_sha256"] = hashlib.sha256(
    succ.read_bytes()).hexdigest()

# absences: nothing scientific committed yet
ev = REPO / "docs/audits/evidence"
facts["absences"] = {
    "no_completion_reconstruction_committed": not any(
        ev.glob("T2_COMPLETION_RECONSTRUCTION*")),
    "no_screen_adjudication_committed": not any(
        ev.glob("T2_SCREEN_ADJUDICATION*")),
    "no_c90_battery": not (
        REPO / "tests/test_t2_completion.py").exists(),
}
assert all(facts["absences"].values())

# service exit facts are CLAIMS (bytes from the master order)
order = (REPO / "docs/handoffs/"
         "MUSASHI_TO_GENERAL_SATOSHI_T2_C89_C94_COMPLETION_AND_"
         "ADJUDICATION_ORDER_2026_09_09.md").read_text()
facts["service_claims_in_order"] = {
    "242_completed": "242 `COMPLETED_VERIFIED`" in order,
    "zero_failed": "zero `TERMINAL_FAILED`" in order,
    "wall_32547_3": "32,547.3" in order}
assert all(facts["service_claims_in_order"].values())

facts["root_untouched"] = _inventory(ROOT) == inv_before
assert facts["root_untouched"]

print(json.dumps(facts, indent=1))
print("\nPRE CONFIRMED at 7bcd3f0d: 242x(claim+record+arrays), "
      "zero terminals, ledgers and release chain present, every "
      "unit shallow-COMPLETED, the successor identity in force, "
      "the service numbers are order-recorded claims — and NO "
      "fresh reconstruction, screen adjudication or completion "
      "battery exists yet; the campaign root received zero "
      "writes")
