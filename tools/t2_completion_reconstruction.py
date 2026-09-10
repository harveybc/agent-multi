"""T2 C89/C91: fresh completion reconstruction and screen
adjudication of the executed resource-successor campaign.

From a fresh process: (1) the external review/execution records
and the complete successor diff verify at their final point of
use (the frozen-evidence gate chain); (2) the 4,650-unit source
census and the exact 242-unit population reconstruct through the
fresh verifier inside those gates; (3) every claim, raw array
and unit record verifies by descriptor-bound reads with EVERY
metric, extreme, timing and per-phase cost recomputed from the
arrays (the productive deep adjudicator); (4) the inventory is
exact with zero foreign, duplicate or missing units; (5) the
wall ledger replays through the productive grammar and the final
release sequence is proven; (6) the reviewed screen adjudicator
runs from the freshly verified observation records — never from
producer aggregates — under the sealed six-panel rules.

Zero writes to the campaign root. CPU only. The output is a
CANDIDATE adjudication for Musashi review; no T4/T5 artifact and
no PUBLICLY_ELIGIBLE claim is produced.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# C89: the execution record authorizes EXACTLY ONE executor
# identity (pinned commit). The reconstruction therefore runs
# the PINNED checkout's own modules — the same bytes that
# executed — via --pinned-checkout; without it, the live
# checkout is used and the gate itself decides.
_pin = None
for _i, _a in enumerate(sys.argv):
    if _a == "--pinned-checkout" and _i + 1 < len(sys.argv):
        _pin = Path(sys.argv[_i + 1])
if _pin is not None:
    sys.path.insert(0, str(_pin))
    sys.path.insert(0, str(_pin / "tools"))
else:
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "tools"))

import t2_confirmatory as conf  # noqa: E402
import t2_confirmatory_executor as ex  # noqa: E402

DEFAULT_ROOT = (
    Path.home() / ".local/share/agent-multi/"
    "t2_confirmatory_results_resource_successor_v1_20260909")


class ReconstructionRefusal(SystemExit):
    def __init__(self, msg):
        super().__init__(f"REFUSED: {msg}")


def replay_wall_ledger(root: Path, design: dict) -> dict:
    """C89.6: replay the whole wall ledger through the
    PRODUCTIVE grammar/state machine, read-only."""
    raw = (Path(root) / "T2_WALL_LEDGER.jsonl").read_bytes()
    w = object.__new__(ex.WallAuthority)
    w.limits = {"max_wall_seconds":
                design["resource_contract"]["max_wall_seconds"]}
    # historical read-only replay: the ledger's own recorded
    # boot identity is the replay's boot (C68 guards LIVE
    # execution across boots; a post-hoc verification validates
    # the ledger's internal consistency)
    first = raw.split(b"\n", 1)[0]
    w.boot = json.loads(first).get("boot_id") if first else None
    w.session = None
    state = w._replay(raw)
    charged = state[0] if isinstance(state, tuple) else state
    if isinstance(charged, dict):
        charged = charged.get("charged")
    return {"replayed": True,
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "records": raw.count(b"\n"),
            "charged_state": charged
            if isinstance(charged, (int, float)) else str(
                type(state).__name__)}


def verify_release_sequence(root: Path) -> dict:
    locks = Path(root) / "locks"
    epochs = sorted(int(p.name.split("_")[-1].split(".")[0])
                    for p in locks.glob("SESSION_*.json"))
    if not epochs:
        raise ReconstructionRefusal("no lock epoch exists")
    last = epochs[-1]
    intent = locks / f"RELEASE_INTENT_{last:06d}.json"
    done = locks / f"RELEASE_DONE_{last:06d}.json"
    for p, what in ((intent, "release intent"),
                    (done, "release done")):
        if not p.is_file():
            raise ReconstructionRefusal(
                f"final {what} witness absent — the release "
                "sequence does not prove")
    return {"final_epoch": last,
            "release_intent_sha256": conf._sha_file(intent),
            "release_done_sha256": conf._sha_file(done),
            "epochs": epochs}


def reconstruct(root: Path = DEFAULT_ROOT) -> dict:
    t0 = time.monotonic()
    root = Path(root)
    # (1)+(2): the frozen-evidence gates — external records,
    # successor diff, census + population fresh verification
    facts = conf.verify_confirmatory_gates(
        ex.MANIFEST_PATH, ex.active_design_path(),
        census_path=ex.CENSUS_PATH)
    design = facts["design"].doc
    manifest = facts["manifest"].doc
    if design.get("schema") == conf.T2_SUCCESSOR_SCHEMA:
        conf.verify_resource_successor(design)   # final use
    authority = {
        "sealed_design_file_sha256": facts["design_file_sha256"],
        "sealed_design_self_sha256": facts["design_self_sha256"],
        "design_review_record_sha256":
            facts["review_record_sha256"],
        "execution_record_sha256":
            facts["execution_record_sha256"],
        "manifest_sha256": facts["manifest_sha256"],
        "census_sha256": facts["census_sha256"]}
    uids = design["task_population"]["series_ids"]
    # (3)+(4): fresh-chain deep adjudication of all units with
    # full metric recomputation and the exact inventory
    rr = ex.ResultsRoot(root, create=False)
    import numpy as np
    raw_root = ex.STATE / "t2_public_raw"

    def _rebuild(uid):
        return np.asarray(
            ex.load_bank_unit(design, uid, manifest,
                              raw_root)["y"],
            dtype=np.float64)

    counts = ex.final_adjudication(rr, uids, design, authority,
                                   "confirmatory", _rebuild)
    # (5): wall replay + release sequence
    wall = replay_wall_ledger(root, design)
    release = verify_release_sequence(root)
    # (6): collect the freshly verified records for the screen
    records = []
    cost_totals = {"train_seconds": 0.0}
    for uid in uids:
        safe = ex._safe_name(uid)
        rec = conf.strict_json_load(
            root / "units" / f"RECORD_{safe}.json",
            f"unit record {uid}")
        ar = rec["assay_record"]
        # C89.4: cost coherence has a PHYSICAL anchor — the sum
        # of the per-phase cost measurements can never exceed the
        # unit's recorded wall (with a small tolerance); a cost
        # edited under a repaired digest dies here.
        def _numsum(node):
            if isinstance(node, (int, float)):
                return float(node)
            if isinstance(node, dict):
                return sum(_numsum(v) for v in node.values())
            return 0.0
        phase_sum = _numsum(ar.get("costs_by_phase", {}))
        wall_u = float(rec.get("wall_seconds", 0.0))
        if wall_u <= 0 or phase_sum > wall_u * 1.10 + 2.0:
            raise ReconstructionRefusal(
                f"{uid}: per-phase cost sum {phase_sum:.2f}s is "
                f"incoherent with the unit wall {wall_u:.2f}s — "
                "an altered cost never adjudicates")
        cost_totals["train_seconds"] += phase_sum
        records.append(ar)
    charged = None
    if isinstance(wall.get("charged_state"), (int, float)):
        charged = wall["charged_state"]
    screen = conf.adjudicate_screen(records, design)
    doc = {
        "schema": ("agent_multi.t2_completion_reconstruction_"
                   "and_screen_adjudication.candidate.v1"),
        "authority": ("CANDIDATE_FOR_MUSASHI_REVIEW — no T4/T5, "
                      "no PUBLICLY_ELIGIBLE claim"),
        "campaign_root_logical": root.name,
        "gate_facts": {k: facts[k] for k in
                       ("design_file_sha256",
                        "design_self_sha256",
                        "review_record_sha256",
                        "execution_record_sha256",
                        "manifest_sha256", "census_sha256")},
        "final_adjudication_counts": counts,
        "wall_ledger": wall,
        "release_sequence": release,
        "screen_adjudication": screen,
        "wall_seconds_reconstruction": round(
            time.monotonic() - t0, 2),
    }
    doc["record_sha256"] = hashlib.sha256(json.dumps(
        {k: doc[k] for k in sorted(doc)},
        sort_keys=True, default=str).encode()).hexdigest()
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--pinned-checkout", type=Path,
                    default=None)
    a = ap.parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.nice(15)
    doc = reconstruct(a.root)
    if a.out:
        fd = os.open(str(a.out), os.O_CREAT | os.O_EXCL
                     | os.O_WRONLY, 0o600)
        try:
            os.write(fd, json.dumps(doc, indent=1,
                                    default=str).encode())
            os.fsync(fd)
        finally:
            os.close(fd)
    print(json.dumps({
        "counts": doc["final_adjudication_counts"],
        "wall_records": doc["wall_ledger"]["records"],
        "release_epoch": doc["release_sequence"]["final_epoch"],
        "screen_verdict":
            doc["screen_adjudication"].get("verdict"),
        "screen_reason":
            doc["screen_adjudication"].get("reason"),
        "reconstruction_seconds":
            doc["wall_seconds_reconstruction"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
