#!/usr/bin/env python3
"""C111 (order 2026-09-12): the B4 campaign closure, additively, in the cube.

B4 is closed: `B4_V4_ACCEPTED_AS_NON_AUTHORIZING_FINAL_CLOSURE`. This
module turns that closure into ONE campaign envelope for the CRISP-DM
outbox, from the committed v4 submission only. It never reopens the
preserved root.

Before anything is built:

* the submission file hashes to the accepted digest;
* its self digest re-derives;
* the counts, verdict and costs are the accepted ones.

The envelope carries the campaign-level counts (2 / 1 / 9) and the
declared costs, as campaign units. It carries no per-cell scientific
result. The quarantined partial cell is counted as neither a failure
nor a result: `units_failed` is 0 and its charge appears only as the
declared lower bound.

Emission goes through the predictor outbox (content-addressed,
append-only), so emitting twice writes once. The loader drains it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SUBMISSION = REPO / "docs/audits/evidence/B4_READJUDICATION_SUBMISSION_V4_2026_09_12.json"
DECISION = "B4_V4_ACCEPTED_AS_NON_AUTHORIZING_FINAL_CLOSURE"
ACCEPTED_FILE_SHA256 = "63ea37901c8756cd9669d18548176d337efeb08f1bb457a3e395eb90c7be7317"
ACCEPTED_SELF_SHA256 = "9e7047ea077a3dfebc24999654d57c87a623a7345557ad308bfb8202e23a109b"
CODE_A = "4c842dd10da9f4956eea4ffc9435492fcd36243e"
PUBLICATION_B = "ff52ca7d9b771978dfe60bfd15e351ffbe3d1d22"
ACCEPTED_COUNTS = {"COMPLETED_VERIFIED": 2, "QUARANTINED_PARTIAL": 1, "NOT_STARTED": 9}
ACCEPTED_VERDICT = "SCIENTIFICALLY_INSUFFICIENT_NO_VERDICT"
CAMPAIGN_KEY = "b4::screen_b_v7_campaign"
PREDICTOR = Path.home() / "Documents/GitHub/predictor"


class EnvelopeBuildRefusal(SystemExit):
    def __init__(self, msg: str) -> None:
        super().__init__(f"REFUSED: {msg}")


def sha_obj(obj) -> str:
    # The submission's own self-digest contract (tools/b4_campaign_closure.py).
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def _olap(predictor: Path = PREDICTOR):
    if not (predictor / "olap/outbox.py").is_file():
        raise EnvelopeBuildRefusal(f"no outbox implementation at {predictor.name}/olap")
    sys.path.insert(0, str(predictor))
    try:
        from olap import campaign_envelope as ce  # noqa: E402
        from olap import outbox as ob  # noqa: E402
    finally:
        sys.path.remove(str(predictor))
    return ce, ob


def verified_submission(path: Path = SUBMISSION) -> dict:
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != ACCEPTED_FILE_SHA256:
        raise EnvelopeBuildRefusal("the submission bytes are not the accepted v4 file")
    doc = json.loads(raw)
    if sha_obj({k: doc[k] for k in sorted(doc) if k != "submission_sha256"}) != doc["submission_sha256"] \
            or doc["submission_sha256"] != ACCEPTED_SELF_SHA256:
        raise EnvelopeBuildRefusal("the submission self digest does not re-derive to the accepted one")
    adj = doc["adjudication"]
    if adj["counts"] != ACCEPTED_COUNTS or adj["verdict"] != ACCEPTED_VERDICT \
            or adj["declared_cells"] != sum(ACCEPTED_COUNTS.values()):
        raise EnvelopeBuildRefusal("counts or verdict differ from the accepted closure")
    if doc["code_identity"]["commit"] != CODE_A:
        raise EnvelopeBuildRefusal("code identity is not the accepted commit A")
    return doc


def build(path: Path = SUBMISSION, predictor: Path = PREDICTOR) -> dict:
    ce, _ = _olap(predictor)
    doc = verified_submission(path)
    adj, costs = doc["adjudication"], doc["costs"]
    partial = costs["quarantined_partial"]
    if len(partial) != 1:
        raise EnvelopeBuildRefusal("exactly one quarantined partial cell was accepted")
    (partial_cell, partial_cost), = partial.items()
    if partial_cost["bound"] != "LOWER_BOUND":
        raise EnvelopeBuildRefusal("the partial charge is only ever a lower bound")
    generation = doc["campaign_generation"]
    unit = {"cell_key": "CAMPAIGN", "candidate_key": generation, "terminal_state": "CLOSED"}
    units = [
        dict(unit, metric_name="cells_completed_verified", metric_value=adj["counts"]["COMPLETED_VERIFIED"]),
        dict(unit, metric_name="cells_quarantined_partial", metric_value=adj["counts"]["QUARANTINED_PARTIAL"]),
        dict(unit, metric_name="cells_not_started", metric_value=adj["counts"]["NOT_STARTED"]),
        dict(unit, metric_name="declared_wall_seconds_completed_cells",
             metric_value=costs["completed_wall_seconds_total"]),
        dict(unit, metric_name="declared_wall_seconds_lower_bound_quarantined_partial",
             metric_value=partial_cost["claim_to_stop_signal_seconds"]),
    ]
    return ce.build_envelope(
        campaign_key=CAMPAIGN_KEY, producer="agent-multi", result_class="DEVELOPMENT",
        identity={"run_id": generation, "code_identity": CODE_A,
                  "design_sha256": doc["campaign_ledger_sha256"],
                  "record_sha256": doc["submission_sha256"]},
        data_consumed={"datasets": [{
            "id": doc["preserved_root_logical"],
            "digest": doc["custody"]["leaf_binding"]["bindings_sha256"],
            "eligibility_state": "CLOSED_NON_AUTHORIZING"}]},
        partitions={"exposure": "SCREEN_DEVELOPMENT_NON_CONFIRMATORY", "splits": "UNAVAILABLE"},
        budget={"device": costs["device"], "wall_seconds": costs["completed_wall_seconds_total"],
                "cost_units": costs["cost_units"],
                "units_verified": adj["counts"]["COMPLETED_VERIFIED"], "units_failed": 0},
        terminal={"state": "CLOSED", "adjudication": ACCEPTED_VERDICT,
                  "eligible_slots": adj["counts"]["COMPLETED_VERIFIED"],
                  "total_slots": adj["declared_cells"],
                  "reason": (f"{DECISION}: 2 COMPLETED_VERIFIED / 1 QUARANTINED_PARTIAL "
                             f"({partial_cell}, neither failure nor result; charge declared only as a "
                             f"lower bound) / 9 NOT_STARTED; campaign closed, no comparison computed")},
        artifacts={"verification": "SCHEMA_EXACT_AND_SELF_DIGEST_REDERIVED",
                   "closure_record": Path(path).name, "results_root": doc["preserved_root_logical"],
                   "decision": DECISION, "publication_commit": PUBLICATION_B},
        units=units)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--emit", action="store_true", help="append to the outbox (default: print only)")
    ap.add_argument("--outbox-root", type=Path)
    a = ap.parse_args(argv)
    doc = build()
    text = json.dumps(doc, sort_keys=True)
    if str(Path.home()) in text:
        raise EnvelopeBuildRefusal("absolute home path in the envelope")
    if not a.emit:
        print(json.dumps(doc, indent=1, sort_keys=True))
        return 0
    _, ob = _olap()
    print(json.dumps(ob.emit(doc, kind="envelope", root=a.outbox_root), sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
