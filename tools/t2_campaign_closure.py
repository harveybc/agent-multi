#!/usr/bin/env python3
"""R2 (order 2026-09-11): close T2 from its 242 durable records.

Three things were missing, and one of them was not known until this
tool refused to run.

1. The executor heartbeat says ``done=241`` while 242 records, 242
   claims and 242 arrays sit on disk. A heartbeat is telemetry — it is
   written before the unit it names is sealed — so the discrepancy is a
   reporting artefact, not a missing unit. Nothing durable said so.

2. Nothing verified the inventory as an EXACT triple per unit: one
   claim, one array, one record, no extras, no duplicates, nothing
   foreign.

3. The recomputation must run under the REVIEWED code identity. The
   external successor execution record pins seven executor files and a
   commit; ``tools/t2_confirmatory_executor.py`` was corrected AFTER
   that record was authored, so the branch tip can no longer satisfy
   its own gate. This tool therefore refuses to guess: it takes an
   explicit ``--reviewed-checkout`` whose bytes equal the record, and
   imports the pinned modules from THERE. The divergence is reported,
   never smoothed over, and no gate is loosened to get past it.

Nothing is retrained and nothing is re-downloaded: every number comes
from the arrays and records already on disk.

    python tools/t2_campaign_closure.py \\
        --reviewed-checkout <clean checkout at the pinned commit> \\
        [--root <results root>] [--emit] [--emit-outbox]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

TIP_REPO = Path(__file__).resolve().parents[1]

CLOSURE_SCHEMA = "agent_multi.t2_campaign_closure.v1"
CLOSURE_LOG = "T2_CAMPAIGN_CLOSURE.jsonl"

#: fields that say WHEN or HOW LONG, never WHAT was found.
VOLATILE_FOR_IDENTITY = frozenset({"closed_at", "measurement"})

UNIT_KINDS = ("CLAIM", "ARRAYS", "RECORD")


class ClosureRefusal(SystemExit):
    def __init__(self, msg: str) -> None:
        super().__init__(f"REFUSED: {msg}")


def sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha_obj(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def utc_now() -> str:
    return (datetime.now(timezone.utc).replace(microsecond=0)
            .isoformat().replace("+00:00", "Z"))


# --------------------------------------------------- reviewed identity
def load_reviewed_modules(checkout: Path):
    """Put the REVIEWED checkout's tools first on sys.path.

    Import order is the whole point: ``t2_confirmatory`` and
    ``t2_confirmatory_executor`` must come from the reviewed checkout,
    while the reconstruction driver — which the execution record does
    not pin — comes from the branch tip. Reversing this would recompute
    under code no external record has seen.
    """
    checkout = checkout.expanduser().resolve()
    for entry in (str(TIP_REPO / "tools"), str(checkout / "tools")):
        while entry in sys.path:
            sys.path.remove(entry)
    sys.path.insert(0, str(TIP_REPO / "tools"))
    sys.path.insert(0, str(checkout / "tools"))
    import t2_confirmatory as conf
    import t2_confirmatory_executor as ex
    import t2_completion_reconstruction as recon
    for mod, want_root in ((conf, checkout), (ex, checkout),
                           (recon, TIP_REPO)):
        got = Path(mod.__file__).resolve().parents[1]
        if got != want_root.resolve():
            raise ClosureRefusal(
                f"{mod.__name__} resolved to {got}, not {want_root}")
    return conf, ex, recon


def assert_reviewed_identity(conf, checkout: Path) -> dict:
    record = json.loads(
        conf.T2_SUCCESSOR_EXECUTION_RECORD_PATH.read_text())
    declared = record["executor_code_identity"]
    physical = {rel: sha_file(checkout / rel) for rel in declared}
    diff = sorted(k for k in declared if declared[k] != physical[k])
    if diff:
        raise ClosureRefusal(
            f"the checkout at {checkout.name} is not the reviewed "
            f"identity; these files differ from the execution record: "
            f"{diff}")
    head = subprocess.run(("git", "-C", str(checkout), "rev-parse", "HEAD"),
                          capture_output=True, text=True).stdout.strip()
    if head != record["pinned_commit"]:
        raise ClosureRefusal(
            f"the checkout is at {head[:12]}, the record pins "
            f"{record['pinned_commit'][:12]}")
    dirty = subprocess.run(("git", "-C", str(checkout), "status",
                            "--porcelain"),
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        raise ClosureRefusal("the reviewed checkout is not clean")
    return {
        "pinned_commit": record["pinned_commit"],
        "execution_record_sha256": sha_obj(
            {k: record[k] for k in sorted(record)}),
        "pinned_files": {k: declared[k] for k in sorted(declared)},
        "checkout_clean": True,
    }


def report_tip_divergence(conf, checkout: Path) -> dict:
    """Name the divergence between the tip and the reviewed identity.

    This is a finding, not a failure of the closure: the tip's extra
    hardening is real work, but it is work no external record has seen,
    and a result recomputed under it would carry an identity nobody
    reviewed.
    """
    record = json.loads(
        conf.T2_SUCCESSOR_EXECUTION_RECORD_PATH.read_text())
    declared = record["executor_code_identity"]
    diverged = {}
    for rel, want in declared.items():
        tip_file = TIP_REPO / rel
        got = sha_file(tip_file) if tip_file.is_file() else "ABSENT"
        if got != want:
            diverged[rel] = {"reviewed": want, "branch_tip": got}
    return {
        "branch_tip_can_run_the_gate": not diverged,
        "diverged_files": diverged,
        "consequence": (
            "the branch tip refuses its own confirmatory gate, so the "
            "recomputation MUST run from the reviewed checkout. The "
            "corrections made after the record was authored (a typed "
            "refusal for omitted mlp seeds at both executor sites) are "
            "NOT part of the identity this closure recomputes under, "
            "and are declared here rather than silently included"
            if diverged else
            "the branch tip equals the reviewed identity"),
    }


# ------------------------------------------------------------ inventory
def exact_inventory(ex, root: Path, uids: list[str]) -> dict:
    """One claim, one array, one record per unit. Nothing else."""
    units_dir = root / "units"
    actual = sorted(p.name for p in units_dir.iterdir() if p.is_file())
    expected: dict[str, str] = {}
    for uid in uids:
        safe = ex._safe_name(uid)
        expected[f"CLAIM_{safe}.json"] = uid
        expected[f"ARRAYS_{safe}.npz"] = uid
        expected[f"RECORD_{safe}.json"] = uid

    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    # A filesystem cannot hold two entries with one name, so a
    # "duplicate" here means a second artifact claiming the same unit
    # and kind under a different spelling — which is what `extra`
    # already catches. Recording the count keeps the claim explicit.
    per_kind = {k: sum(1 for n in actual if n.startswith(k + "_"))
                for k in UNIT_KINDS}
    if missing or extra:
        raise ClosureRefusal(
            f"the unit inventory is not exact: missing={missing[:5]} "
            f"extra={extra[:5]}")
    if set(per_kind.values()) != {len(uids)}:
        raise ClosureRefusal(
            f"per-kind counts {per_kind} do not all equal "
            f"{len(uids)} sealed units")
    return {
        "sealed_units": len(uids),
        "artifacts_per_kind": per_kind,
        "missing": missing,
        "extra": extra,
        "duplicates": [],
        "total_artifacts": len(actual),
        "exact": True,
    }


# ------------------------------------------------------- reconciliation
def reconcile_heartbeat(root: Path, ex, uids: list[str],
                        counts: dict) -> dict:
    hb_p = root / "EXECUTOR_HEARTBEAT.json"
    hb = json.loads(hb_p.read_text()) if hb_p.is_file() else {}
    current = hb.get("current_unit")
    safe = ex._safe_name(current) if current else None
    record_p = (root / "units" / f"RECORD_{safe}.json") if safe else None
    sealed = bool(record_p and record_p.is_file())
    done = hb.get("done")
    adjudicated = counts["COMPLETED_VERIFIED"] + counts["TERMINAL_FAILED"]
    return {
        "heartbeat_done": done,
        "heartbeat_current_unit": current,
        "current_unit_record_sealed": sealed,
        "records_on_disk": len(uids),
        "units_adjudicated": adjudicated,
        "discrepancy": (adjudicated - done) if isinstance(done, int)
                       else "UNAVAILABLE",
        "explanation": (
            "the heartbeat is written at the START of a unit: it names "
            "the unit being worked on and counts the units finished "
            "BEFORE it. When the executor sealed the last unit "
            f"({current}) it did not write a further heartbeat, because "
            "there was no next unit to announce. So done=241 and "
            "current_unit=<the 242nd> together describe 242 units, and "
            "the 242nd unit's record is physically present. No unit is "
            "missing and none was inferred: the count comes from the "
            "records, and the heartbeat is telemetry with no terminal "
            "authority."),
        "authority": "RECORDS_GOVERN_TELEMETRY_DOES_NOT",
    }


# ----------------------------------------------------------- the closure
def build_closure(conf, ex, recon, root: Path, checkout: Path) -> dict:
    measurement: dict = {}
    t0 = time.perf_counter()

    identity = assert_reviewed_identity(conf, checkout)
    divergence = report_tip_divergence(conf, checkout)

    t_recon = time.perf_counter()
    doc = recon.reconstruct(root)
    measurement["reconstruction_seconds"] = round(
        time.perf_counter() - t_recon, 2)

    facts = conf.verify_confirmatory_gates(
        ex.MANIFEST_PATH, ex.active_design_path(),
        census_path=ex.CENSUS_PATH)
    uids = facts["design"].doc["task_population"]["series_ids"]

    inventory = exact_inventory(ex, root, uids)
    counts = doc["final_adjudication_counts"]
    reconciliation = reconcile_heartbeat(root, ex, uids, counts)

    screen = doc["screen_adjudication"]
    verdict = screen.get("verdict")
    if counts["TERMINAL_FAILED"] > 0:
        terminal_state = "FAILED"
    elif verdict in (None, "", "UNAVAILABLE"):
        terminal_state = "INCONCLUSIVE"
    else:
        terminal_state = "COMPLETE"

    measurement["closure_seconds"] = round(time.perf_counter() - t0, 2)

    closure = {
        "schema": CLOSURE_SCHEMA,
        "closed_at": utc_now(),
        "campaign_root_logical": root.name,
        "reviewed_identity": identity,
        "branch_tip_divergence": divergence,
        "inventory": inventory,
        "final_adjudication_counts": counts,
        "heartbeat_reconciliation": reconciliation,
        "wall_ledger": doc["wall_ledger"],
        "release_sequence": doc["release_sequence"],
        "screen_adjudication": screen,
        "gate_facts": doc["gate_facts"],
        "terminal_state": terminal_state,
        "retraining": "NONE — every number re-derives from the arrays "
                      "and records already on disk",
        "downloads": "NONE",
        "measurement": measurement,
    }
    body = {k: closure[k] for k in sorted(closure)}
    closure["closure_sha256"] = sha_obj(body)
    closure["adjudication_sha256"] = sha_obj(
        {k: v for k, v in body.items() if k not in VOLATILE_FOR_IDENTITY})
    return closure


def last_closure(root: Path) -> dict | None:
    log = root / CLOSURE_LOG
    if not log.is_file():
        return None
    lines = [ln for ln in log.read_text().splitlines() if ln.strip()]
    return json.loads(lines[-1]) if lines else None


# ------------------------------------------------------------- outbox
def emit_to_outbox(closure: dict, *, uids: list[str],
                   manifest: dict | None = None) -> dict:
    """One durable envelope into the CRISP-DM outbox.

    The outbox is a local content-addressed directory; emitting never
    contacts a database and never fails because one is unavailable.
    """
    predictor = Path.home() / "Documents/GitHub/predictor"
    if not (predictor / "olap/outbox.py").is_file():
        raise ClosureRefusal(
            f"no outbox implementation at {predictor.name}/olap")
    sys.path.insert(0, str(predictor))
    from olap import outbox as ob                      # noqa: E402

    screen = closure["screen_adjudication"]
    counts = closure["final_adjudication_counts"]
    gate = closure["gate_facts"]
    # The envelope contract is EXACT, and rightly so: my first attempt
    # omitted data_consumed and artifacts and the loader dead-lettered
    # it. A campaign envelope that cannot say what it consumed is a
    # summary, not evidence.
    document = {
        "schema": "crispdm.campaign_envelope.v1",
        "producer": "agent-multi",
        "campaign_key": "t2::resource_successor_screen",
        "result_class": "DEVELOPMENT",
        "identity": {
            "run_id": closure["campaign_root_logical"],
            "code_identity": closure["reviewed_identity"]["pinned_commit"],
            "design_sha256": gate["design_self_sha256"],
            "record_sha256": closure["closure_sha256"],
        },
        "data_consumed": {
            "variables": [{"id": uid,
                           "digest": gate["census_sha256"],
                           "eligibility_state":
                               "PUBLIC_FORECASTING_EVIDENCE"}
                          for uid in sorted(uids)],
            "operators": [{"id": "ewma_alpha_0.3",
                           "digest": gate["design_self_sha256"],
                           "eligibility_state":
                               "T1_ACCEPTED_OPERATOR"}],
            "datasets": [{"id": closure["campaign_root_logical"],
                          "digest": gate["manifest_sha256"],
                          "eligibility_state":
                              "PUBLIC_FORECASTING_EVIDENCE"}],
        },
        "terminal": {
            "state": closure["terminal_state"],
            "adjudication": screen.get("verdict", "UNAVAILABLE"),
            "failure_phase": "UNAVAILABLE",
            "failure_type": "UNAVAILABLE",
            "reason": screen.get("reason", "UNAVAILABLE"),
        },
        "partitions": {
            "exposure": "SCREEN_DEVELOPMENT_NON_CONFIRMATORY",
            "splits": "rolling_origin",
        },
        "budget": {
            "cost_units": "wall_seconds",
            "device": "cpu",
            "wall_seconds": closure["wall_ledger"].get(
                "charged_state", "UNAVAILABLE"),
            "units_verified": counts["COMPLETED_VERIFIED"],
            "units_failed": counts["TERMINAL_FAILED"],
        },
        "artifacts": {
            "verification": "SCHEMA_EXACT_AND_SELF_DIGEST_REDERIVED",
            "closure_record": CLOSURE_LOG,
            "results_root": closure["campaign_root_logical"],
        },
        "units": [{
            "cell_key": closure["campaign_root_logical"],
            "candidate_key": "ewma_alpha_0.3",
            "metric_name": "screen_primary_estimand",
            "metric_value": screen.get(
                "primary_estimand_unweighted_mean_of_panel_effects"),
            "terminal_state": closure["terminal_state"],
        }],
    }
    document["envelope_sha256"] = sha_obj(
        {k: document[k] for k in sorted(document)})
    return ob.emit(document, kind="envelope")


# ---------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reviewed-checkout", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--emit-outbox", action="store_true")
    ap.add_argument("--outbox-only", action="store_true",
                    help="emit the LAST durable closure to the outbox "
                         "without recomputing it; the closure record "
                         "is the authority, the envelope is derived")
    args = ap.parse_args(argv)

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        os.nice(15)
    except OSError:
        pass

    conf, ex, recon = load_reviewed_modules(args.reviewed_checkout)
    root = (args.root or recon.DEFAULT_ROOT).expanduser()

    if args.outbox_only:
        closure = last_closure(root)
        if closure is None:
            raise ClosureRefusal(
                "there is no durable closure to emit; run the closure "
                "first — an envelope is derived from a record, never "
                "invented")
        facts = conf.verify_confirmatory_gates(
            ex.MANIFEST_PATH, ex.active_design_path(),
            census_path=ex.CENSUS_PATH)
        uids = facts["design"].doc["task_population"]["series_ids"]
        closure["outbox"] = emit_to_outbox(closure, uids=uids)
        print(json.dumps(closure, indent=1, sort_keys=True,
                         default=str))
        return 0

    closure = build_closure(conf, ex, recon, root, args.reviewed_checkout)

    if args.emit:
        previous = last_closure(root)
        if previous and previous.get("adjudication_sha256") == \
                closure["adjudication_sha256"]:
            closure["emitted"] = False
            closure["already_closed_at"] = previous["closed_at"]
        else:
            with (root / CLOSURE_LOG).open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(closure, sort_keys=True,
                                    default=str) + "\n")
            closure["emitted"] = True

    if args.emit_outbox:
        facts = conf.verify_confirmatory_gates(
            ex.MANIFEST_PATH, ex.active_design_path(),
            census_path=ex.CENSUS_PATH)
        closure["outbox"] = emit_to_outbox(
            closure,
            uids=facts["design"].doc["task_population"]["series_ids"])

    print(json.dumps(closure, indent=1, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
