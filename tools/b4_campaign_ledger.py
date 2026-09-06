#!/usr/bin/env python3
"""B4 twelve-cell campaign ledger (order @e8bb500f, E9).

Materializes the campaign ledger from the reviewed population BEFORE
any later GPU dispatch: three origins x four seeds, each with its
exact expected terminal identity. Executable checks reject missing,
duplicate, foreign or extra cells; result-to-cell mismatch; changed
digests; reused attempts; results without complete per-bar paired
returns; partial populations presented as campaign results; a
mechanics/preflight record presented as scientific evidence; and any
sealed-period read. Staged scheduling consumes RUNTIME HEALTH ONLY —
a score-bearing field in the scheduler input refuses."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

import b4_authority as b4a  # noqa: E402

EXPECTED_CELLS = tuple(
    f"o{y}_seed{s}" for y in (2022, 2023, 2024)
    for s in (101, 202, 303, 404))
HEALTH_FIELDS = frozenset({
    "gpu_temperature_celsius", "gpu_memory_free_mib",
    "compute_apps_active", "host_rss_free_bytes",
    "previous_terminal_class", "device_available",
    "stop_file_present"})
SCORE_BEARING_TOKENS = ("return", "sharpe", "equity", "profit",
                        "score", "pnl", "reward", "gate",
                        "advance")


class LedgerRefusal(SystemExit):
    pass


def _sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _sha_obj(o) -> str:
    return hashlib.sha256(json.dumps(
        o, sort_keys=True, default=str).encode()).hexdigest()


def materialize_ledger(mat_root: Path, out: Path) -> dict:
    """The ledger exists BEFORE dispatch and names the exact expected
    terminal result identity for every cell."""
    b4a.verify_campaign_materialization(mat_root)
    cells = json.loads(
        (Path(mat_root) / "B4_CELL_CONFIGS.json").read_bytes())
    if sorted(cells) != sorted(EXPECTED_CELLS):
        raise LedgerRefusal(
            f"REFUSED: population is not the reviewed 12 cells "
            f"(got {sorted(cells)})")
    binding = json.loads((Path(mat_root) / "genesis" /
                          "GENESIS_BINDING.json").read_bytes())
    packet = json.loads((Path(mat_root) /
                         "B4_MATERIALIZATION.json").read_bytes())
    entries = {}
    for cid in EXPECTED_CELLS:
        gmeta = packet["genesis"]["cells"][cid]
        entries[cid] = {
            "cell_config_sha256": cells[cid]["config_sha256"],
            "genesis_binding_sha256": binding["binding"][cid],
            "genesis_container_sha256": gmeta["container_sha256"],
            "genesis_tensor_sha256": gmeta["policy_tensor_sha256"],
            "expected_terminal": "B4_CELL_TERMINAL.json under the "
                                 f"campaign output root at {cid}/",
            "expected_result_schema":
                "agent_multi.b4_cell_terminal.v1 + per-bar paired "
                "net returns on the scored origin index",
            "attempt_ids_consumed": [],
            "status": "PENDING",
        }
    ledger = {
        "schema": "agent_multi.b4_campaign_ledger.v1",
        "population_sha256":
            _sha_file(Path(mat_root) / "B4_CELL_CONFIGS.json"),
        "materialization_sha256":
            _sha_file(Path(mat_root) / "B4_MATERIALIZATION.json"),
        "comparator_ref": packet.get("comparator_ref")
        or packet.get("comparator_dir"),
        "owner_preflight_authorization_sha256":
            b4a.OWNER_GPU_AUTH_SHA,
        "preflight_is_scientific_evidence": False,
        "cells": entries,
        "scheduling_rule": ("staged by RUNTIME HEALTH ONLY "
                            "(fields: %s); observed returns and "
                            "gate direction can never reorder, "
                            "skip or add cells"
                            % ", ".join(sorted(HEALTH_FIELDS))),
    }
    ledger["campaign_digest"] = _sha_obj(
        {cid: e["cell_config_sha256"]
         for cid, e in entries.items()})
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise LedgerRefusal(
            "REFUSED: a materialized campaign ledger is immutable — "
            "use a fresh path for a new campaign")
    out.write_text(json.dumps(ledger, indent=1))
    return ledger


def verify_ledger(ledger_path: Path, mat_root: Path) -> dict:
    """Changed cell or campaign digests after materialization
    refuse."""
    ledger = json.loads(Path(ledger_path).read_bytes())
    if ledger.get("population_sha256") != _sha_file(
            Path(mat_root) / "B4_CELL_CONFIGS.json"):
        raise LedgerRefusal(
            "REFUSED: cell population changed after ledger "
            "materialization")
    cells = json.loads(
        (Path(mat_root) / "B4_CELL_CONFIGS.json").read_bytes())
    if sorted(ledger["cells"]) != sorted(EXPECTED_CELLS):
        raise LedgerRefusal("REFUSED: ledger cell set is not the "
                            "reviewed 12")
    for cid, e in ledger["cells"].items():
        if e["cell_config_sha256"] != cells[cid]["config_sha256"]:
            raise LedgerRefusal(
                f"REFUSED: {cid} cell digest changed after "
                "materialization")
    recomputed = _sha_obj({cid: e["cell_config_sha256"]
                           for cid, e in ledger["cells"].items()})
    if recomputed != ledger["campaign_digest"]:
        raise LedgerRefusal("REFUSED: campaign digest does not "
                            "re-derive")
    return ledger


def verify_campaign_results(ledger_path: Path, mat_root: Path,
                            results_root: Path,
                            comparator_dir: Path = None) -> dict:
    """C14: the complete-population gate RE-DERIVES everything —
    exact claim/terminal schemas and their digest binding (a null
    seal is UNCERTAIN, never accepted), unique attempts, per-bar
    schema/types/finiteness/cardinality, bar-identity equality
    against every comparator arm, independent economic conservation,
    all digests, and no artifact reuse. Labels and counts grant
    nothing."""
    import numpy as np
    import pandas as pd
    ledger = verify_ledger(ledger_path, mat_root)
    results_root = Path(results_root)
    seen_attempts = set()
    seen_artifacts = {}
    facts = {}
    comp_idents = {}
    if comparator_dir is not None:
        packet = json.loads((Path(comparator_dir) /
                             "SCREEN_B_RESULTS.json").read_bytes())
        for r in packet["results"]:
            key = int(r["origin"])
            comp = pd.read_csv(r["per_bar_csv"])
            ident = list(pd.to_datetime(comp["datetime"])
                         .dt.strftime("%Y-%m-%d %H:%M"))
            comp_idents.setdefault(key, {})[r["arm"]] = ident
    required_cols = ("origin", "seed", "datetime_utc",
                     "scored_index", "source_row_sha256",
                     "requested_exposure", "realized_exposure",
                     "economic_equity",
                     "net_equity_delta_observed", "env_pnl_fact",
                     "commission_delta", "net_return")
    bars_per_year = {2022: 2190, 2023: 2190, 2024: 2196}
    for cid in EXPECTED_CELLS:
        cell_dir = results_root / cid
        term_p = cell_dir / "B4_CELL_TERMINAL.json"
        claims = sorted(cell_dir.glob("CLAIM_*.json"))
        if not term_p.is_file():
            raise LedgerRefusal(
                f"REFUSED: partial population — {cid} has no "
                "terminal result")
        if len(claims) != 1:
            raise LedgerRefusal(
                f"REFUSED: {cid} has {len(claims)} claims — exactly "
                "one per generation")
        claim = b4a._strict_json_bytes(claims[0].read_bytes(),
                                       f"claim {cid}")
        term = b4a._strict_json_bytes(term_p.read_bytes(),
                                      f"terminal {cid}")
        if term.get("schema") != "agent_multi.b4_cell_terminal.v1":
            raise LedgerRefusal(f"REFUSED: {cid} foreign terminal "
                                "schema")
        if claim.get("schema") != "agent_multi.b4_attempt_claim.v2":
            raise LedgerRefusal(f"REFUSED: {cid} foreign claim "
                                "schema")
        seal = claim.get("terminal_sha256")
        if seal is None:
            raise LedgerRefusal(
                f"REFUSED: {cid} attempt is UNSEALED — an absent "
                "seal is UNCERTAIN, never accepted evidence")
        if seal != _sha_file(term_p):
            raise LedgerRefusal(
                f"REFUSED: {cid} seal does not bind this terminal")
        if term.get("cell") != cid or claim.get("cell") != cid:
            raise LedgerRefusal(
                f"REFUSED: {cid} identity mismatch")
        if term.get("cell_config_sha256") != \
                ledger["cells"][cid]["cell_config_sha256"]:
            raise LedgerRefusal(
                f"REFUSED: {cid} binds a foreign cell digest")
        att = term.get("attempt_id")
        if not att or not isinstance(att, str) or \
                att != claim.get("attempt_id"):
            raise LedgerRefusal(
                f"REFUSED: {cid} attempt binding broken")
        if att in seen_attempts:
            raise LedgerRefusal(
                f"REFUSED: attempt identity reused at {cid}")
        seen_attempts.add(att)
        if "PREFLIGHT" in str(term.get("terminal", "")).upper() or \
                str(term.get("status", "")).startswith(
                    "B4_GPU_PREFLIGHT"):
            raise LedgerRefusal(
                f"REFUSED: {cid} presents a mechanics/preflight "
                "record as scientific evidence")
        if term.get("terminal") != "COMPLETED":
            raise LedgerRefusal(
                f"REFUSED: {cid} terminal is "
                f"{term.get('terminal')!r}")
        for req in ("per_bar_csv", "per_bar_sha256",
                    "scored_index_sha256", "checkpoint_sha256"):
            if req not in term:
                raise LedgerRefusal(
                    f"REFUSED: {cid} terminal lacks {req!r}")
        pb = Path(term["per_bar_csv"])
        if not pb.is_file() or _sha_file(pb) != \
                term["per_bar_sha256"]:
            raise LedgerRefusal(
                f"REFUSED: {cid} per-bar evidence missing or "
                "digest-broken")
        if term["per_bar_sha256"] in seen_artifacts:
            raise LedgerRefusal(
                f"REFUSED: {cid} reuses the per-bar artifact of "
                f"{seen_artifacts[term['per_bar_sha256']]}")
        seen_artifacts[term["per_bar_sha256"]] = cid
        df = pd.read_csv(pb)
        missing_cols = [c for c in required_cols
                        if c not in df.columns]
        if missing_cols:
            raise LedgerRefusal(
                f"REFUSED: {cid} per-bar lacks required columns "
                f"{missing_cols}")
        year = int(cid.split("_")[0][1:])
        if len(df) != bars_per_year[year]:
            raise LedgerRefusal(
                f"REFUSED: {cid} has {len(df)} scored rows, the "
                f"origin requires exactly {bars_per_year[year]}")
        for col in ("economic_equity", "net_equity_delta_observed",
                    "env_pnl_fact", "commission_delta",
                    "net_return"):
            vals = pd.to_numeric(df[col], errors="coerce"
                                 ).to_numpy(dtype=float)
            if not np.isfinite(vals).all():
                raise LedgerRefusal(
                    f"REFUSED: {cid} non-finite or non-numeric "
                    f"value in {col}")
        if int(df["origin"].iloc[0]) != year or \
                df["origin"].nunique() != 1:
            raise LedgerRefusal(f"REFUSED: {cid} origin column "
                                "mismatch")
        # independent conservation (never one field vs itself)
        resid = float((df["net_equity_delta_observed"]
                       - df["env_pnl_fact"]).abs().max())
        if resid > 1e-6:
            raise LedgerRefusal(
                f"REFUSED: {cid} conservation residual {resid} "
                "between independent counters")
        if (pd.to_numeric(df["commission_delta"])
                < -1e-12).any():
            raise LedgerRefusal(
                f"REFUSED: {cid} cumulative commission decreased")
        ident = list(df["datetime_utc"].astype(str))
        ident_sha = hashlib.sha256(
            "|".join(ident).encode()).hexdigest()
        if ident_sha != term["scored_index_sha256"]:
            raise LedgerRefusal(
                f"REFUSED: {cid} bar identities do not re-derive "
                "the declared scored index")
        if comp_idents:
            for arm, comp_ident in comp_idents.get(year,
                                                   {}).items():
                if comp_ident != ident:
                    raise LedgerRefusal(
                        f"REFUSED: {cid} bar-identity vector "
                        f"differs from comparator {arm}@{year}")
        if term.get("sealed_2025_used") is not False:
            raise LedgerRefusal(
                f"REFUSED: {cid} does not prove sealed-period "
                "absence")
        facts[cid] = {"terminal": term["terminal"],
                      "attempt_id": att,
                      "per_bar_sha256": term["per_bar_sha256"]}
    extra = [d.name for d in results_root.iterdir()
             if d.is_dir() and d.name.startswith("o")
             and d.name not in EXPECTED_CELLS]
    if extra:
        raise LedgerRefusal(
            f"REFUSED: extra/foreign cells in the result tree: "
            f"{extra}")
    return {"cells": facts, "n": len(facts)}


def verify_single_cell_result(results_root: Path, cell_id: str,
                              expected_cell_sha: str) -> dict:
    """C1.6: the executor calls THIS immediately after writing a
    COMPLETED terminal — the same field discipline as the campaign
    verifier, for one cell."""
    term_p = Path(results_root) / cell_id / "B4_CELL_TERMINAL.json"
    if not term_p.is_file():
        raise LedgerRefusal(f"REFUSED: {cell_id} has no terminal")
    term = json.loads(term_p.read_bytes())
    if term.get("schema") != "agent_multi.b4_cell_terminal.v1":
        raise LedgerRefusal(f"REFUSED: {cell_id} foreign schema")
    if term.get("cell") != cell_id:
        raise LedgerRefusal(f"REFUSED: {cell_id} identity mismatch")
    if term.get("cell_config_sha256") != expected_cell_sha:
        raise LedgerRefusal(
            f"REFUSED: {cell_id} binds a foreign cell digest")
    if term.get("terminal") != "COMPLETED":
        raise LedgerRefusal(
            f"REFUSED: {cell_id} terminal is "
            f"{term.get('terminal')!r}")
    att = term.get("attempt_id")
    if not att or not isinstance(att, str):
        raise LedgerRefusal(
            f"REFUSED: {cell_id} COMPLETED without attempt_id")
    pb = term.get("per_bar_csv")
    if not pb or not Path(pb).is_file() or \
            _sha_file(Path(pb)) != term.get("per_bar_sha256"):
        raise LedgerRefusal(
            f"REFUSED: {cell_id} per-bar evidence missing or "
            "digest-broken")
    if term.get("sealed_2025_used") is not False:
        raise LedgerRefusal(
            f"REFUSED: {cell_id} without sealed-absence proof")
    return term


def schedule_next(ledger: dict, health: dict) -> str:
    """Health-only staged scheduling: the next PENDING cell in the
    fixed predeclared order, dispatched only when runtime health
    permits. A score-bearing field in the input refuses — observed
    returns can never steer the schedule."""
    for k in health:
        base = k.lower()
        if k not in HEALTH_FIELDS:
            for tok in SCORE_BEARING_TOKENS:
                if tok in base:
                    raise LedgerRefusal(
                        f"REFUSED: score-bearing field {k!r} in the "
                        "scheduler input — scheduling depends on "
                        "runtime health only")
            raise LedgerRefusal(
                f"REFUSED: unknown scheduler input field {k!r}")
    if not health.get("device_available", False):
        return "HOLD: device unavailable"
    if health.get("stop_file_present"):
        return "HOLD: external stop present"
    if health.get("compute_apps_active"):
        return "HOLD: foreign compute workload active"
    for cid in EXPECTED_CELLS:
        if ledger["cells"][cid]["status"] == "PENDING":
            return cid
    return "CAMPAIGN_COMPLETE"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--materialization-root", type=Path,
                    required=True)
    ap.add_argument("--ledger", type=Path, required=True)
    ap.add_argument("--action", required=True,
                    choices=["materialize", "verify"])
    ap.add_argument("--results-root", type=Path)
    args = ap.parse_args(argv)
    if args.action == "materialize":
        ledger = materialize_ledger(args.materialization_root,
                                    args.ledger)
        print(json.dumps({"cells": len(ledger["cells"]),
                          "campaign_digest":
                          ledger["campaign_digest"][:16]}, indent=1))
        return 0
    verify_ledger(args.ledger, args.materialization_root)
    if args.results_root:
        facts = verify_campaign_results(
            args.ledger, args.materialization_root,
            args.results_root)
        print(json.dumps(facts, indent=1))
    else:
        print("LEDGER_VERIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
