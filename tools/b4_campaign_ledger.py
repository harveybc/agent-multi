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
import os
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
        # C32: generation provenance — the v6 ledger is a FRESH
        # materialization (never a copy of the superseded v5
        # mutable ledger); it names the incident, the fixed prior
        # charge, and the append-only lineage. Its expected
        # identity is recomputable from the same materialization.
        "generation_provenance": {
            "campaign_generation": b4a.CAMPAIGN_GENERATION,
            "supersedes_generation": b4a.V6_GENERATION,
            "authorized_generation":
                b4a.AUTHORIZED_CAMPAIGN_GENERATION,
            "supersedes_results_root_logical":
                b4a.V6_RESULTS_ROOT_LOGICAL,
            "incident_lineage": {
                "v5_environment_incident_sha256":
                    b4a.INCIDENT_RECORD_SHA,
                "v6_runtime_incident_order_sha256":
                    b4a.V6_INCIDENT_ORDER_SHA},
            "prior_generations_gpu_seconds_charged":
                b4a.PRIOR_GENERATIONS_GPU_SECONDS,
            "scientific_change": "NONE",
            "failed_attempt_artifacts_reusable": False},
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
    # C32: a ledger of the CURRENT generation must carry truthful
    # provenance — the incident, the lineage and the fixed prior
    # charge. Superseded-generation ledgers (v5) are immutable
    # history and are never consumed by the v6 execution path.
    gp = ledger.get("generation_provenance")
    if gp is not None:
        lin = gp.get("incident_lineage") or {}
        if gp.get("campaign_generation") != \
                b4a.CAMPAIGN_GENERATION or \
                gp.get("supersedes_generation") != \
                b4a.V6_GENERATION or \
                gp.get("authorized_generation") != \
                b4a.AUTHORIZED_CAMPAIGN_GENERATION or \
                lin.get("v5_environment_incident_sha256") != \
                b4a.INCIDENT_RECORD_SHA or \
                lin.get("v6_runtime_incident_order_sha256") != \
                b4a.V6_INCIDENT_ORDER_SHA or \
                gp.get("prior_generations_gpu_seconds_charged") != \
                b4a.PRIOR_GENERATIONS_GPU_SECONDS or \
                gp.get("scientific_change") != "NONE" or \
                gp.get("failed_attempt_artifacts_reusable") \
                is not False:
            raise LedgerRefusal(
                "REFUSED: ledger generation provenance differs "
                "from the live chain constants")
    elif b4a.CAMPAIGN_GENERATION != \
            b4a.AUTHORIZED_CAMPAIGN_GENERATION:
        raise LedgerRefusal(
            "REFUSED: a recovered-generation campaign requires a "
            "ledger with explicit generation provenance — the "
            "superseded mutable ledger is never reused as genesis")
    return ledger


def _load_orch():
    import importlib.util as ilu
    spec = ilu.spec_from_file_location(
        "b4orch_led", REPO / "tools/b4_campaign_orchestrator.py")
    m = ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _derive_comparator_dir(mat_root: Path) -> Path:
    """C20: comparator evidence is MANDATORY and derived from the
    reviewed materialization — it cannot be omitted by API or CLI."""
    packet = json.loads((Path(mat_root) /
                         "B4_MATERIALIZATION.json").read_bytes())
    ref = packet.get("comparator_ref") or packet.get(
        "comparator_dir")
    if not ref:
        raise LedgerRefusal(
            "REFUSED: materialization carries no comparator anchor")
    d = (b4a.resolve_source_ref(str(ref))
         if ":" in str(ref)[:12] else Path(ref))
    if not (Path(d) / "SCREEN_B_RESULTS.json").is_file():
        raise LedgerRefusal(
            "REFUSED: derived comparator population is absent — "
            "completion cannot be claimed without it")
    return Path(d)


TERMINAL_SCHEMA_KEYS = {
    "schema", "cell", "terminal", "g1_eligible",
    "checkpoint_promotable", "attempt_id", "cell_config_sha256",
    "artifact_class", "checkpoint_sha256", "checkpoint_path",
    "per_bar_csv", "per_bar_sha256", "scored_index_sha256",
    "scored_bars", "counter_semantics", "sealed_2025_used",
    "wall_seconds", "effective_limits",
    "authorization_record_sha256", "amendment_11_sha256",
    # C37: the RECOVERED authority — a terminal carrying only
    # a11/a12 refuses under the v6 generation
    "campaign_generation", "recovery_acta_sha256",
    "pinned_execution_commit", "latest_amendment_sha256"}
CLAIM_SCHEMA_KEYS = {
    "schema", "campaign_generation", "attempt_id", "cell",
    "claimed_wall", "claimed_monotonic", "holder_pid",
    "terminal_sha256", "recovery_acta_sha256", "claim_sha256"}
PER_BAR_SCHEMA = {
    "origin": "int", "seed": "int", "datetime_utc": "str",
    "scored_index": "int", "source_row_sha256": "str",
    "requested_exposure": "float", "realized_exposure": "float",
    "gross_equity": "str", "economic_equity": "float",
    "net_equity_delta_observed": "float", "env_pnl_fact": "float",
    "commission_delta": "float",
    "pre_commission_equity_delta_derived": "float",
    "slippage_declared": "str", "net_return": "float"}


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
    orch = _load_orch()
    if comparator_dir is None:
        comparator_dir = _derive_comparator_dir(mat_root)
    seen_attempts = set()
    seen_artifacts = {}
    seen_checkpoints = {}
    facts = {}
    comp_idents = {}
    packet = json.loads((Path(comparator_dir) /
                         "SCREEN_B_RESULTS.json").read_bytes())
    for r in packet["results"]:
        key = int(r["origin"])
        comp = pd.read_csv(r["per_bar_csv"])
        ident = list(pd.to_datetime(comp["datetime"])
                     .dt.strftime("%Y-%m-%d %H:%M"))
        comp_idents.setdefault(key, {})[r["arm"]] = ident
    # C21: expected frozen-source identity per origin (datetime +
    # close), recomputed from the resolved contract source.
    expected_rows_by_origin = {}
    for year in (2022, 2023, 2024):
        contract = json.loads((Path(mat_root) / "contracts" /
                               f"b4_causal_origin_{year}"
                               "_contract.json").read_bytes())
        srcp = b4a.resolve_source_ref(contract["source_ref"])
        full = pd.read_csv(srcp, parse_dates=["DATE_TIME"])
        in_year = full.index[full["DATE_TIME"].dt.year == year]
        lo = max(0, int(in_year[0]) - 540)
        sl = full.iloc[lo: int(in_year[-1]) + 1]
        scored = sl.iloc[540:]
        expected_rows_by_origin[year] = {
            "datetimes": list(scored["DATE_TIME"]
                              .dt.strftime("%Y-%m-%d %H:%M")),
            "row_shas": [hashlib.sha256(
                f"{d}|{c:.10g}".encode()).hexdigest()
                for d, c in zip(
                    scored["DATE_TIME"].dt.strftime(
                        "%Y-%m-%d %H:%M"),
                    scored["CLOSE"].astype(float))],
            "start_index": 540}
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
        claim = orch._secure_json(claims[0], f"claim {cid}")
        term = orch._secure_json(term_p, f"terminal {cid}")
        if term.get("schema") != "agent_multi.b4_cell_terminal.v1":
            raise LedgerRefusal(f"REFUSED: {cid} foreign terminal "
                                "schema")
        if claim.get("schema") != "agent_multi.b4_attempt_claim.v2":
            raise LedgerRefusal(f"REFUSED: {cid} foreign claim "
                                "schema")
        # C21: EXACT schemas at the consuming boundary
        if set(claim) != CLAIM_SCHEMA_KEYS:
            raise LedgerRefusal(
                f"REFUSED: {cid} claim keys are not the exact "
                "schema")
        if set(term) != TERMINAL_SCHEMA_KEYS:
            raise LedgerRefusal(
                f"REFUSED: {cid} terminal keys are not the exact "
                f"schema (diff: "
                f"{sorted(set(term) ^ TERMINAL_SCHEMA_KEYS)})")
        # C18: the seal is the PHYSICAL intent/completion witness
        if orch.seal_state(results_root, cid) != "SEALED":
            raise LedgerRefusal(
                f"REFUSED: {cid} attempt is UNSEALED or UNCERTAIN "
                "— never accepted evidence")
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
        try:
            pb_bytes = orch._secure_read(pb, expected_mode=None)
        except SystemExit:
            raise LedgerRefusal(
                f"REFUSED: {cid} per-bar evidence missing or "
                "unreadable")
        if hashlib.sha256(pb_bytes).hexdigest() != \
                term["per_bar_sha256"]:
            raise LedgerRefusal(
                f"REFUSED: {cid} per-bar evidence digest-broken")
        if term["per_bar_sha256"] in seen_artifacts:
            raise LedgerRefusal(
                f"REFUSED: {cid} reuses the per-bar artifact of "
                f"{seen_artifacts[term['per_bar_sha256']]}")
        seen_artifacts[term["per_bar_sha256"]] = cid
        import io as _io
        df = pd.read_csv(_io.BytesIO(pb_bytes))
        missing_cols = [c for c in required_cols
                        if c not in df.columns]
        if missing_cols:
            raise LedgerRefusal(
                f"REFUSED: {cid} per-bar lacks required columns "
                f"{missing_cols}")
        # C21: EXACT per-bar schema and primitive column types
        if set(df.columns) != set(PER_BAR_SCHEMA):
            raise LedgerRefusal(
                f"REFUSED: {cid} per-bar columns are not the exact "
                f"schema (diff: "
                f"{sorted(set(df.columns) ^ set(PER_BAR_SCHEMA))})")
        for col, want in PER_BAR_SCHEMA.items():
            kind = df[col].dtype.kind
            ok = {"int": kind == "i",
                  "float": kind == "f",
                  "str": kind in ("O", "U")}[want]
            if not ok:
                raise LedgerRefusal(
                    f"REFUSED: {cid} per-bar column {col!r} has "
                    f"kind {kind!r}, the schema requires {want}")
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
        # C21: seed column must equal the cell seed
        cell_seed = int(cid.split("seed")[1])
        if df["seed"].nunique() != 1 or \
                int(df["seed"].iloc[0]) != cell_seed:
            raise LedgerRefusal(
                f"REFUSED: {cid} seed column does not equal the "
                "cell seed")
        # C21: exact absolute scored-index sequence
        exp = expected_rows_by_origin[year]
        want_idx = list(range(exp["start_index"],
                              exp["start_index"] + len(df)))
        if list(df["scored_index"].astype(int)) != want_idx:
            raise LedgerRefusal(
                f"REFUSED: {cid} scored_index is not the expected "
                "absolute sequence")
        # C21: timestamps ordered, unique, equal to frozen source
        dts = list(df["datetime_utc"].astype(str))
        if dts != exp["datetimes"]:
            raise LedgerRefusal(
                f"REFUSED: {cid} timestamps differ from the frozen "
                "source origin")
        if len(set(dts)) != len(dts):
            raise LedgerRefusal(f"REFUSED: {cid} duplicate "
                                "timestamps")
        # C21: source_row_sha256 recomputed from frozen source rows
        if list(df["source_row_sha256"].astype(str)) != \
                exp["row_shas"]:
            raise LedgerRefusal(
                f"REFUSED: {cid} source_row_sha256 does not "
                "recompute from the frozen source rows")
        # C21: net_return recomputed from the equity path
        eq = df["economic_equity"].to_numpy(dtype=float)
        nr = df["net_return"].to_numpy(dtype=float)
        recomputed_nr = np.zeros_like(nr)
        recomputed_nr[1:] = eq[1:] / np.where(eq[:-1] == 0.0, 1.0,
                                              eq[:-1]) - 1.0
        if float(np.max(np.abs(nr[1:] - recomputed_nr[1:]))) > 1e-9:
            raise LedgerRefusal(
                f"REFUSED: {cid} net_return does not recompute "
                "from the economic equity path")
        # C25: the checkpoint MUST exist and verify from one
        # descriptor — a terminal cannot prove an artifact by
        # naming bytes that are no longer present.
        ck = term.get("checkpoint_path")
        if not isinstance(ck, str) or not ck:
            raise LedgerRefusal(
                f"REFUSED: {cid} terminal lacks checkpoint_path")
        try:
            ckfd = os.open(ck, os.O_RDONLY | os.O_NOFOLLOW)
        except FileNotFoundError:
            raise LedgerRefusal(
                f"REFUSED: {cid} checkpoint artifact is ABSENT — "
                "a declared digest of missing bytes is not "
                "evidence")
        except OSError as exc:
            raise LedgerRefusal(
                f"REFUSED: {cid} checkpoint unopenable ({exc}) — "
                "symlinks and races fail closed")
        try:
            ckst = os.fstat(ckfd)
            import stat as _stat
            if not _stat.S_ISREG(ckst.st_mode):
                raise LedgerRefusal(
                    f"REFUSED: {cid} checkpoint is not a regular "
                    "file")
            if ckst.st_uid != os.getuid():
                raise LedgerRefusal(
                    f"REFUSED: {cid} checkpoint has a foreign "
                    "owner")
            if _stat.S_IMODE(ckst.st_mode) & 0o022:
                raise LedgerRefusal(
                    f"REFUSED: {cid} checkpoint is group/world "
                    "writable — unsafe artifact mode")
            h = hashlib.sha256()
            while True:
                chunk = os.read(ckfd, 1 << 20)
                if not chunk:
                    break
                h.update(chunk)
        finally:
            os.close(ckfd)
        if h.hexdigest() != term["checkpoint_sha256"]:
            raise LedgerRefusal(
                f"REFUSED: {cid} checkpoint bytes differ from "
                "the declared digest")
        if term["checkpoint_sha256"] in seen_checkpoints:
            raise LedgerRefusal(
                f"REFUSED: {cid} reuses the checkpoint of "
                f"{seen_checkpoints[term['checkpoint_sha256']]}")
        seen_checkpoints[term["checkpoint_sha256"]] = cid
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
        # C27.7: the terminal's authorization/amendment digests are
        # RE-DERIVED from the live reviewer record and chain files;
        # producer labels grant nothing.
        want_auth = _sha_file(
            b4a.CAMPAIGN_AUTHORIZATION_RECORD_PATH)
        want_a11 = _sha_file(b4a.AMENDMENT_11_PATH)
        if term.get("authorization_record_sha256") != want_auth:
            raise LedgerRefusal(
                f"REFUSED: {cid} terminal authorization digest "
                "does not re-derive from the reviewer record")
        if term.get("amendment_11_sha256") != want_a11:
            raise LedgerRefusal(
                f"REFUSED: {cid} terminal amendment-11 digest "
                "does not re-derive from the live chain")
        # C37: the RECOVERED authority is re-derived from the
        # reviewed acta — producer labels grant nothing; a
        # terminal with a stale, transplanted or absent recovery
        # binding refuses.
        wit = b4a.require_v6_launch_open()
        if term.get("campaign_generation") != \
                wit["campaign_generation"] or \
                term.get("recovery_acta_sha256") != \
                wit["acta_sha256"] or \
                term.get("pinned_execution_commit") != \
                wit["pinned_commit"] or \
                term.get("latest_amendment_sha256") != \
                wit["latest_amendment_sha256"]:
            raise LedgerRefusal(
                f"REFUSED: {cid} terminal recovery bindings "
                "(generation/acta/pinned commit/latest amendment) "
                "do not re-derive from the reviewed recovery "
                "authority")
        facts[cid] = {"terminal": term["terminal"],
                      "attempt_id": att,
                      "per_bar_sha256": term["per_bar_sha256"],
                      "checkpoint_sha256_verified": h.hexdigest()}
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
    term = _load_orch()._secure_json(term_p,
                                     f"terminal {cell_id}")
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
    # C37: the RECOVERED authority verifies BEFORE any evidence —
    # a stale or transplanted binding never gets to per-bar data.
    wit = b4a.require_v6_launch_open()
    for k, want in (("campaign_generation",
                     wit["campaign_generation"]),
                    ("recovery_acta_sha256", wit["acta_sha256"]),
                    ("pinned_execution_commit",
                     wit["pinned_commit"]),
                    ("latest_amendment_sha256",
                     wit["latest_amendment_sha256"])):
        if term.get(k) != want:
            raise LedgerRefusal(
                f"REFUSED: {cell_id} terminal {k} does not "
                "re-derive from the reviewed recovery authority")
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
