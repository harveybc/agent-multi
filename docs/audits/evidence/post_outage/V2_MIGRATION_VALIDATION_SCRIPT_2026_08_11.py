"""WP5 (order of 2026-08-11, section 9): validate the prepared v2 chain
migration manifest in a CLEAN TEMPORARY state directory, and derive the
facts for one new explicit-ID v2 genesis proposal.

Validation and preparation ONLY. No fleet host is touched, no legacy
database is opened for write, nothing is deployed, no chain is resumed.
All state lives under the session scratchpad and is created fresh here.

Checks (each typed, each recorded):
  A. manifest parse + explicit identity present + well-formed;
  B. manifest identity is self-consistent with the deterministic v2
     genesis at doin-core 00397f5 (genesis_hash == Block.genesis().hash,
     chain_id == "doin-" + hash[:12]);
  C. the actual config materializer (doin_node.cli.load_config) ACCEPTS
     the manifest (shared_population domain + explicit identity);
  D. a fresh v2 chain instantiates from the manifest in a clean temp
     state dir via the same calls UnifiedNode.start() makes:
     ChainDB.open -> initialize -> bind_verified_cursor ->
     set_chain_identity; stored identity round-trips;
  E. PROTOCOL_VERSION == 2 and ChainStatus exposes chain_id +
     genesis_hash (via SyncManager.get_our_status, the node's own
     status construction); validate_peer_chain_status accepts the
     attested status and refuses an unattested legacy status with the
     typed errors;
  F. the 10-check chain verifier CLI (python -m
     doin_node.blockchain.verify) exits 0 / fully_verified against the
     fresh chain with --expect-chain-id and --expect-genesis;
  G. a shared-population config WITHOUT explicit chain identity is
     REFUSED by the materializer with the typed
     ChainIdentityConfigError (three variants: both missing, chain_id
     missing, genesis_hash missing) — and the refusal happens before
     any node object exists;
  H. genesis-proposal derivation: a NEW job-specific deterministic
     genesis (explicit generator_id) instantiated in a SECOND clean
     temp state dir also fully verifies (exit 0) under its own new
     chain_id/genesis pair, proving the explicit-ID mechanism supports
     a fresh, non-colliding chain for the next DOIN component job.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRATCH = Path(
    "/tmp/claude-1000/-home-harveybc-Documents-GitHub-predictor/"
    "94c1b43d-d764-48d5-885f-68470ae06b5f/scratchpad"
)
CORE_SRC = SCRATCH / "doin-core-wp5" / "src"
NODE_SRC = SCRATCH / "doin-node-wp5" / "src"
PLUGINS_SRC = Path("/home/harveybc/Documents/GitHub/doin-plugins/src")
MANIFEST = (
    SCRATCH
    / "doin-node-wp5"
    / "examples"
    / "fleet_shared_population_identity_template.json"
)
STATE_A = SCRATCH / "v2-state-validate"        # manifest-identity chain
STATE_B = SCRATCH / "v2-state-genesis-proposal"  # new job genesis chain

sys.path.insert(0, str(CORE_SRC))
sys.path.insert(0, str(NODE_SRC))
sys.path.insert(0, str(PLUGINS_SRC))

REPORT: dict = {
    "work_package": "WP5 v2 migration manifest validation (209-211 accepted)",
    "utc": datetime.now(timezone.utc).isoformat(),
    "doin_core_worktree": str(CORE_SRC.parent),
    "doin_node_worktree": str(NODE_SRC.parent),
    "doin_core_commit": "00397f5390649280aab7ba9b6420e71ff299a9da",
    "doin_node_commit": "0821ec236e85040d9ab45c89b01437f4cbaeb9ab",
    "manifest_path_worktree": str(MANIFEST),
    "manifest_path_canonical": (
        "doin-node branch fix/tx-content-binding-20260810 @ 0821ec2:"
        "examples/fleet_shared_population_identity_template.json"
    ),
    "checks": [],
}


def record(check: str, ok: bool, detail: dict) -> None:
    REPORT["checks"].append({"check": check, "ok": bool(ok), **detail})
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {check}")
    if not ok:
        print(json.dumps(detail, indent=2, default=str))


def main() -> int:
    # Fresh, clean temp state dirs — never a legacy database.
    for d in (STATE_A, STATE_B):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    # ── A. manifest parse + explicit identity ────────────────────────
    raw_bytes = MANIFEST.read_bytes()
    REPORT["manifest_sha256"] = hashlib.sha256(raw_bytes).hexdigest()
    manifest = json.loads(raw_bytes)
    chain_id = manifest.get("chain_id", "")
    genesis_hash = manifest.get("genesis_hash", "")
    ok_a = (
        isinstance(chain_id, str)
        and isinstance(genesis_hash, str)
        and len(chain_id) > 0
        and len(genesis_hash) == 64
        and all(c in "0123456789abcdef" for c in genesis_hash)
        and any(
            (d.get("optimization_config") or {}).get("shared_population")
            for d in manifest.get("domains", [])
        )
    )
    record(
        "A_manifest_explicit_identity",
        ok_a,
        {
            "chain_id": chain_id,
            "genesis_hash": genesis_hash,
            "sha256": REPORT["manifest_sha256"],
        },
    )

    # ── B. identity consistent with deterministic v2 genesis ─────────
    from doin_core.models.block import Block

    det = Block.genesis("genesis")
    ok_b = det.hash == genesis_hash and chain_id == f"doin-{det.hash[:12]}"
    record(
        "B_identity_matches_deterministic_genesis",
        ok_b,
        {
            "computed_genesis_hash": det.hash,
            "computed_chain_id": f"doin-{det.hash[:12]}",
        },
    )

    # ── C. real materializer ACCEPTS the manifest ────────────────────
    from doin_node.cli import load_config

    cfg = load_config(str(MANIFEST), {"data_dir": str(STATE_A)})
    ok_c = cfg.chain_id == chain_id and cfg.genesis_hash == genesis_hash
    record(
        "C_materializer_accepts_manifest",
        ok_c,
        {"config_chain_id": cfg.chain_id, "config_genesis_hash": cfg.genesis_hash},
    )

    # ── D. fresh v2 chain instantiation in the clean temp state dir ──
    # Mirrors UnifiedNode.start(): open -> initialize (empty DB) ->
    # bind_verified_cursor -> verify -> set_chain_identity.
    from doin_node.storage.chaindb import ChainDB

    db_path = STATE_A / "chain.db"
    db = ChainDB(db_path)
    db.open()
    assert db.height == 0
    genesis_block = db.initialize("genesis")
    db.bind_verified_cursor(db.height, db.tip_hash)
    db.set_chain_identity(chain_id, genesis_hash)
    stored = db.get_chain_identity()
    tip_hash = db.tip_hash
    height = db.height
    db.close()
    ok_d = (
        genesis_block.hash == genesis_hash
        and stored == (chain_id, genesis_hash)
        and height == 1
        and tip_hash == genesis_hash
    )
    record(
        "D_fresh_chain_instantiated_and_identity_stamped",
        ok_d,
        {
            "state_dir": str(STATE_A),
            "db_path": str(db_path),
            "height": height,
            "tip_hash": tip_hash,
            "stored_identity": list(stored) if stored else None,
        },
    )

    # ── E. PROTOCOL_VERSION=2 ChainStatus exposes identity ───────────
    from doin_core.protocol.messages import (
        PROTOCOL_VERSION,
        ChainIdentityMismatchError,
        ChainStatus,
        ProtocolVersionMismatchError,
        validate_peer_chain_status,
    )
    from doin_node.network.sync import SyncManager

    sm = SyncManager()
    sm.protocol_version = PROTOCOL_VERSION
    sm.chain_id = chain_id
    sm.genesis_hash = genesis_hash
    sm.update_our_state(height=height, tip_hash=tip_hash)
    status = sm.get_our_status()
    e_details: dict = {
        "PROTOCOL_VERSION": PROTOCOL_VERSION,
        "status_protocol_version": status.protocol_version,
        "status_chain_id": status.chain_id,
        "status_genesis_hash": status.genesis_hash,
    }
    ok_e = (
        PROTOCOL_VERSION == 2
        and status.protocol_version == 2
        and status.chain_id == chain_id
        and status.genesis_hash == genesis_hash
    )
    # attested status is accepted...
    try:
        validate_peer_chain_status(
            status,
            expected_chain_id=chain_id,
            expected_genesis_hash=genesis_hash,
        )
        e_details["attested_accepted"] = True
    except Exception as e:  # pragma: no cover - failure path
        ok_e = False
        e_details["attested_accepted"] = f"REFUSED: {type(e).__name__}: {e}"
    # ...an unattested legacy status is refused with the typed error...
    legacy = ChainStatus(chain_height=5, tip_hash="ab" * 32, tip_index=4)
    try:
        validate_peer_chain_status(
            legacy,
            expected_chain_id=chain_id,
            expected_genesis_hash=genesis_hash,
        )
        ok_e = False
        e_details["legacy_refused"] = False
    except ProtocolVersionMismatchError as e:
        e_details["legacy_refused"] = f"{type(e).__name__}: {e}"
    # ...and a v2 status for a DIFFERENT chain is refused as identity mismatch.
    alien = ChainStatus(
        chain_height=5,
        tip_hash="ab" * 32,
        tip_index=4,
        protocol_version=2,
        chain_id="doin-000000000000",
        genesis_hash="0" * 64,
    )
    try:
        validate_peer_chain_status(
            alien,
            expected_chain_id=chain_id,
            expected_genesis_hash=genesis_hash,
        )
        ok_e = False
        e_details["alien_refused"] = False
    except ChainIdentityMismatchError as e:
        e_details["alien_refused"] = f"{type(e).__name__}: {e}"
    record("E_chainstatus_v2_exposes_identity", ok_e, e_details)

    # ── F. 10-check verifier CLI, exit 0 against the fresh chain ─────
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{CORE_SRC}:{NODE_SRC}:{PLUGINS_SRC}"
    cli = subprocess.run(
        [
            sys.executable,
            "-m",
            "doin_node.blockchain.verify",
            "--db",
            str(db_path),
            "--expect-chain-id",
            chain_id,
            "--expect-genesis",
            genesis_hash,
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    cli_report = json.loads(cli.stdout) if cli.stdout.strip() else {}
    checks_passed = [
        c for c in cli_report.get("checks", []) if c.get("status") == "pass"
    ]
    ok_f = (
        cli.returncode == 0
        and cli_report.get("outcome") == "fully_verified"
        and len(cli_report.get("checks", [])) == 10
        and len(checks_passed) == 10
    )
    record(
        "F_verifier_cli_10_checks_exit_0",
        ok_f,
        {
            "exit_code": cli.returncode,
            "outcome": cli_report.get("outcome"),
            "checks_pass": f"{len(checks_passed)}/10",
            "stderr_summary": cli.stderr.strip().splitlines()[-1]
            if cli.stderr.strip()
            else "",
        },
    )
    REPORT["verifier_cli_report_manifest_chain"] = cli_report

    # ── G. shared-population config WITHOUT identity is refused ──────
    from doin_node.unified import ChainIdentityConfigError

    variants = {
        "both_missing": ("chain_id", "genesis_hash"),
        "chain_id_missing": ("chain_id",),
        "genesis_hash_missing": ("genesis_hash",),
    }
    g_details: dict = {}
    ok_g = True
    for name, drop in variants.items():
        stripped = {k: v for k, v in manifest.items() if k not in drop}
        p = STATE_A / f"stripped_{name}.json"
        p.write_text(json.dumps(stripped, indent=2))
        try:
            load_config(str(p), {"data_dir": str(STATE_A / "never-used")})
            ok_g = False
            g_details[name] = "ACCEPTED (defect: must be refused)"
        except ChainIdentityConfigError as e:
            g_details[name] = f"typed refusal: ChainIdentityConfigError: {e}"
        except Exception as e:  # wrong type = not the contract
            ok_g = False
            g_details[name] = f"WRONG ERROR TYPE {type(e).__name__}: {e}"
    record("G_shared_population_without_identity_refused", ok_g, g_details)

    # ── H. NEW explicit-ID genesis for the next DOIN component job ───
    generator_id = "doin-v2-component-job-001-20260811"
    proposal_genesis = Block.genesis(generator_id)
    proposal_hash = proposal_genesis.hash
    proposal_chain_id = f"doin-{proposal_hash[:12]}"
    db2_path = STATE_B / "chain.db"
    db2 = ChainDB(db2_path)
    db2.open()
    gb2 = db2.initialize(generator_id)  # pre-seed at job materialization
    db2.bind_verified_cursor(db2.height, db2.tip_hash)
    db2.set_chain_identity(proposal_chain_id, proposal_hash)
    stored2 = db2.get_chain_identity()
    db2.close()
    cli2 = subprocess.run(
        [
            sys.executable,
            "-m",
            "doin_node.blockchain.verify",
            "--db",
            str(db2_path),
            "--expect-chain-id",
            proposal_chain_id,
            "--expect-genesis",
            proposal_hash,
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    cli2_report = json.loads(cli2.stdout) if cli2.stdout.strip() else {}
    # Reproducibility: an independent recomputation yields the same hash.
    recomputed = Block.genesis(generator_id).hash
    ok_h = (
        gb2.hash == proposal_hash
        and recomputed == proposal_hash
        and stored2 == (proposal_chain_id, proposal_hash)
        and proposal_hash != genesis_hash  # never collides with manifest/legacy
        and cli2.returncode == 0
        and cli2_report.get("outcome") == "fully_verified"
        and len([c for c in cli2_report.get("checks", []) if c.get("status") == "pass"]) == 10
    )
    record(
        "H_new_explicit_id_genesis_instantiates_and_verifies",
        ok_h,
        {
            "generator_id": generator_id,
            "proposal_genesis_hash": proposal_hash,
            "proposal_chain_id": proposal_chain_id,
            "distinct_from_manifest_identity": proposal_hash != genesis_hash,
            "verifier_exit": cli2.returncode,
            "verifier_outcome": cli2_report.get("outcome"),
            "state_dir": str(STATE_B),
        },
    )
    REPORT["verifier_cli_report_proposal_chain"] = cli2_report
    REPORT["genesis_proposal"] = {
        "generator_id": generator_id,
        "chain_id": proposal_chain_id,
        "genesis_hash": proposal_hash,
    }

    REPORT["all_pass"] = all(c["ok"] for c in REPORT["checks"])
    out = SCRATCH / "V2_MIGRATION_VALIDATION_REPORT.json"
    out.write_text(json.dumps(REPORT, indent=2, default=str))
    print(f"report: {out}")
    print(f"ALL_PASS={REPORT['all_pass']}")
    return 0 if REPORT["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
