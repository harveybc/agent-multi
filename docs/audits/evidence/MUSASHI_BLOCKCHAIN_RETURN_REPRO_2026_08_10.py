#!/usr/bin/env python3
"""Socket-free temporary-DB reproducers for findings 209 and 210."""

from __future__ import annotations

import json
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from doin_core.crypto.hashing import compute_merkle_root
from doin_core.models import Block, BlockHeader, Transaction, TransactionType
from doin_node.blockchain.verify import verify_chain_db
from doin_node.storage.chaindb import ChainDB

TS = datetime(2026, 8, 10, tzinfo=timezone.utc)


def _new_db() -> tuple[Path, ChainDB, Block]:
    path = Path(tempfile.mkdtemp(prefix="doin-audit-")) / "chain.db"
    db = ChainDB(path)
    db.open()
    return path, db, db.initialize("genesis")


def _append(db: ChainDB, previous: Block, number: int) -> Block:
    tx = Transaction(
        tx_type=TransactionType.OPTIMAE_ANNOUNCED,
        domain_id="synthetic-audit-domain",
        peer_id="synthetic-audit-peer",
        payload={"number": number},
        timestamp=TS,
    )
    header = BlockHeader(
        index=previous.header.index + 1,
        previous_hash=previous.hash,
        timestamp=TS,
        merkle_root=compute_merkle_root([tx.id]),
        generator_id="synthetic-audit-generator",
        weighted_performance_sum=1.0,
        threshold=0.5,
    )
    block = Block(header=header, transactions=[tx])
    db.append_block(block)
    return block


def metadata_cases() -> dict[str, object]:
    outcomes: dict[str, object] = {}
    for case in ("missing_tip_hash", "missing_height", "malformed_height"):
        path, db, _ = _new_db()
        db.close()
        conn = sqlite3.connect(path)
        if case == "missing_tip_hash":
            conn.execute("DELETE FROM metadata WHERE key = 'tip_hash'")
        elif case == "missing_height":
            conn.execute("DELETE FROM metadata WHERE key = 'height'")
        else:
            conn.execute(
                "UPDATE metadata SET value = 'not-an-int' WHERE key = 'height'"
            )
        conn.commit()
        conn.close()
        try:
            report = verify_chain_db(path)
            outcomes[case] = {
                "returned": True,
                "outcome": report.outcome.value,
                "check_10": report.checks[-1].status.value,
            }
        except Exception as exc:  # evidence captures the untyped escape
            outcomes[case] = {
                "returned": False,
                "raised": type(exc).__name__,
            }
    return outcomes


def post_start_tamper_case() -> dict[str, object]:
    path, db, genesis = _new_db()
    first = _append(db, genesis, 1)
    before = verify_chain_db(path)

    conn = sqlite3.connect(path)
    conn.execute(
        "UPDATE transactions SET payload = ? WHERE block_index = 1",
        (json.dumps({"number": "tampered"}),),
    )
    conn.commit()
    conn.close()

    append_accepted = False
    try:
        _append(db, first, 2)
        append_accepted = True
    finally:
        height = db.height
        after = verify_chain_db(path)
        db.close()
    return {
        "startup_verification": before.outcome.value,
        "append_after_historical_tamper_accepted": append_accepted,
        "height_after_append": height,
        "subsequent_verification": after.outcome.value,
    }


def main() -> None:
    print(
        json.dumps(
            {
                "schema": "agent_multi.audit.blockchain_return_repro.v1",
                "network_used": False,
                "metadata_cases": metadata_cases(),
                "post_start_tamper": post_start_tamper_case(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
