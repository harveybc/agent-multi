#!/usr/bin/env python3
"""Reproduce the 2026-08-03 IBKR Paper L1 adapter audit findings.

This program is local-only. It imports the LTS adapter and its fixtures, opens
no socket, creates no broker order and needs no credential.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path


def _load_test_helpers(lts_root: Path):
    test_path = lts_root / "tests/unit/test_ibkr_l1_adapter.py"
    spec = importlib.util.spec_from_file_location("ibkr_l1_audit_fixture", test_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load audit fixture: {test_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lts-root",
        type=Path,
        default=Path("/home/harveybc/Documents/GitHub/lts"),
    )
    args = parser.parse_args()
    lts_root = args.lts_root.resolve()
    sys.path.insert(0, str(lts_root))

    helpers = _load_test_helpers(lts_root)
    from app.ibkr_l1_adapter import (  # pylint: disable=import-outside-toplevel
        IbkrPaperL1Sink,
        verify_bracket_acknowledgement,
    )

    plan = helpers._plan()
    altered = helpers._ack(
        plan,
        mutate={
            "take_profit": {
                "status": "Cancelled",
                "orderType": "MKT",
                "lmtPrice": 9.99,
            },
            "stop_loss": {
                "status": "Rejected",
                "orderType": "LMT",
                "auxPrice": 9.98,
            },
        },
    )
    verdict = verify_bracket_acknowledgement(plan=plan, open_orders=altered)

    with tempfile.TemporaryDirectory() as directory:
        profile = helpers._profile(Path(directory))
        sink = IbkrPaperL1Sink(
            helpers._auth(profile, token="arbitrary-local-token"),
            ledger=helpers.FakeLedger(),
            dry_run=False,
        )

        class BrokerObjectWithoutSubmissionMethods:
            pass

        sink._ib = BrokerObjectWithoutSubmissionMethods()  # audit injection
        submission = sink.submit_bracket(helpers._intent(), plan)

    with tempfile.TemporaryDirectory() as directory:
        invalid = helpers._profile(
            Path(directory),
            venue="anything",
            host="0.0.0.0",
            max_orders_this_activation=0,
            quantity=-1,
            stop_distance_price=-2,
            take_profit_distance_price=-3,
            max_spread_price=-4,
        )

    result = {
        "schema": "agent_multi.audit.ibkr_l1_adapter_repro.v1",
        "network_used": False,
        "altered_cancelled_rejected_bracket_marked_protected": verdict["protected"],
        "broker_object_without_place_order_marked_submitted": submission["submitted"],
        "network_submission_counter_without_broker_call": sink.network_submissions,
        "invalid_profile_accepted": {
            "venue": invalid.venue,
            "host": invalid.host,
            "max_orders": invalid.max_orders_this_activation,
            "quantity": invalid.quantity,
            "stop_distance": invalid.stop_distance_price,
            "take_profit_distance": invalid.take_profit_distance_price,
            "max_spread": invalid.max_spread_price,
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
