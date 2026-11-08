"""Offline counterexamples for Retsu's forward-pool handoff; no broker access."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lts-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.lts_root).resolve()))
    from app.forward_pool.accounts import Position, VirtualAccount
    from app.forward_pool.cohort import PoolService
    from app.forward_pool.core import BusinessContract, PoolRefusal
    from app.forward_pool.snapshot import HourlyBar

    stamp = 1_700_000_000_000_000_000
    bar = HourlyBar(
        snapshot_id="completed-hour", bar_close_utc_ns=stamp,
        decision_time_utc_ns=stamp, availability_watermark_utc_ns=stamp,
        finalized=True, instrument="EURUSD", columns=("RAW_CLOSE_LOG_RETURN",),
        values=(0.01,), published_utc_ns=(stamp - 1,), bid=1.1000, ask=1.1002,
        high=1.1010, low=1.0990, session_id="fixture-week", operational=True,
        before_force_flat=True, same_session_next=True, civil_offset_minutes=0,
    )
    findings = []
    with tempfile.TemporaryDirectory(prefix="forward-pool-audit-") as directory:
        root = Path(directory)
        pool = PoolService(root)
        hashes = {k: "sha256:" + "ab" * 32 for k in (
            "recipe_sha256", "scaler_sha256", "strategy_sha256", "code_sha256")}
        record = pool.seed_selected70(now_ns=stamp, **hashes)
        findings.append(dict(
            id="F01", expected_columns=70, actual_columns=len(record.spec.columns),
            fixed_signal=record.spec.fixed_signal,
            violates_expected=len(record.spec.columns) != 70,
        ))
        pool.route.writer = record.candidate_id
        try:
            pool.route.submit(
                effect_id="unknown-fixture", owner=record.candidate_id,
                command=1, snapshot_id=bar.snapshot_id, now_ns=stamp,
                transport=lambda effect: "unknown",
            )
        except PoolRefusal as exc:
            if exc.code != "UNKNOWN_OUTCOME":
                raise
        before = dict(halted=pool.route.halted_unknown, effects=len(pool.route.effects))
        pool._save()
        pool.close()
        pool = PoolService(root)
        after = dict(halted=pool.route.halted_unknown, effects=len(pool.route.effects))
        findings.append(dict(id="F02", before=before, after=after,
                             violates_expected=before != after))
        candidate = pool.records[record.candidate_id]
        candidate.account.balance = 12_000.0
        candidate.account.position = None
        pool.route.cash = pool.route_account.balance = 8_500.0
        pool.route.position = Position(1, 100, 1.1, 1.0, 1.2, record.candidate_id)
        pool.route_account.position = pool.route.position
        pool._mirror_route(record.candidate_id, bar)
        findings.append(dict(
            id="F03", route_cash_before=8500.0, route_cash_after=pool.route.cash,
            shadow_cash=12000.0, broker_fill_events=0,
            violates_expected=pool.route.cash != 8500.0,
        ))
        pool.close()

    account = VirtualAccount("fixture", "fixture")
    # The high/low happened before this completed-bar decision, not after entry.
    result = account.apply(1, bar, BusinessContract(), entry_sl=1.0995, entry_tp=1.1020)
    findings.append(dict(
        id="F04", entries=account.entries, close_reason=result["close_reason"],
        post_entry_market_events=0, balance=account.balance,
        violates_expected=account.entries == 1 and result["close_reason"] is not None,
    ))
    payload = dict(
        schema="forward_pool.counterexamples.v1", evidence_class="OFFLINE_FIXTURE",
        broker_orders=0, live_ledger_access=False, findings=findings,
        verdict="REVISE" if any(x["violates_expected"] for x in findings) else "WITNESSES_PASS",
    )
    Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))
    return 1 if payload["verdict"] == "REVISE" else 0


if __name__ == "__main__":
    raise SystemExit(main())
