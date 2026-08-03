#!/usr/bin/env python3
"""Socket-free independent counterexamples for IBKR L1 Milestones A-E.

Run with the canonical trading-stack interpreter.  The construction helper is
imported from the delivered LTS tests, but every mutation and assertion in this
file is auditor-authored.  No real broker client is constructed.
"""
from __future__ import annotations

import json
import socket
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

LTS_ROOT = Path("/home/harveybc/Documents/GitHub/lts")
sys.path.insert(0, str(LTS_ROOT))


from tests.unit.test_ibkr_l1_outbox import (  # noqa: E402
    Env,
    NOW,
    QUOTE,
    _asset_intent,
)
from tools.mint_paper_capability import (  # noqa: E402
    mint_payload,
    write_capability,
)
import ib_async  # noqa: E402,F401 -- load ssl/socket subclasses before guard


def _network_refused(*_args, **_kwargs):
    raise AssertionError("network access attempted by socket-free audit")


# Import object-construction dependencies first (ssl subclasses socket.socket
# during import), then forbid every subsequent socket construction/connection.
socket.socket = _network_refused
socket.create_connection = _network_refused


def _accepted_entry(env: Env) -> dict:
    env.mint()
    env.decide(_asset_intent())
    return env.consumer.consume_entries(
        quote=QUOTE, now=NOW + timedelta(seconds=2)
    )[0]


def post_ack_stop_loss_disappears() -> dict:
    with tempfile.TemporaryDirectory() as directory:
        env = Env(Path(directory))
        try:
            entry = _accepted_entry(env)
            parent_id, _, stop_id = entry["order_ids"]
            env.client.drop_order(stop_id)
            env.client.fill_parent(parent_id, 20_000.0)
            result = env.consumer.sync_parent_fill(
                entry["effect_id"], now=NOW + timedelta(seconds=3)
            )
            exposures = env.olap.open_exposures()
            reproduced = (
                result is not None
                and bool(exposures)
                and env.olap.get_state("halt", "none") == "none"
            )
            return {
                "reproduced": reproduced,
                "dropped_stop_id": stop_id,
                "effect_state": env.olap.effect_row(entry["effect_id"])["state"],
                "broker_position_units": env.client.position_facts()[0]["units"],
                "l0_exposure_units": exposures[0]["units_open"],
                "halt": env.olap.get_state("halt", "none"),
            }
        finally:
            env.olap.close()


def partial_fill_is_invisible() -> dict:
    with tempfile.TemporaryDirectory() as directory:
        env = Env(Path(directory))
        try:
            entry = _accepted_entry(env)
            env.client.fill_parent(entry["order_ids"][0], 5_000.0)
            result = env.consumer.sync_parent_fill(
                entry["effect_id"], now=NOW + timedelta(seconds=3)
            )
            positions = env.client.position_facts()
            exposures = env.olap.open_exposures()
            return {
                "reproduced": result is None and bool(positions) and not exposures,
                "sync_result": result,
                "broker_position_units": positions[0]["units"],
                "l0_exposures": len(exposures),
                "halt": env.olap.get_state("halt", "none"),
            }
        finally:
            env.olap.close()


def flatten_overcloses_and_reverses() -> dict:
    with tempfile.TemporaryDirectory() as directory:
        env = Env(Path(directory))
        try:
            entry = _accepted_entry(env)
            env.client.fill_parent(entry["order_ids"][0], 20_000.0)
            env.consumer.sync_parent_fill(
                entry["effect_id"], now=NOW + timedelta(seconds=3)
            )
            env.flatten_all("audit-overclose")
            pending = env.olap.l1_pending_decisions("would_be_flatten")[0]
            intent = json.loads(pending["intent_json"])
            original_delta = intent["delta_units"]
            intent["delta_units"] = -40_000.0
            env.olap._con.execute(
                "UPDATE decisions SET intent_json=? WHERE idempotency_key=?",
                (json.dumps(intent), pending["idempotency_key"]),
            )
            result = env.consumer.consume_flattens(
                now=NOW + timedelta(seconds=11)
            )[0]
            placed = [
                fact for name, fact in env.client.calls if name == "place_order"
            ]
            positions = env.client.position_facts()
            return {
                "reproduced": (
                    result["state"] == "effect_unknown"
                    and positions[0]["units"] == -20_000.0
                    and placed[-1]["totalQuantity"] == 40_000.0
                ),
                "original_delta": original_delta,
                "altered_delta": intent["delta_units"],
                "submitted_action": placed[-1]["action"],
                "submitted_quantity": placed[-1]["totalQuantity"],
                "broker_position_after": positions[0]["units"],
                "halt": env.olap.get_state("halt", "none"),
            }
        finally:
            env.olap.close()


def restart_drops_authorized_contract_id() -> dict:
    with tempfile.TemporaryDirectory() as directory:
        env = Env(Path(directory))
        try:
            expected_con_id = 12_087_792
            payload = mint_payload(
                env.profile,
                quantity_ceiling=20_000.0,
                max_risk_fraction_at_stop=0.005,
                validity_seconds=900,
                contract_con_id=expected_con_id,
                now=NOW,
            )
            write_capability(payload, env.store)
            env.decide(_asset_intent())
            original_ack = env.consumer.controller.acknowledge

            def crash_before_ack(*_args, **_kwargs):
                raise ConnectionError("audit crash before acknowledgement")

            env.consumer.controller.acknowledge = crash_before_ack
            try:
                env.consumer.consume_entries(
                    quote=QUOTE, now=NOW + timedelta(seconds=2)
                )
            except ConnectionError:
                pass
            finally:
                env.consumer.controller.acknowledge = original_ack

            effect = env.olap.nonterminal_effects()[0]
            wrong_con_id = 999
            for order_id in effect["order_ids"]:
                env.client._orders[order_id]["contract"]["conId"] = wrong_con_id
            outcomes = env.consumer.resume(now=NOW + timedelta(seconds=3))
            state = env.olap.effect_row(effect["effect_id"])["state"]
            return {
                "reproduced": state == "acknowledged",
                "authorized_con_id": expected_con_id,
                "observed_con_id": wrong_con_id,
                "effect_state": state,
                "reacknowledged": outcomes[0].get("reacknowledged"),
            }
        finally:
            env.olap.close()


def pre_call_crash_stalls_forever() -> dict:
    with tempfile.TemporaryDirectory() as directory:
        env = Env(Path(directory))
        try:
            env.mint()
            env.decide(_asset_intent())
            original_record = env.olap.record_broker_fact

            def crash_after_atomic_commit(*_args, **_kwargs):
                raise RuntimeError("audit crash after capability/effect commit")

            env.olap.record_broker_fact = crash_after_atomic_commit
            try:
                env.consumer.consume_entries(
                    quote=QUOTE, now=NOW + timedelta(seconds=2)
                )
            except RuntimeError:
                pass
            finally:
                env.olap.record_broker_fact = original_record

            resumed = env.consumer.resume(now=NOW + timedelta(seconds=3))
            effects = env.olap.nonterminal_effects()
            place_calls = [
                1 for name, _ in env.client.calls if name == "place_order"
            ]
            return {
                "reproduced": (
                    len(effects) == 1
                    and effects[0]["state"] == "journaled_pending"
                    and resumed[0]["classification"] == "consumed_before_effect"
                    and not place_calls
                ),
                "effect_state": effects[0]["state"],
                "classification": resumed[0]["classification"],
                "broker_place_calls": len(place_calls),
                "pending_decisions": len(
                    env.olap.l1_pending_decisions("would_be_order")
                ),
            }
        finally:
            env.olap.close()


def main() -> int:
    evidence = {
        "schema": "agent_multi.audit.ibkr_l1_a_e_repro.v1",
        "network_used": False,
        "scenarios": {
            "post_ack_stop_loss_disappears": post_ack_stop_loss_disappears(),
            "partial_fill_is_invisible": partial_fill_is_invisible(),
            "flatten_overcloses_and_reverses": flatten_overcloses_and_reverses(),
            "restart_drops_authorized_contract_id": (
                restart_drops_authorized_contract_id()
            ),
            "pre_call_crash_stalls_forever": pre_call_crash_stalls_forever(),
        },
    }
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0 if all(
        scenario["reproduced"] for scenario in evidence["scenarios"].values()
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
