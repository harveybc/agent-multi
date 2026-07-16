from __future__ import annotations

from tools.gpu_temperature_watchdog import evaluate, parse_nvidia_smi


def test_parse_nvidia_smi_rows() -> None:
    rows = parse_nvidia_smi(
        "0, NVIDIA GeForce RTX 4090 Laptop GPU, 84, 35, 42.65\n"
        "1, NVIDIA GeForce RTX 5090, 52, 5, 85.62\n"
    )
    assert rows[0]["temperature_c"] == 84
    assert rows[1]["name"] == "NVIDIA GeForce RTX 5090"


def test_alert_repeats_and_recovers_with_hysteresis() -> None:
    state = {"events": {}}
    hot = [{
        "index": 0,
        "name": "GPU",
        "temperature_c": 84.0,
        "utilization_pct": 50.0,
        "power_w": 100.0,
    }]
    messages, keys = evaluate(
        machine="dragon",
        gpus=hot,
        state=state,
        threshold=78,
        recovery_threshold=72,
        expected_gpus=1,
        repeat_seconds=3600,
        now=1000,
    )
    assert "TEMPERATURE ALERT" in messages[0]
    assert keys == ["temperature:0"]
    state["events"]["temperature:0"]["last_sent_at"] = 1000

    messages, keys = evaluate(
        machine="dragon",
        gpus=hot,
        state=state,
        threshold=78,
        recovery_threshold=72,
        expected_gpus=1,
        repeat_seconds=3600,
        now=1100,
    )
    assert messages == []
    assert keys == []

    cool = [{**hot[0], "temperature_c": 70.0}]
    messages, keys = evaluate(
        machine="dragon",
        gpus=cool,
        state=state,
        threshold=78,
        recovery_threshold=72,
        expected_gpus=1,
        repeat_seconds=3600,
        now=1200,
    )
    assert "RECOVERED" in messages[0]
    assert keys == ["temperature:0"]


def test_missing_egpu_produces_count_alert() -> None:
    state = {"events": {}}
    messages, keys = evaluate(
        machine="gamma",
        gpus=[{
            "index": 0,
            "name": "Laptop GPU",
            "temperature_c": 50.0,
            "utilization_pct": 10.0,
            "power_w": 30.0,
        }],
        state=state,
        threshold=78,
        recovery_threshold=72,
        expected_gpus=2,
        repeat_seconds=3600,
        now=1000,
    )
    assert "GPU COUNT ALERT" in messages[0]
    assert keys == ["gpu_count"]
