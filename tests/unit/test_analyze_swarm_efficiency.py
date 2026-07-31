from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from analyze_swarm_efficiency import (  # noqa: E402
    measure_generations,
    parse_worker_log,
)


def _line(timestamp: str, message: str) -> str:
    return f"{timestamp} [doin_node.unified] INFO: {message}"


def test_generation_tail_idle_is_measured_from_last_worker_finish() -> None:
    campaign = "domain-v1"
    omega, _ = parse_worker_log(
        "omega",
        [
            _line(
                "2026-07-31 00:00:00",
                f"[SHARED] Evaluating candidate 0/2 gen=0 for {campaign}",
            ),
            _line(
                "2026-07-31 00:00:10",
                f"[SHARED] Candidate 0/2 result: fitness=1 gen=0 {campaign}",
            ),
        ],
    )
    dragon, _ = parse_worker_log(
        "dragon",
        [
            _line(
                "2026-07-31 00:00:00",
                f"[SHARED] Evaluating candidate 1/2 gen=0 for {campaign}",
            ),
            _line(
                "2026-07-31 00:00:20",
                f"[SHARED] Candidate 1/2 result: fitness=2 gen=0 {campaign}",
            ),
        ],
    )

    measured = measure_generations(omega + dragon, ["omega", "dragon"])[0]

    assert measured["complete"] is True
    assert measured["tail_barrier_idle_seconds"] == pytest.approx(10.0)
    assert measured["tail_barrier_idle_fraction"] == pytest.approx(0.25)
    assert measured["non_evaluation_gap_fraction"] == pytest.approx(0.25)


def test_duplicate_candidate_prevents_utilization_claim() -> None:
    campaign = "domain-v1"
    lines = [
        _line(
            "2026-07-31 00:00:00",
            f"[SHARED] Evaluating candidate 0/1 gen=0 for {campaign}",
        ),
        _line(
            "2026-07-31 00:00:10",
            f"[SHARED] Candidate 0/1 result: fitness=1 gen=0 {campaign}",
        ),
    ]
    first, _ = parse_worker_log("omega", lines)
    second, _ = parse_worker_log("dragon", lines)

    measured = measure_generations(first + second, ["omega", "dragon"])[0]

    assert measured["complete"] is False
    assert measured["duplicate_candidates"] == [0]


def test_fork_adoption_pairs_announcement_route_and_height() -> None:
    _, forks = parse_worker_log(
        "gamma",
        [
            _line("2026-07-31 01:58:19", "Block #9 announced by 100.99.54.79"),
            _line(
                "2026-07-31 01:58:29",
                "Equal-height fork selected peer 100.99.54.79:8470 tip "
                "75a5add46f551637 over local ac9c1dcac46d8a6e; rolling back",
            ),
            _line(
                "2026-07-31 01:58:30",
                "Equal-height fork converged to 75a5add46f551637 at height 10",
            ),
        ],
    )

    event = forks["peer_adoptions"][0]
    assert event["announcement_to_convergence_seconds"] == pytest.approx(11.0)
    assert event["selection_to_convergence_seconds"] == pytest.approx(1.0)
