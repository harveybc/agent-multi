#!/usr/bin/env python3
"""Measure shared-population utilization and fork convergence from node logs."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable


TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
LINE_TIMESTAMP = re.compile(r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
EVALUATION_START = re.compile(
    r"\[SHARED\] Evaluating candidate (?P<candidate>\d+)/(?P<population>\d+) "
    r"gen=(?P<generation>\d+) for (?P<campaign>\S+)"
)
EVALUATION_RESULT = re.compile(
    r"\[SHARED\] Candidate (?P<candidate>\d+)/(?P<population>\d+) result: "
    r"fitness=(?P<fitness>\S+) gen=(?P<generation>\d+) (?P<campaign>\S+)"
)
BLOCK_ANNOUNCEMENT = re.compile(
    r"Block #(?P<height>\d+) announced by (?P<route>\S+)"
)
FORK_SELECTED = re.compile(
    r"Equal-height fork selected peer (?P<route>\S+) tip (?P<tip>[0-9a-f]+) "
    r"over local (?P<local_tip>[0-9a-f]+)"
)
FORK_CONVERGED = re.compile(
    r"Equal-height fork converged to (?P<tip>[0-9a-f]+) at height (?P<height>\d+)"
)
FORK_LOCAL = re.compile(
    r"Equal-height fork resolved in favor of local tip (?P<tip>[0-9a-f]+)"
)


@dataclass(frozen=True)
class EvaluationInterval:
    worker: str
    campaign: str
    generation: int
    candidate: int
    population: int
    started_at: datetime
    completed_at: datetime
    fitness: float

    @property
    def duration_seconds(self) -> float:
        return (self.completed_at - self.started_at).total_seconds()


def _timestamp(line: str) -> datetime | None:
    match = LINE_TIMESTAMP.match(line)
    if not match:
        return None
    return datetime.strptime(match.group("timestamp"), TIMESTAMP_FORMAT)


def _route_host(route: str) -> str:
    return route.rstrip(".").split(":", 1)[0]


def parse_worker_log(
    worker: str,
    lines: Iterable[str],
    *,
    campaign_filter: str | None = None,
) -> tuple[list[EvaluationInterval], dict[str, object]]:
    open_starts: dict[tuple[str, int, int], list[tuple[datetime, int]]] = defaultdict(list)
    intervals: list[EvaluationInterval] = []
    announcements: list[dict[str, object]] = []
    selected: dict[str, object] | None = None
    convergence: list[dict[str, object]] = []
    local_resolutions = 0

    for line in lines:
        observed_at = _timestamp(line)
        if observed_at is None:
            continue

        match = EVALUATION_START.search(line)
        if match:
            campaign = match.group("campaign")
            if campaign_filter is None or campaign_filter == campaign:
                key = (
                    campaign,
                    int(match.group("generation")),
                    int(match.group("candidate")),
                )
                open_starts[key].append(
                    (observed_at, int(match.group("population")))
                )
            continue

        match = EVALUATION_RESULT.search(line)
        if match:
            campaign = match.group("campaign")
            if campaign_filter is not None and campaign_filter != campaign:
                continue
            key = (
                campaign,
                int(match.group("generation")),
                int(match.group("candidate")),
            )
            starts = open_starts.get(key)
            if starts:
                started_at, population = starts.pop()
                if observed_at >= started_at:
                    intervals.append(
                        EvaluationInterval(
                            worker=worker,
                            campaign=campaign,
                            generation=key[1],
                            candidate=key[2],
                            population=population,
                            started_at=started_at,
                            completed_at=observed_at,
                            fitness=float(match.group("fitness")),
                        )
                    )
            continue

        match = BLOCK_ANNOUNCEMENT.search(line)
        if match:
            announcements.append(
                {
                    "observed_at": observed_at,
                    "height": int(match.group("height")),
                    "route": match.group("route"),
                }
            )
            continue

        match = FORK_SELECTED.search(line)
        if match:
            selected = {
                "observed_at": observed_at,
                "route": match.group("route"),
                "tip": match.group("tip"),
                "local_tip": match.group("local_tip"),
            }
            continue

        match = FORK_CONVERGED.search(line)
        if match and selected is not None:
            height = int(match.group("height"))
            selected_route = str(selected["route"])
            compatible = [
                item
                for item in announcements
                if item["height"] == height - 1
                and _route_host(str(item["route"])) == _route_host(selected_route)
                and item["observed_at"] <= selected["observed_at"]
            ]
            announcement = compatible[-1] if compatible else None
            convergence.append(
                {
                    "worker": worker,
                    "height": height,
                    "route": selected_route,
                    "tip": match.group("tip"),
                    "local_tip": selected["local_tip"],
                    "selected_at": selected["observed_at"].isoformat(),
                    "converged_at": observed_at.isoformat(),
                    "selection_to_convergence_seconds": (
                        observed_at - selected["observed_at"]
                    ).total_seconds(),
                    "announcement_to_convergence_seconds": (
                        (observed_at - announcement["observed_at"]).total_seconds()
                        if announcement is not None
                        else None
                    ),
                }
            )
            selected = None
            continue

        if FORK_LOCAL.search(line):
            local_resolutions += 1

    unmatched_starts = sum(len(value) for value in open_starts.values())
    return intervals, {
        "peer_adoptions": convergence,
        "local_tip_resolutions": local_resolutions,
        "unmatched_evaluation_starts": unmatched_starts,
    }


def measure_generations(
    intervals: list[EvaluationInterval],
    workers: list[str],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[EvaluationInterval]] = defaultdict(list)
    for interval in intervals:
        grouped[(interval.campaign, interval.generation)].append(interval)

    measurements: list[dict[str, object]] = []
    for (campaign, generation), rows in sorted(grouped.items()):
        population_values = {row.population for row in rows}
        expected_population = max(population_values)
        by_candidate: dict[int, list[EvaluationInterval]] = defaultdict(list)
        for row in rows:
            by_candidate[row.candidate].append(row)
        duplicates = sorted(
            candidate for candidate, values in by_candidate.items() if len(values) > 1
        )
        candidate_ids = set(by_candidate)
        zero_based = set(range(expected_population))
        one_based = set(range(1, expected_population + 1))
        if candidate_ids <= zero_based:
            expected_ids = zero_based
            index_base = 0
        elif candidate_ids <= one_based:
            expected_ids = one_based
            index_base = 1
        else:
            expected_ids = zero_based
            index_base = None
        missing = sorted(expected_ids - candidate_ids)
        complete = (
            len(population_values) == 1
            and not duplicates
            and not missing
            and len(rows) == expected_population
        )
        item: dict[str, object] = {
            "campaign": campaign,
            "generation": generation,
            "expected_population": expected_population,
            "candidate_index_base": index_base,
            "completed_candidates": len(by_candidate),
            "complete": complete,
            "missing_candidates": missing,
            "duplicate_candidates": duplicates,
        }
        if complete:
            generation_start = min(row.started_at for row in rows)
            generation_end = max(row.completed_at for row in rows)
            makespan = (generation_end - generation_start).total_seconds()
            available = makespan * len(workers)
            active = sum(row.duration_seconds for row in rows)
            last_finish = {
                worker: max(
                    (
                        row.completed_at
                        for row in rows
                        if row.worker == worker
                    ),
                    default=generation_start,
                )
                for worker in workers
            }
            tail_idle_by_worker = {
                worker: (generation_end - completed_at).total_seconds()
                for worker, completed_at in last_finish.items()
            }
            tail_idle = sum(tail_idle_by_worker.values())
            item.update(
                {
                    "started_at": generation_start.isoformat(),
                    "completed_at": generation_end.isoformat(),
                    "makespan_seconds": makespan,
                    "fleet_available_seconds": available,
                    "candidate_evaluation_seconds": active,
                    "non_evaluation_gap_seconds": max(0.0, available - active),
                    "non_evaluation_gap_fraction": (
                        max(0.0, available - active) / available
                        if available > 0.0
                        else 0.0
                    ),
                    "tail_barrier_idle_seconds": tail_idle,
                    "tail_barrier_idle_fraction": (
                        tail_idle / available if available > 0.0 else 0.0
                    ),
                    "tail_barrier_idle_by_worker_seconds": tail_idle_by_worker,
                    "candidate_counts_by_worker": {
                        worker: sum(row.worker == worker for row in rows)
                        for worker in workers
                    },
                }
            )
        measurements.append(item)
    return measurements


def analyze_logs(
    log_paths: dict[str, Path],
    *,
    campaign_filter: str | None = None,
) -> dict[str, object]:
    intervals: list[EvaluationInterval] = []
    fork_summary: dict[str, object] = {}
    input_logs: dict[str, object] = {}
    for worker, path in log_paths.items():
        payload = path.read_bytes()
        worker_intervals, worker_forks = parse_worker_log(
            worker,
            payload.decode("utf-8", errors="replace").splitlines(),
            campaign_filter=campaign_filter,
        )
        intervals.extend(worker_intervals)
        fork_summary[worker] = worker_forks
        input_logs[worker] = {
            "path": str(path),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    generations = measure_generations(intervals, list(log_paths))
    complete = [item for item in generations if item["complete"]]
    total_available = sum(
        float(item["fleet_available_seconds"]) for item in complete
    )
    total_tail = sum(float(item["tail_barrier_idle_seconds"]) for item in complete)
    total_gap = sum(float(item["non_evaluation_gap_seconds"]) for item in complete)
    adoptions = [
        event
        for worker in fork_summary.values()
        for event in worker["peer_adoptions"]
    ]
    return {
        "schema": "agent_multi.swarm_efficiency_evidence.v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "campaign_filter": campaign_filter,
        "workers": list(log_paths),
        "input_logs": input_logs,
        "summary": {
            "complete_generations": len(complete),
            "observed_generations": len(generations),
            "completed_local_evaluations": len(intervals),
            "tail_barrier_idle_seconds": total_tail,
            "tail_barrier_idle_fraction": (
                total_tail / total_available if total_available > 0.0 else None
            ),
            "non_evaluation_gap_seconds": total_gap,
            "non_evaluation_gap_fraction": (
                total_gap / total_available if total_available > 0.0 else None
            ),
            "peer_tip_adoptions": len(adoptions),
            "median_announcement_to_convergence_seconds": _median(
                [
                    float(event["announcement_to_convergence_seconds"])
                    for event in adoptions
                    if event["announcement_to_convergence_seconds"] is not None
                ]
            ),
        },
        "generations": generations,
        "forks": fork_summary,
        "evaluation_intervals": [
            {
                **asdict(interval),
                "started_at": interval.started_at.isoformat(),
                "completed_at": interval.completed_at.isoformat(),
                "duration_seconds": interval.duration_seconds,
            }
            for interval in intervals
        ],
    }


def capture_clock_samples(
    clock_hosts: dict[str, str],
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    now: Callable[[], datetime] | None = None,
) -> dict[str, dict[str, object]]:
    """Measure each log host's UTC offset against the collector midpoint."""
    clock = now or (lambda: datetime.now(timezone.utc))
    samples: dict[str, dict[str, object]] = {}
    for worker, target in clock_hosts.items():
        command = (
            ["date", "-u", "+%Y-%m-%dT%H:%M:%S.%6NZ"]
            if target == "local"
            else [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=5",
                target,
                "date -u +%Y-%m-%dT%H:%M:%S.%6NZ",
            ]
        )
        started_at = clock().astimezone(timezone.utc)
        try:
            completed = runner(
                command,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            samples[worker] = {
                "target": target,
                "status": "error",
                "error": type(exc).__name__,
                "requested_at": started_at.isoformat(),
            }
            continue
        received_at = clock().astimezone(timezone.utc)
        raw_remote = completed.stdout.strip()
        try:
            remote_at = datetime.fromisoformat(
                raw_remote.replace("Z", "+00:00")
            ).astimezone(timezone.utc)
        except ValueError:
            samples[worker] = {
                "target": target,
                "status": "error",
                "returncode": completed.returncode,
                "error": "invalid_remote_utc",
                "requested_at": started_at.isoformat(),
                "received_at": received_at.isoformat(),
            }
            continue
        if completed.returncode != 0:
            samples[worker] = {
                "target": target,
                "status": "error",
                "returncode": completed.returncode,
                "error": "date_command_failed",
                "requested_at": started_at.isoformat(),
                "received_at": received_at.isoformat(),
            }
            continue
        midpoint = started_at + (received_at - started_at) / 2
        samples[worker] = {
            "target": target,
            "status": "ok",
            "requested_at": started_at.isoformat(),
            "received_at": received_at.isoformat(),
            "remote_utc": remote_at.isoformat(),
            "round_trip_seconds": (
                received_at - started_at
            ).total_seconds(),
            "offset_from_collector_midpoint_seconds": (
                remote_at - midpoint
            ).total_seconds(),
        }
    return samples


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def render_markdown(result: dict[str, object]) -> str:
    summary = result["summary"]
    complete = [
        item for item in result["generations"] if item["complete"]
    ]
    lines = [
        "# Swarm Efficiency Measurement",
        "",
        f"Generated: `{result['generated_at']}`",
        f"Campaign filter: `{result['campaign_filter'] or 'all'}`",
        "",
        "## Result",
        "",
    ]
    if complete:
        lines.extend(
            [
                f"- Complete generations measured: **{summary['complete_generations']}**.",
                "- Tail-barrier idle: "
                f"**{float(summary['tail_barrier_idle_fraction']):.2%}** "
                "of measured fleet wall-clock capacity.",
                "- Total non-evaluation gap: "
                f"**{float(summary['non_evaluation_gap_fraction']):.2%}**. "
                "This includes barrier waits, scheduling, communication, restarts, "
                "and any unlogged work; it is not attributed to one cause.",
            ]
        )
    else:
        lines.append(
            "- No generation had exactly one completed interval for every candidate; "
            "no utilization percentage is asserted."
        )
    lines.extend(
        [
            f"- Peer-tip adoptions observed: **{summary['peer_tip_adoptions']}**.",
            "- Median announcement-to-convergence latency: "
            f"`{summary['median_announcement_to_convergence_seconds']}` seconds.",
            "",
            "## Clock Samples",
            "",
        ]
    )
    clock_samples = result.get("clock_samples", {})
    if clock_samples:
        lines.extend(
            [
                "| Worker | Target | Status | Offset vs collector midpoint | RTT |",
                "| --- | --- | --- | ---: | ---: |",
            ]
        )
        for worker, sample in clock_samples.items():
            offset = sample.get("offset_from_collector_midpoint_seconds")
            rtt = sample.get("round_trip_seconds")
            lines.append(
                f"| {worker} | {sample.get('target')} | {sample.get('status')} | "
                f"{float(offset):.6f}s | {float(rtt):.6f}s |"
                if offset is not None and rtt is not None
                else f"| {worker} | {sample.get('target')} | "
                f"{sample.get('status')} | N/A | N/A |"
            )
    else:
        lines.append(
            "No per-host clock samples were supplied; cross-host timing "
            "assumptions remain unmeasured."
        )
    lines.extend(
        [
            "",
            "## Generation Detail",
            "",
            "| Generation | Candidates | Complete | Tail idle | Non-evaluation gap |",
            "| ---: | ---: | :---: | ---: | ---: |",
        ]
    )
    for item in result["generations"]:
        tail = (
            f"{float(item['tail_barrier_idle_fraction']):.2%}"
            if item["complete"]
            else "N/A"
        )
        gap = (
            f"{float(item['non_evaluation_gap_fraction']):.2%}"
            if item["complete"]
            else "N/A"
        )
        lines.append(
            f"| {item['generation']} | "
            f"{item['completed_candidates']}/{item['expected_population']} | "
            f"{'yes' if item['complete'] else 'no'} | {tail} | {gap} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "The tail metric starts at each worker's last completed candidate and ends "
            "at the generation's final candidate. It directly measures generational "
            "straggler waiting. The broader gap is descriptive only and must not be "
            "called barrier loss without additional instrumentation.",
            "",
            "Input log hashes are embedded in the JSON evidence packet.",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_log(value: str) -> tuple[str, Path]:
    worker, separator, path = value.partition("=")
    if not separator or not worker or not path:
        raise argparse.ArgumentTypeError("--log must be WORKER=/path/to/log")
    return worker, Path(path).expanduser().resolve()


def _parse_clock_host(value: str) -> tuple[str, str]:
    worker, separator, target = value.partition("=")
    if not separator or not worker or not target:
        raise argparse.ArgumentTypeError(
            "--clock-host must be WORKER=local-or-ssh-target"
        )
    return worker, target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log",
        action="append",
        required=True,
        type=_parse_log,
        metavar="WORKER=PATH",
    )
    parser.add_argument("--campaign")
    parser.add_argument(
        "--clock-host",
        action="append",
        default=[],
        type=_parse_clock_host,
        metavar="WORKER=local-or-ssh-target",
        help="capture one UTC clock sample per log worker",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()
    log_paths = dict(args.log)
    if len(log_paths) != len(args.log):
        parser.error("worker names in --log must be unique")
    for path in log_paths.values():
        if not path.is_file():
            parser.error(f"log does not exist: {path}")
    clock_hosts = dict(args.clock_host)
    if len(clock_hosts) != len(args.clock_host):
        parser.error("worker names in --clock-host must be unique")
    if clock_hosts and set(clock_hosts) != set(log_paths):
        parser.error("--clock-host workers must exactly match --log workers")

    result = analyze_logs(log_paths, campaign_filter=args.campaign)
    result["clock_samples"] = capture_clock_samples(clock_hosts)
    json_text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    markdown = render_markdown(result)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json_text, encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown, encoding="utf-8")
    if not args.output_json and not args.output_md:
        print(json_text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
