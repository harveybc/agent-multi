#!/usr/bin/env python3
"""Consolidated machine-readable multi-front status contract.

Implements owner-approved improvements 1 (consolidated status) and 4
(queue-state taxonomy) from the 2026-08-01 acceptance contract. Aggregates
existing tier-0 evidence sources read-only; never invents a value — a source
that cannot be read yields an explicit `unavailable` entry instead.

Every numeric field carries `unit` and `horizon`; every section records its
source, fetch time and freshness. `basis` distinguishes `observed` (read
directly from a source) from `derived` (computed here, formula named).

Finding 204 (order 2026-08-10 §6/WP3): Front 1 describes the work that is
ACTUALLY running. The first-class `l1_factorial` source reads the durable
launcher heartbeats, cell records and training logs of the four assigned
factorial workers (local filesystem + read-only ssh for remote hosts); the
paused DOIN campaign supervisor renders separately as history and can never
replace the active factorial in `f1_optimization` or the executable queue.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import socket
import sqlite3
import subprocess
import time as _time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

SCHEMA = "agent_multi.multifront_status.v2"

QUEUE_STATES = {
    "running",
    "materialized",
    "dependency_blocked",
    "proposed",
    "owner_blocked",
}

# States that can never be simultaneously true for one item, and field
# requirements per state (acceptance contract section 4).
_REQUIRES_HASHES = {"running", "materialized"}


class QueueStateError(ValueError):
    """A queue item violates the canonical taxonomy."""


_SHA256_HEX = 64

# Per-state field contract (finding 036): a field allowed in one state is
# forbidden in every state that does not list it.
_STATE_FIELDS: dict[str, dict[str, set[str]]] = {
    "running": {"required": set(), "forbidden": {"dependency", "owner_blocked_reason"}},
    "materialized": {"required": set(), "forbidden": {"dependency", "owner_blocked_reason"}},
    "dependency_blocked": {"required": {"dependency"}, "forbidden": {"owner_blocked_reason"}},
    "proposed": {"required": set(), "forbidden": {"dependency", "owner_blocked_reason"}},
    "owner_blocked": {"required": {"owner_blocked_reason"}, "forbidden": {"dependency"}},
}


def _valid_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == _SHA256_HEX and all(
        character in "0123456789abcdef" for character in text.lower()
    )


def validate_queue_item(item: Mapping[str, Any]) -> None:
    state = item.get("state")
    if state not in QUEUE_STATES:
        raise QueueStateError(f"unknown queue state: {state!r}")
    states_claimed = item.get("also_states") or []
    if states_claimed:
        raise QueueStateError(
            "a queue item has exactly one canonical state; "
            f"got extra states {states_claimed!r}"
        )
    contract = _STATE_FIELDS[state]
    for field in contract["forbidden"]:
        if item.get(field):
            raise QueueStateError(f"{state} item cannot carry {field}")
    for field in contract["required"]:
        if not item.get(field):
            raise QueueStateError(f"{state} item must carry {field}")
    hashes = item.get("hashes") or {}
    for key, value in hashes.items():
        if key.endswith("_sha256") and value and not _valid_sha256(value):
            raise QueueStateError(f"{key} is not a valid SHA-256 hex digest")
    if state in _REQUIRES_HASHES:
        if not _valid_sha256(hashes.get("config_sha256")) and not _valid_sha256(
            hashes.get("plan_sha256")
        ):
            raise QueueStateError(
                f"{state} item requires a syntactically valid "
                "config_sha256 or plan_sha256"
            )


def validate_queue(items: list[Mapping[str, Any]]) -> None:
    seen: set[str] = set()
    for item in items:
        item_id = str(item.get("id") or "")
        if not item_id:
            raise QueueStateError("queue item requires an id")
        if item_id in seen:
            raise QueueStateError(f"duplicate queue item id: {item_id}")
        seen.add(item_id)
        validate_queue_item(item)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _age_seconds(iso_ts: Optional[str]) -> Optional[float]:
    if not iso_ts:
        return None
    try:
        ts = datetime.fromisoformat(str(iso_ts).replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return round((datetime.now(timezone.utc) - ts).total_seconds(), 1)
    except ValueError:
        return None


def _sha256_file(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _load_json_file(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _get_url(url: str, timeout: float) -> Optional[dict]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read())
    except Exception:
        return None


def _as_dict(value: Any) -> dict:
    """Finding 037: valid JSON of unexpected shape must degrade, not raise.

    Any nested section accessed with `.get()` goes through this, so a truthy
    list/str/number in place of an object reads as an empty section and the
    downstream field becomes explicitly unavailable.
    """
    return value if isinstance(value, dict) else {}


def _direct_count(value: Any) -> Optional[int]:
    """Parse a direct venue count; only non-negative true integers qualify.

    Booleans, strings, floats and negatives carry no documented coercion
    contract and therefore read as unavailable (finding 037), never as a
    number invented by coercion.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value >= 0 else None


def _venue_execution_truth(
    heartbeat_path: Path, olap_path: Path, now: float
) -> dict[str, Any]:
    """Findings 098/102: venue truth derives from the CURRENT execution
    heartbeat plus the accepted lifecycle OLAP — never old read-only
    preflight labels. Cumulative Paper history is preserved as counts."""
    truth: dict[str, Any] = {"mode": "unknown"}
    heartbeat = _load_json_file(heartbeat_path)
    if isinstance(heartbeat, dict) and heartbeat:
        observed = heartbeat.get("observed_at")
        age = None
        try:
            age = round(now - datetime.fromisoformat(
                str(observed)).timestamp(), 1)
        except (TypeError, ValueError):
            pass
        read_only = heartbeat.get("read_only")
        truth.update({
            "mode": ("write_enabled" if read_only is False
                     else "read_only" if read_only is True else "unknown"),
            "environment": heartbeat.get("environment"),
            "account_fingerprint": heartbeat.get("account_fingerprint"),
            "heartbeat_state": heartbeat.get("state"),
            "heartbeat_age_seconds": age,
            "model_id": (heartbeat.get("model_id")
                         or _as_dict(heartbeat.get("inference")).get(
                             "model_id")),
        })
    else:
        truth["mode_reason"] = "heartbeat unreadable"
    try:
        con = sqlite3.connect(f"file:{olap_path}?mode=ro", uri=True)
        try:
            decisions = dict(con.execute(
                "SELECT outcome, COUNT(*) FROM decisions GROUP BY outcome"))
            last = con.execute(
                "SELECT outcome, reason FROM decisions"
                " ORDER BY decided_at DESC LIMIT 1").fetchone()
            truth.update({
                "decisions_cumulative": decisions,
                "last_decision": (
                    {"outcome": last[0], "reason": last[1]}
                    if last else None),
                "lifecycles_cumulative": con.execute(
                    "SELECT COUNT(*) FROM l1_effects").fetchone()[0],
                "open_exposures": con.execute(
                    "SELECT COUNT(*) FROM exposures WHERE state='open'"
                ).fetchone()[0],
                "sessions_cumulative": con.execute(
                    "SELECT COUNT(*) FROM live_model_sessions"
                ).fetchone()[0],
                "halt": (con.execute(
                    "SELECT value FROM service_state WHERE key='halt'"
                ).fetchone() or [None])[0],
            })
        finally:
            con.close()
    except sqlite3.Error:
        truth["olap_reason"] = "execution OLAP unreadable"
    return truth


# ── Front 1 first-class source: L1 matched factorial (finding 204) ──────────

L1_HEARTBEAT_SCHEMA = "agent_multi.l1_launcher_heartbeat.v2"
L1_RECORD_SCHEMA = "agent_multi.l1_factorial_cell_record.v2"

# '[epoch  34/1996] L1 no-activity 0/40 ...' — the launcher log's per-epoch
# line. It carries NO timestamp; epoch timing therefore only exists as
# cross-observation deltas (see _l1_eta_from_samples), never invented.
_L1_EPOCH_RE = re.compile(
    r"\[epoch\s+(\d+)/(\d+)\]\s+L1 no-activity\s+(\d+)/(\d+)")
# '            TRAIN trades=   0 win%= 0.00 ... profit=+0.00% bal=10000.00'
_L1_TRADES_RE = re.compile(
    r"^\s*(TRAIN_TAIL|TRAIN|VAL)\s+trades=\s*(\d+)"
    r"(?:\s+win%=\s*([+\-]?[0-9.]+|[+\-]?nan))?"
    r"(?:.*?profit=([+\-]?[0-9.]+)%)?"
    r"(?:\s+bal=([0-9.]+))?",
    re.MULTILINE,
)


def _float_or_none(text: Any) -> Optional[float]:
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if value == value else None  # NaN reads as unavailable


class DefaultL1Reader:
    """Worker-host fact reader: local filesystem, remote read-only ssh.

    Remote commands run as ``ssh -o BatchMode=yes <host> '<cmd>' </dev/null``
    (read-only: cat/tail/stat/ls/systemctl show). An ssh transport failure
    (exit 255 or a timeout) marks the host unreachable in ``errors`` and
    short-circuits every later call for that host, so one dead host costs
    one connection attempt, not eight. A missing file is simply ``None``.
    Tests never construct this class — they inject a fake reader.
    """

    def __init__(self, local_hostname: Optional[str] = None,
                 connect_timeout: int = 6, command_timeout: int = 25):
        self.local_hostname = local_hostname or socket.gethostname()
        self.connect_timeout = connect_timeout
        self.command_timeout = command_timeout
        self.errors: dict[str, str] = {}

    def _is_local(self, host: str) -> bool:
        return host == self.local_hostname

    def _ssh(self, host: str, command: str) -> Optional[str]:
        if host in self.errors:
            return None
        try:
            proc = subprocess.run(
                ["ssh", "-o", "BatchMode=yes",
                 "-o", f"ConnectTimeout={self.connect_timeout}",
                 host, command],
                stdin=subprocess.DEVNULL, capture_output=True, text=True,
                timeout=self.command_timeout)
        except Exception as exc:  # timeout, missing ssh binary, ...
            self.errors[host] = f"{type(exc).__name__}: {exc}"[:200]
            return None
        if proc.returncode == 255:
            self.errors[host] = (proc.stderr.strip()
                                 or "ssh transport failure (exit 255)")[:200]
            return None
        if proc.returncode != 0:
            return None
        return proc.stdout

    def read_text(self, host: str, path: str) -> Optional[str]:
        if self._is_local(host):
            try:
                return Path(path).expanduser().read_text()
            except OSError:
                return None
        return self._ssh(host, f"cat {path}")

    def read_tail(self, host: str, path: str,
                  max_bytes: int = 131072) -> Optional[str]:
        if self._is_local(host):
            try:
                p = Path(path).expanduser()
                with p.open("rb") as fh:
                    fh.seek(max(0, p.stat().st_size - max_bytes))
                    return fh.read().decode("utf-8", "replace")
            except OSError:
                return None
        return self._ssh(host, f"tail -c {int(max_bytes)} {path}")

    def mtime(self, host: str, path: str) -> Optional[float]:
        if self._is_local(host):
            try:
                return Path(path).expanduser().stat().st_mtime
            except OSError:
                return None
        out = self._ssh(host, f"stat -c %Y {path}")
        try:
            return float(out.strip()) if out else None
        except ValueError:
            return None

    def nrestarts(self, host: str, unit: str) -> Optional[int]:
        if self._is_local(host):
            try:
                out = subprocess.run(
                    ["systemctl", "--user", "show", unit,
                     "-p", "NRestarts", "--value"],
                    capture_output=True, text=True, timeout=10).stdout
            except Exception:
                return None
        else:
            out = self._ssh(
                host, f"systemctl --user show {unit} -p NRestarts --value")
        try:
            return int(out.strip()) if out and out.strip() else None
        except ValueError:
            return None

    def latest_heartbeat(self, host: str,
                         output_root: str) -> Optional[str]:
        if self._is_local(host):
            root = Path(output_root).expanduser()
            try:
                candidates = sorted(
                    root.glob("*/seed*/launcher_heartbeat.json"),
                    key=lambda p: p.stat().st_mtime, reverse=True)
            except OSError:
                return None
            return str(candidates[0]) if candidates else None
        out = self._ssh(
            host,
            f"ls -t {output_root}/*/seed*/launcher_heartbeat.json"
            " 2>/dev/null | head -1")
        if not out:
            return None
        return out.strip() or None


def _l1_parse_log_tail(text: str) -> dict[str, Any]:
    """Parse the LAST epoch line and last per-split trade lines."""
    parsed: dict[str, Any] = {}
    epochs = list(_L1_EPOCH_RE.finditer(text))
    if epochs:
        m = epochs[-1]
        parsed["epoch"] = int(m.group(1))
        parsed["epoch_max"] = int(m.group(2))
        parsed["no_activity"] = int(m.group(3))
        parsed["no_activity_of"] = int(m.group(4))
    trades: dict[str, Any] = {}
    for m in _L1_TRADES_RE.finditer(text):
        trades[m.group(1)] = {
            "trades": int(m.group(2)),
            "win_pct": _float_or_none(m.group(3)),
            "profit_pct": _float_or_none(m.group(4)),
            "balance": _float_or_none(m.group(5)),
        }
    if trades:
        parsed["trades"] = trades
    return parsed


def _l1_iso(value: Any) -> Optional[datetime]:
    try:
        ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def _l1_record_eta_sample(state_dir: Path, identity: str, seed: int,
                          cell: Optional[str], epoch: Optional[int],
                          now: datetime) -> list[dict[str, Any]]:
    """Append an (observation-time, epoch) sample when the epoch advanced;
    return all samples. Timing samples live in the STATUS TOOL's own state
    dir — never inside the run root, which stays read-only."""
    path = state_dir / "eta_samples" / f"{identity}.seed{seed}.jsonl"
    samples: list[dict[str, Any]] = []
    try:
        for line in path.read_text().splitlines()[-500:]:
            try:
                samples.append(json.loads(line))
            except ValueError:
                continue
    except OSError:
        pass
    if epoch is None:
        return samples
    last = samples[-1] if samples else None
    if not last or last.get("epoch") != epoch or last.get("cell") != cell:
        entry = {"observed_utc": now.isoformat(), "cell": cell,
                 "epoch": epoch}
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as fh:
                fh.write(json.dumps(entry, sort_keys=True) + "\n")
            samples.append(entry)
        except OSError:
            pass
    return samples


def _l1_eta_from_samples(samples: list[dict[str, Any]], cell: Optional[str],
                         epoch: Optional[int],
                         epoch_max: Optional[int]) -> dict[str, Any]:
    """Epoch-rate ETA from OBSERVED cross-observation deltas only.

    Epoch log lines carry no timestamps, so each sample's timestamp is the
    collector's observation time; a pair's error is bounded by the spacing
    between observations, which the uncertainty note declares.
    """
    if epoch is None or epoch_max is None:
        return {"value": "unavailable",
                "missing": "no epoch line parsed from the worker log"}
    relevant = [s for s in samples if s.get("cell") == cell
                and isinstance(s.get("epoch"), int)]
    rates: list[float] = []
    for before, after in zip(relevant, relevant[1:]):
        t0, t1 = _l1_iso(before.get("observed_utc")), _l1_iso(
            after.get("observed_utc"))
        depoch = after["epoch"] - before["epoch"]
        if t0 and t1 and depoch > 0:
            dt = (t1 - t0).total_seconds()
            if dt > 0:
                rates.append(dt / depoch)
    if len(rates) < 2:
        return {
            "value": "unavailable",
            "missing": (
                "fewer than 2 observed epoch-duration deltas for this cell "
                f"({len(rates)} so far; epoch log lines carry no timestamps, "
                "so durations only exist across repeated status "
                "observations)"),
        }
    rates.sort()
    n = len(rates)
    median = (rates[n // 2] if n % 2 else
              (rates[n // 2 - 1] + rates[n // 2]) / 2)
    remaining = max(0, epoch_max - epoch)
    return {
        "basis": "derived",
        "eta_seconds": {"value": round(median * remaining, 1),
                        "low": round(rates[0] * remaining, 1),
                        "high": round(rates[-1] * remaining, 1),
                        "unit": "seconds"},
        "seconds_per_epoch": {"median": round(median, 1),
                              "min": round(rates[0], 1),
                              "max": round(rates[-1], 1)},
        "remaining_epochs": remaining,
        "sample_size": {"value": n, "unit": "observation_pairs"},
        "formula": ("median(delta_seconds/delta_epochs over adjacent status "
                    "observations of the same cell) * remaining_epochs; "
                    "remaining_epochs = epoch_max - last_epoch"),
        "horizon": ("current cell upper bound; L1 patience or activity "
                    "stopping may end the cell earlier"),
        "uncertainty": ("range [low, high] from min/max observed pair "
                        "rates; sample timestamps are collector observation "
                        "times, so each pair carries error up to the "
                        "observation spacing"),
    }


def _l1_cell_eta(durations: list[float], cells_remaining: int) -> dict[str, Any]:
    if len(durations) < 2:
        return {
            "value": "unavailable",
            "missing": (
                "fewer than 2 completed cell records under the active "
                f"identity ({len(durations)} so far); cell-level ETA derives "
                "only from observed started_utc→finished_utc durations"),
        }
    mean = sum(durations) / len(durations)
    return {
        "basis": "derived",
        "eta_seconds": {"value": round(mean * cells_remaining, 1),
                        "low": round(min(durations) * cells_remaining, 1),
                        "high": round(max(durations) * cells_remaining, 1),
                        "unit": "seconds"},
        "mean_cell_seconds": round(mean, 1),
        "cells_remaining": cells_remaining,
        "sample_size": {"value": len(durations), "unit": "completed_cells"},
        "formula": ("mean(finished_utc - started_utc of completed cell "
                    "records under the active identity) * remaining cells"),
        "horizon": "remaining cells across all workers",
        "uncertainty": "range [low, high] from min/max observed cell durations",
    }


def collect_l1_factorial(
    *,
    contract_path: Path,
    reader: Any,
    identity: Optional[str] = None,
    state_dir: Optional[Path] = None,
    local_hostname: Optional[str] = None,
    stale_after_seconds: float = 900.0,
    alert_emitter: Optional[Callable[..., bool]] = None,
    now_fn: Optional[Callable[[], datetime]] = None,
) -> tuple[dict[str, Any], list[dict[str, str]], Optional[dict[str, Any]]]:
    """First-class Front-1 source: the active L1 matched factorial.

    Returns (front block, unavailable entries, executable-queue entry).
    Strictly read-only towards the run: heartbeats, cell records, logs and
    systemd restart counters are read; nothing under the output root is
    ever written. The only writes are the tool's own ETA samples and
    alert-dedup markers under ``state_dir``, plus at most ONE bounded
    incident-ledger observation per (identity, seed, cell) when a worker
    reaches terminal inactivity at the declared patience boundary.
    """
    now = (now_fn or (lambda: datetime.now(timezone.utc)))()
    unavailable: list[dict[str, str]] = []

    try:
        contract = json.loads(contract_path.read_text())
    except (OSError, ValueError) as exc:
        reason = f"l1 factorial contract unreadable: {type(exc).__name__}"
        unavailable.append(
            {"field": "f1_optimization.active_l1_factorial", "reason": reason})
        return {"source": "l1_factorial", "state": "unavailable",
                "reason": reason}, unavailable, None
    contract_sha = _sha256_file(contract_path)
    assignments = _as_dict(contract.get("assignments"))
    cells = list(_as_dict(contract.get("cells")))
    seeds = [s for s in (contract.get("seeds") or []) if str(s) in assignments]
    output_root = str(contract.get("output_root") or "").rstrip("/")
    stopping = _as_dict(contract.get("stopping"))
    declared_patience = stopping.get("l1_activity_patience")
    local = local_hostname or getattr(reader, "local_hostname", None) \
        or socket.gethostname()
    if not (seeds and cells and output_root):
        reason = "contract lacks seeds/cells/output_root"
        unavailable.append(
            {"field": "f1_optimization.active_l1_factorial", "reason": reason})
        return {"source": "l1_factorial", "state": "unavailable",
                "reason": reason,
                "contract_path": str(contract_path)}, unavailable, None

    hosts: list[str] = []
    for seed in seeds:
        host = _as_dict(assignments.get(str(seed))).get("hostname")
        if host and host not in hosts:
            hosts.append(host)
    hosts.sort(key=lambda h: h != local)  # local first: no ssh to discover

    identity_basis = "explicit_parameter"
    if not identity:
        for host in hosts:
            latest = reader.latest_heartbeat(host, output_root)
            if latest:
                parts = str(latest).rstrip("/").split("/")
                if len(parts) >= 3:
                    identity = parts[-3]
                    identity_basis = (
                        f"discovered_latest_heartbeat_mtime({host})")
                    break
    if not identity:
        reason = ("no experiment identity: none supplied and no launcher "
                  "heartbeat discoverable on any assigned host "
                  f"(reader errors: {getattr(reader, 'errors', {}) or 'none'})")
        unavailable.append(
            {"field": "f1_optimization.active_l1_factorial", "reason": reason})
        return {"source": "l1_factorial", "state": "unavailable",
                "reason": reason,
                "contract_path": str(contract_path)}, unavailable, None

    workers: dict[str, Any] = {}
    cell_durations: list[float] = []
    records_landed_total = 0
    running_fresh = 0
    any_fact = False
    zero_trade_alerts: list[dict[str, Any]] = []

    for seed in seeds:
        assignment = _as_dict(assignments.get(str(seed)))
        host = assignment.get("hostname") or "unknown"
        unit = f"l1-factorial@{seed}.service"
        seed_dir = f"{output_root}/{identity}/seed{seed}"
        entry: dict[str, Any] = {
            "identity": identity, "seed": seed, "host": host,
            "unit": unit, "basis": "observed",
            "assigned_gpu_uuid": assignment.get("gpu_uuid"),
        }
        heartbeat_raw = reader.read_text(
            host, f"{seed_dir}/launcher_heartbeat.json")
        heartbeat: Optional[dict] = None
        if heartbeat_raw:
            try:
                loaded = json.loads(heartbeat_raw)
                heartbeat = loaded if isinstance(loaded, dict) else None
            except ValueError:
                heartbeat = None
        host_error = _as_dict(getattr(reader, "errors", {})).get(host)
        if heartbeat is None:
            entry["terminal_state"] = "unavailable"
            entry["unavailable_reason"] = (
                f"host unreachable: {host_error}" if host_error
                else f"launcher heartbeat missing or unparsable at "
                     f"{seed_dir}/launcher_heartbeat.json")
            unavailable.append({
                "field": f"f1_optimization.active_l1_factorial."
                         f"workers.{seed}",
                "reason": entry["unavailable_reason"]})
            workers[str(seed)] = entry
            continue
        any_fact = True
        hb_updated = _l1_iso(heartbeat.get("updated_utc"))
        hb_age = (round((now - hb_updated).total_seconds(), 1)
                  if hb_updated else None)
        cell = heartbeat.get("cell")
        cell_spec = _as_dict(_as_dict(contract.get("cells")).get(cell))
        entry.update({
            "heartbeat_schema": heartbeat.get("schema"),
            "terminal_state": heartbeat.get("terminal_state") or "unknown",
            "error": heartbeat.get("error"),
            "cell": cell,
            "cell_factors": {
                "phase1_mode": cell_spec.get("phase1_mode"),
                "phase2_lr_multiplier": cell_spec.get("phase2_lr_multiplier"),
            } if cell_spec else None,
            "attempt": heartbeat.get("attempt"),
            "pid": heartbeat.get("pid"),
            "pid_start_identity": heartbeat.get("pid_start_identity"),
            "cuda_visible_devices": heartbeat.get("cuda_visible_devices"),
            "observed_gpu_uuids": heartbeat.get("observed_gpu_uuids"),
            "heartbeat_assigned_gpu_uuid": heartbeat.get("assigned_gpu_uuid"),
            "progress": heartbeat.get("progress"),
            "heartbeat_updated_utc": heartbeat.get("updated_utc"),
            "heartbeat_age_seconds": hb_age,
        })

        # Training log: '<root>/logs/seed<seed>.log' is the deployed name;
        # the '.launcher.log' variant is tolerated for forward-compat.
        parsed: dict[str, Any] = {}
        log_path = None
        for candidate in (f"{output_root}/logs/seed{seed}.log",
                          f"{output_root}/logs/seed{seed}.launcher.log"):
            tail = reader.read_tail(host, candidate)
            if tail:
                log_path = candidate
                parsed = _l1_parse_log_tail(tail)
                break
        entry["log_path"] = log_path
        if log_path:
            log_mtime = reader.mtime(host, log_path)
            if log_mtime:
                entry["log_mtime_utc"] = datetime.fromtimestamp(
                    log_mtime, timezone.utc).isoformat()
                entry["log_age_seconds"] = round(
                    now.timestamp() - log_mtime, 1)
        epoch = parsed.get("epoch")
        epoch_max = parsed.get("epoch_max")
        entry["epoch"] = ({"value": epoch, "of": epoch_max,
                           "unit": "epochs", "horizon": "cell"}
                          if epoch is not None else None)
        entry["activity_patience"] = (
            {"value": parsed.get("no_activity"),
             "of": parsed.get("no_activity_of"),
             "declared_patience": declared_patience,
             "unit": "activity_ineligible_epochs", "horizon": "cell"}
            if parsed.get("no_activity") is not None else None)
        entry["trades"] = parsed.get("trades")
        progress_times = [t for t in (hb_updated, _l1_iso(
            entry.get("log_mtime_utc"))) if t]
        entry["last_progress_utc"] = (
            max(progress_times).isoformat() if progress_times else None)

        entry["restart_count"] = {
            "value": reader.nrestarts(host, unit),
            "unit": "systemd_restarts",
            "source": f"systemctl --user show {unit} -p NRestarts",
        }

        landed: dict[str, Any] = {}
        record_inactivity_cell: Optional[str] = None
        for cell_name in cells:
            raw = reader.read_text(
                host, f"{seed_dir}/{cell_name}/l1_cell_record.json")
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except ValueError:
                continue
            if not isinstance(record, dict):
                continue
            started = _l1_iso(record.get("started_utc"))
            finished = _l1_iso(record.get("finished_utc"))
            duration = (round((finished - started).total_seconds(), 1)
                        if started and finished else None)
            if duration is not None:
                cell_durations.append(duration)
            landed[cell_name] = {
                "schema": record.get("schema"),
                "stop_reason": record.get("stop_reason"),
                "termination_cause": record.get("termination_cause"),
                "history_len": record.get("history_len"),
                "phase1_mode": record.get("phase1_mode"),
                "phase2_lr_multiplier": record.get("phase2_lr_multiplier"),
                "decision_eligible": record.get("decision_eligible"),
                "finished_utc": record.get("finished_utc"),
                "duration_seconds": duration,
            }
            if (record.get("activity_stopped_without_eligible_checkpoint")
                    or "activity" in str(record.get("stop_reason") or "")):
                record_inactivity_cell = cell_name
        entry["landed_cells"] = landed or None
        entry["records_landed"] = {"value": len(landed), "of": len(cells),
                                   "unit": "cell_records",
                                   "horizon": "seed"}
        records_landed_total += len(landed)

        if (entry["terminal_state"] == "RUNNING" and hb_age is not None
                and hb_age <= stale_after_seconds):
            running_fresh += 1

        samples: list[dict[str, Any]] = []
        if state_dir is not None:
            samples = _l1_record_eta_sample(
                state_dir, identity, seed, cell, epoch, now)
        entry["eta"] = _l1_eta_from_samples(samples, cell, epoch, epoch_max)
        if state_dir is None and entry["eta"].get("value") == "unavailable":
            entry["eta"]["missing"] += "; no state dir for timing samples"

        # Zero-trade monitoring (WP3.6): ONE bounded alert exactly at the
        # declared patience boundary — terminal inactivity — deduplicated
        # by an emitted-marker keyed (identity, seed, cell). The run itself
        # is NEVER mutated.
        streak = parsed.get("no_activity")
        threshold = (declared_patience
                     if isinstance(declared_patience, int)
                     else parsed.get("no_activity_of"))
        boundary_cell = None
        condition = None
        if (streak is not None and isinstance(threshold, int)
                and threshold > 0 and streak >= threshold):
            boundary_cell = cell or "unknown-cell"
            condition = (f"parsed no-activity {streak}/{threshold} reached "
                         "the declared patience boundary")
        elif record_inactivity_cell:
            boundary_cell = record_inactivity_cell
            condition = ("landed cell record reports terminal activity "
                         "stop")
        if boundary_cell:
            alert = {"seed": seed, "cell": boundary_cell,
                     "condition": condition, "emitted": False,
                     "deduped": False}
            marker = None
            if state_dir is not None:
                marker = (state_dir / "alerts" /
                          f"l1_zero_trade.{identity}.seed{seed}."
                          f"{boundary_cell}.json")
            if marker is not None and marker.exists():
                alert["deduped"] = True
            elif alert_emitter is None:
                alert["skipped"] = "no alert emitter configured"
            elif marker is None:
                alert["skipped"] = ("no state dir for dedup markers; "
                                    "refusing to emit unbounded alerts")
            else:
                ok = bool(alert_emitter(
                    source="multifront_status",
                    event_code=(f"l1_zero_trade_terminal.seed{seed}."
                                f"{boundary_cell}"),
                    severity="P2",
                    summary=(f"L1 factorial {identity} seed {seed} cell "
                             f"{boundary_cell}: terminal inactivity — "
                             f"{condition}; run not mutated"),
                    front="front1",
                    machine=host,
                    affected_object=f"{identity}/seed{seed}/{boundary_cell}",
                    payload={"identity": identity, "seed": seed,
                             "cell": boundary_cell,
                             "no_activity": streak,
                             "declared_patience": threshold,
                             "trades": parsed.get("trades"),
                             "epoch": epoch, "epoch_max": epoch_max},
                ))
                alert["emitted"] = ok
                if ok:
                    try:
                        marker.parent.mkdir(parents=True, exist_ok=True)
                        marker.write_text(json.dumps(
                            {"emitted_utc": now.isoformat(),
                             "condition": condition}, sort_keys=True) + "\n")
                    except OSError:
                        pass
            zero_trade_alerts.append(alert)
        workers[str(seed)] = entry

    if running_fresh:
        state = "active"
        state_basis = (f"{running_fresh} worker(s) RUNNING with launcher "
                       f"heartbeat age <= {stale_after_seconds:.0f}s")
    elif any_fact:
        state = "inactive_or_unknown"
        state_basis = ("no worker is RUNNING with a fresh heartbeat; "
                       "per-worker terminal states carry the facts")
    else:
        state = "unavailable"
        state_basis = "no worker fact readable on any assigned host"
        unavailable.append({
            "field": "f1_optimization.active_l1_factorial",
            "reason": state_basis})

    total_cells = len(cells) * len(seeds)
    block: dict[str, Any] = {
        "source": "l1_factorial",
        "basis": "observed",
        "state": state,
        "state_basis": state_basis,
        "experiment": contract.get("experiment"),
        "asset": contract.get("asset"),
        "identity": identity,
        "identity_basis": identity_basis,
        "contract_path": str(contract_path),
        "contract_sha256": contract_sha,
        "output_root": output_root,
        "declared_stopping": {
            "l1_activity_patience": declared_patience,
            "l1_patience": stopping.get("l1_patience"),
            "max_epochs": stopping.get("max_epochs"),
            "unit": "epochs", "horizon": "cell",
        },
        "workers": workers,
        "workers_running_fresh": {"value": running_fresh, "of": len(seeds),
                                  "unit": "workers", "horizon": "instant"},
        "records_landed": {"value": records_landed_total, "of": total_cells,
                           "unit": "cell_records", "horizon": "experiment"},
        "cells_eta": _l1_cell_eta(
            cell_durations, total_cells - records_landed_total),
    }
    if zero_trade_alerts:
        block["zero_trade_alerts"] = zero_trade_alerts

    queue_entry = None
    if state == "active":
        queue_entry = {
            "id": f"l1-matched-factorial-{identity}",
            "front": "f1",
            "state": "running",
            "hashes": {"config_sha256": contract_sha},
        }
    return block, unavailable, queue_entry


def collect(
    *,
    snapshot_path: Path,
    watchdog_path: Path,
    social_db_path: Path,
    supervisor_url: str,
    l0_heartbeat_path: Path = Path.home()
    / ".local/state/lts/demo-execution-l0/heartbeat.json",
    l0_db_path: Path = Path.home() / ".local/state/lts/demo-execution-l0.sqlite",
    execution_state_dir: Path = Path.home() / ".local/state/lts",
    timeout: float = 6.0,
    l1_contract_path: Optional[Path] = None,
    l1_identity: Optional[str] = None,
    l1_reader: Optional[Any] = None,
    l1_state_dir: Optional[Path] = None,
    l1_local_hostname: Optional[str] = None,
    l1_stale_after_seconds: float = 900.0,
    l1_alert_emitter: Optional[Callable[..., bool]] = None,
    l1_now_fn: Optional[Callable[[], datetime]] = None,
) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    unavailable: list[dict[str, str]] = []
    fronts: dict[str, Any] = {}

    def register(name: str, locator: str, payload_ts: Optional[str]) -> None:
        sources.append(
            {
                "name": name,
                "locator": locator,
                "fetched_at": _now(),
                "payload_generated_at": payload_ts,
                "freshness_seconds": _age_seconds(payload_ts),
            }
        )

    # ── Front 1: the ACTIVE work — L1 matched factorial (finding 204) ──
    f1: dict[str, Any] = {"basis": "observed"}
    fronts["f1_optimization"] = f1
    l1_queue_entry: Optional[dict[str, Any]] = None
    if l1_contract_path is not None:
        l1_active_reader = l1_reader or DefaultL1Reader(
            local_hostname=l1_local_hostname)
        l1_block, l1_unavailable, l1_queue_entry = collect_l1_factorial(
            contract_path=l1_contract_path,
            reader=l1_active_reader,
            identity=l1_identity,
            state_dir=l1_state_dir,
            local_hostname=l1_local_hostname,
            stale_after_seconds=l1_stale_after_seconds,
            alert_emitter=l1_alert_emitter,
            now_fn=l1_now_fn,
        )
        f1["active_l1_factorial"] = l1_block
        unavailable.extend(l1_unavailable)
        if l1_block.get("state") != "unavailable":
            register("l1_factorial", str(l1_contract_path), None)
    else:
        f1["active_l1_factorial"] = {
            "source": "l1_factorial",
            "state": "unavailable",
            "reason": "l1 factorial source not configured (no contract path)",
        }
        unavailable.append(
            {"field": "f1_optimization.active_l1_factorial",
             "reason": "l1 factorial source not configured"})

    # ── Front 1 history: paused DOIN campaign (supervisor API, observed) ──
    # The paused campaign renders as HISTORY only; it can never replace the
    # active factorial above (finding 204).
    status = _get_url(f"{supervisor_url}/api/status", timeout)
    network = _get_url(f"{supervisor_url}/api/network", timeout)
    if not isinstance(status, dict):
        status = None  # truthy wrong-type payload degrades to unavailable
    workers = _as_dict(status.get("workers")) if status else {}
    worker = next(iter(workers.values()), None) if workers else None
    if status and isinstance(worker, dict):
        population = _as_dict(worker.get("shared_population"))
        candidate = _as_dict(worker.get("candidate"))
        eta = _as_dict(worker.get("candidate_eta"))
        register("supervisor_status", f"{supervisor_url}/api/status", status.get("updated_at"))
        f1["doin_campaign_history"] = {
            "basis": "observed",
            "note": ("paused DOIN campaign shown as HISTORY only; never the "
                     "active Front-1 work (finding 204)"),
            "plan_id": status.get("plan_id"),
            "plan_sha256": status.get("plan_hash"),
            "job_id": status.get("job_id"),
            "phase": status.get("phase"),
            "stage": {"value": candidate.get("stage"), "of": candidate.get("total_stages"), "name": candidate.get("stage_name"), "unit": "ordinal", "horizon": "campaign"},
            "generation": {"value": population.get("generation"), "unit": "count", "horizon": "job"},
            "generation_evaluated": {"value": population.get("evaluated"), "of": population.get("pop_size"), "unit": "candidates", "horizon": "generation"},
            "best_fitness": {"value": worker.get("best_performance"), "unit": "dimensionless_full_period_proxy", "horizon": "job_0", "note": "owner-ratified Alternative A: initialization evidence only; job 1 selects with robust_weekly_rap_fitness (fraction/week)"},
            "candidates_per_hour_recent": {"value": eta.get("candidates_per_hour"), "unit": "candidates/hour", "horizon": "recent_window", "basis": "derived", "formula": "median of matched start/result log pairs (supervisor)"},
        }
    else:
        unavailable.append({"field": "f1_optimization.doin_campaign_history", "reason": "supervisor status unreachable, empty, or wrong type"})
    if (f1["active_l1_factorial"].get("state") == "unavailable"
            and "doin_campaign_history" not in f1):
        unavailable.append(
            {"field": "f1_optimization",
             "reason": "neither the active L1 factorial nor the paused "
                       "campaign history is readable"})

    if not isinstance(network, dict):
        network = None  # truthy wrong-type payload degrades to unavailable
    if network:
        register("supervisor_network", f"{supervisor_url}/api/network", None)
        anchors = set()
        tips = set()
        for participant in _as_dict(network.get("participants")).values():
            nested = _as_dict(_as_dict(_as_dict(participant).get("status")).get("workers"))
            for w in nested.values():
                w = _as_dict(w)
                anchors.add((w.get("finalized_height"), str(w.get("finalized_hash"))[:12]))
                tips.add(str(w.get("tip_hash"))[:12])
        f1.setdefault("doin_campaign_history", {})["chain_coherence"] = {
            "basis": "observed",
            "source": "supervisor_network",
            "distinct_unfinalized_tips": {"value": len(tips), "unit": "count", "horizon": "instant"},
            "distinct_finalized_anchors": {
                "value": sorted(
                    [list(a) for a in anchors], key=lambda x: (x[0] is None, x)
                ),
                "unit": "(block_height, hash_prefix12) pairs",
                "horizon": "instant",
            },
            "note": "anchor divergence must converge before archive; no mutation",
        }

    # ── Front 2: venues (watchdog packet, observed) ──
    watchdog = _load_json_file(watchdog_path)
    if isinstance(watchdog, dict) and watchdog:
        register("paper_execution_watchdog", str(watchdog_path), watchdog.get("generated_at"))
        alpaca = _as_dict(watchdog.get("alpaca"))
        ibkr = _as_dict(watchdog.get("ibkr"))
        mt5 = _as_dict(watchdog.get("mt5"))
        heartbeat = _as_dict(mt5.get("heartbeat"))
        # Finding 035: order/position counts come ONLY from direct venue
        # payloads. If any venue count is missing, the aggregate is
        # unavailable — never zero-by-absence. Alerts stay a separate field.
        # Finding 037: wrong-type sections and non-numeric counts also
        # degrade to unavailability instead of crashing or coercing.
        venue_orders: dict[str, Any] = {
            "alpaca": _as_dict(alpaca.get("detail")).get("open_orders"),
            "ibkr": _as_dict(ibkr.get("latest_complete")).get("open_orders"),
            "mt5": _as_dict(mt5.get("latest_snapshot")).get("orders_total"),
        }
        venue_positions: dict[str, Any] = {
            "alpaca": _as_dict(alpaca.get("detail")).get("open_positions"),
            "ibkr": _as_dict(ibkr.get("latest_complete")).get("open_positions"),
            "mt5": _as_dict(mt5.get("latest_snapshot")).get("positions_total"),
        }

        def _aggregate(counts: Mapping[str, Any], label: str) -> dict[str, Any]:
            parsed = {k: _direct_count(v) for k, v in counts.items()}
            missing = sorted(k for k, v in counts.items() if v is None)
            invalid = sorted(
                k for k, v in counts.items() if v is not None and parsed[k] is None
            )
            if missing or invalid:
                reasons = []
                if missing:
                    reasons.append(f"missing direct counts from: {', '.join(missing)}")
                if invalid:
                    reasons.append(
                        f"non-numeric direct counts from: {', '.join(invalid)}"
                    )
                unavailable.append(
                    {
                        "field": f"f2_business_reality.{label}.aggregate",
                        "reason": "; ".join(reasons),
                    }
                )
                total: Optional[int] = None
            else:
                total = sum(v for v in parsed.values() if v is not None)
            return {
                "per_venue": parsed,
                "aggregate": total,
                "unit": label,
                "horizon": "instant",
                "basis": "observed",
                "source": "paper_execution_watchdog",
            }

        alpaca_detail = _as_dict(alpaca.get("detail"))
        ibkr_latest = _as_dict(ibkr.get("latest_complete"))
        mt5_snapshot = _as_dict(mt5.get("latest_snapshot"))
        fronts["f2_business_reality"] = {
            "basis": "observed",
            "active_events": watchdog.get("active_event_keys"),
            "alpaca_sessions": {"value": alpaca.get("complete_sessions"), "unit": "sessions", "horizon": "cumulative", "note": "cumulative, not continuous-window"},
            "ibkr_sessions": {"value": ibkr.get("complete_sessions"), "unit": "sessions", "horizon": "cumulative"},
            "mt5_heartbeat_age": {"value": heartbeat.get("age_seconds"), "unit": "seconds", "horizon": "instant"},
            "mt5_read_only": mt5.get("read_only"),
            "open_orders": _aggregate(venue_orders, "orders"),
            "open_positions": _aggregate(venue_positions, "positions"),
            # Findings 098/102: per-account truth derives from current
            # execution heartbeats plus accepted lifecycle OLAP, never old
            # read-only preflight labels. Identity stays fingerprint-only
            # and balances are excluded (doc 09 §5); the old preflight
            # inspectors remain visible as observer_* context only.
            "accounts": {
                "alpaca_paper": {
                    **_venue_execution_truth(
                        execution_state_dir
                        / "alpaca-model-runner-heartbeat.json",
                        execution_state_dir
                        / "alpaca-model-execution.sqlite",
                        _time.time()),
                    "observer_status": alpaca_detail.get("account_status"),
                    "observer_quotes_received": {"value": alpaca_detail.get("quotes_received"), "unit": "quotes", "horizon": "cumulative"},
                },
                "ibkr_paper": {
                    **_venue_execution_truth(
                        execution_state_dir
                        / "ibkr-model-runner-heartbeat.json",
                        execution_state_dir
                        / "ibkr-model-execution.sqlite",
                        _time.time()),
                    "observer_last_session_id": ibkr_latest.get("session_id"),
                    "observer_last_reconciliation_at": ibkr_latest.get("reconciliation_observed_at"),
                },
                "oanda_mt5_demo": {
                    "environment": heartbeat.get("environment"),
                    "connected": heartbeat.get("connected"),
                    "terminal_build": heartbeat.get("terminal_build"),
                    "trade_allowed_by_terminal": heartbeat.get("trade_allowed"),
                    "mode": ("write_enabled"
                             if heartbeat.get("read_only") is False
                             or mt5.get("read_only") is False
                             else "read_only"
                             if heartbeat.get("read_only") is True
                             else "unknown"),
                    "execution_enabled": (heartbeat.get("execution_enabled")
                                          if heartbeat.get("execution_enabled")
                                          is not None
                                          else mt5.get("execution_enabled")),
                    "symbols_total": mt5_snapshot.get("symbols_total"),
                    "heartbeats": {"value": _as_dict(mt5.get("counts")).get("heartbeats"), "unit": "heartbeats", "horizon": "cumulative"},
                },
                "note": "balances excluded by redaction policy; mode derived"
                        " from current execution heartbeats and lifecycle"
                        " OLAP (finding 098)",
            },
        }
    else:
        unavailable.append({"field": "f2_business_reality", "reason": "watchdog packet unreadable or wrong type"})

    # ── Front 2b: L0 demo-execution runner (heartbeat + ledger, observed) ──
    l0_heartbeat = _load_json_file(l0_heartbeat_path)
    if isinstance(l0_heartbeat, dict) and l0_heartbeat:
        register("l0_demo_execution", str(l0_heartbeat_path), l0_heartbeat.get("at"))
        l0_counts: dict[str, Any] = {}
        try:
            l0_con = sqlite3.connect(f"file:{l0_db_path}?mode=ro", uri=True)
            l0_counts = {
                "decisions": l0_con.execute(
                    "SELECT COUNT(*) FROM decisions"
                ).fetchone()[0],
                "would_be_orders": l0_con.execute(
                    "SELECT COUNT(*) FROM decisions WHERE outcome LIKE 'would_be%'"
                ).fetchone()[0],
                "lifecycle_events": l0_con.execute(
                    "SELECT COUNT(*) FROM lifecycle_events"
                ).fetchone()[0],
            }
            l0_con.close()
        except sqlite3.Error:
            unavailable.append(
                {"field": "f2_business_reality.l0_demo_execution.ledger",
                 "reason": "L0 ledger unreadable"}
            )
        fronts.setdefault("f2_business_reality", {})["l0_demo_execution"] = {
            "basis": "observed",
            "source": "l0_demo_execution",
            "heartbeat_age": {"value": _age_seconds(l0_heartbeat.get("at")),
                              "unit": "seconds", "horizon": "instant"},
            "last_outcome": l0_heartbeat.get("outcome"),
            "halt_state": l0_heartbeat.get("halt_state"),
            "capability_evidence": l0_heartbeat.get("capability_evidence"),
            "network_submissions": {
                "value": l0_heartbeat.get("network_submissions_session"),
                "unit": "submissions", "horizon": "runner_session",
                "note": "structurally zero: the sink has no network path",
            },
            "ledger": {"value": l0_counts or None, "unit": "rows",
                       "horizon": "cumulative"},
        }
    else:
        unavailable.append(
            {"field": "f2_business_reality.l0_demo_execution",
             "reason": "L0 runner heartbeat missing or wrong type"}
        )

    # ── Front 3: social (OLAP counts, observed) ──
    try:
        con = sqlite3.connect(f"file:{social_db_path}?mode=ro", uri=True)
        posts = con.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
        runs = con.execute("SELECT COUNT(*) FROM collection_runs").fetchone()[0]
        drafts = con.execute("SELECT COUNT(*) FROM drafts").fetchone()[0]
        review_states = dict(
            con.execute(
                "SELECT review_state,COUNT(*) FROM posts GROUP BY review_state"
            ).fetchall()
        )
        tables = {
            row[0]
            for row in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        enrichment = {
            "available": "post_enrichments" in tables,
            "enriched_total": 0,
            "eligible_backlog_remaining": None,
            "actions": {},
            "runs_by_status": {},
        }
        if enrichment["available"]:
            enrichment["enriched_total"] = con.execute(
                "SELECT COUNT(*) FROM post_enrichments"
            ).fetchone()[0]
            enrichment["eligible_backlog_remaining"] = con.execute(
                """
                SELECT COUNT(*) FROM posts p
                LEFT JOIN post_enrichments e USING(external_id)
                WHERE e.external_id IS NULL AND p.injection_flags_json='[]'
                  AND p.relevance_score>=0.25
                """
            ).fetchone()[0]
            enrichment["actions"] = dict(
                con.execute(
                    """
                    SELECT recommended_action,COUNT(*) FROM post_enrichments
                    GROUP BY recommended_action
                    """
                ).fetchall()
            )
            enrichment["runs_by_status"] = dict(
                con.execute(
                    """
                    SELECT status,COUNT(*) FROM social_enrichment_runs
                    GROUP BY status
                    """
                ).fetchall()
            )
        con.close()
        register("social_intelligence_olap", str(social_db_path), None)
        fronts["f3_social"] = {
            "basis": "observed",
            "collection_runs": {"value": runs, "unit": "runs", "horizon": "cumulative"},
            "posts_collected": {"value": posts, "unit": "posts", "horizon": "cumulative"},
            "drafts": {"value": drafts, "unit": "drafts", "horizon": "cumulative", "note": "publishing gated on human approval"},
            "review_states": review_states,
            "enrichment": enrichment,
        }
    except sqlite3.Error:
        unavailable.append({"field": "f3_social", "reason": "social OLAP unreadable"})

    # ── Front 4: audit/evidence (snapshot packet, observed) ──
    snapshot = _load_json_file(snapshot_path)
    if isinstance(snapshot, dict) and isinstance(snapshot.get("meta"), dict):
        meta = snapshot["meta"]
        register("audit_snapshot", str(snapshot_path), meta.get("generated_at"))
        fronts["f4_audit_evidence"] = {
            "basis": "observed",
            "source": "audit_snapshot",
            "snapshot_sha256": meta.get("snapshot_sha256"),
            "tests_packet_available": bool(_as_dict(snapshot.get("tests")).get("available")),
        }
    else:
        unavailable.append(
            {
                "field": "f4_audit_evidence",
                "reason": "audit snapshot unreadable, wrong type, or missing meta",
            }
        )

    # ── Queue (taxonomy of section 4) ──
    # Finding 036: only explicitly known supervisor states enter the
    # executable queue; anything else (failed, completed, unknown) is exposed
    # in queue_excluded, never disguised as materialized work.
    _SUPERVISOR_STATE_MAP = {"running": "running", "queued": "dependency_blocked"}
    queue: list[dict[str, Any]] = []
    queue_excluded: list[dict[str, Any]] = []
    if l1_queue_entry is not None:
        # Finding 204: the ACTIVE factorial leads the executable queue;
        # paused-campaign jobs can never displace it.
        queue.append(l1_queue_entry)
    plan_jobs = network.get("plan_jobs") if network else None
    if plan_jobs is not None and not isinstance(plan_jobs, list):
        # Finding 037: a wrong-type plan_jobs section degrades explicitly.
        unavailable.append(
            {"field": "queue.f1", "reason": "plan_jobs is not a list"}
        )
        plan_jobs = []
    if plan_jobs:
        for job in plan_jobs:
            if not isinstance(job, dict):
                queue_excluded.append(
                    {
                        "id": repr(job)[:80],
                        "front": "f1",
                        "supervisor_status": "wrong_type",
                        "reason": "plan job entry is not an object; recorded as history/error",
                    }
                )
                continue
            job_status = str(job.get("status") or "")
            state = _SUPERVISOR_STATE_MAP.get(job_status)
            if state is None:
                queue_excluded.append(
                    {
                        "id": str(job.get("job_id")),
                        "front": "f1",
                        "supervisor_status": job_status or "unknown",
                        "reason": "not an executable-queue state; recorded as history/error",
                    }
                )
                continue
            plan_hash = network.get("plan_hash")
            if state in _REQUIRES_HASHES and not _valid_sha256(plan_hash):
                # Finding 037: a malformed live hash must not make our own
                # taxonomy validator raise; the job is excluded explicitly.
                queue_excluded.append(
                    {
                        "id": str(job.get("job_id")),
                        "front": "f1",
                        "supervisor_status": job_status,
                        "reason": "plan_sha256 missing or malformed; cannot enter executable queue",
                    }
                )
                continue
            entry: dict[str, Any] = {
                "id": str(job.get("job_id")),
                "front": "f1",
                "state": state,
                "hashes": {"plan_sha256": plan_hash},
            }
            if state == "dependency_blocked":
                entry["dependency"] = "job-0 champion/elite archive (fail-closed materializer)"
            queue.append(entry)
    queue.append(
        {
            "id": "ibkr-paper-l1-canary",
            "front": "f2",
            "state": "dependency_blocked",
            "dependency": (
                "IBKR write adapter + independent zero-submit preflight + "
                "owner single-use activation (doc 29 L1)"
            ),
            "hashes": {},
        }
    )
    queue.append(
        {
            "id": "darwinex-zero-subscription",
            "front": "f2",
            "state": "owner_blocked",
            "owner_blocked_reason": "recurring spending not approved (owner, 2026-08-01)",
            "hashes": {},
        }
    )
    validate_queue(queue)

    return {
        "schema": SCHEMA,
        "generated_at": _now(),
        "sources": sources,
        "fronts": fronts,
        "queue": queue,
        "queue_excluded": queue_excluded,
        "unavailable": unavailable,
    }


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path.home() / ".local/state/agent-multi/audit-snapshots/latest.json")
    parser.add_argument("--watchdog", type=Path, default=Path.home() / ".local/state/lts/paper-execution-watchdog/latest.json")
    parser.add_argument("--social-db", type=Path, default=Path.home() / ".local/state/agent-multi/social-intelligence.sqlite")
    parser.add_argument("--supervisor-url", default="http://127.0.0.1:8795")
    parser.add_argument(
        "--l1-contract", type=Path,
        default=repo / "examples/config/phase_3_eth_sac_dynamics/"
                       "l1_factorial_contract_v3.json",
        help="L1 factorial contract; the first-class Front-1 source")
    parser.add_argument("--no-l1", action="store_true",
                        help="disable the L1 factorial source")
    parser.add_argument("--l1-identity", default=None,
                        help="active experiment identity; discovered from "
                             "the newest launcher heartbeat when omitted")
    parser.add_argument(
        "--l1-state-dir", type=Path,
        default=Path.home() / ".local/state/agent-multi/multifront-l1",
        help="status-tool state (ETA samples, alert dedup markers); "
             "NEVER inside the run's output root")
    parser.add_argument("--no-emit-alerts", action="store_true",
                        help="report zero-trade boundary facts without "
                             "emitting the bounded incident observation")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    l1_alert_emitter = None
    if not args.no_l1 and not args.no_emit_alerts:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent))
        import incident_emit  # noqa: E402  (lazy: only the CLI emits)
        l1_alert_emitter = incident_emit.observe_incident
    packet = collect(
        snapshot_path=args.snapshot,
        watchdog_path=args.watchdog,
        social_db_path=args.social_db,
        supervisor_url=args.supervisor_url,
        l1_contract_path=None if args.no_l1 else args.l1_contract,
        l1_identity=args.l1_identity,
        l1_state_dir=None if args.no_l1 else args.l1_state_dir,
        l1_alert_emitter=l1_alert_emitter,
    )
    text = json.dumps(packet, indent=1, sort_keys=True)
    if args.output:
        args.output.write_text(text)
        digest = hashlib.sha256(text.encode()).hexdigest()
        print(json.dumps({"written": str(args.output), "sha256": digest}))
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
