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

Finding 212: epoch/trade/patience facts bind to (identity, seed, cell,
attempt) with source freshness; a stale or differently bound source renders
typed unavailability with its age, never current telemetry, and attempt
paths are shown only for the CURRENT heartbeat's attempt.

Finding 213: workers run cells concurrently, so the full-experiment ETA is
the MAXIMUM per-worker remaining path (active + queued cells), reported
separately from each worker's current-cell ETA, each with sample count and
uncertainty.

Finding 214: the IBKR L1 queue item derives from execution heartbeat and
journal facts; a broker hold is operational-but-held with its exact reason
and owner action, never a hardcoded development dependency.

Finding 228: the fresh durable halt state and fresh direct broker facts are
authoritative; the latest decision is historical context only. A cleared
hold (halt='none') with zero open exposures reports
operational_waiting_next_decision and never asks the owner to clear an
already-cleared hold.

Finding 273: IBKR execution identity accepts the current top-level
``artifact_sha256`` heartbeat field and the legacy ``inference`` field. If
both are present they must be valid and identical; schema disagreement fails
closed instead of falsely classifying an operational runner.

Order 2026-08-11 §7.7 (finding 229): Front 1 now also carries the RUNNING
P1 difficulty x P1 LR factorial mechanics screen as a first-class source —
current seed/cell/checkpoint per worker from the runner's per-cell
heartbeats with freshness, per-seed and fleet cell-record counts,
current-cell ETA and the finding-213 critical-path experiment ETA, and the
runner-sampled GPU utilization/temperature. The completed old L1 matched
factorial (2de49ea9) renders history-only through its own block: with no
fresh RUNNING launcher heartbeat it is never `active` and never enters the
executable queue.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import socket
import sqlite3
import subprocess
import sys
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


IBKR_QUEUE_ID = "ibkr-paper-l1-canary"


def _ibkr_l1_queue_entry(
    execution_state_dir: Path,
    now: float,
    heartbeat_stale_after_seconds: float = 43200.0,
) -> tuple[dict[str, Any], Optional[dict[str, str]]]:
    """Finding 214: the IBKR L1 queue item derives from execution
    heartbeat/journal facts, never a hardcoded development dependency.

    A broker hold is an OPERATIONAL state: the item renders
    operational-but-held with the exact hold reason and the owner action
    that clears it — never `dependency-blocked missing-adapter`. Only when
    no durable evidence is readable does the item name the missing
    evidence itself as the dependency.

    Finding 228 authority order: (1) the FRESH durable halt state and
    fresh direct broker facts are authoritative; (2) the latest decision
    is historical context only. A durable ``halt='none'`` (the owner's
    signed clear) with zero open exposures reports
    ``operational_waiting_next_decision`` with NO owner action — a stale
    ``halted:hold`` rejection can never outrank current durable state or
    re-ask the owner to clear an already-cleared hold. The old rejection
    stays visible as history in ``evidence.last_decision``.

    Returns (queue item, unavailable entry or None).
    """
    heartbeat_path = execution_state_dir / "ibkr-model-runner-heartbeat.json"
    olap_path = execution_state_dir / "ibkr-model-execution.sqlite"
    truth = _venue_execution_truth(heartbeat_path, olap_path, now)
    heartbeat_readable = "heartbeat_state" in truth
    journal_readable = "decisions_cumulative" in truth
    base: dict[str, Any] = {
        "id": IBKR_QUEUE_ID, "front": "f2", "basis": "observed",
        "evidence_sources": {"heartbeat": str(heartbeat_path),
                             "journal": str(olap_path)},
        "hashes": {},
    }
    if not heartbeat_readable and not journal_readable:
        item = {**base, "state": "dependency_blocked",
                "dependency": (
                    "fresh IBKR execution heartbeat/journal evidence: both "
                    "sources unreadable at collection time, and no "
                    "operational state may be asserted without observed "
                    "facts (finding 214)")}
        return item, {"field": f"queue.{IBKR_QUEUE_ID}",
                      "reason": ("IBKR execution heartbeat and journal "
                                 "unreadable")}
    hb_age = truth.get("heartbeat_age_seconds")
    hb_fresh = (heartbeat_readable and hb_age is not None
                and hb_age <= heartbeat_stale_after_seconds)
    halt = truth.get("halt")
    # Finding 228: the durable halt VALUE decides, not its truthiness as a
    # string — 'none' (or an absent key) means no hold is active. Only an
    # actual halt marker ('hold', 'kill', …) renders a held state.
    hold_active = halt is not None and str(halt).strip().lower() not in (
        "", "none")
    open_exposures = truth.get("open_exposures")
    evidence = {
        "mode": truth.get("mode"),
        "heartbeat_state": truth.get("heartbeat_state"),
        "heartbeat_age_seconds": hb_age,
        "halt": halt,
        "open_exposures": open_exposures,
        "lifecycles_cumulative": truth.get("lifecycles_cumulative"),
        # Historical context ONLY (finding 228): the latest decision never
        # outranks the fresh durable halt state above.
        "last_decision": truth.get("last_decision"),
    }
    if hold_active:
        last = _as_dict(truth.get("last_decision"))
        held_operational = truth.get("mode") == "write_enabled" and hb_fresh
        reason = (f"broker hold enforced: execution journal service_state "
                  f"halt={halt!r}")
        if last:
            reason += (f"; last decision {last.get('outcome')!r} with reason "
                       f"{last.get('reason')!r}")
        reason += (" — the runner is operational and rejecting decisions "
                   "for exactly this hold, not a missing adapter"
                   if held_operational else
                   " — the hold is durable journal state; current "
                   "operational freshness is recorded in `evidence`")
        item = {**base, "state": "owner_blocked",
                "operational_state": ("operational_but_held"
                                      if held_operational else "held"),
                "owner_blocked_reason": reason,
                "owner_action": (
                    "review the flat-reconciliation packet and clear the "
                    "hold via the one-time authenticated hold-clear packet "
                    "(order 2026-08-10 WP4.3); the hold stays set until "
                    "explicit authenticated owner action"),
                "evidence": evidence}
        return item, None
    if not heartbeat_readable or not hb_fresh:
        age_text = ("unreadable" if not heartbeat_readable else
                    f"stale: age {hb_age}s exceeds "
                    f"{heartbeat_stale_after_seconds:.0f}s")
        item = {**base, "state": "dependency_blocked",
                "dependency": (f"fresh IBKR execution heartbeat (heartbeat "
                               f"{age_text}); journal shows no hold"),
                "evidence": evidence}
        return item, None
    if truth.get("mode") != "write_enabled":
        item = {**base, "state": "dependency_blocked",
                "dependency": ("write-enabled IBKR execution runner: the "
                               "current fresh heartbeat reports "
                               f"{truth.get('mode')!r}"),
                "evidence": evidence}
        return item, None
    # Write-enabled, fresh, unheld: the canary runs as the runner's normal
    # operation; the executing model artifact hash is its config identity.
    heartbeat = _as_dict(_load_json_file(heartbeat_path))
    top_level_artifact_sha = heartbeat.get("artifact_sha256")
    nested_artifact_sha = _as_dict(
        heartbeat.get("inference")).get("artifact_sha256")
    top_level_present = top_level_artifact_sha is not None
    nested_present = nested_artifact_sha is not None
    if top_level_present and nested_present and (
        not _valid_sha256(top_level_artifact_sha)
        or not _valid_sha256(nested_artifact_sha)
        or top_level_artifact_sha != nested_artifact_sha
    ):
        item = {**base, "state": "dependency_blocked",
                "dependency": (
                    "unambiguous content-addressed execution identity: "
                    "top-level and legacy nested artifact SHA-256 fields "
                    "are both present but are invalid or disagree "
                    "(finding 273)"),
                "evidence": evidence}
        return item, None
    artifact_sha = (top_level_artifact_sha
                    if _valid_sha256(top_level_artifact_sha)
                    else nested_artifact_sha)
    if _valid_sha256(artifact_sha):
        # Finding 228: halt cleared + flat (zero open exposures) means the
        # runner simply waits for its next H4 decision; any prior
        # 'halted:hold' rejection in evidence.last_decision is history and
        # must never generate an owner action item.
        if isinstance(open_exposures, int) and open_exposures == 0:
            operational_state = "operational_waiting_next_decision"
            state_text = ("write-enabled, flat (zero open exposures), no "
                          "durable halt — waiting for the next H4 decision;"
                          " any prior rejection in last_decision is "
                          "historical context only (finding 228)")
        elif isinstance(open_exposures, int):
            operational_state = "operational_with_open_exposure"
            state_text = (f"write-enabled, unheld, {open_exposures} open "
                          "exposure(s) under management")
        else:
            operational_state = "operational_exposure_unknown"
            state_text = ("write-enabled and unheld; open-exposure count "
                          "unavailable from the journal")
        item = {**base, "state": "running",
                "operational_state": operational_state,
                "hashes": {"config_sha256": artifact_sha},
                "note": ("config_sha256 = executing model artifact SHA-256 "
                         "from the live write-enabled heartbeat; "
                         + state_text),
                "evidence": evidence}
        return item, None
    item = {**base, "state": "dependency_blocked",
            "dependency": ("content-addressed execution identity: the fresh "
                           "write-enabled heartbeat carries no valid model "
                           "artifact SHA-256"),
            "evidence": evidence}
    return item, None


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

    def unit_loaded(self, host: str, unit: str) -> Optional[bool]:
        """Is this systemd user unit actually loaded on ``host``?

        Finding 233: ``systemctl show -p NRestarts`` answers ``0`` for a
        unit that does not exist, so a restart count alone cannot tell a
        supervised worker from a direct ``nohup`` process. LoadState is
        the discriminator; ``None`` means the question was unanswerable.
        """
        if self._is_local(host):
            try:
                out: Optional[str] = subprocess.run(
                    ["systemctl", "--user", "show", unit,
                     "-p", "LoadState", "--value"],
                    capture_output=True, text=True, timeout=10).stdout
            except Exception:
                return None
        else:
            out = self._ssh(
                host, f"systemctl --user show {unit} -p LoadState --value")
        if not out or not out.strip():
            return None
        return out.strip() == "loaded"

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


def _l1_attempt_cell(attempt_path: Any) -> Optional[str]:
    """Cell directory an attempt path is bound to. Attempt dirs are
    ``<root>/<identity>/seed<seed>/<cell>/attempt-<cell_id>-NN``."""
    parts = str(attempt_path or "").rstrip("/").split("/")
    if len(parts) < 2 or not parts[-1].startswith("attempt-"):
        return None
    return parts[-2]


def _l1_attempt_is_current(attempt_path: Any, identity: str, seed: int,
                           cell: Optional[str]) -> bool:
    """Finding 212: an attempt path may be shown only when it binds to the
    CURRENT heartbeat's (identity, seed, cell). The launcher records the
    last COMPLETED cell's attempt dir in the heartbeat, which is history,
    never the current attempt."""
    if not attempt_path or cell is None:
        return False
    parts = str(attempt_path).rstrip("/").split("/")
    return (_l1_attempt_cell(attempt_path) == cell
            and identity in parts and f"seed{seed}" in parts)


def _l1_bound_telemetry(
    *,
    reader: Any,
    host: str,
    identity: str,
    seed: int,
    cell: Optional[str],
    current_attempt: Optional[str],
    terminal_state: str,
    hb_age: Optional[float],
    hb_progress: Any,
    output_root: str,
    now: datetime,
    stale_after_seconds: float,
    telemetry_stale_after_seconds: float,
) -> dict[str, Any]:
    """Finding 212: epoch/trade/patience facts must bind to
    ``(identity, seed, cell, attempt)`` and carry source freshness.

    Source preference: (1) structured heartbeat progress published by the
    launcher, bound and fresh by construction; (2) a per-attempt log inside
    the CURRENT attempt directory; (3) the global ``logs/seed<seed>.log``,
    which carries no binding metadata and therefore binds only while it is
    FRESH and co-temporal with a fresh RUNNING heartbeat (one exclusive
    launcher per seed writes it serially, so a fresh tail belongs to the
    current cell). A stale or differently bound source yields a typed
    unbound result with its age — never current telemetry.
    """
    hb_fresh = hb_age is not None and hb_age <= stale_after_seconds
    result: dict[str, Any] = {"parsed": {}, "log_path": None,
                              "log_age_seconds": None, "log_mtime_utc": None}

    def _bound(source: str, age: Optional[float], parsed: dict[str, Any],
               basis: str) -> dict[str, Any]:
        result["parsed"] = parsed
        result["binding"] = {
            "bound": True, "source": source,
            "source_age_seconds": age, "basis": basis,
            "binds": {"identity": identity, "seed": seed, "cell": cell,
                      "attempt": current_attempt},
        }
        return result

    # 1) Structured launcher progress: bound facts by construction.
    if isinstance(hb_progress, dict) and hb_progress.get("epoch") is not None:
        prog_cell = hb_progress.get("cell", cell)
        if hb_fresh and cell is not None and prog_cell == cell:
            parsed: dict[str, Any] = {
                key: hb_progress.get(key)
                for key in ("epoch", "epoch_max", "no_activity",
                            "no_activity_of")
                if hb_progress.get(key) is not None}
            trades = hb_progress.get("trades")
            if isinstance(trades, dict):
                parsed["trades"] = trades
            return _bound(
                "heartbeat_progress", hb_age, parsed,
                "structured progress published inside the fresh current "
                "heartbeat: bound to its (identity, seed, cell, attempt)")

    # 2) Per-attempt log inside the CURRENT attempt directory.
    if current_attempt:
        for name in ("train.log", "launcher.log"):
            candidate = f"{current_attempt}/{name}"
            tail = reader.read_tail(host, candidate)
            if not tail:
                continue
            mtime = reader.mtime(host, candidate)
            age = (round(now.timestamp() - mtime, 1)
                   if mtime is not None else None)
            if age is not None and age <= telemetry_stale_after_seconds:
                result.update({"log_path": candidate,
                               "log_age_seconds": age,
                               "log_mtime_utc": datetime.fromtimestamp(
                                   mtime, timezone.utc).isoformat()})
                return _bound(
                    "per_attempt_log", age, _l1_parse_log_tail(tail),
                    "timestamp-fresh log inside the current heartbeat's "
                    "attempt directory: path-bound to "
                    "(identity, seed, cell, attempt)")

    # 3) Global seed log: binds only fresh AND co-temporal with a fresh
    #    RUNNING heartbeat that names a current cell.
    log_path, tail = None, None
    for candidate in (f"{output_root}/logs/seed{seed}.log",
                      f"{output_root}/logs/seed{seed}.launcher.log"):
        text = reader.read_tail(host, candidate)
        if text:
            log_path, tail = candidate, text
            break
    log_age = None
    if log_path:
        mtime = reader.mtime(host, log_path)
        if mtime is not None:
            log_age = round(now.timestamp() - mtime, 1)
            result["log_mtime_utc"] = datetime.fromtimestamp(
                mtime, timezone.utc).isoformat()
    result.update({"log_path": log_path, "log_age_seconds": log_age})

    if log_path is None:
        reason = ("no readable worker log (per-attempt or global) on the "
                  "assigned host")
    elif log_age is None:
        reason = (f"worker log {log_path} has no readable mtime; freshness "
                  "cannot be established, so its facts cannot be bound to "
                  "the current attempt")
    elif log_age > telemetry_stale_after_seconds:
        reason = (f"worker log is stale (age {log_age:.0f}s > "
                  f"{telemetry_stale_after_seconds:.0f}s): its last telemetry "
                  "belongs to an earlier cell/attempt, not the current "
                  "heartbeat's cell "
                  f"{cell!r} (finding 212)")
    elif not hb_fresh:
        reason = ("launcher heartbeat is stale "
                  f"(age {hb_age}s > {stale_after_seconds:.0f}s), so the "
                  "global log cannot be bound to a current attempt")
    elif cell is None or terminal_state != "RUNNING":
        reason = (f"heartbeat names no current cell (terminal_state="
                  f"{terminal_state!r}): the global log has no current "
                  "attempt to bind to")
    else:
        parsed = _l1_parse_log_tail(tail)
        return _bound(
            "global_seed_log", log_age, parsed,
            "global seed log carries no binding metadata; bound because it "
            "is fresh and co-temporal with the fresh RUNNING heartbeat of "
            "the single exclusive per-seed launcher, whose current cell is "
            f"{cell!r}")

    result["binding"] = {
        "bound": False, "source": log_path,
        "source_age_seconds": log_age, "reason": reason,
        "binds": None,
    }
    return result


def _l1_experiment_eta(worker_states: Mapping[str, Mapping[str, Any]],
                       durations: list[float],
                       active_eta_source_label: str =
                       "current_cell_epoch_eta") -> dict[str, Any]:
    """Finding 213: workers run cells CONCURRENTLY. The full-experiment ETA
    is the MAXIMUM per-worker remaining path (active cell + queued cells),
    never the serial sum of all remaining cells across workers."""
    unknown = sorted(seed for seed, ws in worker_states.items()
                     if ws.get("remaining") is None)
    if unknown:
        return {
            "value": "unavailable",
            "missing": ("per-worker remaining path unknown for seed(s) "
                        f"{', '.join(unknown)}: worker facts unreadable, so "
                        "the critical-path maximum cannot be established"),
        }
    if all((ws.get("remaining") or 0) == 0 for ws in worker_states.values()):
        return {
            "basis": "derived",
            "eta_seconds": {"value": 0.0, "low": 0.0, "high": 0.0,
                            "unit": "seconds"},
            "note": "no remaining cells on any worker; experiment complete",
        }
    if len(durations) < 2:
        return {
            "value": "unavailable",
            "missing": (
                "fewer than 2 completed cell records under the active "
                f"identity ({len(durations)} so far); per-worker remaining "
                "paths derive only from observed started_utc→finished_utc "
                "cell durations"),
        }
    mean = sum(durations) / len(durations)
    lo, hi = min(durations), max(durations)
    per_worker: dict[str, dict[str, Any]] = {}
    for seed, ws in sorted(worker_states.items()):
        remaining = int(ws.get("remaining") or 0)
        active = bool(ws.get("active")) and remaining > 0
        queued = max(0, remaining - (1 if active else 0))
        value, low, high = queued * mean, queued * lo, queued * hi
        active_source = None
        if active:
            active_eta = _as_dict(ws.get("active_eta"))
            eta_seconds = _as_dict(active_eta.get("eta_seconds"))
            if isinstance(eta_seconds.get("value"), (int, float)):
                value += eta_seconds["value"]
                low += eta_seconds.get("low", eta_seconds["value"])
                high += eta_seconds.get("high", eta_seconds["value"])
                active_source = active_eta_source_label
            else:
                value, low, high = value + mean, low + lo, high + hi
                active_source = ("mean_completed_cell_duration (no "
                                 "epoch-rate samples for the active cell)")
        per_worker[seed] = {
            "remaining_cells": remaining,
            "queued_cells": queued,
            "active_cell": ws.get("cell") if active else None,
            "active_cell_eta_source": active_source,
            "path_seconds": {"value": round(value, 1), "low": round(low, 1),
                             "high": round(high, 1), "unit": "seconds"},
        }
    critical = max(per_worker,
                   key=lambda s: per_worker[s]["path_seconds"]["value"])
    return {
        "basis": "derived",
        "eta_seconds": dict(per_worker[critical]["path_seconds"]),
        "critical_path_seed": critical,
        "per_worker_paths": per_worker,
        "mean_cell_seconds": round(mean, 1),
        "sample_size": {"value": len(durations), "unit": "completed_cells"},
        "formula": (
            "max over workers of (active-cell remaining + queued cells * "
            "mean observed completed-cell duration); workers run cells "
            "concurrently, so the experiment ETA is the longest "
            "single-worker path and never the serial sum of all remaining "
            "cells (finding 213)"),
        "horizon": ("full experiment: all remaining cells across concurrent "
                    "workers"),
        "uncertainty": ("range [low, high] propagates min/max observed cell "
                        "durations (and the active-cell epoch-rate range "
                        "where available) along each worker path"),
    }


def collect_l1_factorial(
    *,
    contract_path: Path,
    reader: Any,
    identity: Optional[str] = None,
    state_dir: Optional[Path] = None,
    local_hostname: Optional[str] = None,
    stale_after_seconds: float = 900.0,
    telemetry_stale_after_seconds: float = 3600.0,
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
    worker_states: dict[str, dict[str, Any]] = {}
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
            worker_states[str(seed)] = {"remaining": None, "active": False,
                                        "cell": None, "active_eta": None}
            continue
        any_fact = True
        hb_updated = _l1_iso(heartbeat.get("updated_utc"))
        hb_age = (round((now - hb_updated).total_seconds(), 1)
                  if hb_updated else None)
        cell = heartbeat.get("cell")
        cell_spec = _as_dict(_as_dict(contract.get("cells")).get(cell))
        # Finding 212: the heartbeat's `attempt` field records the last
        # COMPLETED cell's attempt directory. An attempt path is shown
        # only when it binds to the CURRENT heartbeat's cell; anything
        # else is withheld with a typed reason (the cell it belongs to),
        # never displayed as if it were current.
        raw_attempt = heartbeat.get("attempt")
        attempt_is_current = _l1_attempt_is_current(
            raw_attempt, identity, seed, cell)
        entry.update({
            "heartbeat_schema": heartbeat.get("schema"),
            "terminal_state": heartbeat.get("terminal_state") or "unknown",
            "error": heartbeat.get("error"),
            "cell": cell,
            "cell_factors": {
                "phase1_mode": cell_spec.get("phase1_mode"),
                "phase2_lr_multiplier": cell_spec.get("phase2_lr_multiplier"),
            } if cell_spec else None,
            "attempt": raw_attempt if attempt_is_current else None,
            "pid": heartbeat.get("pid"),
            "pid_start_identity": heartbeat.get("pid_start_identity"),
            "cuda_visible_devices": heartbeat.get("cuda_visible_devices"),
            "observed_gpu_uuids": heartbeat.get("observed_gpu_uuids"),
            "heartbeat_assigned_gpu_uuid": heartbeat.get("assigned_gpu_uuid"),
            "progress": heartbeat.get("progress"),
            "heartbeat_updated_utc": heartbeat.get("updated_utc"),
            "heartbeat_age_seconds": hb_age,
        })
        if raw_attempt and not attempt_is_current:
            entry["attempt_withheld"] = {
                "reason": ("heartbeat attempt path is not the CURRENT "
                           "attempt: it belongs to cell "
                           f"{_l1_attempt_cell(raw_attempt)!r} while the "
                           f"current heartbeat cell is {cell!r} (the "
                           "launcher records the last COMPLETED attempt); "
                           "path withheld (finding 212)"),
                "bound_cell": _l1_attempt_cell(raw_attempt),
            }

        # Finding 212: telemetry facts bind to (identity, seed, cell,
        # attempt) with source freshness, or render typed unavailability.
        telemetry = _l1_bound_telemetry(
            reader=reader, host=host, identity=identity, seed=seed,
            cell=cell,
            current_attempt=raw_attempt if attempt_is_current else None,
            terminal_state=entry["terminal_state"], hb_age=hb_age,
            hb_progress=heartbeat.get("progress"),
            output_root=output_root, now=now,
            stale_after_seconds=stale_after_seconds,
            telemetry_stale_after_seconds=telemetry_stale_after_seconds)
        parsed = telemetry["parsed"]
        binding = telemetry["binding"]
        entry["telemetry_binding"] = binding
        entry["log_path"] = telemetry["log_path"]
        if telemetry["log_mtime_utc"]:
            entry["log_mtime_utc"] = telemetry["log_mtime_utc"]
        if telemetry["log_age_seconds"] is not None:
            entry["log_age_seconds"] = telemetry["log_age_seconds"]
        if binding["bound"]:
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
        else:
            epoch = epoch_max = None
            unbound_fact = {
                "value": "unavailable",
                "reason": binding.get("reason"),
                "source": telemetry["log_path"],
                "source_age_seconds": telemetry["log_age_seconds"],
            }
            entry["epoch"] = dict(unbound_fact)
            entry["activity_patience"] = dict(unbound_fact)
            entry["trades"] = dict(unbound_fact)
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
        entry["current_cell_eta"] = _l1_eta_from_samples(
            samples, cell, epoch, epoch_max)
        if state_dir is None and \
                entry["current_cell_eta"].get("value") == "unavailable":
            entry["current_cell_eta"]["missing"] += \
                "; no state dir for timing samples"
        worker_states[str(seed)] = {
            "remaining": len(cells) - len(landed),
            "active": (entry["terminal_state"] == "RUNNING"
                       and cell is not None and hb_age is not None
                       and hb_age <= stale_after_seconds),
            "cell": cell,
            "active_eta": entry["current_cell_eta"],
        }

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
        "experiment_eta": _l1_experiment_eta(worker_states, cell_durations),
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


# ── Front 1 first-class source: P1 difficulty x P1 LR factorial (§7.7) ──────

P1LR_HEARTBEAT_SCHEMA = "agent_multi.p1_difficulty_lr_heartbeat.v1"
P1LR_RECORD_SCHEMA = "agent_multi.p1_difficulty_lr_cell_record.v1"
P1LR_CHECKPOINT_STAGES = ("materializing", "nested-role-verification",
                          "training", "terminal-custody",
                          "outer-validation-final", "complete")

# ── Finding 233: EVERY mode-dependent fact derives from ONE validated
# mode. Screen and decision are distinct experiments with distinct
# content-addressed identities, distinct output roots, distinct units
# and distinct evidence classes; reading one while the other runs is
# what rendered a false 0/16, 0/4 idle picture over four busy GPUs.
P1LR_MODES = ("screen", "decision")
P1LR_UNIT_TEMPLATES = {"screen": "p1lr-screen@{seed}.service",
                       "decision": "p1lr-decision@{seed}.service"}
P1LR_MODE_EVIDENCE_CLASS = {"screen": "mechanics_screen",
                            "decision": "decision_run"}
# Back-compat alias: the screen template stays importable under its old
# name (the guard and older callers referenced it directly).
P1LR_UNIT_TEMPLATE = P1LR_UNIT_TEMPLATES["screen"]


class P1lrModeRefusal(ValueError):
    """A typed P1LR mode/root refusal.

    Finding 233: a mode or identity that does not bind to the root being
    read is a REFUSAL, never a rendered zero. ``as_block()`` therefore
    publishes the reason and the corrective command and deliberately
    carries NO ``workers_running_fresh`` and NO ``records_landed`` — a
    0/4 or 0/16 under the wrong root is a false idle picture, not a
    degraded one.
    """

    def __init__(self, code: str, reason: str, **facts: Any):
        super().__init__(f"{code}: {reason}")
        self.code = code
        self.reason = reason
        self.facts = facts

    def as_block(self, **extra: Any) -> dict[str, Any]:
        block: dict[str, Any] = {
            "source": "p1lr_factorial",
            "basis": "refusal",
            "state": "refused",
            "error_code": self.code,
            "reason": self.reason,
            "refusal_contract": (
                "a refused P1LR mode/identity binding renders NO worker "
                "and NO record counts: a zero under the wrong output "
                "root is a false idle picture, not a measurement "
                "(finding 233)"),
        }
        block.update(self.facts)
        block.update(extra)
        return block


def p1lr_mode_binding(
    contract: dict,
    mode: str,
    *,
    unit_template: Optional[str] = None,
) -> dict[str, Any]:
    """Everything mode-dependent, derived from ONE validated mode.

    Returns the output root, systemd unit template, expected heartbeat/
    record mode, expected evidence class and the per-seed and total cell
    counts for ``mode``; raises :class:`P1lrModeRefusal` for an unknown
    mode, a missing/colliding decision root or an incomplete contract.
    """
    if mode not in P1LR_MODES:
        raise P1lrModeRefusal(
            "P1LR_MODE_INVALID",
            f"unknown P1LR execution mode {mode!r}: the mode is explicit "
            f"and validated, one of {list(P1LR_MODES)}",
            supplied_mode=mode, known_modes=list(P1LR_MODES))

    screen_root = str(contract.get("output_root") or "").rstrip("/")
    decision = _as_dict(contract.get("decision_run"))
    decision_root = str(decision.get("output_root") or "").rstrip("/")

    if mode == "decision":
        if not decision_root:
            raise P1lrModeRefusal(
                "P1LR_DECISION_ROOT_MISSING",
                "decision mode requires contract decision_run.output_root; "
                "the contract declares none, so there is no decision root "
                "to read (never fall back to the screen root)",
                mode=mode, screen_output_root=screen_root or None)
        if screen_root and decision_root == screen_root:
            raise P1lrModeRefusal(
                "P1LR_MODE_ROOTS_COLLIDE",
                "decision_run.output_root equals the screen output_root: "
                "the two modes are not separable, so no mode-bound "
                "reading is possible",
                mode=mode, output_root=decision_root)
        output_root, other_root = decision_root, (screen_root or None)
    else:
        if not screen_root:
            raise P1lrModeRefusal(
                "P1LR_SCREEN_ROOT_MISSING",
                "screen mode requires contract output_root; the contract "
                "declares none",
                mode=mode)
        output_root, other_root = screen_root, (decision_root or None)

    assignments = _as_dict(contract.get("assignments"))
    cells_map = _as_dict(contract.get("cells"))
    cell_order = _as_dict(contract.get("cell_order"))
    seeds = [s for s in (contract.get("seeds") or []) if str(s) in assignments]
    if not (seeds and cells_map):
        raise P1lrModeRefusal(
            "P1LR_CONTRACT_INCOMPLETE",
            "contract lacks assigned seeds or cells, so no per-mode cell "
            "total can be derived",
            mode=mode, output_root=output_root)
    cells_per_seed = {
        str(seed): ([c for c in (cell_order.get(str(seed)) or [])
                     if c in cells_map] or list(cells_map))
        for seed in seeds}
    other_mode = "screen" if mode == "decision" else "decision"
    resolved_unit_template = unit_template or P1LR_UNIT_TEMPLATES[mode]
    if "{seed}" not in resolved_unit_template:
        raise P1lrModeRefusal(
            "P1LR_UNIT_TEMPLATE_INVALID",
            "runtime unit template must contain the literal {seed} placeholder",
            mode=mode,
            unit_template=resolved_unit_template,
        )
    return {
        "mode": mode,
        "output_root": output_root,
        "other_mode": other_mode,
        "other_mode_output_root": other_root,
        "unit_template": resolved_unit_template,
        "unit_example": resolved_unit_template.format(seed=seeds[0]),
        "heartbeat_mode_expected": mode,
        "record_mode_expected": mode,
        "evidence_class_expected": P1LR_MODE_EVIDENCE_CLASS[mode],
        "decision_eligible_expected": mode == "decision",
        "seeds": [int(s) for s in seeds],
        "cells_per_seed": cells_per_seed,
        "total_cells": sum(len(v) for v in cells_per_seed.values()),
    }


def _p1lr_newest_local_heartbeat(
        output_root: str) -> tuple[Optional[float], Optional[str]]:
    """(newest heartbeat mtime, its experiment identity) under the LOCAL
    output root — cell heartbeats plus seed-level refusal heartbeats.
    Remote roots are never globbed: each host discovers its own."""
    root = Path(output_root).expanduser()
    newest: tuple[Optional[float], Optional[str]] = (None, None)
    try:
        candidates = (list(root.glob("*/seed*/*/heartbeat.json"))
                      + list(root.glob("*/seed*/runner_heartbeat.json")))
    except OSError:
        return newest
    for path in candidates:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if newest[0] is None or mtime > newest[0]:
            newest = (mtime, path.relative_to(root).parts[0])
    return newest


def _p1lr_discover_identity(output_root: str) -> Optional[str]:
    """Newest experiment identity under the LOCAL output root."""
    return _p1lr_newest_local_heartbeat(output_root)[1]


def _p1lr_identity_local_presence(output_root: Optional[str],
                                  identity: str) -> bool:
    """Does this identity have a directory under the LOCAL root?"""
    if not output_root:
        return False
    try:
        return (Path(output_root).expanduser() / identity).is_dir()
    except OSError:
        return False


def p1lr_verify_identity_binding(binding: dict, identity: str,
                                 *, presence_fn: Optional[
                                     Callable[[Optional[str], str], bool]]
                                 = None) -> dict[str, Any]:
    """Bind an identity to the mode's root, or REFUSE (finding 233).

    An identity that lives under the OTHER mode's root and not under the
    selected one is the exact defect: the auditor's decision identity
    read under the screen root rendered 0/16 and 0/4 while four decision
    processes trained. That case raises :class:`P1lrModeRefusal`; an
    identity absent from both LOCAL roots is not refused (a host may
    legitimately hold no local seed directory) but its presence facts
    are published so the reading is never mistaken for proof.
    """
    presence = presence_fn or _p1lr_identity_local_presence
    here = presence(binding["output_root"], identity)
    there = presence(binding["other_mode_output_root"], identity)
    if there and not here:
        other = binding["other_mode"]
        raise P1lrModeRefusal(
            "P1LR_IDENTITY_MODE_MISMATCH",
            f"experiment identity {identity!r} exists under the "
            f"{other} root {binding['other_mode_output_root']} and NOT "
            f"under the requested {binding['mode']} root "
            f"{binding['output_root']}: it is a {other}-mode identity. "
            f"Reading it here would render a false empty state; rerun "
            f"with --p1lr-mode {other}",
            mode=binding["mode"], requested_mode=binding["mode"],
            identity=identity, identity_mode=other,
            output_root=binding["output_root"],
            other_mode_output_root=binding["other_mode_output_root"],
            corrective_command=(
                f"tools/multifront_status.py --p1lr-mode {other} "
                f"--p1lr-identity {identity}"))
    return {
        "identity_under_mode_root": here,
        "identity_under_other_mode_root": there,
        "basis": ("local filesystem directory presence under each mode's "
                  "output root on this host"),
    }


def _p1lr_other_mode_activity(binding: dict, now: datetime,
                              stale_after_seconds: float,
                              ) -> Optional[dict[str, Any]]:
    """Advisory: FRESH local heartbeat activity under the OTHER mode's
    root while this mode is being read (finding 233).

    The defect the auditor hit was silent: the screen root was read
    while the decision root was the live one. When the other root shows
    fresh local activity, the reading says so explicitly and names the
    command that would observe it, instead of leaving an operator with
    a quiet empty screen."""
    other_root = binding.get("other_mode_output_root")
    if not other_root:
        return None
    mtime, other_identity = _p1lr_newest_local_heartbeat(str(other_root))
    if mtime is None or other_identity is None:
        return None
    age = round(now.timestamp() - mtime, 1)
    if age > stale_after_seconds:
        return None
    other_mode = binding["other_mode"]
    return {
        "value": "fresh_activity_under_other_mode_root",
        "other_mode": other_mode,
        "other_mode_output_root": other_root,
        "other_mode_identity": other_identity,
        "heartbeat_age_seconds": age,
        "note": (f"this reading is bound to the {binding['mode']} root; a "
                 f"{other_mode}-mode run is ALSO writing fresh heartbeats "
                 "locally and is NOT described by these counts"),
        "corrective_command": (f"tools/multifront_status.py --p1lr-mode "
                               f"{other_mode} --p1lr-identity "
                               f"{other_identity}"),
    }


def _transition_queue():
    """The durable transition-queue module, however this file was loaded.

    ``tools/multifront_status.py`` runs both as a script (``tools/`` is
    sys.path[0]) and as ``tools.multifront_status``; the import is
    written for both instead of assuming one.
    """
    try:
        from tools import experiment_transition_queue as etq
    except ImportError:  # running with tools/ itself on sys.path
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import experiment_transition_queue as etq  # type: ignore
    return etq


def p1lr_transition_view(*, queue_dir: Optional[Path], experiment: str,
                         mode: str, identity: str, now: datetime,
                         terminal: bool, records_landed: int,
                         cells_total: int,
                         output_root: Optional[str] = None,
                         contract_path: Optional[str] = None,
                         contract_sha256: Optional[str] = None,
                         terminal_utc: Optional[str] = None,
                         enrol: bool = True,
                         budget_seconds: Optional[float] = None,
                         ) -> dict[str, Any]:
    """The fleet-level terminal-to-next-job answer (order 2026-08-15 §3).

    This collector is the only observer that reads EVERY assigned host,
    so it is the one that may declare an experiment terminal and enrol
    the durable transition record. Enrolment is idempotent and
    content-addressed on the ending job, so repeated status runs — and
    runs from different hosts — converge on ONE record and never restart
    the transition budget.

    A terminal experiment whose successor is not proven dispatched
    returns ``completed_untransitioned``. That is the state the fleet
    must surface; ``inactive_or_unknown`` (the old answer) described the
    previous experiment's stillness instead of the fleet's idleness.
    """
    etq = _transition_queue()
    if not terminal:
        return {
            "value": "current_job_running",
            "basis": "experiment_not_terminal",
            "reason": ("the current experiment is not terminal; transition "
                       "status does not apply until all assigned cells land "
                       "and no worker remains RUNNING"),
            "transition_id": None,
            "next_job_id": None,
            "materialization_state": None,
            "dispatch_state": None,
            "blockers": [],
            "over_budget": None,
            "successor_dispatched": False,
            "queue_dir": (str(Path(queue_dir).expanduser())
                          if queue_dir is not None else None),
        }
    if queue_dir is None:
        status = etq.transition_status(None, now=now)
        status.update({
            "queue_dir": None,
            "basis": "transition_queue_not_configured",
            "reason": ("no durable transition queue is configured, so no "
                       "successor can be proven dispatched; a terminal "
                       "experiment is un-transitioned by default"),
        })
        return status
    record = None
    if terminal and enrol:
        try:
            record = etq.ensure_terminal_record(
                queue_dir, experiment=experiment, mode=mode,
                identity=identity, records_landed=records_landed,
                cells_total=cells_total, output_root=output_root,
                contract_path=contract_path,
                contract_sha256=contract_sha256,
                terminal_utc=terminal_utc, evidence_root=output_root,
                now=now,
                transition_budget_seconds=(
                    budget_seconds
                    if budget_seconds is not None
                    else etq.DEFAULT_TRANSITION_BUDGET_SECONDS))
        except OSError:
            # An unwritable queue directory must degrade to "cannot
            # prove a dispatch", never to a fabricated healthy state.
            record = None
    if record is None:
        records, _unreadable = etq.load_records(queue_dir)
        record = etq.find_record(records, experiment=experiment,
                                 mode=mode, identity=identity)
    status = etq.transition_status(record, now=now)
    status["queue_dir"] = str(Path(queue_dir).expanduser())
    return status


def _p1lr_transition_queue_entry(status: Mapping[str, Any], *,
                                 identity: str,
                                 contract_sha: Optional[str],
                                 ) -> Optional[dict[str, Any]]:
    """The SUCCESSOR's executable-queue item for an un-transitioned run.

    Finding 204 discipline in the transition case: a terminal experiment
    with no dispatched successor must not vanish from the executable
    queue — that silence is exactly what hid the fleet's idle time. The
    successor enters the canonical taxonomy according to what actually
    blocks it, so an owner reading the queue sees whether the fleet is
    waiting on materialization, on a dependency, or on the owner.
    """
    blockers = list(status.get("blockers") or [])
    owner_blockers = [b for b in blockers if b.get("owner_action_required")]
    dependency_blockers = [b for b in blockers
                           if b.get("dependency") and b not in owner_blockers]
    next_job = _as_dict(status.get("next_job"))
    item: dict[str, Any] = {
        "id": f"p1lr-transition-{status.get('transition_id') or identity}",
        "front": "f1",
        "transition_state": status.get("value"),
        "predecessor_identity": identity,
        "next_job_id": status.get("next_job_id"),
        "materialization_state": status.get("materialization_state"),
        "dispatch_state": status.get("dispatch_state"),
        "elapsed_since_terminal_seconds":
            status.get("elapsed_since_terminal_seconds"),
        "transition_budget_seconds": status.get("transition_budget_seconds"),
        "over_budget": status.get("over_budget"),
    }
    # Only syntactically valid digests may ride in `hashes`: the queue
    # taxonomy rejects a malformed digest outright, and a transition item
    # must never be the reason the whole packet fails validation.
    hashes = {k: v for k, v in
              {"config_sha256": next_job.get("contract_sha256"),
               "plan_sha256": next_job.get("plan_sha256")}.items()
              if _valid_sha256(v)}
    if owner_blockers:
        item["state"] = "owner_blocked"
        item["owner_blocked_reason"] = "; ".join(
            f"{b.get('code')}: {b.get('detail')}" for b in owner_blockers)
    elif dependency_blockers:
        item["state"] = "dependency_blocked"
        item["dependency"] = "; ".join(
            str(b.get("dependency")) for b in dependency_blockers)
    elif (status.get("materialization_state") == "materialized"
            and hashes):
        item["state"] = "materialized"
    else:
        item["state"] = "proposed"
    if hashes:
        item["hashes"] = hashes
    if contract_sha:
        item["predecessor_contract_sha256"] = contract_sha
    return item


def _p1lr_cell_eta(durations: list[float], elapsed: Optional[float],
                   elapsed_reason: Optional[str]) -> dict[str, Any]:
    """Current-cell ETA from OBSERVED facts only: mean/min/max completed
    P1LR cell durations minus the running cell's observed elapsed time
    (the exclusive-claim sidecar's acquired_utc). The screen heartbeat
    carries stage progress, not epoch counts, so completed-cell durations
    are the only observed duration source (finding 213 style)."""
    if elapsed is None:
        return {"value": "unavailable",
                "missing": elapsed_reason or "no observed cell start time"}
    if len(durations) < 2:
        return {
            "value": "unavailable",
            "missing": (
                "fewer than 2 completed cell records under the active "
                f"identity ({len(durations)} so far); the current-cell ETA "
                "derives only from observed completed-cell durations minus "
                "the observed cell runtime"),
            "elapsed_seconds": elapsed,
        }
    mean = sum(durations) / len(durations)
    lo, hi = min(durations), max(durations)
    return {
        "basis": "derived",
        "eta_seconds": {"value": round(max(0.0, mean - elapsed), 1),
                        "low": round(max(0.0, lo - elapsed), 1),
                        "high": round(max(0.0, hi - elapsed), 1),
                        "unit": "seconds"},
        "elapsed_seconds": elapsed,
        "mean_cell_seconds": round(mean, 1),
        "sample_size": {"value": len(durations), "unit": "completed_cells"},
        "formula": ("max(0, mean observed completed-cell duration - elapsed "
                    "since the running cell's exclusive claim was acquired)"),
        "horizon": "current cell",
        "uncertainty": ("range [low, high] from min/max observed "
                        "completed-cell durations; 0 floors a cell already "
                        "running longer than the observed durations"),
    }


def collect_p1lr_factorial(
    *,
    contract_path: Path,
    reader: Any,
    identity: Optional[str] = None,
    local_hostname: Optional[str] = None,
    stale_after_seconds: float = 900.0,
    now_fn: Optional[Callable[[], datetime]] = None,
    mode: str = "screen",
    identity_presence_fn: Optional[
        Callable[[Optional[str], str], bool]] = None,
    transition_queue_dir: Optional[Path] = None,
    transition_enrol: bool = True,
    transition_budget_seconds: Optional[float] = None,
    runtime_authority_path: Optional[Path] = None,
) -> tuple[dict[str, Any], list[dict[str, str]], Optional[dict[str, Any]]]:
    """First-class Front-1 source: the RUNNING P1 difficulty x P1 LR
    factorial mechanics screen (order 2026-08-11 §7.7 / finding 229).

    Strictly read-only towards the run: per-cell runner heartbeats
    (``seed<seed>/<cell>/heartbeat.json``), seed-level refusal heartbeats,
    cell records and the exclusive-claim sidecars are read; nothing is
    ever written anywhere. Facts on remote hosts arrive via the shared
    read-only reader; an unreachable host renders a typed unavailable
    worker, never a fabricated count — each assigned host's own status
    run stays authoritative for its local seeds (per-host N/4), so the
    fleet 16 carries an explicit reachability note.

    Finding-212 discipline: a stale heartbeat renders typed staleness
    (its age plus last-known facts) and NEVER current seed/cell/checkpoint
    claims. Finding-213 discipline: the experiment ETA is the MAXIMUM
    per-worker remaining path, reusing ``_l1_experiment_eta``.

    Finding-233 discipline: ``mode`` is explicit and VALIDATED, and the
    output root, systemd unit, expected heartbeat/record mode and total
    cell count all derive from it. An identity that belongs to the other
    mode's root REFUSES (typed, count-free) instead of rendering a false
    empty state.

    Order 2026-08-15 §3 discipline: an experiment whose every assigned
    host was readable and whose every cell record has landed is TERMINAL.
    A terminal experiment renders ``completed_untransitioned`` until the
    durable transition queue proves a successor dispatched — a completed
    predecessor is never a reason to render the fleet as healthy — and
    the successor keeps a first-class executable-queue item so the idle
    time is visible where an owner looks for work.

    Returns (front block, unavailable entries, executable-queue entry).
    """
    now = (now_fn or (lambda: datetime.now(timezone.utc)))()
    unavailable: list[dict[str, str]] = []
    field = "f1_optimization.active_p1lr_factorial"

    def _refuse(refusal: P1lrModeRefusal) -> tuple[
            dict[str, Any], list[dict[str, str]], None]:
        unavailable.append({"field": field,
                            "reason": f"{refusal.code}: {refusal.reason}"})
        return (refusal.as_block(contract_path=str(contract_path)),
                unavailable, None)

    try:
        contract = json.loads(contract_path.read_text())
    except (OSError, ValueError) as exc:
        reason = f"p1lr factorial contract unreadable: {type(exc).__name__}"
        unavailable.append({"field": field, "reason": reason})
        return {"source": "p1lr_factorial", "state": "unavailable",
                "reason": reason}, unavailable, None
    contract_sha = _sha256_file(contract_path)

    # ── ONE validated mode; every mode-dependent fact derives from it ──
    try:
        binding = p1lr_mode_binding(contract, mode)
    except P1lrModeRefusal as refusal:
        return _refuse(refusal)
    output_root = binding["output_root"]
    cells_per_seed = binding["cells_per_seed"]
    seeds = binding["seeds"]
    assignments = _as_dict(contract.get("assignments"))
    cells_map = _as_dict(contract.get("cells"))

    identity_basis = "explicit_parameter"
    if not identity:
        identity = _p1lr_discover_identity(output_root)
        identity_basis = (f"discovered_latest_heartbeat_mtime(local,"
                          f"{binding['mode']}_root)")
    if not identity:
        reason = ("no experiment identity: none supplied and no runner "
                  f"heartbeat discoverable under the local {binding['mode']} "
                  f"output root {output_root}")
        unavailable.append({"field": field, "reason": reason})
        return {"source": "p1lr_factorial", "state": "unavailable",
                "mode": binding["mode"], "output_root": output_root,
                "reason": reason,
                "contract_path": str(contract_path)}, unavailable, None

    # AUD-F1-20260816-272: an incident-specific accepted unit family may
    # intentionally differ from the generic p1lr-decision@ template. Status
    # binds it from durable authority rather than declaring supervised workers
    # to be direct/nohup workers.
    authority_facts: Optional[dict[str, Any]] = None
    if runtime_authority_path is not None:
        try:
            authority = json.loads(runtime_authority_path.read_text())
            accepted = _as_dict(_as_dict(authority).get("accepted_runtime"))
        except (OSError, ValueError) as exc:
            return _refuse(P1lrModeRefusal(
                "P1LR_RUNTIME_AUTHORITY_UNREADABLE",
                f"runtime authority is explicit but unreadable: {type(exc).__name__}",
                runtime_authority_path=str(runtime_authority_path),
            ))
        expected_identity = accepted.get(f"{binding['mode']}_identity")
        expected_contract = (
            accepted.get(f"{binding['mode']}_contract_sha256")
            or accepted.get("contract_sha256")
            or accepted.get("screen_contract_sha256")
        )
        if expected_identity != identity or expected_contract != contract_sha:
            return _refuse(P1lrModeRefusal(
                "P1LR_RUNTIME_AUTHORITY_MISMATCH",
                "runtime authority does not bind the selected identity and contract",
                runtime_authority_path=str(runtime_authority_path),
                authority_identity=expected_identity,
                selected_identity=identity,
                authority_contract_sha256=expected_contract,
                selected_contract_sha256=contract_sha,
            ))
        raw_pattern = accepted.get(f"{binding['mode']}_unit_pattern")
        if not isinstance(raw_pattern, str) or raw_pattern.count("<seed>") != 1:
            return _refuse(P1lrModeRefusal(
                "P1LR_RUNTIME_UNIT_PATTERN_INVALID",
                "authority unit pattern must contain exactly one <seed> placeholder",
                runtime_authority_path=str(runtime_authority_path),
                unit_pattern=raw_pattern,
            ))
        try:
            binding = p1lr_mode_binding(
                contract,
                binding["mode"],
                unit_template=raw_pattern.replace("<seed>", "{seed}"),
            )
        except P1lrModeRefusal as refusal:
            return _refuse(refusal)
        authority_facts = {
            "path": str(runtime_authority_path),
            "sha256": _sha256_file(runtime_authority_path),
            "unit_pattern": raw_pattern,
            "binding": "identity_contract_and_unit_pattern_exact",
        }

    # ── identity ↔ mode-root binding: mismatch REFUSES, never renders 0 ──
    try:
        identity_presence = p1lr_verify_identity_binding(
            binding, identity, presence_fn=identity_presence_fn)
    except P1lrModeRefusal as refusal:
        return _refuse(refusal)

    workers: dict[str, Any] = {}
    worker_states: dict[str, dict[str, Any]] = {}
    pending_eta: dict[str, tuple[Optional[float], Optional[str]]] = {}
    durations: list[float] = []
    records_total = 0
    total_cells = binding["total_cells"]
    running_fresh = 0
    any_fact = False
    rejected_records: list[dict[str, Any]] = []
    unreadable_seeds: list[int] = []
    newest_finished: Optional[datetime] = None

    for seed in seeds:
        assignment = _as_dict(assignments.get(str(seed)))
        host = assignment.get("hostname") or "unknown"
        unit = binding["unit_template"].format(seed=seed)
        seed_dir = f"{output_root}/{identity}/seed{seed}"
        seed_cells = cells_per_seed[str(seed)]
        entry: dict[str, Any] = {
            "identity": identity, "seed": seed, "host": host,
            "mode": binding["mode"], "unit": unit, "basis": "observed",
            "assigned_gpu_uuid": assignment.get("gpu_uuid"),
            "cell_order": seed_cells,
        }

        # Newest runner heartbeat: the running cell rewrites its
        # heartbeat every minute, so the max updated_utc across the four
        # cell heartbeats plus the seed-level refusal heartbeat is the
        # seed's current fact.
        heartbeats: list[tuple[Optional[datetime], dict]] = []
        for name in ([f"{seed_dir}/{cell}/heartbeat.json"
                      for cell in seed_cells]
                     + [f"{seed_dir}/runner_heartbeat.json"]):
            raw = reader.read_text(host, name)
            if not raw:
                continue
            try:
                loaded = json.loads(raw)
            except ValueError:
                continue
            if isinstance(loaded, dict) and loaded:
                heartbeats.append((_l1_iso(loaded.get("updated_utc")),
                                   loaded))
        host_error = _as_dict(getattr(reader, "errors", {})).get(host)
        if not heartbeats:
            entry["terminal_state"] = "unavailable"
            entry["unavailable_reason"] = (
                f"host unreachable: {host_error}" if host_error
                else "no cell or runner heartbeat readable under "
                     f"{seed_dir}")
            unavailable.append({"field": f"{field}.workers.{seed}",
                                "reason": entry["unavailable_reason"]})
            # An unreadable host can never contribute to a TERMINAL
            # verdict: 16/16 is only 16/16 when all four were readable.
            unreadable_seeds.append(seed)
            workers[str(seed)] = entry
            worker_states[str(seed)] = {"remaining": None, "active": False,
                                        "cell": None, "active_eta": None}
            continue
        any_fact = True
        heartbeats.sort(key=lambda pair: (pair[0] is not None, pair[0]
                                          or datetime.min.replace(
                                              tzinfo=timezone.utc)))
        hb_updated, hb = heartbeats[-1]
        # Remote reads are sequential and can take longer than one heartbeat
        # interval. Age each fact when it has actually arrived instead of
        # against the collector-start timestamp, which can produce impossible
        # negative ages for later hosts even when every clock is synchronized.
        observed_now = (now_fn or (lambda: datetime.now(timezone.utc)))()
        raw_hb_age = ((observed_now - hb_updated).total_seconds()
                      if hb_updated else None)
        hb_clock_ahead = (round(max(0.0, -raw_hb_age), 1)
                          if raw_hb_age is not None else None)
        hb_age = (round(max(0.0, raw_hb_age), 1)
                  if raw_hb_age is not None else None)
        hb_fresh = hb_age is not None and hb_age <= stale_after_seconds
        terminal_state = hb.get("terminal_state") or "unknown"
        cell = hb.get("cell")
        stage = hb.get("progress")
        entry.update({
            "heartbeat_schema": hb.get("schema"),
            "heartbeat_updated_utc": hb.get("updated_utc"),
            "heartbeat_age_seconds": hb_age,
            "heartbeat_clock_ahead_seconds": hb_clock_ahead,
            "heartbeat_fresh": hb_fresh,
            "terminal_state": terminal_state,
            "error": hb.get("error"),
            "pid": hb.get("pid"),
            "pid_start_identity": hb.get("pid_start_identity"),
            "cell_identity": hb.get("cell_identity"),
            "cuda_visible_devices": hb.get("cuda_visible_devices"),
            "observed_gpu_uuids": hb.get("observed_gpu_uuids"),
            "heartbeat_assigned_gpu_uuid": hb.get("assigned_gpu_uuid"),
            # Finding 233: the expected mode is derived from the
            # validated mode; the heartbeat schema carries no mode field
            # today, so the binding is POSITIONAL (mode root + identity)
            # and the verified mode field lives on the cell records.
            "heartbeat_mode_expected": binding["heartbeat_mode_expected"],
            "heartbeat_mode_declared": hb.get("mode"),
        })
        if hb.get("mode") is not None and hb.get("mode") != binding["mode"]:
            entry["heartbeat_mode_mismatch"] = (
                f"heartbeat declares mode {hb.get('mode')!r} under the "
                f"{binding['mode']} root {output_root}: the artifact does "
                "not belong to the mode being read")

        running_now = bool(hb_fresh and terminal_state == "RUNNING"
                           and cell)
        if running_now:
            spec = _as_dict(cells_map.get(cell))
            entry["current_cell"] = cell
            entry["current_cell_factors"] = ({
                "phase1_dynamics": spec.get("phase1_dynamics"),
                "phase1_learning_rate": spec.get("phase1_learning_rate"),
            } if spec else None)
            entry["checkpoint"] = {
                "stage": stage,
                "known_stages": list(P1LR_CHECKPOINT_STAGES),
                "unit": "runner_stage",
                "horizon": "cell",
                "source": ("cell heartbeat progress published by the "
                           "runner; bound to (identity, seed, cell) by "
                           "construction"),
                "source_age_seconds": hb_age,
            }
            entry["attempt"] = hb.get("attempt")
            running_fresh += 1
        elif hb_age is None:
            typed = {
                "value": "unavailable",
                "reason": ("runner heartbeat carries no parseable "
                           "updated_utc; freshness cannot be established, "
                           "so no current seed/cell/checkpoint claim is "
                           "made (finding 212)"),
                "heartbeat_age_seconds": None,
                "last_known": {"cell": cell, "stage": stage,
                               "terminal_state": terminal_state},
            }
            entry["current_cell"] = dict(typed)
            entry["checkpoint"] = dict(typed)
        elif not hb_fresh:
            typed = {
                "value": "unavailable",
                "reason": (f"runner heartbeat is stale (age {hb_age:.0f}s "
                           f"> {stale_after_seconds:.0f}s): its last facts "
                           f"(terminal_state {terminal_state!r}, cell "
                           f"{cell!r}, stage {stage!r}) are history, never "
                           "current claims (finding 212)"),
                "heartbeat_age_seconds": hb_age,
                "last_known": {"cell": cell, "stage": stage,
                               "terminal_state": terminal_state},
            }
            entry["current_cell"] = dict(typed)
            entry["checkpoint"] = dict(typed)
        else:
            entry["current_cell"] = None
            entry["checkpoint"] = None
            entry["not_running_reason"] = (
                "heartbeat is fresh but terminal_state is "
                f"{terminal_state!r}, so no current cell is claimed")

        # GPU utilization/temperature: the runner heartbeat embeds a
        # per-minute nvidia-smi sample for the ASSIGNED GPU taken on its
        # own host (ladder.gpu_telemetry) — the only source that is
        # correct for remote seeds too.
        util = hb.get("gpu_utilization_pct")
        temp = hb.get("gpu_temperature_c")
        if hb_fresh and (util is not None or temp is not None):
            entry["gpu"] = {
                "basis": "observed",
                "source": ("nvidia-smi sampled by the runner heartbeat on "
                           "the assigned host"),
                "source_age_seconds": hb_age,
                "utilization_pct": {"value": util, "unit": "percent",
                                    "horizon": "instant"},
                "temperature_c": {"value": temp, "unit": "celsius",
                                  "horizon": "instant"},
            }
        else:
            entry["gpu"] = {
                "value": "unavailable",
                "reason": ("heartbeat carries no GPU telemetry sample"
                           if hb_fresh else
                           "heartbeat stale: its GPU sample is history, "
                           "not a current reading"),
                "heartbeat_age_seconds": hb_age,
            }

        landed: dict[str, Any] = {}
        for cell_name in seed_cells:
            raw = reader.read_text(
                host, f"{seed_dir}/{cell_name}/cell_record.json")
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except ValueError:
                continue
            if not isinstance(record, dict):
                continue
            # Finding 233: cell records DO carry the executed mode; a
            # record of the other mode under this root is contamination
            # and is never counted as landed evidence for this mode.
            record_mode = record.get("mode")
            if record_mode is not None and record_mode != binding["mode"]:
                rejected_records.append({
                    "seed": seed, "cell": cell_name,
                    "record_mode": record_mode,
                    "expected_mode": binding["mode"],
                    "reason": ("cell record declares another mode under "
                               f"the {binding['mode']} output root; it is "
                               "not evidence for this mode"),
                })
                continue
            duration: Optional[float] = None
            elapsed_field = record.get("elapsed_seconds")
            if isinstance(elapsed_field, (int, float)) \
                    and not isinstance(elapsed_field, bool):
                duration = round(float(elapsed_field), 1)
            else:
                started = _l1_iso(record.get("started_utc"))
                finished = _l1_iso(record.get("finished_utc"))
                if started and finished:
                    duration = round(
                        (finished - started).total_seconds(), 1)
            if duration is not None:
                durations.append(duration)
            finished_at = _l1_iso(record.get("finished_utc"))
            if finished_at is not None and (newest_finished is None
                                            or finished_at > newest_finished):
                newest_finished = finished_at
            best_checkpoint_available = bool(
                record.get("best_model_path")
                and record.get("best_model_sha256"))
            raw_stop_reason = record.get("stop_reason")
            effective_stop_reason = raw_stop_reason
            if (
                raw_stop_reason == "activity_stop_no_eligible_checkpoint"
                and record.get("activity_status") == "active"
                and best_checkpoint_available
            ):
                # Records emitted before the stop-label correction used the
                # same raw label both with and without a retained checkpoint.
                # Preserve that durable fact while exposing its truthful
                # operational interpretation.
                effective_stop_reason = (
                    "activity_stop_after_best_checkpoint")
            landed[cell_name] = {
                "schema": record.get("schema"),
                "stop_reason": raw_stop_reason,
                "effective_stop_reason": effective_stop_reason,
                "termination_cause": record.get("termination_cause"),
                "activity_status": record.get("activity_status"),
                "activity_inactive_cause": record.get(
                    "activity_inactive_cause"),
                "decision_eligible": record.get("decision_eligible"),
                "promotion_eligible": record.get("promotion_eligible"),
                "best_checkpoint_available": best_checkpoint_available,
                "finished_utc": record.get("finished_utc"),
                "duration_seconds": duration,
                "terminal_model_sha256": record.get(
                    "terminal_model_sha256"),
                "mode": record_mode,
                "evidence_class": record.get("evidence_class"),
            }
        entry["landed_cells"] = landed or None
        entry["landed_cell_semantics"] = (
            "decision_eligible means the record may enter the factorial "
            "decision, including a measured inactive outcome; model "
            "viability is reported by activity_status, promotion_eligible "
            "and best_checkpoint_available; stop_reason is the immutable "
            "as-run label and effective_stop_reason disambiguates legacy "
            "activity-stop records"
        )
        entry["records_landed"] = {"value": len(landed),
                                   "of": len(seed_cells),
                                   "unit": "cell_records",
                                   "horizon": "seed",
                                   "mode": binding["mode"],
                                   "output_root": output_root}
        records_total += len(landed)

        # Finding 233: `systemctl show -p NRestarts` answers 0 even for a
        # unit that does not exist, so a restart count without LoadState
        # would render a DIRECT nohup worker as a supervised one. The
        # unit file that WOULD supervise it is named in the remediation.
        unit_file = f"examples/systemd/{binding['unit_template'].format(seed='')}"
        loaded_fn = getattr(reader, "unit_loaded", None)
        unit_loaded = loaded_fn(host, unit) if callable(loaded_fn) else None
        nrestarts = reader.nrestarts(host, unit)
        entry["unit_loaded"] = unit_loaded
        if unit_loaded is False:
            entry["restart_count"] = {
                "value": "unavailable",
                "reason": (f"{unit} is not loaded on {host}: systemd "
                           "reports NRestarts=0 for unknown units, so no "
                           "restart count exists to report"),
                "unit_name": unit,
                "source": f"systemctl --user show {unit} -p LoadState",
            }
            entry["launch_durability"] = {
                "value": "no_unit_loaded",
                "reason": (f"no systemd unit supervises seed {seed} on "
                           f"{host}; a DIRECT (nohup) worker does not "
                           "survive logout/reboot and is never restarted "
                           "automatically"),
                "remediation": (f"install {unit_file} and run "
                                f"systemctl --user enable --now {unit}"),
            }
        else:
            entry["restart_count"] = {
                "value": nrestarts if nrestarts is not None else "unavailable",
                "unit": "systemd_restarts",
                "unit_name": unit,
                "unit_loaded": unit_loaded,
                "source": f"systemctl --user show {unit} -p NRestarts",
                **({} if nrestarts is not None else
                   {"reason": f"NRestarts unreadable for {unit} on {host}"}),
                **({} if unit_loaded is True else
                   {"load_state_note": (
                       "unit load state unknown: this count is only "
                       "meaningful if the unit exists")}),
            }

        eta_elapsed: Optional[float] = None
        eta_reason: Optional[str] = None
        if running_now:
            lock_raw = reader.read_text(
                host, f"{output_root}/{identity}/locks/"
                      f"exclusive_claim.seed{seed}.{cell}.lock")
            acquired = None
            if lock_raw:
                try:
                    acquired = _l1_iso(
                        _as_dict(json.loads(lock_raw)).get("acquired_utc"))
                except ValueError:
                    acquired = None
            if acquired:
                eta_elapsed = max(
                    0.0, round((now - acquired).total_seconds(), 1))
            else:
                eta_reason = (
                    "running cell start time unobservable: the "
                    "exclusive-claim sidecar for the current cell is "
                    "missing or carries no acquired_utc")
        else:
            eta_reason = ("no fresh RUNNING heartbeat names a current "
                          "cell, so there is no current-cell ETA")
        pending_eta[str(seed)] = (eta_elapsed, eta_reason)
        workers[str(seed)] = entry
        worker_states[str(seed)] = {
            "remaining": len(seed_cells) - len(landed),
            "active": running_now,
            "cell": cell if running_now else None,
            "active_eta": None,  # filled once all durations are gathered
        }

    for key, (elapsed, reason) in pending_eta.items():
        eta = _p1lr_cell_eta(durations, elapsed, reason)
        workers[key]["current_cell_eta"] = eta
        worker_states[key]["active_eta"] = eta

    # ── §3: TERMINAL is a fleet fact, and it needs every host readable ──
    experiment_terminal = bool(
        total_cells > 0 and records_total >= total_cells
        and not unreadable_seeds and not running_fresh)
    transition = p1lr_transition_view(
        queue_dir=transition_queue_dir, experiment=str(
            contract.get("experiment") or ""),
        mode=binding["mode"], identity=identity, now=now,
        terminal=experiment_terminal, records_landed=records_total,
        cells_total=total_cells, output_root=output_root,
        contract_path=str(contract_path), contract_sha256=contract_sha,
        terminal_utc=(newest_finished.isoformat()
                      if newest_finished else None),
        enrol=transition_enrol, budget_seconds=transition_budget_seconds)

    if running_fresh:
        state = "active"
        state_basis = (f"{running_fresh} worker(s) RUNNING with runner "
                       f"heartbeat age <= {stale_after_seconds:.0f}s")
    elif experiment_terminal:
        # The defect (2026-08-15): a terminal 16/16 with no successor
        # rendered as quiet inactivity, so fleet idle time after a
        # completed experiment was invisible. It is now a NAMED state.
        # Only a POSITIVE transition verdict may lift it: anything else
        # (including a record that never recorded the terminal result)
        # is un-transitioned, because unproven is not dispatched.
        state = (transition["value"]
                 if transition["value"] in ("transitioned",
                                            "transition_dispatch_in_progress",
                                            "superseded")
                 else "completed_untransitioned")
        if state == "completed_untransitioned":
            state_basis = (
                f"every assigned host was readable and all "
                f"{records_total}/{total_cells} cell records landed, so "
                f"identity {identity} is TERMINAL — and NO successor is "
                f"proven dispatched by the durable transition queue "
                f"({transition['reason']}). This is fleet idle time after "
                "a terminal completion, not healthy inactivity")
        else:
            state_basis = (
                f"identity {identity} is TERMINAL "
                f"({records_total}/{total_cells} records) and the durable "
                f"transition queue reports {state}: {transition['reason']}")
    elif any_fact:
        state = "inactive_or_unknown"
        state_basis = ("no worker is RUNNING with a fresh heartbeat; "
                       "per-worker terminal states carry the facts")
    else:
        state = "unavailable"
        state_basis = ("no worker fact readable on any assigned host "
                       f"under the {binding['mode']} output root "
                       f"{output_root}")
        unavailable.append({"field": field, "reason": state_basis})

    block: dict[str, Any] = {
        "source": "p1lr_factorial",
        "basis": "observed",
        "state": state,
        "state_basis": state_basis,
        "experiment": contract.get("experiment"),
        "asset": contract.get("asset"),
        "mode": binding["mode"],
        "mode_basis": "explicit_validated_parameter",
        "identity": identity,
        "identity_basis": identity_basis,
        "identity_presence": identity_presence,
        "contract_path": str(contract_path),
        "contract_sha256": contract_sha,
        "output_root": output_root,
        "other_mode": binding["other_mode"],
        "other_mode_output_root": binding["other_mode_output_root"],
        "unit_template": binding["unit_template"],
        "heartbeat_schema_expected": P1LR_HEARTBEAT_SCHEMA,
        "heartbeat_mode_expected": binding["heartbeat_mode_expected"],
        "record_mode_expected": binding["record_mode_expected"],
        "evidence_class_expected": binding["evidence_class_expected"],
        "decision_eligible_expected": binding["decision_eligible_expected"],
        "workers": workers,
        "experiment_terminal": experiment_terminal,
        # §3 bullets 1/3/4: the transition is a first-class fact taken
        # from durable records, never from a heartbeat, a shell process,
        # a chat message or operator memory.
        "transition": transition,
        "experiment_eta": _l1_experiment_eta(
            worker_states, durations,
            active_eta_source_label="current_cell_duration_eta"),
    }
    if authority_facts is not None:
        block["runtime_authority"] = authority_facts
    if state == "unavailable":
        # Finding 233: with NO readable fact there is nothing to count.
        # Rendering 0/4 workers and 0/16 records here is exactly the
        # false idle picture the auditor observed over four busy GPUs.
        typed_empty = {
            "value": "unavailable",
            "reason": state_basis,
            "of": None,
        }
        block["workers_running_fresh"] = dict(
            typed_empty, of=len(seeds), unit="workers", horizon="instant")
        block["records_landed"] = dict(
            typed_empty, of=total_cells, unit="cell_records",
            horizon="experiment", mode=binding["mode"],
            output_root=output_root)
    else:
        block["workers_running_fresh"] = {
            "value": running_fresh, "of": len(seeds),
            "unit": "workers", "horizon": "instant"}
        block["records_landed"] = {
            "value": records_total, "of": total_cells,
            "unit": "cell_records", "horizon": "experiment",
            "mode": binding["mode"], "output_root": output_root,
            "fleet_note": (
                "each assigned host's own status run counts its seeds' "
                "records from its LOCAL output root (per-host N/4 per "
                "assigned seed); this run reaches remote seeds only "
                "through the read-only reader, and an unreachable host "
                "renders a typed unavailable worker, never a fabricated "
                "count — the fleet 16 is complete only when every "
                "assigned host was readable at collection time"),
        }
    if rejected_records:
        block["records_rejected_mode_mismatch"] = rejected_records
    other_activity = _p1lr_other_mode_activity(binding, now,
                                               stale_after_seconds)
    if other_activity is not None:
        block["other_mode_activity"] = other_activity

    queue_entry = None
    if state == "active":
        queue_entry = {
            "id": f"p1lr-factorial-{identity}",
            "front": "f1",
            "state": "running",
            "mode": binding["mode"],
            "output_root": output_root,
            "hashes": {"config_sha256": contract_sha},
        }
    elif state == "completed_untransitioned":
        # The successor takes the queue slot the terminal predecessor
        # vacated, so an un-transitioned fleet is never an EMPTY queue.
        queue_entry = _p1lr_transition_queue_entry(
            transition, identity=identity, contract_sha=contract_sha)
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
    p1lr_contract_path: Optional[Path] = None,
    p1lr_identity: Optional[str] = None,
    p1lr_reader: Optional[Any] = None,
    p1lr_local_hostname: Optional[str] = None,
    p1lr_stale_after_seconds: float = 900.0,
    p1lr_now_fn: Optional[Callable[[], datetime]] = None,
    p1lr_mode: str = "screen",
    p1lr_transition_queue_dir: Optional[Path] = None,
    p1lr_transition_enrol: bool = True,
    p1lr_transition_budget_seconds: Optional[float] = None,
    p1lr_runtime_authority_path: Optional[Path] = None,
    p1lr_identity_presence_fn: Optional[
        Callable[[Optional[str], str], bool]] = None,
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

    # ── Front 1: the ACTIVE work — P1LR factorial screen (order
    # 2026-08-11 §7.7), then the completed L1 factorial (finding 204) ──
    f1: dict[str, Any] = {
        "basis": "observed",
        "current_work": {
            "state": "unavailable",
            "reason": "active Front-1 sources have not been collected yet",
        },
    }
    fronts["f1_optimization"] = f1
    p1lr_queue_entry: Optional[dict[str, Any]] = None
    if p1lr_contract_path is not None:
        p1lr_active_reader = (p1lr_reader or l1_reader or DefaultL1Reader(
            local_hostname=p1lr_local_hostname or l1_local_hostname))
        p1lr_block, p1lr_unavailable, p1lr_queue_entry = \
            collect_p1lr_factorial(
                contract_path=p1lr_contract_path,
                reader=p1lr_active_reader,
                identity=p1lr_identity,
                local_hostname=p1lr_local_hostname or l1_local_hostname,
                stale_after_seconds=p1lr_stale_after_seconds,
                now_fn=p1lr_now_fn or l1_now_fn,
                mode=p1lr_mode,
                transition_queue_dir=p1lr_transition_queue_dir,
                transition_enrol=p1lr_transition_enrol,
                transition_budget_seconds=p1lr_transition_budget_seconds,
                runtime_authority_path=p1lr_runtime_authority_path,
                identity_presence_fn=p1lr_identity_presence_fn,
            )
        f1["active_p1lr_factorial"] = p1lr_block
        unavailable.extend(p1lr_unavailable)
        if p1lr_block.get("state") not in ("unavailable", "refused"):
            register("p1lr_factorial", str(p1lr_contract_path), None)
    else:
        f1["active_p1lr_factorial"] = {
            "source": "p1lr_factorial",
            "state": "unavailable",
            "reason": ("p1lr factorial source not configured "
                       "(no contract path)"),
        }
        unavailable.append(
            {"field": "f1_optimization.active_p1lr_factorial",
             "reason": "p1lr factorial source not configured"})

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

    p1lr_state = _as_dict(f1.get("active_p1lr_factorial")).get("state")
    l1_state = _as_dict(f1.get("active_l1_factorial")).get("state")
    if p1lr_state == "active":
        active = _as_dict(f1["active_p1lr_factorial"])
        f1["current_work"] = {
            "state": "active",
            "source_key": "active_p1lr_factorial",
            "experiment": active.get("experiment"),
            "identity": active.get("identity"),
            "mode": active.get("mode"),
        }
        legacy = _as_dict(f1.get("active_l1_factorial"))
        if l1_state != "active":
            legacy["role"] = "history"
            legacy["compatibility_note"] = (
                "active_l1_factorial is a legacy schema key; current_work "
                "is authoritative and this block is not running")
    elif l1_state == "active":
        active = _as_dict(f1["active_l1_factorial"])
        f1["current_work"] = {
            "state": "active",
            "source_key": "active_l1_factorial",
            "experiment": active.get("experiment"),
            "identity": active.get("identity"),
            "mode": active.get("mode"),
        }
    else:
        f1["current_work"] = {
            "state": "none_running",
            "source_states": {
                "active_p1lr_factorial": p1lr_state,
                "active_l1_factorial": l1_state,
            },
        }

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
    if (f1["active_p1lr_factorial"].get("state") in ("unavailable",
                                                     "refused")
            and f1["active_l1_factorial"].get("state") == "unavailable"
            and "doin_campaign_history" not in f1):
        unavailable.append(
            {"field": "f1_optimization",
             "reason": "neither the active P1LR screen, the L1 factorial "
                       "nor the paused campaign history is readable"})

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
    if p1lr_queue_entry is not None:
        # Order 2026-08-11 §7.7: the RUNNING P1LR screen leads the
        # executable queue; the completed L1 factorial is history only
        # (no fresh RUNNING heartbeat -> no queue entry at all).
        # Order 2026-08-15 §3: when the P1LR run itself is TERMINAL and
        # un-transitioned, its SUCCESSOR takes that slot — an idle fleet
        # after a completion must never look like an empty queue.
        queue.append(p1lr_queue_entry)
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
    # Finding 214: the IBKR L1 item derives from execution heartbeat and
    # journal facts. A broker hold renders operational-but-held with its
    # exact reason and owner action; the old hardcoded development
    # dependency ('write adapter + preflight') is retired.
    ibkr_item, ibkr_unavailable = _ibkr_l1_queue_entry(
        execution_state_dir, _time.time())
    queue.append(ibkr_item)
    if ibkr_unavailable:
        unavailable.append(ibkr_unavailable)
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
    parser.add_argument(
        "--p1lr-contract", type=Path,
        default=repo / "examples/config/phase_3_eth_sac_dynamics/"
                       "p1_difficulty_lr_factorial_v2.json",
        help="P1 difficulty x P1 LR factorial contract; defaults to the "
             "corrected-observation v2 Front-1 source (pass an explicit "
             "path when inspecting a historical v1 run)")
    parser.add_argument("--no-p1lr", action="store_true",
                        help="disable the P1LR factorial source")
    parser.add_argument("--p1lr-identity", default=None,
                        help="active P1LR experiment identity; discovered "
                             "from the newest local runner heartbeat under "
                             "the SELECTED mode's root when omitted")
    parser.add_argument("--p1lr-mode", choices=list(P1LR_MODES),
                        default="screen",
                        help="P1LR execution mode to read (finding 233): "
                             "output root, systemd unit, expected "
                             "heartbeat/record mode and total cells all "
                             "derive from it — 'screen' reads "
                             "output_root, 'decision' reads "
                             "decision_run.output_root; an identity "
                             "belonging to the other mode's root is a "
                             "typed refusal, never a rendered zero")
    parser.add_argument(
        "--p1lr-runtime-authority", type=Path,
        help="optional durable authority record binding the selected P1LR "
             "identity/contract to an incident-specific systemd unit pattern")
    parser.add_argument(
        "--transition-queue-dir", type=Path,
        default=Path.home() / ".local/state/agent-multi/"
                              "experiment-transition-queue",
        help="durable terminal-to-next-job queue (order 2026-08-15 §3): "
             "a TERMINAL experiment renders completed_untransitioned "
             "until these records prove a successor dispatched; NEVER "
             "inside a run's output root")
    parser.add_argument("--no-transition-queue", action="store_true",
                        help="do not read or enrol durable transition "
                             "records (a terminal experiment then always "
                             "renders completed_untransitioned)")
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
        p1lr_contract_path=None if args.no_p1lr else args.p1lr_contract,
        p1lr_identity=args.p1lr_identity,
        p1lr_mode=args.p1lr_mode,
        p1lr_runtime_authority_path=args.p1lr_runtime_authority,
        p1lr_transition_queue_dir=(None if args.no_transition_queue
                                   else args.transition_queue_dir),
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
