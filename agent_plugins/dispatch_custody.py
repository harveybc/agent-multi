"""Durable single-use execution custody (DATA-SOTA-358).

A dispatched one-shot execution (e.g. the transfer-loader smoke) must
be provably single-use: the run is RESERVED atomically in a durable
ledger BEFORE the model is constructed, evidence goes to a UNIQUE
non-clobbering path, and a completed or uncertain prior attempt
refuses another execution. Presentation is separated from execution: a
renderer reads completed evidence freely without touching any model.

Ledger records live outside the public repository
(``~/.local/state/agent-multi/dispatch_ledger``) and are keyed by the
sha256 of (dispatch id, sealed generation digest, effective
architecture digest, data digest, code identity). State transitions:
``reserved -> running -> completed | failed_before_forward |
interrupted``. Only an explicit ``failed_before_forward`` prior state
(certainly no model forward happened) permits a retry; ``reserved``,
``running``, ``interrupted`` and ``completed`` all refuse — an
uncertain attempt is treated as spent.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from agent_plugins.branch_pretraining import (_fsync_write_bytes,
                                              sha256_obj)

DEFAULT_LEDGER_ROOT = (Path.home()
                       / ".local/state/agent-multi/dispatch_ledger")
RETRYABLE_STATES = ("failed_before_forward",)
TERMINAL_UNCERTAIN = ("reserved", "running", "interrupted")


class ExecutionCustodyError(RuntimeError):
    """Typed refusal: the dispatch identity is already spent, the
    output path is unsafe, or the ledger is inconsistent."""


def dispatch_key(*, dispatch_id: str, generation_digest: str,
                 architecture_digest: str, data_digest: str,
                 code_identity: dict[str, Any]) -> str:
    return sha256_obj({"dispatch_id": dispatch_id,
                       "generation_digest": generation_digest,
                       "architecture_digest": architecture_digest,
                       "data_digest": data_digest,
                       "code_identity": code_identity})


class DispatchLedger:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else DEFAULT_LEDGER_ROOT
        self.root.mkdir(parents=True, exist_ok=True)

    def _record_path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def read(self, key: str) -> dict[str, Any] | None:
        path = self._record_path(key)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def reserve(self, key: str, *, identity: dict[str, Any],
                output_path: Path) -> None:
        """Atomic single-use reservation BEFORE model construction.
        O_CREAT|O_EXCL makes concurrent reservation race-free: exactly
        one caller wins."""
        output_path = Path(output_path)
        if output_path.is_symlink() or output_path.parent.is_symlink():
            raise ExecutionCustodyError(
                "output path or its parent is a symlink — refused")
        if output_path.exists():
            raise ExecutionCustodyError(
                f"output path already exists (no-clobber): "
                f"{output_path.name}")
        existing = self.read(key)
        if existing is not None:
            state = existing.get("state")
            if state in RETRYABLE_STATES:
                pass  # certainly no forward happened; retry permitted
            elif state == "completed":
                raise ExecutionCustodyError(
                    f"dispatch identity already COMPLETED at "
                    f"{existing.get('completed_at') or 'unknown'} — a "
                    f"second execution refuses (DATA-SOTA-358)")
            else:
                raise ExecutionCustodyError(
                    f"dispatch identity in UNCERTAIN state "
                    f"{state!r} — treated as spent; a second "
                    f"execution refuses (DATA-SOTA-358)")
            os.replace(self._record_path(key),
                       self._record_path(key + ".retired-"
                                         + str(existing.get(
                                             "attempt", 0))))
        record = {"schema": "agent_multi.dispatch_ledger.v1",
                  "key": key, "state": "reserved",
                  "attempt": (existing or {}).get("attempt", 0) + 1,
                  "identity": identity,
                  "output_path": f"external:{output_path.name}"}
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            fd = os.open(self._record_path(key), flags, 0o600)
        except FileExistsError as exc:
            raise ExecutionCustodyError(
                "concurrent reservation lost: another caller holds "
                "this dispatch identity (DATA-SOTA-358)") from exc
        try:
            payload = json.dumps(record, indent=1).encode()
            os.write(fd, payload)
            os.fsync(fd)
        finally:
            os.close(fd)

    def transition(self, key: str, state: str,
                   extra: dict[str, Any] | None = None) -> None:
        record = self.read(key)
        if record is None:
            raise ExecutionCustodyError(
                f"no ledger record for {key[:12]}")
        record["state"] = state
        if extra:
            record.update(extra)
        _fsync_write_bytes(self._record_path(key),
                           json.dumps(record, indent=1).encode())

    def record_protocol_deviation(self, key: str,
                                  facts: dict[str, Any]) -> None:
        """Preserve a historical deviation with ONLY the facts actually
        known — never invented metrics."""
        record = {"schema": "agent_multi.dispatch_ledger.v1",
                  "key": key, "state": "DISCLOSED_PROTOCOL_DEVIATION",
                  "facts": facts}
        path = self._record_path(key)
        if path.exists():
            raise ExecutionCustodyError(
                "deviation record already present; never overwrite "
                "history")
        _fsync_write_bytes(path, json.dumps(record, indent=1).encode())
