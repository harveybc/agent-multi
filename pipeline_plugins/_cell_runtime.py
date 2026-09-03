"""Observable, resumable CELL runtime for long SAC trainings
(Musashi correction 3, 2026-09-03; PERMANENT order @95e088da).

The nested trainer, when handed ``cell_runtime_dir``:

* writes a machine-readable heartbeat/status EVERY epoch (current
  epoch, completed/total, elapsed, median/p90 epoch duration, ETA
  including pessimistic, patience state, best composite, last durable
  artifact) — no process attachment needed;
* persists a RESUME BUNDLE every ``resume_checkpoint_every_epochs``
  epochs and at every improvement: model (policy + critics +
  optimizers via the SB3 archive), replay buffer, counters
  (num_timesteps, _n_updates, episode counters), RNG states (torch
  CPU+CUDA, numpy, python), patience/early-stop state, and the full
  evaluation histories — everything Musashi's correction names;
* on ``resume_from_cell_runtime`` restores ALL of the above exactly
  and continues from the next epoch. The environment stream restarts
  at an episode boundary (declared semantic: backtrader env state is
  not snapshotted; every LEARNING state listed above is exact).

Every write is atomic (tmp + fsync + rename + dir fsync)."""
from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path
from typing import Any

STATE_NAME = "resume_state.json"
MODEL_NAME = "resume_model.zip"
REPLAY_NAME = "resume_replay.pkl"
RNG_NAME = "resume_rng.pt"
STATUS_NAME = "status.json"


class CellRuntimeError(RuntimeError):
    """Typed refusal from the observable cell runtime."""


def _atomic_write(path: Path, payload: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)
    dfd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


class CellRuntime:
    """Per-attempt observable runtime context."""

    def __init__(self, config: dict, max_epochs: int):
        raw = config.get("cell_runtime_dir")
        self.enabled = bool(raw)
        self.dir = Path(str(raw)) if raw else None
        self.every = int(config.get(
            "resume_checkpoint_every_epochs", 5) or 5)
        self.max_epochs = int(max_epochs)
        self.epoch_durations: list[float] = []
        self.wall_started = time.time()
        self.resumed_elapsed = 0.0
        if self.enabled:
            self.dir.mkdir(parents=True, exist_ok=True)

    # ---- status / heartbeat -------------------------------------- #
    def heartbeat(self, *, epoch: int, best_composite: float,
                  no_improve: int, patience: int,
                  stop_reason: str | None,
                  last_artifact: str | None,
                  extra: dict | None = None) -> None:
        if not self.enabled:
            return
        durations = sorted(self.epoch_durations)
        eta = None
        remaining = max(0, self.max_epochs - epoch)
        if durations:
            median = durations[len(durations) // 2]
            p90 = durations[min(len(durations) - 1,
                                int(len(durations) * 0.9))]
            eta = {"median_epoch_s": round(median, 1),
                   "p90_epoch_s": round(p90, 1),
                   "remaining_epochs_max": remaining,
                   "eta_median_s": round(median * remaining, 1),
                   "eta_p90_s": round(p90 * remaining, 1),
                   "eta_pessimistic_s": round(
                       durations[-1] * remaining, 1),
                   "note": "remaining assumes the max-epoch budget; "
                           "early stop can only shorten it"}
        status = {
            "schema": "agent_multi.sac_cell_runtime_status.v1",
            "timestamp": time.time(),
            "epoch_completed": epoch,
            "max_epochs": self.max_epochs,
            "elapsed_s": round(time.time() - self.wall_started
                               + self.resumed_elapsed, 1),
            "eta": eta,
            "patience": {"no_improve": no_improve,
                         "budget": patience},
            "best_composite": best_composite,
            "stop_reason": stop_reason,
            "last_durable_artifact": last_artifact,
            **(extra or {}),
        }
        _atomic_write(self.dir / STATUS_NAME,
                      json.dumps(status, indent=1,
                                 default=str).encode())

    # ---- resume bundle ------------------------------------------- #
    def write_bundle(self, *, epoch: int, model: Any,
                     state: dict) -> str:
        if not self.enabled:
            return ""
        import random

        import numpy as np
        import torch
        model.save(str(self.dir / MODEL_NAME))
        if getattr(model, "replay_buffer", None) is not None:
            model.save_replay_buffer(str(self.dir / REPLAY_NAME))
        rng = {"torch": torch.get_rng_state(),
               "numpy": np.random.get_state(),
               "python": random.getstate()}
        if torch.cuda.is_available():
            rng["torch_cuda"] = torch.cuda.get_rng_state_all()
        buf = io.BytesIO()
        torch.save(rng, buf)
        _atomic_write(self.dir / RNG_NAME, buf.getvalue())
        payload = {
            "schema": "agent_multi.sac_cell_resume_state.v1",
            "epoch": int(epoch),
            "elapsed_s": round(time.time() - self.wall_started
                               + self.resumed_elapsed, 1),
            "num_timesteps": int(getattr(model, "num_timesteps", 0)),
            "n_updates": int(getattr(model, "_n_updates", 0) or 0),
            "episode_num": int(getattr(model, "_episode_num", 0)
                               or 0),
            "replay_transitions": (
                int(model.replay_buffer.size())
                if getattr(model, "replay_buffer", None) is not None
                else None),
            **state,
        }
        _atomic_write(self.dir / STATE_NAME,
                      json.dumps(payload, indent=1,
                                 default=str).encode())
        return str(self.dir / STATE_NAME)

    # ---- restore ------------------------------------------------- #
    def read_state(self) -> dict | None:
        if not self.enabled:
            return None
        path = self.dir / STATE_NAME
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def restore_into(self, model: Any, saved: dict, *,
                     expected_config_sha256: str) -> dict:
        """Exact-state restoration. Refuses on identity drift."""
        import random

        import numpy as np
        import torch
        if saved.get("config_sha256") != expected_config_sha256:
            raise CellRuntimeError(
                "resume REFUSED: the effective config digest changed "
                f"({saved.get('config_sha256', '')[:12]}… vs "
                f"{expected_config_sha256[:12]}…) — a resumed attempt "
                "must execute the identical cell")
        model.set_parameters(str(self.dir / MODEL_NAME),
                             exact_match=True)
        replay = self.dir / REPLAY_NAME
        if replay.exists():
            model.load_replay_buffer(str(replay))
        model.num_timesteps = int(saved["num_timesteps"])
        model._n_updates = int(saved["n_updates"])
        model._episode_num = int(saved.get("episode_num", 0))
        rng = torch.load(str(self.dir / RNG_NAME),
                         weights_only=False)
        torch.set_rng_state(rng["torch"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
        if "torch_cuda" in rng and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["torch_cuda"])
        self.resumed_elapsed = float(saved.get("elapsed_s", 0.0))
        return {"resumed_from_epoch": int(saved["epoch"]),
                "start_epoch": int(saved["epoch"]) + 1,
                "restored": ["model+optimizers", "replay_buffer",
                             "num_timesteps", "n_updates",
                             "episode_num", "rng(torch,cuda,numpy,"
                             "python)", "patience", "histories"],
                "environment_stream": (
                    "restarts at an episode boundary — declared, "
                    "not snapshotted")}
