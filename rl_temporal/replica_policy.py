"""Fail-closed lane G replica admission, without importing the ML stack.

Run `python -m rl_temporal.replica_policy --help` for local operator commands.
The state root must be shared by all runners of a campaign. This is an
operator-controlled experiment ledger, not a distributed security authority.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile


SCHEMA = "rl_temporal.replica_policy.v1"
ARMS = ("RL-S0", "RL-S1", "RL-D0", "RL-D1")
SEEDS = (101, 202, 303)
REASONS = ("published_protocol", "final_contrast", "stochastic_variability", "nondeterminism_diagnosis")
ALLOWED = {"ALLOW_SCREENING", "ALLOW_JUSTIFIED_REPLICA"}


def declaration(seed):
    if type(seed) is not int or seed not in SEEDS:
        status = "REJECTED_FOURTH_REPLICA" if seed == 404 else "REJECTED_UNDECLARED_SEED"
    else:
        status = "ALLOW_SCREENING" if seed == SEEDS[0] else "HELD_EXTRA_REPLICA"
    return {"schema": SCHEMA, "status": status, "screening_seed": SEEDS[0],
            "maximum_replicas": 3, "requires_persisted_authorization": seed != SEEDS[0]}


def validate_matrix_seeds(seeds):
    if not seeds or len(seeds) > 3 or any(type(s) is not int for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("declare one screening seed, at most three unique paired seeds")
    if 404 in seeds:
        raise ValueError("REJECTED_FOURTH_REPLICA: historical lane G seed 404 is not admissible")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _now():
    return datetime.now(timezone.utc).isoformat()


class ReplicaPolicy:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.state_path = self.root / "REPLICA_POLICY.json"

    @contextmanager
    def _locked(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".replica_policy.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = {"schema": SCHEMA, "seed_plan": list(SEEDS), "authorizations": {}, "claims": {}}
            if self.state_path.exists():
                state = json.loads(self.state_path.read_text())
                if (not isinstance(state, dict) or state.get("schema") != SCHEMA or
                        state.get("seed_plan") != list(SEEDS) or
                        not isinstance(state.get("claims"), dict) or
                        not isinstance(state.get("authorizations"), dict)):
                    raise ValueError("invalid replica policy state")
            yield state

    def _save(self, state):
        # Atomic replacement under flock: no partially-written admission state.
        fd, name = tempfile.mkstemp(prefix=".replica-", dir=self.root)
        try:
            with os.fdopen(fd, "w") as stream:
                stream.write(_canonical(state) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, self.state_path)
            directory_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def _identity(self, cfg, out):
        arm, seed = cfg.get("arm"), cfg.get("train_seed")
        if arm not in ARMS or type(seed) is not int or cfg.get("eval_seed") != seed:
            raise ValueError("known arm and identical integer train/eval seeds required")
        out = Path(out).resolve()
        if out == self.root or not out.is_relative_to(self.root):
            raise ValueError("cell output must be within the shared campaign policy root")
        digest = hashlib.sha256(_canonical(cfg).encode()).hexdigest()
        return f"{arm}:{seed}", digest, out

    def _existing(self, slot, out):
        arm, seed = slot.split(":")
        # Preserve legacy results without calling them newly validated evidence.
        candidates = {out, self.root / f"{arm}_seed{seed}"}
        for directory in sorted(candidates):
            if (directory / "RESULT.json").exists():
                return {"status": "PRESERVED_EXISTING_RESULT", "path": str(directory / "RESULT.json")}
        # Legacy output names/device directories are not scientific identities.
        for path in sorted(self.root.rglob("RESULT.json")):
            record = json.loads(path.read_text())
            if not isinstance(record, dict) or record.get("arm") not in ARMS or type(record.get("seed")) is not int:
                raise ValueError(f"cannot establish historical result identity: {path}")
            if record["arm"] == arm and record["seed"] == int(seed):
                return {"status": "PRESERVED_EXISTING_RESULT", "path": str(path)}
        for path in sorted(self.root.rglob("heartbeat.json")):
            heartbeat = json.loads(path.read_text())
            if not isinstance(heartbeat, dict):
                raise ValueError(f"cannot establish historical attempt identity: {path}")
            if heartbeat.get("arm") == arm and heartbeat.get("train_seed") == int(seed):
                return {"status": "HELD_EXISTING_ATTEMPT", "path": str(path)}
        for directory in sorted(candidates):
            if directory.exists() and any(p.name != "run.log" for p in directory.iterdir()):
                return {"status": "HELD_EXISTING_ATTEMPT", "path": str(directory)}
        return None

    def authorize(self, cfg, out, *, reason_code, reason, declared_by):
        slot, digest, out = self._identity(cfg, out)
        if cfg["train_seed"] not in SEEDS[1:]:
            raise ValueError("only the second/third declared replica can be authorized")
        if reason_code not in REASONS or not isinstance(reason, str) or not reason.strip() or not isinstance(declared_by, str) or not declared_by.strip():
            raise ValueError("predeclared reason code, justification and owner required")
        with self._locked() as state:
            if slot in state["claims"] or self._existing(slot, out):
                raise ValueError("cannot authorize after an attempt or result exists")
            if slot in state["authorizations"]:
                raise ValueError("authorization already persisted; cannot rewrite history")
            record = {"config_sha256": digest, "reason_code": reason_code,
                      "reason": reason.strip(), "declared_by": declared_by.strip(), "declared_at": _now()}
            state["authorizations"][slot] = record
            self._save(state)
            return record

    def admit(self, cfg, out, *, claim=False):
        slot, digest, out = self._identity(cfg, out)
        with self._locked() as state:
            existing = self._existing(slot, out)
            if existing:
                return existing
            if slot in state["claims"]:
                return {"status": "HELD_EXISTING_ATTEMPT", "slot": slot}
            decision = declaration(cfg["train_seed"])
            authorization = state["authorizations"].get(slot)
            if decision["status"] == "HELD_EXTRA_REPLICA" and authorization:
                if (not isinstance(authorization, dict) or
                        authorization.get("reason_code") not in REASONS or
                        not all(isinstance(authorization.get(k), str) and authorization[k].strip()
                                for k in ("reason", "declared_by", "declared_at"))):
                    raise ValueError("invalid persisted replica justification")
                declared = datetime.fromisoformat(authorization["declared_at"])
                if declared.tzinfo is None or declared > datetime.now(timezone.utc):
                    raise ValueError("justification must precede admission")
                if authorization.get("config_sha256") == digest:
                    decision["status"] = "ALLOW_JUSTIFIED_REPLICA"
            decision.update(slot=slot, config_sha256=digest)
            if claim and decision["status"] in ALLOWED:
                state["claims"][slot] = {"config_sha256": digest, "output_dir": str(out),
                                         "claimed_at": _now(), "authorization": authorization}
                self._save(state)
            return decision


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "authorize"))
    parser.add_argument("--cell", required=True)
    parser.add_argument("--root", required=True, help="Shared campaign output/state root")
    parser.add_argument("--out", required=True)
    parser.add_argument("--reason-code", choices=REASONS)
    parser.add_argument("--reason")
    parser.add_argument("--declared-by")
    args = parser.parse_args(argv)
    try:
        policy = ReplicaPolicy(args.root)
        cfg = json.loads(Path(args.cell).read_text())
        if args.action == "authorize":
            result = policy.authorize(cfg, args.out, reason_code=args.reason_code,
                                      reason=args.reason, declared_by=args.declared_by)
            print(_canonical(result))
            return 0
        result = policy.admit(cfg, args.out)
        print(_canonical(result))
        return 0 if result["status"] in ALLOWED else 3
    except (OSError, ValueError, TypeError) as exc:
        print(_canonical({"status": "REFUSED_POLICY_ERROR", "error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
