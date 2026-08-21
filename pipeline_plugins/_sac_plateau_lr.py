"""Epoch-level Reduce-on-Plateau learning-rate controller for SAC.

Order: MUSASHI_TO_GENERAL_SATOSHI_SAC_PLATEAU_LR_AND_LONG_HORIZON_ORDER_2026_08_21 §3.

Contract, stated once:

- The controller is OPTIONAL and separately versioned. When it is not
  configured the training path is byte/decision identical to fixed LR.
- It is driven ONLY by the easy checkpoint monitor scalar at epoch
  boundaries. It never receives a summary dict, a candidate-fitness
  ranking, or any test-split fact — ``observe`` accepts exactly one
  scalar and nothing else, so test facts are structurally inaccessible.
- Every contract number is explicit. There are no library defaults to
  inherit: factor, LR patience, minimum LR, improvement threshold
  (min-delta) and cooldown are all required constructor arguments.
- A reduction updates EVERY intended SAC optimizer explicitly (actor,
  critic and — when entropy is learned — the entropy-coefficient
  optimizer) AND replaces ``model.lr_schedule`` with a constant, because
  SB3's ``train()`` re-applies ``lr_schedule`` to those optimizers on
  every call; editing param groups alone would be silently undone.
- A reduction never resets and never masquerades as a monitor
  improvement: the controller keeps its own bookkeeping and does not
  touch checkpoint/patience state; its internal best is NOT reset by a
  reduction.
- State serialization (``state_dict``/``load_state_dict``) exists for
  AUDIT DERIVABILITY and the per-epoch sidecar only. The executing
  pipeline NEVER loads a sidecar, so plateau runs are NON-RESUMABLE and
  fail closed: warm-starting from a checkpoint that has a plateau
  sidecar beside it is refused, because a fresh controller would
  silently discard best value, bad-epoch count, cooldown, reduction
  count and current LR while appearing to continue
  (AUD-F1-20260821-PLR-01).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

CONTRACT_ID = "agent_multi.sac_plateau_lr.v1"

# The three optimizers this contract governs on an SB3 SAC model, in
# declaration order. ent_coef_optimizer is absent when entropy is fixed;
# that absence is recorded, never silently ignored.
GOVERNED_OPTIMIZERS = ("actor", "critic", "ent_coef")


class SacPlateauLrError(ValueError):
    """Typed refusal for malformed plateau-LR contracts or facts."""


def _require_int(name: str, value: Any, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SacPlateauLrError(
            f"{name} must be a non-boolean integer, got {value!r}")
    if value < minimum:
        raise SacPlateauLrError(
            f"{name} must be >= {minimum}, got {value!r}")
    return int(value)


def _require_finite_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SacPlateauLrError(
            f"{name} must be a finite number, got {value!r}")
    out = float(value)
    if out != out or out in (float("inf"), float("-inf")):
        raise SacPlateauLrError(
            f"{name} must be finite, got {value!r}")
    return out


class ConstantLr:
    """Picklable constant schedule installed on the model at a reduction."""

    def __init__(self, lr: float):
        self.lr = float(lr)

    def __call__(self, _progress_remaining: float) -> float:
        return self.lr

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return f"ConstantLr({self.lr!r})"


def observed_sac_lrs(model: Any) -> Dict[str, Optional[float]]:
    """Read the CURRENT per-optimizer learning rates off a SAC model.

    Absent optimizers report None — never zero (an absence is not a
    measurement). This is the per-epoch report fact ordered in §2.
    """
    out: Dict[str, Optional[float]] = {}
    for name in GOVERNED_OPTIMIZERS:
        opt = _optimizer_for(model, name)
        if opt is None:
            out[name] = None
            continue
        groups = getattr(opt, "param_groups", None)
        out[name] = float(groups[0]["lr"]) if groups else None
    return out


def _optimizer_for(model: Any, name: str) -> Any:
    if name == "actor":
        return getattr(getattr(model, "actor", None), "optimizer", None)
    if name == "critic":
        return getattr(getattr(model, "critic", None), "optimizer", None)
    if name == "ent_coef":
        return getattr(model, "ent_coef_optimizer", None)
    raise SacPlateauLrError(f"unknown governed optimizer {name!r}")


def apply_lr_to_sac(model: Any, new_lr: float) -> Dict[str, Any]:
    """Set ``new_lr`` on every governed optimizer AND the model schedule.

    Returns the applied-policy record: which optimizers were updated,
    each one's previous LR, and confirmation the schedule was replaced.
    Actor and critic optimizers are REQUIRED — a SAC model without them
    is refused rather than partially updated.
    """
    lr = _require_finite_float("new_lr", new_lr)
    if lr <= 0.0:
        raise SacPlateauLrError(f"new_lr must be > 0, got {new_lr!r}")
    updated: List[str] = []
    absent: List[str] = []
    old_lrs: Dict[str, Optional[float]] = {}
    for name in GOVERNED_OPTIMIZERS:
        opt = _optimizer_for(model, name)
        if opt is None:
            if name in ("actor", "critic"):
                raise SacPlateauLrError(
                    f"SAC model has no {name} optimizer; refusing a "
                    "partial learning-rate update")
            absent.append(name)
            old_lrs[name] = None
            continue
        groups = getattr(opt, "param_groups", None)
        if not groups:
            raise SacPlateauLrError(
                f"{name} optimizer has no param_groups; refusing a "
                "partial learning-rate update")
        old_lrs[name] = float(groups[0]["lr"])
        for group in groups:
            group["lr"] = lr
        updated.append(name)
    # SB3 SAC.train() calls _update_learning_rate(...) with
    # lr_schedule(progress) on every train step batch; without replacing
    # the schedule the param-group edit above is reverted immediately.
    model.lr_schedule = ConstantLr(lr)
    return {
        "optimizers_updated": updated,
        "optimizers_absent": absent,
        "old_lrs": old_lrs,
        "new_lr": lr,
        "lr_schedule_replaced": True,
    }


class SacPlateauLrController:
    """Reduce-on-plateau over the easy checkpoint monitor (mode: max)."""

    def __init__(
        self,
        *,
        factor: float,
        lr_patience: int,
        min_lr: float,
        threshold: float,
        cooldown: int,
        start_epoch: int,
        initial_lr: float,
    ):
        self.factor = _require_finite_float("factor", factor)
        if not (0.0 < self.factor < 1.0):
            raise SacPlateauLrError(
                f"factor must be in (0, 1), got {factor!r}")
        self.lr_patience = _require_int("lr_patience", lr_patience, minimum=1)
        self.min_lr = _require_finite_float("min_lr", min_lr)
        if self.min_lr <= 0.0:
            raise SacPlateauLrError(f"min_lr must be > 0, got {min_lr!r}")
        self.threshold = _require_finite_float("threshold", threshold)
        if self.threshold < 0.0:
            raise SacPlateauLrError(
                f"threshold must be >= 0, got {threshold!r}")
        self.cooldown = _require_int("cooldown", cooldown, minimum=0)
        self.start_epoch = _require_int("start_epoch", start_epoch, minimum=0)
        self.current_lr = _require_finite_float("initial_lr", initial_lr)
        if self.current_lr <= 0.0:
            raise SacPlateauLrError(
                f"initial_lr must be > 0, got {initial_lr!r}")
        if self.current_lr < self.min_lr:
            raise SacPlateauLrError(
                f"initial_lr {initial_lr!r} is below min_lr {min_lr!r}")
        self.best_value: Optional[float] = None
        self.num_bad_epochs = 0
        self.cooldown_remaining = 0
        self.reductions_total = 0
        self.last_epoch = 0

    def contract(self) -> Dict[str, Any]:
        return {
            "contract_id": CONTRACT_ID,
            "mode": "max",
            "driven_by": "easy_checkpoint_monitor",
            "factor": self.factor,
            "lr_patience": self.lr_patience,
            "min_lr": self.min_lr,
            "threshold": self.threshold,
            "cooldown": self.cooldown,
            "start_epoch": self.start_epoch,
            "governed_optimizers": list(GOVERNED_OPTIMIZERS),
        }

    def state_dict(self) -> Dict[str, Any]:
        """Serialization for audit derivability and the sidecar ONLY.

        No executing path loads this back into a training run
        (PLR-01): plateau runs are non-resumable and refuse
        warm-starts that look like resumes.
        """
        return {
            "contract_id": CONTRACT_ID,
            "current_lr": self.current_lr,
            "best_value": self.best_value,
            "num_bad_epochs": self.num_bad_epochs,
            "cooldown_remaining": self.cooldown_remaining,
            "reductions_total": self.reductions_total,
            "last_epoch": self.last_epoch,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if state.get("contract_id") != CONTRACT_ID:
            raise SacPlateauLrError(
                "refusing scheduler state from contract "
                f"{state.get('contract_id')!r}; this controller is "
                f"{CONTRACT_ID}")
        self.current_lr = _require_finite_float(
            "current_lr", state["current_lr"])
        best = state["best_value"]
        self.best_value = (
            None if best is None else _require_finite_float(
                "best_value", best))
        self.num_bad_epochs = _require_int(
            "num_bad_epochs", state["num_bad_epochs"], minimum=0)
        self.cooldown_remaining = _require_int(
            "cooldown_remaining", state["cooldown_remaining"], minimum=0)
        self.reductions_total = _require_int(
            "reductions_total", state["reductions_total"], minimum=0)
        self.last_epoch = _require_int(
            "last_epoch", state["last_epoch"], minimum=0)

    def observe(
        self,
        *,
        epoch: int,
        monitor_value: float,
        apply_fn: Callable[[float], Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Consume one epoch-boundary monitor scalar; maybe reduce.

        ``apply_fn(new_lr)`` performs the actual optimizer update (in
        production: ``apply_lr_to_sac``) and returns its policy record.
        The signature is closed: there is no argument through which a
        summary, ranking or test fact can reach this controller.
        """
        epoch = _require_int("epoch", epoch, minimum=1)
        if epoch <= self.last_epoch:
            raise SacPlateauLrError(
                f"epoch {epoch} not after last observed {self.last_epoch}; "
                "scheduler state would be corrupted")
        value = _require_finite_float("monitor_value", monitor_value)
        self.last_epoch = epoch

        improved = (
            self.best_value is None
            or value > self.best_value + self.threshold
        )
        if improved:
            self.best_value = value
            self.num_bad_epochs = 0
        in_warmup = epoch <= self.start_epoch
        in_cooldown = self.cooldown_remaining > 0
        if in_cooldown:
            self.cooldown_remaining -= 1
        elif not improved and not in_warmup:
            self.num_bad_epochs += 1

        reduced = False
        reason = "improved" if improved else (
            "warmup" if in_warmup else (
                "cooldown" if in_cooldown else "waiting"))
        old_lr = self.current_lr
        apply_record: Optional[Dict[str, Any]] = None
        if (
            not improved and not in_warmup and not in_cooldown
            and self.num_bad_epochs >= self.lr_patience
        ):
            target = max(self.current_lr * self.factor, self.min_lr)
            if target < self.current_lr:
                apply_record = apply_fn(target)
                self.current_lr = target
                self.cooldown_remaining = self.cooldown
                self.num_bad_epochs = 0
                self.reductions_total += 1
                reduced = True
                reason = "plateau_reduction"
            else:
                reason = "at_min_lr"
        return {
            "contract_id": CONTRACT_ID,
            "epoch": epoch,
            "monitor_value": value,
            "best_value": self.best_value,
            "monitor_improved": improved,
            "num_bad_epochs": self.num_bad_epochs,
            "cooldown_remaining": self.cooldown_remaining,
            "reduced": reduced,
            "reason": reason,
            "old_lr": old_lr,
            "new_lr": self.current_lr,
            "reductions_total": self.reductions_total,
            "apply_record": apply_record,
        }


def build_controller_from_config(
    config: Dict[str, Any],
    *,
    selection_metric: str,
    default_start_epoch: int,
    initial_lr: float,
) -> Optional[SacPlateauLrController]:
    """Construct the controller from ``config['plateau_lr']`` or return None.

    Refuses BEFORE training when the contract is malformed or when the
    selection metric is not the easy checkpoint monitor — this scheduler
    is defined over that monitor only (order §3).
    """
    spec = config.get("plateau_lr")
    if spec is None:
        return None
    if not isinstance(spec, dict):
        raise SacPlateauLrError(
            f"plateau_lr must be a mapping, got {type(spec).__name__}")
    if selection_metric != "easy_checkpoint_monitor_v1":
        raise SacPlateauLrError(
            "plateau_lr is driven only by easy_checkpoint_monitor; "
            f"refusing selection_metric={selection_metric!r} "
            "(order 2026-08-21 §3)")
    required = ("factor", "lr_patience", "min_lr", "threshold", "cooldown")
    missing = [k for k in required if k not in spec]
    if missing:
        raise SacPlateauLrError(
            "plateau_lr contract is explicit; missing required "
            f"keys {missing} (library defaults are refused)")
    unknown = sorted(
        set(spec) - set(required) - {"start_epoch"})
    if unknown:
        raise SacPlateauLrError(
            f"plateau_lr has unknown keys {unknown}")
    start_epoch = spec.get("start_epoch", default_start_epoch)
    return SacPlateauLrController(
        factor=spec["factor"],
        lr_patience=spec["lr_patience"],
        min_lr=spec["min_lr"],
        threshold=spec["threshold"],
        cooldown=spec["cooldown"],
        start_epoch=start_epoch,
        initial_lr=initial_lr,
    )


def assert_not_resuming_plateau_run(warm_start_model: Any) -> None:
    """PLR-01 fail-closed guard: plateau runs are non-resumable.

    A plateau sidecar sitting beside the warm-start checkpoint is
    evidence of an interrupted or continued plateau run. Constructing a
    fresh controller there would silently reset scheduler state while
    appearing to resume — refuse instead. A warm start WITHOUT a
    sidecar is a legitimately new scheduler lifecycle (e.g. the
    curriculum handoff) and passes. Nothing is ever loaded merely
    because it exists.
    """
    if not warm_start_model:
        return
    from pathlib import Path
    sidecar = Path(
        str(Path(str(warm_start_model)).with_suffix(""))
        + ".plateau_lr_state.json")
    if sidecar.exists():
        raise SacPlateauLrError(
            "REFUSED_PLATEAU_RESUME: plateau runs are non-resumable "
            f"(PLR-01). A scheduler sidecar exists at {sidecar} beside "
            "the warm-start checkpoint; a fresh controller would "
            "silently discard its state. Start a new run directory or "
            "disable plateau_lr for this warm start.")
