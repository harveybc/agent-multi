"""Pure execution-cost curriculum and robust weekly-fitness primitives.

Execution costs are an environment contract, never optimizer genes.  This
module deliberately has no dependency on an agent, simulator, or optimizer so
the same immutable contract can be used by training and validation adapters.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence


CONTRACT_VERSION = "execution_cost_curriculum.v1"
PHASE_NAMES = ("easy_nonzero", "nominal_randomized", "stress")
COST_KEYS = (
    "commission_fraction_per_side",
    "full_spread_rate",
    "slippage_bps_per_side",
)
OBSERVATION_NAMES = (
    *(f"execution_cost_{key}_normalized" for key in COST_KEYS),
    "execution_cost_financing_enabled",
    "execution_cost_phase_progress",
)


def _finite_float(value: Any, *, field: str, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    return result


def _required_string(value: Any, *, field: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{field} must be a non-empty string")
    return result


@dataclass(frozen=True)
class CostScenario:
    """One fixed execution-cost scenario in return-fraction conventions."""

    scenario_id: str
    commission_fraction_per_side: float
    full_spread_rate: float
    slippage_bps_per_side: float
    financing_enabled: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "scenario_id",
            _required_string(self.scenario_id, field="scenario_id"),
        )
        for key in COST_KEYS:
            object.__setattr__(
                self,
                key,
                _finite_float(getattr(self, key), field=key, minimum=0.0),
            )
        if not isinstance(self.financing_enabled, bool):
            raise ValueError("financing_enabled must be boolean")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "CostScenario":
        if not isinstance(raw, Mapping):
            raise ValueError("each cost scenario must be an object")
        try:
            return cls(
                scenario_id=raw["scenario_id"],
                commission_fraction_per_side=raw["commission_fraction_per_side"],
                full_spread_rate=raw["full_spread_rate"],
                slippage_bps_per_side=raw["slippage_bps_per_side"],
                financing_enabled=raw.get("financing_enabled", False),
            )
        except KeyError as exc:
            raise ValueError(f"cost scenario is missing {exc.args[0]}") from exc

    def cost_patch(self) -> dict[str, float | bool]:
        return {
            **{key: float(getattr(self, key)) for key in COST_KEYS},
            "financing_enabled": self.financing_enabled,
        }


@dataclass(frozen=True)
class ScenarioWeight:
    scenario_id: str
    weight: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "scenario_id",
            _required_string(self.scenario_id, field="scenario weight scenario_id"),
        )
        object.__setattr__(
            self,
            "weight",
            _finite_float(self.weight, field="scenario weight", minimum=0.0),
        )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ScenarioWeight":
        if not isinstance(raw, Mapping):
            raise ValueError("each scenario weight must be an object")
        try:
            return cls(scenario_id=raw["scenario_id"], weight=raw["weight"])
        except KeyError as exc:
            raise ValueError(f"scenario weight is missing {exc.args[0]}") from exc


@dataclass(frozen=True)
class CurriculumPhase:
    name: str
    start_progress: float
    end_progress: float
    scenario_weights: tuple[ScenarioWeight, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _required_string(self.name, field="phase name"))
        start = _finite_float(self.start_progress, field=f"{self.name}.start_progress")
        end = _finite_float(self.end_progress, field=f"{self.name}.end_progress")
        if not 0.0 <= start < end <= 1.0:
            raise ValueError(
                f"{self.name} progress range must satisfy 0 <= start < end <= 1"
            )
        object.__setattr__(self, "start_progress", start)
        object.__setattr__(self, "end_progress", end)
        object.__setattr__(self, "scenario_weights", tuple(self.scenario_weights))
        if not self.scenario_weights:
            raise ValueError(f"{self.name} must reference at least one scenario")
        identifiers = tuple(item.scenario_id for item in self.scenario_weights)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(f"{self.name} contains duplicate scenario weights")
        if sum(item.weight for item in self.scenario_weights) <= 0.0:
            raise ValueError(f"{self.name} scenario weights must have positive total")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "CurriculumPhase":
        if not isinstance(raw, Mapping):
            raise ValueError("each curriculum phase must be an object")
        try:
            weights = tuple(
                ScenarioWeight.from_mapping(item) for item in raw["scenario_weights"]
            )
            return cls(
                name=raw["name"],
                start_progress=raw["start_progress"],
                end_progress=raw["end_progress"],
                scenario_weights=weights,
            )
        except KeyError as exc:
            raise ValueError(f"curriculum phase is missing {exc.args[0]}") from exc
        except TypeError as exc:
            raise ValueError("scenario_weights must be an array") from exc


@dataclass(frozen=True)
class CostSelection:
    """Deterministic selection returned to an environment adapter."""

    contract_version: str
    curriculum_id: str
    contract_fingerprint: str
    phase_name: str
    scenario_id: str
    training_progress: float
    episode_index: int
    _cost_values: tuple[float, float, float]
    financing_enabled: bool
    observable_vector: tuple[float, float, float, float, float]

    @property
    def cost_patch(self) -> dict[str, float | bool]:
        """Return a fresh broker-compatible patch, safe for caller mutation."""
        return {
            **dict(zip(COST_KEYS, self._cost_values)),
            "financing_enabled": self.financing_enabled,
        }

    @property
    def observable_names(self) -> tuple[str, ...]:
        return OBSERVATION_NAMES


@dataclass(frozen=True)
class ExecutionCostCurriculum:
    """Versioned, immutable schedule over fixed execution-cost scenarios."""

    curriculum_id: str
    scenarios: tuple[CostScenario, ...]
    phases: tuple[CurriculumPhase, ...]
    normalization_bounds: tuple[float, float, float]
    contract_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != CONTRACT_VERSION:
            raise ValueError(
                f"unsupported contract_version {self.contract_version!r}; "
                f"expected {CONTRACT_VERSION!r}"
            )
        object.__setattr__(
            self,
            "curriculum_id",
            _required_string(self.curriculum_id, field="curriculum_id"),
        )
        object.__setattr__(self, "scenarios", tuple(self.scenarios))
        object.__setattr__(self, "phases", tuple(self.phases))
        object.__setattr__(self, "normalization_bounds", tuple(self.normalization_bounds))

        if not self.scenarios:
            raise ValueError("curriculum must define at least one fixed cost scenario")
        scenario_ids = tuple(item.scenario_id for item in self.scenarios)
        if len(set(scenario_ids)) != len(scenario_ids):
            raise ValueError("cost scenario_id values must be unique")

        names = tuple(phase.name for phase in self.phases)
        if names != PHASE_NAMES:
            raise ValueError(f"curriculum phases must be exactly {PHASE_NAMES}")
        if self.phases[0].start_progress != 0.0:
            raise ValueError("easy_nonzero must start at progress 0")
        if self.phases[-1].end_progress != 1.0:
            raise ValueError("stress must end at progress 1")
        for left, right in zip(self.phases, self.phases[1:]):
            if left.end_progress != right.start_progress:
                raise ValueError(
                    f"phase ranges must be contiguous: {left.name} ends at "
                    f"{left.end_progress}, {right.name} starts at {right.start_progress}"
                )

        if len(self.normalization_bounds) != len(COST_KEYS):
            raise ValueError(
                f"normalization_bounds must contain {len(COST_KEYS)} values"
            )
        bounds = tuple(
            _finite_float(value, field=f"normalization bound {key}", minimum=0.0)
            for key, value in zip(COST_KEYS, self.normalization_bounds)
        )
        if any(value <= 0.0 for value in bounds):
            raise ValueError("all normalization bounds must be > 0")
        object.__setattr__(self, "normalization_bounds", bounds)

        known = set(scenario_ids)
        for phase in self.phases:
            unknown = {
                item.scenario_id
                for item in phase.scenario_weights
                if item.scenario_id not in known
            }
            if unknown:
                raise ValueError(
                    f"{phase.name} references unknown scenarios: {sorted(unknown)}"
                )
        if (
            sum(item.weight > 0.0 for item in self.phases[1].scenario_weights)
            < 2
        ):
            raise ValueError(
                "nominal_randomized must have at least two positive-weight scenarios"
            )

        by_id = {item.scenario_id: item for item in self.scenarios}
        for item in self.phases[0].scenario_weights:
            if item.weight <= 0.0:
                continue
            scenario = by_id[item.scenario_id]
            if any(getattr(scenario, key) <= 0.0 for key in COST_KEYS):
                raise ValueError(
                    "easy_nonzero may reference only scenarios with nonzero costs"
                )
        for scenario in self.scenarios:
            for key, bound in zip(COST_KEYS, bounds):
                if getattr(scenario, key) > bound:
                    raise ValueError(
                        f"{scenario.scenario_id}.{key} exceeds normalization bound"
                    )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ExecutionCostCurriculum":
        if not isinstance(raw, Mapping):
            raise ValueError("execution cost curriculum must be an object")
        try:
            raw_bounds = raw["normalization_bounds"]
            if not isinstance(raw_bounds, Mapping):
                raise ValueError("normalization_bounds must be an object")
            bounds = tuple(raw_bounds[key] for key in COST_KEYS)
            return cls(
                contract_version=raw["contract_version"],
                curriculum_id=raw["curriculum_id"],
                scenarios=tuple(
                    CostScenario.from_mapping(item) for item in raw["scenarios"]
                ),
                phases=tuple(
                    CurriculumPhase.from_mapping(item) for item in raw["phases"]
                ),
                normalization_bounds=bounds,
            )
        except KeyError as exc:
            raise ValueError(f"execution cost curriculum is missing {exc.args[0]}") from exc
        except TypeError as exc:
            raise ValueError("scenarios and phases must be arrays") from exc

    @classmethod
    def from_json_file(cls, path: str) -> "ExecutionCostCurriculum":
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        return cls.from_mapping(raw)

    def to_canonical_mapping(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "curriculum_id": self.curriculum_id,
            "normalization_bounds": dict(
                zip(COST_KEYS, self.normalization_bounds)
            ),
            "scenarios": [
                {"scenario_id": scenario.scenario_id, **scenario.cost_patch()}
                for scenario in self.scenarios
            ],
            "phases": [
                {
                    "name": phase.name,
                    "start_progress": phase.start_progress,
                    "end_progress": phase.end_progress,
                    "scenario_weights": [
                        {
                            "scenario_id": item.scenario_id,
                            "weight": item.weight,
                        }
                        for item in phase.scenario_weights
                    ],
                }
                for phase in self.phases
            ],
        }

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_canonical_mapping(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def select(
        self,
        *,
        seed: int,
        training_progress: float,
        episode_index: int = 0,
        scenario_id: str | None = None,
    ) -> CostSelection:
        progress = _finite_float(training_progress, field="training_progress")
        if not 0.0 <= progress <= 1.0:
            raise ValueError("training_progress must be within [0, 1]")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        if (
            isinstance(episode_index, bool)
            or not isinstance(episode_index, int)
            or episode_index < 0
        ):
            raise ValueError("episode_index must be a nonnegative integer")

        phase = self.phases[-1]
        for candidate in self.phases:
            if candidate.start_progress <= progress < candidate.end_progress:
                phase = candidate
                break

        if scenario_id is None:
            digest_input = (
                f"{self.fingerprint}|{seed}|{progress.hex()}|"
                f"{phase.name}|{episode_index}"
            ).encode("utf-8")
            draw_int = int.from_bytes(
                hashlib.blake2b(digest_input, digest_size=8).digest(),
                byteorder="big",
                signed=False,
            )
            draw = draw_int / float(1 << 64)
            total_weight = sum(item.weight for item in phase.scenario_weights)
            threshold = draw * total_weight
            cumulative = 0.0
            selected_id = phase.scenario_weights[-1].scenario_id
            for item in phase.scenario_weights:
                cumulative += item.weight
                if threshold < cumulative:
                    selected_id = item.scenario_id
                    break
        else:
            selected_id = _required_string(
                scenario_id, field="fixed validation scenario_id"
            )
            if selected_id not in {item.scenario_id for item in self.scenarios}:
                raise ValueError(f"unknown fixed cost scenario {selected_id!r}")

        scenario = next(
            item for item in self.scenarios if item.scenario_id == selected_id
        )
        values = tuple(float(getattr(scenario, key)) for key in COST_KEYS)
        observable = tuple(
            min(1.0, max(0.0, value / bound))
            for value, bound in zip(values, self.normalization_bounds)
        ) + (
            1.0 if scenario.financing_enabled else 0.0,
            progress,
        )
        return CostSelection(
            contract_version=self.contract_version,
            curriculum_id=self.curriculum_id,
            contract_fingerprint=self.fingerprint,
            phase_name=phase.name,
            scenario_id=scenario.scenario_id,
            training_progress=progress,
            episode_index=episode_index,
            _cost_values=values,
            financing_enabled=scenario.financing_enabled,
            observable_vector=observable,
        )


@dataclass(frozen=True)
class RobustFitnessConfig:
    """Dimensionless weights applied to weekly return-fraction penalties."""

    lower_tail_fraction: float = 0.25
    downside_penalty_weight: float = 1.0
    dispersion_penalty_weight: float = 0.5
    annualization_weeks: float = 52.0

    def __post_init__(self) -> None:
        tail = _finite_float(
            self.lower_tail_fraction,
            field="lower_tail_fraction",
            minimum=0.0,
        )
        if not 0.0 < tail <= 1.0:
            raise ValueError("lower_tail_fraction must be within (0, 1]")
        object.__setattr__(self, "lower_tail_fraction", tail)
        for field in ("downside_penalty_weight", "dispersion_penalty_weight"):
            object.__setattr__(
                self,
                field,
                _finite_float(getattr(self, field), field=field, minimum=0.0),
            )
        weeks = _finite_float(
            self.annualization_weeks,
            field="annualization_weeks",
            minimum=0.0,
        )
        if weeks <= 0.0:
            raise ValueError("annualization_weeks must be > 0")
        object.__setattr__(self, "annualization_weeks", weeks)


def _lower_tail_cvar(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    target_mass = fraction * len(ordered)
    remaining = target_mass
    weighted_sum = 0.0
    for value in ordered:
        if remaining <= 0.0:
            break
        mass = min(1.0, remaining)
        weighted_sum += value * mass
        remaining -= mass
    return weighted_sum / target_mass


def aggregate_robust_weekly_fitness(
    scenario_metrics: Mapping[str, Mapping[str, Any]],
    *,
    config: RobustFitnessConfig | None = None,
) -> dict[str, float | int | str]:
    """Aggregate immutable-scenario validation evidence in weekly units.

    Each scenario must provide finite ``mean_weekly_return``,
    ``annualized_return`` and ``mean_weekly_rap`` values. Missing or nonfinite
    evidence is an error; no zero filling is permitted.
    """

    if not isinstance(scenario_metrics, Mapping) or not scenario_metrics:
        raise ValueError("scenario_metrics must be a non-empty mapping")
    settings = config or RobustFitnessConfig()
    weekly_returns: list[float] = []
    annualized_returns: list[float] = []
    weekly_raps: list[float] = []
    scenario_ids: list[str] = []
    for raw_id, metrics in scenario_metrics.items():
        scenario_id = _required_string(raw_id, field="validation scenario_id")
        if not isinstance(metrics, Mapping):
            raise ValueError(f"{scenario_id} metrics must be an object")
        missing = {
            key
            for key in (
                "mean_weekly_return",
                "annualized_return",
                "mean_weekly_rap",
            )
            if key not in metrics
        }
        if missing:
            raise ValueError(
                f"{scenario_id} is missing canonical weekly metrics: "
                f"{sorted(missing)}"
            )
        weekly_return = _finite_float(
            metrics["mean_weekly_return"],
            field=f"{scenario_id}.mean_weekly_return",
        )
        weekly_rap = _finite_float(
            metrics["mean_weekly_rap"],
            field=f"{scenario_id}.mean_weekly_rap",
        )
        scenario_ids.append(scenario_id)
        weekly_returns.append(weekly_return)
        annualized_returns.append(
            _finite_float(
                metrics["annualized_return"],
                field=f"{scenario_id}.annualized_return",
            )
        )
        weekly_raps.append(weekly_rap)

    mean_weekly_return = math.fsum(weekly_returns) / len(weekly_returns)
    annualized_return = math.fsum(annualized_returns) / len(annualized_returns)
    mean_weekly_rap = math.fsum(weekly_raps) / len(weekly_raps)
    cvar = _lower_tail_cvar(weekly_raps, settings.lower_tail_fraction)
    variance = (
        math.fsum((value - mean_weekly_rap) ** 2 for value in weekly_raps)
        / len(weekly_raps)
    )
    dispersion = math.sqrt(variance)
    downside_shortfall = max(0.0, mean_weekly_rap - cvar)
    downside_penalty = settings.downside_penalty_weight * downside_shortfall
    dispersion_penalty = settings.dispersion_penalty_weight * dispersion
    robust_fitness = mean_weekly_rap - downside_penalty - dispersion_penalty
    worst_index = min(range(len(weekly_raps)), key=weekly_raps.__getitem__)

    return {
        "scenario_count": len(scenario_ids),
        "mean_weekly_return": mean_weekly_return,
        "annualized_return": annualized_return,
        "annual_return": annualized_return,
        "mean_weekly_rap": mean_weekly_rap,
        "annual_rap": mean_weekly_rap * settings.annualization_weeks,
        "worst_scenario_id": scenario_ids[worst_index],
        "worst_scenario_weekly_rap": weekly_raps[worst_index],
        "lower_tail_cvar_weekly_rap": cvar,
        "scenario_weekly_rap_dispersion": dispersion,
        "downside_shortfall_weekly_rap": downside_shortfall,
        "downside_weekly_rap_penalty": downside_penalty,
        "dispersion_weekly_rap_penalty": dispersion_penalty,
        "robust_weekly_rap_fitness": robust_fitness,
    }
