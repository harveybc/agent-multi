#!/usr/bin/env python3
"""Bounded CPU pilot for model capacity, memorization and description.

The pilot implements M0-M2 of work plan 44.  It is intentionally independent
from B4 and T2: no financial data, no GPU, no promotion, and no optimizer gene.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import model_information_contract as mic
import t1_known_truth_bank as t1_bank


DESIGN_SCHEMA = "agent_multi.model_capacity_m0_m2_design.v1"
UNIT_SCHEMA = "agent_multi.model_capacity_m0_m2_unit.v1"
SUMMARY_SCHEMA = "agent_multi.model_capacity_m0_m2_summary.v1"
DEFAULT_DESIGN = (
    REPO
    / "docs/audits/evidence/MODEL_CAPACITY_M0_M2_PILOT_DESIGN_2026_09_07.json"
)
CODE_FILES = (
    "tools/model_information_contract.py",
    "tools/model_capacity_pilot.py",
)
HEX64 = set("0123456789abcdef")


class PilotRefusal(RuntimeError):
    """Typed refusal for invalid design, evidence or resource state."""


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: Any) -> bytes:
    return mic.canonical_json_bytes(value)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    mode = path.stat().st_mode & 0o777
    if mode != 0o700:
        raise PilotRefusal(f"private directory mode must be 0700: {mode:04o}")


def _exclusive_json(path: Path, value: Any) -> None:
    payload = json.dumps(
        value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False
    ).encode("ascii") + b"\n"
    fd = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)


def _replace_json(path: Path, value: Any) -> None:
    payload = json.dumps(
        value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False
    ).encode("ascii") + b"\n"
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    fd = os.open(
        tmp,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)


def current_code_identity() -> dict[str, str]:
    return {name: _sha_file(REPO / name) for name in CODE_FILES}


def make_design() -> dict[str, Any]:
    body = {
        "schema": DESIGN_SCHEMA,
        "date": "2026-09-07",
        "claim_scope": "M0_M2_BOUNDED_CPU_PILOT_ONLY",
        "code_identity": current_code_identity(),
        "resources": {
            "device": "cpu",
            "max_wall_seconds": 7200,
            "max_peak_rss_bytes": 1_073_741_824,
            "torch_threads": 1,
            "heartbeat_seconds": 20,
        },
        "threshold_neuron": {
            "input_counts": [4, 8, 16],
            "association_ratios": [1.0, 1.5, 2.0, 2.5, 3.0],
            "seeds": [11, 12, 13],
            "algorithm": "pocket_perceptron_no_bias",
            "max_updates": 20000,
            "perfect_fit_tolerance": 1.0,
            "theoretical_reference": "approximately_two_random_binary_associations_per_weight_under_stated_single_threshold_neuron_assumptions",
        },
        "boolean_mlp": {
            "input_count": 12,
            "rules": ["majority", "xor3", "three_term_dnf"],
            "train_label_noise": [0.0, 0.2],
            "hidden_widths": [4, 16, 64],
            "seeds": [11, 12, 13],
            "split_sizes": {"train": 1024, "calibration": 512, "evaluation": 512},
        },
        "temporal_mlp": {
            "families": ["sine", "chirp", "heavisine"],
            "snr_db": ["inf", 10, 0],
            "hidden_widths": [8, 32],
            "seeds": [11, 12],
            "lookback": 16,
            "source_contract": "t1_known_truth_bank.clean_signal_and_realized_noise",
        },
        "training": {
            "optimizer": "adam",
            "learning_rate": 0.01,
            "max_epochs": 120,
            "minimum_stop_epoch": 20,
            "patience": 10,
            "minimum_delta": 1e-5,
            "diagnostic_post_stop_epochs": 15,
            "checkpoint_epochs": [0, 1, 2, 4, 8, 16, 32, 64, 96, 120],
            "evaluation_used_for_stopping": False,
        },
        "description_contract": {
            "model_quantization_bits": [4, 8, 16],
            "compressors": ["zlib", "bz2", "lzma"],
            "interpretation": "estimator_specific_upper_bounds_not_exact_kolmogorov_complexity",
            "independent_information_multiplier_for_repeated_epochs": 1,
        },
        "decision": {
            "pilot_only": True,
            "capacity_saturation_requires_both_fit_and_failure_regimes": True,
            "post_stop_branch_may_select": False,
            "confirmatory_claims_authorized": False,
            "doin_gene_authorized": False,
        },
    }
    return mic.seal_document(body, "design_sha256")


DESIGN_KEYS = {
    "schema",
    "date",
    "claim_scope",
    "code_identity",
    "resources",
    "threshold_neuron",
    "boolean_mlp",
    "temporal_mlp",
    "training",
    "description_contract",
    "decision",
    "design_sha256",
}


def verify_design(design: dict[str, Any]) -> None:
    mic.verify_sealed_document(
        design,
        field="design_sha256",
        schema=DESIGN_SCHEMA,
        exact_keys=DESIGN_KEYS,
    )
    if design["code_identity"] != current_code_identity():
        raise PilotRefusal("design code identity does not match executing bytes")
    resources = design["resources"]
    if resources.get("device") != "cpu":
        raise PilotRefusal("pilot device must be cpu")
    mic.require_int("resources.max_wall_seconds", resources.get("max_wall_seconds"), minimum=1)
    mic.require_int(
        "resources.max_peak_rss_bytes", resources.get("max_peak_rss_bytes"), minimum=1
    )
    if design["training"].get("evaluation_used_for_stopping") is not False:
        raise PilotRefusal("evaluation split cannot govern stopping")
    decision = design["decision"]
    expected_false = (
        "post_stop_branch_may_select",
        "confirmatory_claims_authorized",
        "doin_gene_authorized",
    )
    if any(decision.get(key) is not False for key in expected_false):
        raise PilotRefusal("pilot decision grants forbidden authority")
    if design["description_contract"].get("interpretation") != (
        "estimator_specific_upper_bounds_not_exact_kolmogorov_complexity"
    ):
        raise PilotRefusal("description contract overclaims its meaning")


def load_design(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PilotRefusal(f"cannot load design: {exc}") from exc
    if not isinstance(value, dict):
        raise PilotRefusal("design must be a JSON object")
    verify_design(value)
    return value


def _unit_id(kind: str, spec: dict[str, Any]) -> str:
    return f"{kind}__{_sha({'kind': kind, 'spec': spec})[:20]}"


def expected_units(design: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    units: list[tuple[str, dict[str, Any]]] = []
    threshold = design["threshold_neuron"]
    for inputs in threshold["input_counts"]:
        for ratio in threshold["association_ratios"]:
            for seed in threshold["seeds"]:
                spec = {
                    "inputs": inputs,
                    "association_ratio": ratio,
                    "associations": int(round(inputs * ratio)),
                    "seed": seed,
                }
                units.append(("threshold_neuron", spec))
    boolean = design["boolean_mlp"]
    for rule in boolean["rules"]:
        for noise in boolean["train_label_noise"]:
            for width in boolean["hidden_widths"]:
                for seed in boolean["seeds"]:
                    units.append(
                        (
                            "boolean_mlp",
                            {
                                "rule": rule,
                                "train_label_noise": noise,
                                "hidden_width": width,
                                "seed": seed,
                            },
                        )
                    )
    temporal = design["temporal_mlp"]
    for family in temporal["families"]:
        for snr in temporal["snr_db"]:
            for width in temporal["hidden_widths"]:
                for seed in temporal["seeds"]:
                    units.append(
                        (
                            "temporal_mlp",
                            {
                                "family": family,
                                "snr_db": snr,
                                "hidden_width": width,
                                "seed": seed,
                            },
                        )
                    )
    return units


def _seed(spec: dict[str, Any], suffix: str = "") -> int:
    return int(_sha({"spec": spec, "suffix": suffix})[:8], 16)


def _tensor_state(model: nn.Module) -> dict[str, np.ndarray]:
    return {
        name: value.detach().cpu().numpy().copy()
        for name, value in sorted(model.state_dict().items())
    }


def _copy_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _load_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    model.load_state_dict({name: value.clone() for name, value in state.items()})


class TinyMLP(nn.Module):
    def __init__(self, inputs: int, hidden: int) -> None:
        super().__init__()
        self.hidden = nn.Linear(inputs, hidden)
        self.output = nn.Linear(hidden, 1)

    def forward(self, value: torch.Tensor, *, return_hidden: bool = False):
        hidden = torch.tanh(self.hidden(value))
        output = self.output(hidden).squeeze(-1)
        return (output, hidden) if return_hidden else output


def _effective_dimension(hidden: np.ndarray) -> float:
    centered = hidden - hidden.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = singular**2
    if float(energy.sum()) == 0.0:
        return 0.0
    p = energy[energy > 0] / energy.sum()
    return float(np.exp(-np.sum(p * np.log(p))))


def _linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    left = left - left.mean(axis=0, keepdims=True)
    right = right - right.mean(axis=0, keepdims=True)
    cross = np.linalg.norm(left.T @ right, ord="fro") ** 2
    denom = np.linalg.norm(left.T @ left, ord="fro") * np.linalg.norm(
        right.T @ right, ord="fro"
    )
    return float(cross / denom) if denom else 0.0


def _gradient_and_hessian_metrics(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    classification: bool,
) -> dict[str, float]:
    subset = slice(0, min(64, inputs.shape[0]))
    output = model(inputs[subset])
    loss = (
        nn.functional.binary_cross_entropy_with_logits(output, targets[subset])
        if classification
        else nn.functional.mse_loss(output, targets[subset])
    )
    params = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    gradients = torch.autograd.grad(loss, params, create_graph=True)
    flat = torch.cat([gradient.reshape(-1) for gradient in gradients])
    gradient_norm = float(torch.linalg.vector_norm(flat).detach())
    fisher_mean = float(torch.mean(flat.detach() ** 2))
    vector = torch.ones_like(flat)
    vector = vector / torch.linalg.vector_norm(vector)
    eigenvalue = torch.tensor(0.0)
    for _ in range(2):
        dot = torch.dot(flat, vector)
        hv_parts = torch.autograd.grad(dot, params, retain_graph=True)
        hv = torch.cat([part.reshape(-1) for part in hv_parts]).detach()
        norm = torch.linalg.vector_norm(hv)
        if float(norm) == 0.0:
            eigenvalue = torch.tensor(0.0)
            break
        vector = hv / norm
        eigenvalue = torch.dot(vector, hv)
    return {
        "gradient_l2": gradient_norm,
        "fisher_diagonal_mean": fisher_mean,
        "hessian_power_iteration_value": float(eigenvalue),
        "hessian_iterations": 2,
    }


def _loss_and_score(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    classification: bool,
) -> tuple[float, float]:
    with torch.no_grad():
        output = model(inputs)
        if classification:
            loss = nn.functional.binary_cross_entropy_with_logits(output, targets)
            score = ((output >= 0).float() == targets).float().mean()
        else:
            loss = nn.functional.mse_loss(output, targets)
            baseline = torch.mean((targets - targets.mean()) ** 2)
            score = 1.0 - loss / torch.clamp(baseline, min=1e-12)
    return float(loss), float(score)


def _checkpoint(
    model: TinyMLP,
    *,
    epoch: int,
    branch: str,
    train: tuple[torch.Tensor, torch.Tensor],
    calibration: tuple[torch.Tensor, torch.Tensor],
    evaluation: tuple[torch.Tensor, torch.Tensor],
    classification: bool,
    initial_hidden: np.ndarray,
    noisy_train_targets: torch.Tensor | None,
    clean_train_targets: torch.Tensor | None,
) -> dict[str, Any]:
    losses: dict[str, float] = {}
    scores: dict[str, float] = {}
    for name, pair in (
        ("train", train),
        ("calibration", calibration),
        ("evaluation", evaluation),
    ):
        loss, score = _loss_and_score(model, *pair, classification=classification)
        losses[name] = loss
        scores[name] = score
    with torch.no_grad():
        _, hidden = model(calibration[0][:128], return_hidden=True)
    hidden_np = hidden.detach().cpu().numpy()
    if noisy_train_targets is not None and clean_train_targets is not None:
        with torch.no_grad():
            logits = model(train[0])
            changed = noisy_train_targets != clean_train_targets
            if bool(changed.any()):
                remembered = (
                    ((logits[changed] >= 0).float() == noisy_train_targets[changed])
                    .float()
                    .mean()
                )
                memorization: Any = float(remembered)
            else:
                memorization = {
                    "available": False,
                    "reason": "no_sample_specific_exceptions",
                }
    else:
        memorization = {"available": False, "reason": "not_a_noisy_label_task"}
    body = {
        "schema": "agent_multi.model_capacity_checkpoint.v1",
        "epoch": epoch,
        "branch": branch,
        "selection_authority": branch == "selection",
        "losses": losses,
        "scores": scores,
        "sample_specific_memorization": memorization,
        "model_description": mic.describe_model_arrays(_tensor_state(model)),
        "activation": {
            "effective_dimension": _effective_dimension(hidden_np),
            "linear_cka_to_initial": _linear_cka(initial_hidden, hidden_np),
        },
        "differential": _gradient_and_hessian_metrics(
            model,
            *calibration,
            classification=classification,
        ),
    }
    return mic.seal_document(body, "checkpoint_sha256")


def _boolean_rule(name: str, values: np.ndarray) -> np.ndarray:
    bits = values.astype(bool)
    if name == "majority":
        return (bits.sum(axis=1) >= bits.shape[1] / 2).astype(np.float32)
    if name == "xor3":
        return np.logical_xor(np.logical_xor(bits[:, 0], bits[:, 1]), bits[:, 2]).astype(
            np.float32
        )
    if name == "three_term_dnf":
        result = (bits[:, 0] & bits[:, 1]) | (bits[:, 2] & ~bits[:, 3])
        result |= bits[:, 4] & bits[:, 5] & bits[:, 6]
        return result.astype(np.float32)
    raise PilotRefusal(f"unknown Boolean rule {name}")


def _boolean_data(design: dict[str, Any], spec: dict[str, Any]):
    contract = design["boolean_mlp"]
    inputs = contract["input_count"]
    all_patterns = ((np.arange(2**inputs)[:, None] >> np.arange(inputs)) & 1).astype(
        np.float32
    )
    rng = np.random.default_rng(_seed(spec, "boolean_split"))
    order = rng.permutation(len(all_patterns))
    sizes = contract["split_sizes"]
    train_idx = order[: sizes["train"]]
    cal_idx = order[sizes["train"] : sizes["train"] + sizes["calibration"]]
    eval_idx = order[
        sizes["train"]
        + sizes["calibration"] : sizes["train"]
        + sizes["calibration"]
        + sizes["evaluation"]
    ]
    x_train, x_cal, x_eval = (
        all_patterns[train_idx],
        all_patterns[cal_idx],
        all_patterns[eval_idx],
    )
    y_train_clean = _boolean_rule(spec["rule"], x_train)
    y_cal = _boolean_rule(spec["rule"], x_cal)
    y_eval = _boolean_rule(spec["rule"], x_eval)
    flip_rng = np.random.default_rng(_seed(spec, "label_noise"))
    flips = flip_rng.random(len(y_train_clean)) < float(spec["train_label_noise"])
    y_train = np.logical_xor(y_train_clean.astype(bool), flips).astype(np.float32)
    arrays = {
        "train_inputs": x_train,
        "train_clean_targets": y_train_clean,
        "train_observed_targets": y_train,
        "calibration_inputs": x_cal,
        "calibration_targets": y_cal,
        "evaluation_inputs": x_eval,
        "evaluation_targets": y_eval,
    }
    tensors = {
        name: torch.tensor(value, dtype=torch.float32)
        for name, value in arrays.items()
    }
    return tensors, arrays


def _temporal_data(design: dict[str, Any], spec: dict[str, Any]):
    rng = np.random.default_rng(_seed(spec, "temporal_signal"))
    clean = t1_bank.clean_signal(spec["family"], rng)[0]
    perturbation = t1_bank.realized_noise("white", clean[None, :], spec["snr_db"], rng)
    observed = (clean[None, :] + perturbation["noise"])[0]
    lookback = design["temporal_mlp"]["lookback"]
    features = np.array(
        [observed[index - lookback : index] for index in range(lookback, len(clean))],
        dtype=np.float32,
    )
    targets = clean[lookback:].astype(np.float32)
    n = len(targets)
    train_end = int(n * 0.6)
    cal_end = int(n * 0.8)
    x_train, x_cal, x_eval = features[:train_end], features[train_end:cal_end], features[cal_end:]
    y_train, y_cal, y_eval = targets[:train_end], targets[train_end:cal_end], targets[cal_end:]
    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = np.maximum(x_train.std(axis=0, keepdims=True), 1e-6)
    y_mean = float(y_train.mean())
    y_std = max(float(y_train.std()), 1e-6)
    x_train, x_cal, x_eval = (
        (x_train - x_mean) / x_std,
        (x_cal - x_mean) / x_std,
        (x_eval - x_mean) / x_std,
    )
    y_train, y_cal, y_eval = (
        (y_train - y_mean) / y_std,
        (y_cal - y_mean) / y_std,
        (y_eval - y_mean) / y_std,
    )
    arrays = {
        "clean_signal": clean,
        "realized_noise": perturbation["noise"][0],
        "observed_signal": observed,
        "train_inputs": x_train,
        "train_targets": y_train,
        "calibration_inputs": x_cal,
        "calibration_targets": y_cal,
        "evaluation_inputs": x_eval,
        "evaluation_targets": y_eval,
    }
    tensors = {
        name: torch.tensor(value, dtype=torch.float32)
        for name, value in arrays.items()
        if name.endswith("inputs") or name.endswith("targets")
    }
    return tensors, arrays


def choose_stop_from_calibration(
    calibration_losses: list[float],
    *,
    minimum_epoch: int,
    patience: int,
    minimum_delta: float,
) -> tuple[int, int] | None:
    best = math.inf
    best_epoch = -1
    stale = 0
    for epoch, value in enumerate(calibration_losses):
        mic.require_number(f"calibration_losses[{epoch}]", value, minimum=0.0)
        if value < best - minimum_delta:
            best = value
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if epoch >= minimum_epoch and stale >= patience:
            return best_epoch, epoch
    return None


def _train_mlp_unit(
    design: dict[str, Any],
    kind: str,
    spec: dict[str, Any],
) -> dict[str, Any]:
    torch.manual_seed(_seed(spec, "model"))
    torch.use_deterministic_algorithms(True)
    if kind == "boolean_mlp":
        tensors, arrays = _boolean_data(design, spec)
        inputs = design["boolean_mlp"]["input_count"]
        classification = True
        train = (tensors["train_inputs"], tensors["train_observed_targets"])
        calibration = (tensors["calibration_inputs"], tensors["calibration_targets"])
        evaluation = (tensors["evaluation_inputs"], tensors["evaluation_targets"])
        clean_train = tensors["train_clean_targets"]
        observed_train = tensors["train_observed_targets"]
    elif kind == "temporal_mlp":
        tensors, arrays = _temporal_data(design, spec)
        inputs = design["temporal_mlp"]["lookback"]
        classification = False
        train = (tensors["train_inputs"], tensors["train_targets"])
        calibration = (tensors["calibration_inputs"], tensors["calibration_targets"])
        evaluation = (tensors["evaluation_inputs"], tensors["evaluation_targets"])
        clean_train = None
        observed_train = None
    else:
        raise PilotRefusal(f"unknown MLP unit kind {kind}")

    model = TinyMLP(inputs, int(spec["hidden_width"]))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(design["training"]["learning_rate"])
    )
    loss_fn = (
        nn.BCEWithLogitsLoss() if classification else nn.MSELoss()
    )
    with torch.no_grad():
        _, initial_hidden_tensor = model(calibration[0][:128], return_hidden=True)
    initial_hidden = initial_hidden_tensor.numpy().copy()
    initial_state = _copy_state(model)
    training = design["training"]
    checkpoints: list[dict[str, Any]] = []
    checkpoint_epochs = set(training["checkpoint_epochs"])
    calibration_losses: list[float] = []
    best_loss = math.inf
    best_epoch = 0
    best_state = _copy_state(model)
    no_improve = 0
    stop_epoch: int | None = None
    diagnostic_end: int | None = None

    checkpoints.append(
        _checkpoint(
            model,
            epoch=0,
            branch="selection",
            train=train,
            calibration=calibration,
            evaluation=evaluation,
            classification=classification,
            initial_hidden=initial_hidden,
            noisy_train_targets=observed_train,
            clean_train_targets=clean_train,
        )
    )
    for epoch in range(1, int(training["max_epochs"]) + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(train[0])
        loss = loss_fn(prediction, train[1])
        loss.backward()
        optimizer.step()
        calibration_loss, _ = _loss_and_score(
            model, *calibration, classification=classification
        )
        calibration_losses.append(calibration_loss)
        if stop_epoch is None:
            if calibration_loss < best_loss - float(training["minimum_delta"]):
                best_loss = calibration_loss
                best_epoch = epoch
                best_state = _copy_state(model)
                no_improve = 0
            else:
                no_improve += 1
            if (
                epoch >= int(training["minimum_stop_epoch"])
                and no_improve >= int(training["patience"])
            ):
                stop_epoch = epoch
                diagnostic_end = min(
                    int(training["max_epochs"]),
                    epoch + int(training["diagnostic_post_stop_epochs"]),
                )
        is_key = epoch in checkpoint_epochs or epoch == stop_epoch or epoch == diagnostic_end
        if is_key:
            checkpoints.append(
                _checkpoint(
                    model,
                    epoch=epoch,
                    branch="diagnostic_continuation" if stop_epoch is not None else "selection",
                    train=train,
                    calibration=calibration,
                    evaluation=evaluation,
                    classification=classification,
                    initial_hidden=initial_hidden,
                    noisy_train_targets=observed_train,
                    clean_train_targets=clean_train,
                )
            )
        if diagnostic_end is not None and epoch >= diagnostic_end:
            break

    diagnostic_state = _copy_state(model)
    diagnostic_epoch = epoch
    _load_state(model, best_state)
    selection_checkpoint = _checkpoint(
        model,
        epoch=best_epoch,
        branch="selection",
        train=train,
        calibration=calibration,
        evaluation=evaluation,
        classification=classification,
        initial_hidden=initial_hidden,
        noisy_train_targets=observed_train,
        clean_train_targets=clean_train,
    )
    _load_state(model, diagnostic_state)
    diagnostic_checkpoint = _checkpoint(
        model,
        epoch=diagnostic_epoch,
        branch="diagnostic_continuation",
        train=train,
        calibration=calibration,
        evaluation=evaluation,
        classification=classification,
        initial_hidden=initial_hidden,
        noisy_train_targets=observed_train,
        clean_train_targets=clean_train,
    )
    # Restore selected state so no caller can accidentally consume diagnostics.
    _load_state(model, best_state)
    return {
        "data_description": mic.describe_data_arrays(arrays, repeated_exposures=epoch),
        "initial_model_description": mic.describe_model_arrays(initial_state),
        "checkpoints": checkpoints,
        "stopping": {
            "selection_epoch": best_epoch,
            "observed_stop_epoch": stop_epoch,
            "diagnostic_end_epoch": diagnostic_epoch,
            "selection_source": "calibration_loss_only",
            "evaluation_used_for_stopping": False,
            "diagnostic_branch_may_select": False,
        },
        "selection_checkpoint": selection_checkpoint,
        "diagnostic_checkpoint": diagnostic_checkpoint,
    }


def _threshold_unit(design: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    rng = np.random.default_rng(_seed(spec, "threshold"))
    n = int(spec["associations"])
    k = int(spec["inputs"])
    x = rng.normal(size=(n, k))
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    labels = rng.choice(np.array([-1.0, 1.0]), size=n)
    x_eval = rng.normal(size=(max(n, 64), k))
    x_eval /= np.maximum(np.linalg.norm(x_eval, axis=1, keepdims=True), 1e-12)
    labels_eval = rng.choice(np.array([-1.0, 1.0]), size=len(x_eval))
    weight = np.zeros(k, dtype=np.float64)
    best_weight = weight.copy()
    best_accuracy = 0.0
    converged_at: int | None = None
    max_updates = int(design["threshold_neuron"]["max_updates"])
    for update in range(max_updates):
        signed = labels * (x @ weight)
        incorrect = np.flatnonzero(signed <= 0)
        accuracy = 1.0 - len(incorrect) / n
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_weight = weight.copy()
        if len(incorrect) == 0:
            converged_at = update
            best_accuracy = 1.0
            best_weight = weight.copy()
            break
        index = incorrect[update % len(incorrect)]
        weight += labels[index] * x[index]
    weight = best_weight
    train_accuracy = float(np.mean(labels * (x @ weight) > 0))
    eval_accuracy = float(np.mean(labels_eval * (x_eval @ weight) > 0))
    arrays = {"inputs": x, "labels": labels, "evaluation_inputs": x_eval,
              "evaluation_labels": labels_eval}
    return {
        "data_description": mic.describe_data_arrays(
            arrays, repeated_exposures=(converged_at or max_updates) + 1
        ),
        "model_description": mic.describe_model_arrays({"weight": weight}),
        "train_accuracy": train_accuracy,
        "independent_random_label_evaluation_accuracy": eval_accuracy,
        "perfect_fit": train_accuracy == 1.0,
        "converged_update": converged_at,
        "updates_budget": max_updates,
    }


UNIT_KEYS = {
    "schema",
    "unit_id",
    "kind",
    "spec",
    "design_sha256",
    "code_identity",
    "status",
    "result",
    "cpu_wall_seconds",
    "peak_rss_bytes",
    "record_sha256",
}


def _run_unit(design: dict[str, Any], kind: str, spec: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    unit_id = _unit_id(kind, spec)
    result = (
        _threshold_unit(design, spec)
        if kind == "threshold_neuron"
        else _train_mlp_unit(design, kind, spec)
    )
    body = {
        "schema": UNIT_SCHEMA,
        "unit_id": unit_id,
        "kind": kind,
        "spec": spec,
        "design_sha256": design["design_sha256"],
        "code_identity": design["code_identity"],
        "status": "COMPLETED",
        "result": result,
        "cpu_wall_seconds": round(time.perf_counter() - started, 6),
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
    }
    return mic.seal_document(body, "record_sha256")


def verify_unit_record(
    record: dict[str, Any], design: dict[str, Any], kind: str, spec: dict[str, Any]
) -> None:
    mic.verify_sealed_document(
        record,
        field="record_sha256",
        schema=UNIT_SCHEMA,
        exact_keys=UNIT_KEYS,
    )
    expected_id = _unit_id(kind, spec)
    if record["unit_id"] != expected_id or record["kind"] != kind or record["spec"] != spec:
        raise PilotRefusal("unit identity/specification mismatch")
    if record["design_sha256"] != design["design_sha256"]:
        raise PilotRefusal("unit design binding mismatch")
    if record["code_identity"] != design["code_identity"]:
        raise PilotRefusal("unit code binding mismatch")
    if record["status"] != "COMPLETED":
        raise PilotRefusal("unit is not completed")
    mic.require_number("cpu_wall_seconds", record["cpu_wall_seconds"], minimum=0.0)
    mic.require_int("peak_rss_bytes", record["peak_rss_bytes"], minimum=1)
    result = record["result"]
    mic.verify_data_description(result["data_description"])
    if kind == "threshold_neuron":
        mic.verify_model_description(result["model_description"])
        mic.require_number("train_accuracy", result["train_accuracy"], minimum=0.0)
        if result["perfect_fit"] is not (result["train_accuracy"] == 1.0):
            raise PilotRefusal("perfect-fit disposition does not derive from accuracy")
    else:
        mic.verify_model_description(result["initial_model_description"])
        stopping = result["stopping"]
        if stopping.get("evaluation_used_for_stopping") is not False:
            raise PilotRefusal("evaluation split contaminated stopping")
        if stopping.get("diagnostic_branch_may_select") is not False:
            raise PilotRefusal("diagnostic continuation cannot select")
        observed_stop = stopping.get("observed_stop_epoch")
        if observed_stop is not None and stopping.get("selection_epoch") > observed_stop:
            raise PilotRefusal("post-stop checkpoint contaminated selection")
        if result["selection_checkpoint"].get("branch") != "selection":
            raise PilotRefusal("selected checkpoint is not from selection branch")
        if result["diagnostic_checkpoint"].get("selection_authority") is not False:
            raise PilotRefusal("diagnostic checkpoint gained selection authority")
        for checkpoint in result["checkpoints"] + [
            result["selection_checkpoint"],
            result["diagnostic_checkpoint"],
        ]:
            body = dict(checkpoint)
            digest = body.pop("checkpoint_sha256", None)
            if not isinstance(digest, str) or _sha(body) != digest:
                raise PilotRefusal("checkpoint self-digest mismatch")
            mic.verify_model_description(checkpoint["model_description"])
            for group in ("losses", "scores", "activation", "differential"):
                for name, value in checkpoint[group].items():
                    if name == "hessian_iterations":
                        mic.require_int(f"checkpoint.{group}.{name}", value, minimum=1)
                    else:
                        mic.require_number(f"checkpoint.{group}.{name}", value)


def _threshold_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[int, float], list[dict[str, Any]]] = {}
    for record in records:
        if record["kind"] != "threshold_neuron":
            continue
        key = (int(record["spec"]["inputs"]), float(record["spec"]["association_ratio"]))
        groups.setdefault(key, []).append(record)
    curves: list[dict[str, Any]] = []
    by_k: dict[int, list[dict[str, Any]]] = {}
    for (inputs, ratio), rows in sorted(groups.items()):
        item = {
            "inputs": inputs,
            "association_ratio": ratio,
            "associations": rows[0]["spec"]["associations"],
            "replicates": len(rows),
            "perfect_fit_rate": float(np.mean([row["result"]["perfect_fit"] for row in rows])),
            "mean_train_accuracy": float(np.mean([row["result"]["train_accuracy"] for row in rows])),
            "mean_random_label_evaluation_accuracy": float(
                np.mean(
                    [
                        row["result"]["independent_random_label_evaluation_accuracy"]
                        for row in rows
                    ]
                )
            ),
        }
        curves.append(item)
        by_k.setdefault(inputs, []).append(item)
    boundaries: list[dict[str, Any]] = []
    for inputs, rows in sorted(by_k.items()):
        rows = sorted(rows, key=lambda item: item["association_ratio"])
        fit = [row for row in rows if row["perfect_fit_rate"] >= 2 / 3]
        fail = [row for row in rows if row["perfect_fit_rate"] <= 1 / 3]
        observed = bool(fit and fail and max(x["association_ratio"] for x in fit) < max(x["association_ratio"] for x in fail))
        boundaries.append(
            {
                "inputs": inputs,
                "status": "SATURATION_TRANSITION_OBSERVED" if observed else "SATURATION_NOT_IDENTIFIED",
                "last_ratio_with_majority_perfect_fit": max(
                    (row["association_ratio"] for row in fit), default=None
                ),
                "first_ratio_with_majority_failure": min(
                    (row["association_ratio"] for row in fail), default=None
                ),
                "endpoint_is_never_called_capacity_without_both_regimes": True,
            }
        )
    return {"curves": curves, "boundaries": boundaries}


def derive_summary(records: list[dict[str, Any]], design: dict[str, Any]) -> dict[str, Any]:
    expected = expected_units(design)
    by_id = {record["unit_id"]: record for record in records}
    expected_ids = {_unit_id(kind, spec) for kind, spec in expected}
    if set(by_id) != expected_ids or len(by_id) != len(records):
        raise PilotRefusal("record population is incomplete, duplicated or foreign")
    for kind, spec in expected:
        verify_unit_record(by_id[_unit_id(kind, spec)], design, kind, spec)
    counts = {
        kind: sum(record["kind"] == kind for record in records)
        for kind in ("threshold_neuron", "boolean_mlp", "temporal_mlp")
    }
    mlp_records = [record for record in records if record["kind"] != "threshold_neuron"]
    stop_observed = sum(
        record["result"]["stopping"]["observed_stop_epoch"] is not None
        for record in mlp_records
    )
    selection_eval = [
        record["result"]["selection_checkpoint"]["scores"]["evaluation"]
        for record in mlp_records
    ]
    diagnostic_eval = [
        record["result"]["diagnostic_checkpoint"]["scores"]["evaluation"]
        for record in mlp_records
    ]
    body = {
        "schema": SUMMARY_SCHEMA,
        "disposition": "M0_M2_PILOT_COMPLETE_CONFIRMATORY_DESIGN_READY_FOR_REVIEW",
        "scope": "mechanics_and_range_evidence_only",
        "design_sha256": design["design_sha256"],
        "code_identity": design["code_identity"],
        "unit_counts": counts,
        "units_total": len(records),
        "all_units_completed": True,
        "threshold_reference": _threshold_summary(records),
        "training_trajectory": {
            "mlp_units": len(mlp_records),
            "early_stop_observed_units": stop_observed,
            "mean_selected_evaluation_score": float(np.mean(selection_eval)),
            "mean_diagnostic_end_evaluation_score": float(np.mean(diagnostic_eval)),
            "evaluation_governed_stopping": False,
            "diagnostic_continuation_selected": False,
        },
        "resource": {
            "sum_unit_cpu_wall_seconds": float(
                sum(record["cpu_wall_seconds"] for record in records)
            ),
            "peak_rss_bytes": int(max(record["peak_rss_bytes"] for record in records)),
        },
        "claims_not_made": [
            "exact_kolmogorov_complexity",
            "intelligence_in_bits",
            "knowledge_localized_to_individual_weights",
            "residual_capacity_from_repeated_weights",
            "confirmatory_capacity_law",
            "doin_or_live_eligibility",
        ],
    }
    return mic.seal_document(body, "summary_sha256")


def _load_records(records_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(records_dir.glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise PilotRefusal(f"record is not an object: {path.name}")
        records.append(value)
    return records


def execute(design: dict[str, Any], output_root: Path, stop_file: Path | None) -> dict[str, Any]:
    if torch.cuda.is_available():
        raise PilotRefusal("CUDA must be hidden for the CPU pilot")
    torch.set_num_threads(int(design["resources"]["torch_threads"]))
    _private_dir(output_root)
    records_dir = output_root / "records"
    _private_dir(records_dir)
    started = time.monotonic()
    units = expected_units(design)
    for index, (kind, spec) in enumerate(units, start=1):
        if stop_file is not None and stop_file.exists():
            raise PilotRefusal("external stop-file requested")
        elapsed = time.monotonic() - started
        if elapsed >= design["resources"]["max_wall_seconds"]:
            raise PilotRefusal("pilot wall-clock budget reached")
        unit_id = _unit_id(kind, spec)
        path = records_dir / f"{unit_id}.json"
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            verify_unit_record(existing, design, kind, spec)
        else:
            record = _run_unit(design, kind, spec)
            if record["peak_rss_bytes"] > design["resources"]["max_peak_rss_bytes"]:
                raise PilotRefusal("pilot RSS budget reached")
            verify_unit_record(record, design, kind, spec)
            _exclusive_json(path, record)
        _replace_json(
            output_root / "HEARTBEAT.json",
            {
                "schema": "agent_multi.model_capacity_m0_m2_heartbeat.v1",
                "completed": index,
                "total": len(units),
                "current_unit": unit_id,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "device": "cpu",
            },
        )
    records = _load_records(records_dir)
    summary = derive_summary(records, design)
    summary_path = output_root / "SUMMARY.json"
    if summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if existing != summary:
            raise PilotRefusal("existing summary differs from re-derived summary")
    else:
        _exclusive_json(summary_path, summary)
    _replace_json(
        output_root / "STATUS.json",
        {
            "schema": "agent_multi.model_capacity_m0_m2_status.v1",
            "status": "COMPLETED",
            "units_completed": len(records),
            "summary_sha256": summary["summary_sha256"],
            "elapsed_seconds": round(time.monotonic() - started, 3),
        },
    )
    return summary


def verify_run(design: dict[str, Any], output_root: Path) -> dict[str, Any]:
    records = _load_records(output_root / "records")
    derived = derive_summary(records, design)
    published = json.loads((output_root / "SUMMARY.json").read_text(encoding="utf-8"))
    if derived != published:
        raise PilotRefusal("published summary does not match record re-derivation")
    return derived


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--make-design", action="store_true")
    action.add_argument("--execute", action="store_true")
    action.add_argument("--verify", action="store_true")
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--stop-file", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.make_design:
            design = make_design()
            args.design.parent.mkdir(parents=True, exist_ok=True)
            _exclusive_json(args.design, design)
            print(json.dumps(design, sort_keys=True, indent=2))
            return 0
        if args.output_root is None:
            raise PilotRefusal("--output-root is required for execute/verify")
        design = load_design(args.design)
        if args.execute:
            summary = execute(design, args.output_root, args.stop_file)
        else:
            summary = verify_run(design, args.output_root)
        print(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False))
        return 0
    except (PilotRefusal, mic.ModelInformationError, OSError, ValueError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
