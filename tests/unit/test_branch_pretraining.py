"""WP-PRETRAIN v2 regressions (Data-First order @7886de39;
DATA-SOTA-341..346 corrected — the finding-specific frozen
counterexamples live in test_data_sota_341_346_regressions.py).

Proven here: contract v2 refusals, causal per-origin fit boundary,
strictly-forward targets, masked-only reconstruction scoring, pinball
asymmetry, monitor split, end-to-end runner behavior with the executing
preprocessor, EXACT resume across interruption (complete artifact set),
torn-generation refusal, and manifest sanitization.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from agent_plugins.branch_pretraining import (  # noqa: E402
    PretrainContractError, build_step_index, forward_log_return_targets,
    load_fit_slice, masked_reconstruction_loss, pinball_loss,
    refuse_on_identity_drift, sample_span_mask, three_way_split,
    validate_contract)

SOURCE_CONFIG = {
    "window_size": 16,
    "feature_columns": ["f1", "f2", "f3", "f4", "f5", "f6"],
    "feature_binary_columns": [],
    "feature_scaling": "rolling_zscore",
    "feature_scaling_window": 32,
    "include_price_window": False,
    "include_agent_state": False,
}

BASE_CONTRACT = {
    "schema": "agent_multi.pretrain_contract.v3",
    "score_origin": {"origin_id": "o2022",
                     "score_start": "2024-01-01"},
    "objective_domain": "runtime_domain_with_target_adapters",
    "fit_end": "2023-12-31T23:00:00",
    "observation_pipeline": {
        "preprocessor_plugin": "feature_window_preprocessor",
        "source_config": "<set-by-fixture>",
    },
    "date_column": "DATE_TIME", "close_column": "CLOSE",
    "feature_columns": ["f1", "f2", "f3", "f4", "f5", "f6"],
    "window_size": 16, "window_stride": 1, "warmup_bars": 34,
    "branches": [
        {"name": "alpha", "plugin": "gru_branch",
         "params": {"hidden_size": 8},
         "features": ["f1", "f2", "f3"]},
        {"name": "beta", "plugin": "tcn_branch",
         "params": {"channels": [8, 8]},
         "features": ["f4", "f5", "f6"]},
    ],
    "normalization_policies": {
        "alpha": {"policy": "identity_preprocessed"},
        "beta": {"policy": "window_zscore_visible", "eps": 1e-5},
    },
    "objectives": {
        "masked_patch_reconstruction":
            {"weight": 1.0, "mask_ratio": 0.25, "mask_span": 4},
        "multi_horizon_quantile":
            {"weight": 1.0, "horizons": [1, 3],
             "quantiles": [0.1, 0.5, 0.9]},
    },
    "objective_balancing": {"method": "inverse_initial_loss",
                            "floor": 1e-6},
    "partition_fractions": {"calibration": 0.2, "monitor": 0.2},
    "optimizer": {"lr": 0.001, "batch_size": 32},
    "epochs": 2, "seed": 3, "max_windows": None,
}


def contract_with(**overrides):
    merged = json.loads(json.dumps(BASE_CONTRACT))
    merged.update(overrides)
    return merged


# ---------------------------------------------------------------- contract

@pytest.mark.parametrize("mutation, fragment", [
    ({"schema": "agent_multi.pretrain_contract.v2"}, "unsupported"),
    ({"fit_end": "2024-06-01T00:00:00"}, "does not precede"),
    ({"fit_end": ""}, "non-empty ISO-8601"),
    ({"score_origin": {}}, "score_origin"),
    ({"observation_pipeline": {}}, "executing preprocessor"),
    ({"objectives": {}}, "at least one objective"),
    ({"objectives": {"contrastive": {"weight": 1.0}}},
     "unknown objectives"),
    ({"epochs": True}, "non-boolean integer"),
    ({"branches": []}, "branches must not be empty"),
    ({"normalization_policies": None}, "normalization_policies"),
    ({"objective_domain": "mixed_freely"}, "objective_domain"),
    ({"normalization_policies": {"alpha":
        {"policy": "identity_preprocessed"}}}, "cover the branch"),
    ({"objective_balancing": {}}, "inverse_initial_loss"),
    ({"partition_fractions": {"calibration": 0.2, "monitor": 0.0}},
     "partition_fractions.monitor"),
    ({"warmup_bars": 1}, "warmup_bars"),
], ids=["v2-schema", "fit-not-before-score", "fit-missing",
        "origin-missing", "pipeline-missing", "no-objective",
        "unknown-objective", "bool-epochs", "no-branch", "no-policies",
        "no-domain", "policy-gap", "no-balancing", "monitor-0",
        "warmup-1"])
def test_contract_refusals(mutation, fragment):
    with pytest.raises(PretrainContractError, match=fragment):
        validate_contract(contract_with(**mutation))


@pytest.mark.parametrize("spec, fragment", [
    ({"weight": 1.0, "mask_ratio": 0.0, "mask_span": 4}, "mask_ratio"),
    ({"weight": 1.0, "mask_ratio": 1.0, "mask_span": 4}, "mask_ratio"),
    ({"weight": 1.0, "mask_ratio": 0.25, "mask_span": 16}, "mask_span"),
    ({"weight": 0.0, "mask_ratio": 0.25, "mask_span": 4}, "weight"),
    ({"weight": "1", "mask_ratio": 0.25, "mask_span": 4}, "non-boolean"),
], ids=["ratio-0", "ratio-1", "span-eq-window", "weight-0", "str-weight"])
def test_reconstruction_spec_refusals(spec, fragment):
    contract = contract_with()
    contract["objectives"]["masked_patch_reconstruction"] = spec
    with pytest.raises(PretrainContractError, match=fragment):
        validate_contract(contract)


@pytest.mark.parametrize("spec, fragment", [
    ({"weight": 1.0, "horizons": [1, 1], "quantiles": [0.5]}, "unique"),
    ({"weight": 1.0, "horizons": [1], "quantiles": [0.0]}, "quantiles"),
    ({"weight": 1.0, "horizons": [1], "quantiles": [1.0]}, "quantiles"),
    ({"weight": 1.0, "horizons": [], "quantiles": [0.5]}, "horizons"),
    ({"weight": 1.0, "horizons": [1],
      "quantiles": [0.5, 0.5]}, "strictly increasing"),
    ({"weight": 1.0, "horizons": [1],
      "quantiles": [0.9, 0.1]}, "strictly increasing"),
], ids=["dup-horizon", "q0", "q1", "no-horizon", "dup-q", "unsorted-q"])
def test_quantile_spec_refusals(spec, fragment):
    contract = contract_with()
    contract["objectives"]["multi_horizon_quantile"] = spec
    with pytest.raises(PretrainContractError, match=fragment):
        validate_contract(contract)


# ------------------------------------------------------------- fit slice

def synthetic_csv(path: Path, hours: int = 400,
                  start: str = "2023-11-01", nan_at: int | None = None):
    import pandas as pd
    rng = np.random.default_rng(11)
    stamps = pd.date_range(start, periods=hours, freq="4h")
    frame = {"DATE_TIME": stamps.strftime("%Y-%m-%d %H:%M:%S")}
    for i in range(1, 7):
        frame[f"f{i}"] = rng.normal(size=hours) * (10 ** i)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, hours)))
    if nan_at is not None:
        frame["f2"] = np.asarray(frame["f2"], dtype=float)
        frame["f2"][nan_at] = np.nan
    frame["CLOSE"] = close
    pd.DataFrame(frame).to_csv(path, index=False)
    return stamps


def test_fit_slice_structurally_excludes_rows_after_fit_end(tmp_path):
    csv = tmp_path / "d.csv"
    stamps = synthetic_csv(csv, hours=600)  # crosses into 2024
    assert str(stamps[-1]) >= "2024-01-01"
    df, cols, close = load_fit_slice(csv, contract_with())
    import pandas as pd
    assert pd.to_datetime(df["DATE_TIME"]).max() <= pd.Timestamp(
        "2023-12-31T23:00:00")
    assert len(df) < 600  # later rows are ABSENT, not merely unused


def test_fit_slice_refuses_nan(tmp_path):
    csv = tmp_path / "d.csv"
    synthetic_csv(csv, nan_at=37)
    with pytest.raises(PretrainContractError, match="NaN"):
        load_fit_slice(csv, contract_with())


def test_step_index_drops_targets_crossing_fit_end():
    steps = build_step_index(300, warmup_bars=256, stride=1,
                             max_horizon=12, max_windows=None)
    assert steps[0] == 256
    # last observed bar t-1 plus max horizon must stay inside the slice
    assert steps[-1] == 300 - 12
    with pytest.raises(PretrainContractError, match="no eligible"):
        build_step_index(260, warmup_bars=256, stride=1, max_horizon=12,
                         max_windows=None)


def test_step_index_max_windows_keeps_newest():
    steps = build_step_index(300, 256, 1, 2, max_windows=5)
    assert steps == [294, 295, 296, 297, 298]


def test_three_way_split_is_chronological_and_disjoint():
    train, calibration, monitor = three_way_split(
        list(range(100)), 0.2, 0.2)
    assert train == list(range(60))            # OLDEST block
    assert calibration == list(range(60, 80))  # middle
    assert monitor == list(range(80, 100))     # NEWEST fit-tail
    with pytest.raises(PretrainContractError, match="no training"):
        three_way_split([1, 2], 0.4, 0.4)


def test_targets_anchor_on_last_observed_bar():
    close = np.array([100.0, 110.0, 121.0, 133.1, 146.41])
    # step t=2 observes rows [t-w, 2): last observed bar is index 1
    got = forward_log_return_targets(close, steps=[2], horizons=[1, 3])
    expect = np.log(1.1)
    assert np.allclose(got, [[expect, 3 * expect]], atol=1e-6)
    with pytest.raises(PretrainContractError, match="non-positive"):
        forward_log_return_targets(np.array([1.0, -1.0, 2.0]), [1], [1])


# ------------------------------------------------------------- objectives

def test_span_mask_ratio_and_visibility():
    gen = torch.Generator().manual_seed(0)
    mask = sample_span_mask(64, 32, ratio=0.25, span=4, generator=gen)
    frac = mask.float().mean().item()
    assert 0.10 <= frac <= 0.45
    assert bool(mask.any(dim=1).all()), "every sample masks something"
    assert bool((~mask).any(dim=1).all()), "every sample keeps a step"


def test_reconstruction_loss_scores_only_masked_positions():
    windows = torch.randn(4, 8, 3)
    mask = torch.zeros(4, 8, dtype=torch.bool)
    mask[:, 2:4] = True

    class Oracle(torch.nn.Module):
        """Perfect on MASKED steps, garbage on visible ones."""
        def __init__(self):
            super().__init__()
            corrupted = windows.clone()
            corrupted[~mask] = 777.0
            self.out = corrupted.reshape(4, -1)

        def forward(self, x):
            return self.out
    loss = masked_reconstruction_loss(
        torch.nn.Identity(), Oracle(), windows, windows, mask)
    assert float(loss) == pytest.approx(0.0, abs=1e-9)
    inverted = masked_reconstruction_loss(
        torch.nn.Identity(),
        type("Zero", (torch.nn.Module,),
             {"forward": lambda self, x: torch.zeros(4, 24)})(),
        windows, windows, mask)
    assert float(inverted) > 0


def test_pinball_asymmetry_and_median_case():
    target = torch.zeros(1, 1)
    over = torch.full((1, 1, 1), 1.0)   # prediction above target
    under = torch.full((1, 1, 1), -1.0)
    q9_over = float(pinball_loss(over, target, [0.9]))
    q9_under = float(pinball_loss(under, target, [0.9]))
    assert q9_under > q9_over  # q=0.9 punishes UNDER-prediction more
    assert float(pinball_loss(over, target, [0.5])) == pytest.approx(0.5)


# ---------------------------------------------------------------- identity

def test_identity_drift_refusal_names_the_field():
    saved = {"data_sha256": "aa", "seed": 3}
    with pytest.raises(PretrainContractError,
                       match="data_sha256.*saved='aa'.*current='bb'"):
        refuse_on_identity_drift(saved, {"data_sha256": "bb", "seed": 3})
    refuse_on_identity_drift(saved, dict(saved))  # identical passes


# ------------------------------------------------------- executing runner

RUNNER = REPO / "tools/pretrain_branches.py"


def run_runner(csv: Path, contract_file: Path, out_dir: Path, *extra,
               expect_rc=0):
    env = {**os.environ, "AGENT_MULTI_ETH_CSV": str(csv),
           "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(REPO)}
    proc = subprocess.run(
        [sys.executable, str(RUNNER), "--contract", str(contract_file),
         "--output-dir", str(out_dir), *extra],
        capture_output=True, text=True, env=env, cwd=str(REPO))
    assert proc.returncode == expect_rc, proc.stdout + proc.stderr
    return proc


@pytest.fixture(scope="module")
def runner_case(tmp_path_factory):
    root = tmp_path_factory.mktemp("pretrain")
    csv = root / "synthetic.csv"
    synthetic_csv(csv, hours=260)
    source = root / "source_config.json"
    source.write_text(json.dumps(SOURCE_CONFIG))
    contract = contract_with()
    contract["observation_pipeline"]["source_config"] = str(source)
    contract_file = root / "contract.json"
    contract_file.write_text(json.dumps(contract))
    return csv, contract_file, root


def _artifact_states(out_dir: Path):
    return {p.name: torch.load(p, weights_only=True)
            for p in sorted(out_dir.glob("branch_*.pt"))}


def test_runner_end_to_end_and_exact_resume(runner_case):
    csv, contract_file, root = runner_case
    full = root / "full2"
    run_runner(csv, contract_file, full, "--epochs", "2")
    manifest = json.loads((full / "pretrain_manifest.json").read_text())
    assert manifest["completed"] is True
    assert manifest["transfer_eligibility"].startswith(
        "NOT_TRANSFER_ELIGIBLE")
    assert (full / "generation.json").is_file()
    # DATA-SOTA-349: three chronological partitions, digest-bound
    parts = manifest["partitions"]
    assert set(parts) == {"train", "calibration", "monitor"}
    assert (parts["train"]["last_step"]
            < parts["calibration"]["first_step"]
            <= parts["calibration"]["last_step"]
            < parts["monitor"]["first_step"])
    for part in parts.values():
        assert part["windows"] > 0 and part["steps_sha256"]
    for name in ("alpha", "beta"):
        progress = manifest["progress"][name]
        losses = progress["losses"]
        assert [r["epoch"] for r in losses] == [0, 1]
        for r in losses:
            assert np.isfinite(r["train"]["weighted_total"])
            assert set(r["monitor_fit_tail"]) >= {
                "reconstruction", "quantile", "quantile_crossing_rate"}
            assert r["monitor_fit_tail"]["quantile_crossing_rate"] == 0.0
            assert r["gradient_diagnostics"]["norms"]
        weights = progress["effective_weights"]
        assert weights["effective"].keys() == {"reconstruction",
                                               "quantile"}
        assert "calibration" in weights["calibrated_on"]
        assert "initial_calibration_losses" in weights
        assert manifest["artifacts"][name]["encoder_sha256"]

    # interrupt mid-second-branch, then resume the exact trajectory
    split = root / "split"
    proc = run_runner(csv, contract_file, split, "--epochs", "2",
                      "--stop-after-epochs", "3")
    assert "INTERRUPTED after 3 epoch generations" in proc.stdout
    assert "completed" not in json.loads(
        (split / "pretrain_manifest.json").read_text())
    run_runner(csv, contract_file, split, "--resume")

    # DATA-SOTA-346: the COMPLETE artifact set matches bit-for-bit —
    # encoders, heads, artifact digests and per-epoch loss records.
    ref, got = _artifact_states(full), _artifact_states(split)
    assert ref.keys() == got.keys()
    for name in ref:
        for key in ref[name]:
            assert torch.equal(ref[name][key], got[name][key]), (
                f"resume not EXACT at {name}:{key}")
    resumed = json.loads((split / "pretrain_manifest.json").read_text())
    assert resumed["artifacts"] == manifest["artifacts"]

    def strip_wall_clock(progress):
        clean = json.loads(json.dumps(progress))
        for branch in clean.values():
            for record in branch["losses"]:
                record.pop("seconds", None)
        return clean
    assert strip_wall_clock(resumed["progress"]) == \
        strip_wall_clock(manifest["progress"])


def test_runner_refuses_resume_on_data_drift(runner_case, tmp_path):
    csv, contract_file, root = runner_case
    out = root / "full2"  # completed run from the end-to-end test
    drifted = tmp_path / "drifted.csv"
    synthetic_csv(drifted, hours=260, start="2023-10-01")
    proc = run_runner(drifted, contract_file, out, "--resume",
                      expect_rc=1)
    assert "identity drift REFUSED" in proc.stderr
    assert "data_sha256" in proc.stderr


def test_runner_refuses_torn_generation(runner_case):
    csv, contract_file, root = runner_case
    out = root / "full2"
    manifest_path = out / "pretrain_manifest.json"
    original = manifest_path.read_text()
    try:
        manifest_path.write_text(original + " ")  # torn pair
        proc = run_runner(csv, contract_file, out, "--resume",
                          expect_rc=1)
        assert "TORN GENERATION" in proc.stderr
    finally:
        manifest_path.write_text(original)


def test_runner_refuses_silent_overwrite(runner_case):
    csv, contract_file, root = runner_case
    proc = run_runner(csv, contract_file, root / "full2", expect_rc=1)
    assert "never silently overwritten" in proc.stderr


def test_manifest_is_sanitized_no_absolute_paths(runner_case):
    _csv, _contract, root = runner_case
    text = (root / "full2" / "pretrain_manifest.json").read_text()
    for banned in ("/home/", "/tmp/", ".local/", "harveybc",
                   "omega", "dragon", "gamma"):
        assert banned not in text, f"manifest leaks {banned!r}"
    manifest = json.loads(text)
    assert manifest["identity"]["interpreter"].startswith("python:")
    assert "/" not in manifest["identity"]["interpreter"]
    # DATA-SOTA-346: resume identity binds the executing identities
    for key in ("runner_sha256", "code_commit", "torch_version",
                "preprocessor_module_sha256",
                "normalization_policies_digest"):
        assert key in manifest["identity"]
