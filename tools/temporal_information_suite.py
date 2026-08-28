#!/usr/bin/env python3
"""Temporal-information acceptance suite runner (order b06ec0c7 §4;
continuation 2026-08-28). REPRESENTATION_DIAGNOSTIC — never a
promotion authority. CPU, bounded.

Per family (real sealed candidate encoders) and per control encoder
(random init, shuffled-time inputs):

* structural controls: future immutability (bitwise), newest-bar
  sensitivity, time reversal, save/load bit-exactness;
* synthetic-signal diagnostics with KNOWN periodicity, phase, regime
  change, constant/duplicate/noise channels: causal ridge probes for
  next-bar prediction of the periodic channel, against shuffled-time
  and phase-randomized surrogates and a random encoder;
* effective rank / collapse diagnostics;
* lagged cross-correlation preservation (linearly decodable memory).

A family PASSES the temporal gate only if every structural control
holds AND its future-facing synthetic probe beats BOTH the
shuffled-time and the random-encoder controls. Constant/duplicate/
noise channels must not manufacture predictive success (the noise
probe must stay near zero)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-dir", required=True)
    parser.add_argument("--bars", type=int, default=1200)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    import numpy as np
    import torch

    from agent_plugins.grouped_architecture import (
        snapshot_effective_config)
    from agent_plugins.pretrained_branch_loader import verify_source
    from agent_plugins.temporal_information import (
        TemporalControlFailure, control_future_immutability,
        control_newest_bar_sensitivity, control_save_load_bit_exact,
        control_time_reversal, effective_rank,
        lagged_correlation_preservation, phase_randomized_surrogate,
        probe_r2, synthetic_signals, window_embeddings)
    from app.plugin_loader import load_plugin

    pretrain_dir = Path(args.pretrain_dir)
    split_contract = json.loads(
        (REPO / "examples/config/phase_3_eth_sac_dynamics/splits/"
         "eth_nested_split_contract_o2022_paired_v1.json"
         ).read_text())
    data_path = Path(split_contract["source_csv"])
    source = verify_source(pretrain_dir, REPO, data_path)
    contract = source["contract"]
    manifest = source["manifest"]
    snapshot = snapshot_effective_config(
        REPO / "examples/config/"
        "project3_ethusdt_4h_sac_grouped_strong_v1.json")
    window = int(snapshot["env_config"]["window_size"])

    from agent_plugins.component_config import deep_merge_strict
    from agent_plugins.pretrained_branch_loader import (
        strict_load_encoder)

    def build_encoder(branch, load_weights: bool, seed: int = 0):
        plugin_class, _ = load_plugin("feature_branch.plugins",
                                      branch["plugin"])
        params = deep_merge_strict(plugin_class.plugin_params,
                                   branch["params"], path="p")
        torch.manual_seed(seed)
        module, _dim = plugin_class.build(
            len(branch["features"]), window, params)
        if load_weights:
            entry = manifest["artifacts"][branch["name"]]
            state = torch.load(
                pretrain_dir / entry["encoder_file"],
                weights_only=True)
            strict_load_encoder(module, state, branch["name"])
        module.eval()
        return module

    report = {
        "schema": "agent_multi.temporal_information_suite.v1",
        "classification": "REPRESENTATION_DIAGNOSTIC",
        "candidate_seal": json.loads(
            (pretrain_dir / "generation.json").read_text())[
            "manifest_sha256"],
        "window": window,
        "bars": int(args.bars),
        "families": {},
    }
    rng = np.random.default_rng(7)

    for branch in contract["branches"]:
        family = branch["name"]
        features = len(branch["features"])
        encoder = build_encoder(branch, load_weights=True)
        random_encoder = build_encoder(branch, load_weights=False,
                                       seed=1234)
        entry: dict = {"feature_count": features}

        # --- structural controls (typed failures surface as facts)
        controls = {}
        for name, fn in (
                ("future_immutability", control_future_immutability),
                ("newest_bar_sensitivity",
                 control_newest_bar_sensitivity),
                ("time_reversal", control_time_reversal),
                ("save_load_bit_exact", control_save_load_bit_exact)):
            try:
                controls[name] = fn(encoder, window, features)
            except TemporalControlFailure as exc:
                controls[name] = {"control": name, "passed": False,
                                  "failure": str(exc)}
        entry["structural_controls"] = controls

        # --- synthetic-signal diagnostics
        synthetic = synthetic_signals(args.bars, features)
        series = synthetic["series"]
        emb = window_embeddings(encoder, series, window)
        emb_random = window_embeddings(random_encoder, series, window)
        shuffled = series[rng.permutation(len(series))]
        emb_shuffled_time = window_embeddings(encoder, shuffled,
                                              window)
        surrogate = phase_randomized_surrogate(series)
        emb_surrogate = window_embeddings(encoder, surrogate, window)

        # future-facing target: next bar of the PERIODIC channel
        target = series[window:, 0]
        noise_target = rng.standard_normal(len(target))

        def aligned(embeddings):
            m = min(len(embeddings) - 1, len(target))
            return embeddings[:m], target[:m]

        e0, t0 = aligned(emb)
        er, _ = aligned(emb_random)
        es, _ = aligned(emb_shuffled_time)
        ep, _ = aligned(emb_surrogate)
        probes = {
            "pretrained_next_bar_r2": probe_r2(e0, t0),
            "random_encoder_next_bar_r2": probe_r2(er[:len(t0)], t0),
            "shuffled_time_next_bar_r2": probe_r2(es[:len(t0)], t0),
            "phase_surrogate_next_bar_r2": probe_r2(ep[:len(t0)], t0),
            "noise_target_r2_must_be_near_zero": probe_r2(
                e0, noise_target[:len(t0)]),
        }
        entry["synthetic_probes"] = probes
        entry["effective_rank"] = effective_rank(emb)
        entry["lagged_memory_r2"] = lagged_correlation_preservation(
            emb[:-1], series[window - 1:-1, 0])

        structural_pass = all(c.get("passed") for c in
                              controls.values())
        # 374-376 discipline: when the RANDOM-encoder control already
        # saturates the synthetic task (r2 >= 0.95), beating it is not
        # informative — the gate declares the ceiling saturated
        # instead of fabricating a PASS or FAIL from a non-informative
        # diagnostic
        ceiling_saturated = probes[
            "random_encoder_next_bar_r2"] >= 0.95
        beats_controls = (
            probes["pretrained_next_bar_r2"]
            > probes["shuffled_time_next_bar_r2"]
            and probes["pretrained_next_bar_r2"]
            > probes["random_encoder_next_bar_r2"])
        no_false_success = probes[
            "noise_target_r2_must_be_near_zero"] < 0.05
        if not structural_pass or not no_false_success:
            verdict = "FAIL"
        elif ceiling_saturated:
            verdict = "CEILING_SATURATED_INCONCLUSIVE"
        elif beats_controls:
            verdict = "PASS"
        else:
            verdict = "FAIL"
        entry["temporal_gate"] = {
            "structural_controls_pass": structural_pass,
            "random_encoder_ceiling_saturated": ceiling_saturated,
            "beats_shuffled_time_and_random_encoder": beats_controls,
            "no_false_predictive_success": no_false_success,
            "verdict": verdict,
        }
        report["families"][family] = entry

    report["summary"] = {
        family: entry["temporal_gate"]["verdict"]
        for family, entry in report["families"].items()}
    payload = json.dumps(report, indent=1)
    print(json.dumps({"summary": report["summary"],
                      "per_family_pretrained_r2": {
                          f: e["synthetic_probes"][
                              "pretrained_next_bar_r2"]
                          for f, e in report["families"].items()},
                      "shuffled_time_r2": {
                          f: e["synthetic_probes"][
                              "shuffled_time_next_bar_r2"]
                          for f, e in report["families"].items()}},
                     indent=1))
    if args.output:
        Path(args.output).write_text(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
