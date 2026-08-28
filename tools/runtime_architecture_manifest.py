#!/usr/bin/env python3
"""Runtime architecture manifest v2 (steps-1-2 correction order
2026-08-28, points 7-9). Three STRICTLY SEPARATED typed sections:

* ``runtime_measured`` — facts measured from the executing modules:
  every ordered forward-hook record (qualified path, input/output
  shapes, dtype, device, own parameters), introspected topology
  (actual Conv1d kernels/dilations, GRU/attention/transformer
  attributes, registered causal-mask buffers), leaf-parameter
  conservation, and EMPIRICAL per-bar temporal influence measured by
  one-position perturbations over all 32 bars.
* ``module_declared_contracts`` — reduction semantics reviewed from
  the exact executing source, tied to each module file's sha256.
* ``literature_comparison_declaration`` — a REVIEWED DECLARATION,
  explicitly NOT runtime-derived, corrected against the executing
  code (audit findings: channel-INDEPENDENT patching with final-token
  selection; TFT-style HAS variable selection + GRU core + final
  timestep; TimesNet-style takes the final folded cell).

MECHANICS_ONLY. The extractor stays an experimental candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


MODULE_FILES = {
    "patchtst_branch": "feature_branch_plugins/patchtst_branch.py",
    "tft_branch": "feature_branch_plugins/tft_branch.py",
    "timesnet_branch": "feature_branch_plugins/timesnet_branch.py",
    "tcn_branch": "feature_branch_plugins/tcn_branch.py",
    "gru_branch": "feature_branch_plugins/gru_branch.py",
    "mlp_branch": "feature_branch_plugins/mlp_branch.py",
}

# Reviewed against the executing forward() of the exact module file
# whose sha256 is bound next to each entry at generation time.
REDUCTION_CONTRACTS = {
    "patchtst_branch": (
        "per channel: causal-masked transformer over endpoint-"
        "anchored patch tokens; SELECTS THE FINAL PATCH TOKEN "
        "(enc[:, -1, :]) per channel, concatenates channels, linear "
        "head to d_model"),
    "tft_branch": (
        "GRN-softmax variable-selection weights over features, "
        "per-feature embedding summed by those weights, GRU temporal "
        "core, causal multi-head self-attention, output GRN on "
        "core+attended; SELECTS THE FINAL TIMESTEP ([:, -1, :])"),
    "timesnet_branch": (
        "FFT amplitude top-k period selection; per period: fold to "
        "(cycles, period), two Conv2d + GELU, SELECT THE FINAL CELL "
        "(conv[0, :, -1, -1]); amplitude-softmax weighted sum over "
        "periods, LayerNorm"),
    "tcn_branch": "last causal step of the dilated conv stack",
    "gru_branch": "last hidden state of the unidirectional GRU",
    "mlp_branch": "feed-forward over concatenated state scalars "
                  "(no temporal axis)",
}

LITERATURE_DECLARATION = {
    "status": ("REVIEWED_DECLARATION — corrected against the "
               "executing code at the digests bound in "
               "module_declared_contracts; NOT runtime-derived"),
    "patchtst_branch": {
        "style_name": "PatchTST-style",
        "reference": "Nie et al. 2023",
        "shared_with_reference": [
            "channel-INDEPENDENT patching (unfold per channel, "
            "tokens per (batch x channel) sequence)",
            "transformer encoder over patch tokens"],
        "differences": [
            "endpoint-anchored patch offset drops the OLDEST "
            "remainder bars (DATA-SOTA-331), not zero-start unfold",
            "CAUSAL mask across patch tokens (the reference uses "
            "unmasked encoding)",
            "final-patch-token selection + linear head over "
            "concatenated per-channel tokens, not the reference "
            "flatten head",
            "no RevIN (preprocessing is the rolling-zscore feature "
            "contract)"],
    },
    "tft_branch": {
        "style_name": "TFT-style",
        "reference": "Lim et al. 2021",
        "shared_with_reference": [
            "GRN-softmax VARIABLE SELECTION over input features",
            "recurrent temporal core (GRU)",
            "interpretable multi-head attention with gated residual "
            "combination"],
        "differences": [
            "no static covariate encoders, no encoder-decoder split, "
            "no quantile output heads",
            "single GRU layer instead of LSTM encoder-decoder",
            "final-timestep selection as the branch representation"],
    },
    "timesnet_branch": {
        "style_name": "TimesNet-style",
        "reference": "Wu et al. 2023",
        "shared_with_reference": [
            "FFT amplitude top-k period discovery",
            "1D->2D folding by period and 2D convolution",
            "amplitude-weighted aggregation over periods"],
        "differences": [
            "ONE TimesBlock-like stage with a two-layer plain Conv2d "
            "(not stacked blocks with inception towers)",
            "final-cell selection per folded map before weighting "
            "(not full 2D->1D unfolding)"],
    },
    "tcn_branch": {"style_name": "causal TCN",
                   "reference": "Bai et al. 2018",
                   "differences": [
                       "left-padded causal Conv1d stack, additive "
                       "skip when channels match; no weight-norm"]},
    "gru_branch": {"style_name": "unidirectional GRU",
                   "reference": "standard nn.GRU",
                   "differences": ["last-hidden reduction"]},
}


def introspect_topology(module) -> dict:
    """Facts read from the ACTUAL constructed submodules — kernels,
    dilations, heads, layers, registered mask buffers."""
    import torch

    convs1d, convs2d, grus, attentions, transformers = [], [], [], \
        [], []
    masks = []
    for name, sub in module.named_modules():
        if isinstance(sub, torch.nn.Conv1d):
            convs1d.append({"path": name,
                            "kernel_size": list(sub.kernel_size),
                            "dilation": list(sub.dilation),
                            "in_channels": sub.in_channels,
                            "out_channels": sub.out_channels})
        elif isinstance(sub, torch.nn.Conv2d):
            convs2d.append({"path": name,
                            "kernel_size": list(sub.kernel_size),
                            "padding": list(sub.padding)})
        elif isinstance(sub, torch.nn.GRU):
            grus.append({"path": name, "layers": sub.num_layers,
                         "hidden_size": sub.hidden_size,
                         "bidirectional": sub.bidirectional})
        elif isinstance(sub, torch.nn.MultiheadAttention):
            attentions.append({"path": name,
                               "num_heads": sub.num_heads,
                               "embed_dim": sub.embed_dim})
        elif isinstance(sub, torch.nn.TransformerEncoder):
            transformers.append({"path": name,
                                 "num_layers": sub.num_layers})
    for buffer_name, buffer in module.named_buffers():
        if "mask" in buffer_name:
            masks.append({"buffer": buffer_name,
                          "shape": list(buffer.shape)})
    return {"conv1d": convs1d, "conv2d": convs2d, "gru": grus,
            "attention": attentions, "transformer": transformers,
            "registered_mask_buffers": masks}


def theoretical_receptive_field(topology: dict, window: int) -> dict:
    convs = topology["conv1d"]
    if convs:
        span = 1
        for conv in convs:
            span += (conv["kernel_size"][0] - 1) * conv["dilation"][0]
        return {"bars": min(span, window),
                "basis": ("INTROSPECTED Conv1d kernels/dilations: "
                          "1 + sum((k-1)*d) = " + str(span)
                          + ", clipped to the window")}
    if topology["gru"] or topology["attention"] or \
            topology["transformer"] or topology["conv2d"]:
        return {"bars": window,
                "basis": "recurrence/attention/FFT-fold reach the "
                         "whole window STRUCTURALLY; empirical "
                         "influence is measured separately"}
    return {"bars": 0, "basis": "no temporal axis"}


def empirical_influence(module, window: int, features: int,
                        seed: int = 0) -> dict:
    """One-position perturbations over every bar: does mutating bar t
    change the representation? Measured, not assumed."""
    import torch

    torch.manual_seed(seed)
    probe = torch.randn(1, window, features)
    module.eval()
    with torch.no_grad():
        base = module(probe)
        deltas = []
        for t in range(window):
            mutated = probe.clone()
            mutated[:, t, :] += 1.0
            out = module(mutated)
            deltas.append(float(
                torch.linalg.vector_norm(out - base).item()))
    scale = max(max(deltas), 1e-12)
    nonzero = [i for i, d in enumerate(deltas)
               if d > 1e-9 * scale]
    return {
        "per_bar_l2_delta": [round(d, 8) for d in deltas],
        "bars_with_nonzero_influence": len(nonzero),
        "first_influential_bar": nonzero[0] if nonzero else None,
        "last_influential_bar": nonzero[-1] if nonzero else None,
        "newest_bar_influential": (window - 1) in nonzero,
        "method": "add +1.0 to every feature at ONE bar of a fixed "
                  "random probe; L2 of the output change; threshold "
                  "1e-9 relative to the max delta",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    import gymnasium as gym
    import numpy as np
    import torch

    from agent_plugins.dispatch_authorization import (
        resolve_required_entry_points)
    from agent_plugins.grouped_architecture import (
        snapshot_effective_config)
    from agent_plugins.grouped_features_extractor import (
        build_grouped_extractor_class)

    snapshot = snapshot_effective_config(
        REPO / "examples/config/"
        "project3_ethusdt_4h_sac_grouped_strong_v1.json")
    materialized = snapshot["materialized"]
    architecture = materialized["architecture"]
    env_cfg = snapshot["env_config"]
    window = int(env_cfg["window_size"])
    feature_count = len(env_cfg["feature_columns"])
    state_keys = list(architecture.get("state_keys") or [])

    spaces = {"features": gym.spaces.Box(
        -np.inf, np.inf, (window, feature_count), dtype=np.float32)}
    for key in state_keys:
        spaces[key] = gym.spaces.Box(-np.inf, np.inf, (1,),
                                     dtype=np.float32)
    torch.manual_seed(0)
    extractor = build_grouped_extractor_class()(
        gym.spaces.Dict(spaces), architecture=architecture)
    extractor.eval()

    # ---- runtime_measured: FULL ordered hook log --------------------
    hook_log: list = []
    hooks = []

    def register(owner: str, qualified: str, module):
        def hook(mod, inputs, output):
            def shape_of(x):
                if torch.is_tensor(x):
                    return {"shape": list(x.shape),
                            "dtype": str(x.dtype),
                            "device": str(x.device)}
                if isinstance(x, (tuple, list)):
                    return [shape_of(v) for v in x]
                return str(type(x).__name__)
            own_params = sum(
                p.numel() for p in mod.parameters(recurse=False))
            hook_log.append({
                "call_index": len(hook_log),
                "owner": owner,
                "qualified_path": qualified,
                "module_type": type(mod).__name__,
                "input": [shape_of(v) for v in inputs],
                "output": shape_of(output),
                "own_parameters": own_params,
            })
        hooks.append(module.register_forward_hook(hook))

    branches = architecture["branches"]
    for index, branch in enumerate(branches):
        module = extractor.temporal_branches[index]
        for name, sub in module.named_modules():
            register(branch["name"], name or "<root>", sub)
    if state_keys and extractor.state_branch is not None:
        for name, sub in extractor.state_branch.named_modules():
            register("agent_state", name or "<root>", sub)
    for name, sub in extractor.fusion.named_modules():
        register("fusion", name or "<root>", sub)

    batch = {k: torch.zeros((2,) + s.shape)
             for k, s in spaces.items()}
    with torch.no_grad():
        fused = extractor(batch)
    for h in hooks:
        h.remove()

    # leaf-parameter conservation: every parameter counted exactly
    # once by identity — no double counting of shared modules
    seen: set = set()
    leaf_total = 0
    for p in extractor.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            leaf_total += p.numel()
    named_total = sum(p.numel() for p in extractor.parameters())
    assert leaf_total == named_total, "shared-parameter double count"

    def params_of(module):
        return sum(p.numel() for p in module.parameters())

    branch_sections = []
    for index, branch in enumerate(branches):
        module = extractor.temporal_branches[index]
        plugin = branch["plugin"]
        features = list(branch["features"])
        topology = introspect_topology(module)
        with torch.no_grad():
            latent = module(torch.zeros(2, window, len(features)))
        branch_sections.append({
            "family": branch["name"],
            "plugin": plugin,
            "ordered_features": features,
            "input_window_bars": window,
            "trainable_parameters": params_of(module),
            "latent_dimension": int(latent.shape[-1]),
            "compression_ratio_input_values_to_latent": round(
                window * len(features) / int(latent.shape[-1]), 2),
            "introspected_topology": topology,
            "theoretical_receptive_field":
                theoretical_receptive_field(topology, window),
            "empirical_temporal_influence": empirical_influence(
                module, window, len(features)),
        })

    runtime_measured = {
        "derivation": "constructed via the canonical materializer at "
                      "this commit; forward hooks on EVERY submodule; "
                      "one probe forward; per-branch perturbation "
                      "sweeps",
        "architecture_digest": materialized["architecture_digest"],
        "config_sha256": snapshot["config_sha256"],
        "input_window_bars_h4": window,
        "feature_count": feature_count,
        "branches": branch_sections,
        "state_branch": ({
            "family": "agent_state",
            "plugin": architecture.get("state_branch"),
            "note": "randomly initialized, NOT pretrained",
            "state_keys": state_keys,
            "trainable_parameters": params_of(
                extractor.state_branch)}
            if state_keys and extractor.state_branch is not None
            else None),
        "fusion": {
            "declared": architecture.get("fusion"),
            "note": "randomly initialized, NOT pretrained",
            "trainable_parameters": params_of(extractor.fusion),
            "output_dimension": int(fused.shape[-1]),
            "introspected_topology": introspect_topology(
                extractor.fusion),
        },
        "total_trainable_extractor_parameters": leaf_total,
        "leaf_parameter_conservation": (
            f"identity-deduplicated leaf sum {leaf_total} == "
            f"named-parameter sum {named_total} — no shared-module "
            "double counting"),
        "layer_by_layer_hook_log": hook_log,
        "dropout_values_present": sorted({
            float(m.p) for m in extractor.modules()
            if isinstance(m, torch.nn.Dropout)}),
        "normalization_layers_present": sorted({
            type(m).__name__ for m in extractor.modules()
            if "Norm" in type(m).__name__}),
        "entry_point_metadata_digest":
            resolve_required_entry_points(REPO)[
                "entry_point_metadata_digest"],
    }

    module_declared = {
        "status": "explicit module contracts reviewed from the exact "
                  "executing source, bound to each file digest",
        "contracts": {
            plugin: {
                "module_file": MODULE_FILES[plugin],
                "module_sha256": _sha(REPO / MODULE_FILES[plugin]),
                "reduction_semantics": REDUCTION_CONTRACTS[plugin],
            } for plugin in MODULE_FILES
        },
    }

    contract = json.loads(
        (REPO / "examples/config/"
         "pretrain_contract_eth_h4_o2022_full5_pcgrad_v1.json"
         ).read_text())
    manifest = {
        "schema": "agent_multi.runtime_architecture_manifest.v2",
        "classification": "MECHANICS_ONLY",
        "supersedes": "RUNTIME_ARCHITECTURE_MANIFEST_2026_08_28.json",
        "status_claim": ("EXPERIMENTAL CANDIDATE — not a proven SOTA "
                         "trading extractor; branch names are "
                         "'-style'; no reference parity claimed"),
        "runtime_measured": runtime_measured,
        "module_declared_contracts": module_declared,
        "literature_comparison_declaration": LITERATURE_DECLARATION,
        "training_facts": {
            "initialization": "PyTorch defaults under manual_seed("
                              "train_seed) at SAC construction; the "
                              "treatment overwrites the five temporal "
                              "branches from the sealed generation "
                              "with bit parity",
            "optimizers": {
                "sac": "Adam lr 3e-4 (design binding); extractor "
                       "params inside each network's optimizer; "
                       "target critic polyak-tracked",
                "pretraining": {
                    "balancing": contract.get("objective_balancing"),
                    "gradient_combiner": contract.get(
                        "gradient_combiner")}},
            "pretraining_objectives": sorted(
                (contract.get("objectives") or {}).keys()),
            "transfer": {
                "transferred": "five temporal-branch encoders into "
                               "actor+critic+critic_target (separate "
                               "extractors)",
                "excluded": "heads/adapters, optimizer/replay/"
                            "calibration payloads, state MLP, fusion",
            },
        },
    }
    payload = json.dumps(manifest, indent=1)
    print(json.dumps({
        "total_params": leaf_total,
        "hook_records": len(hook_log),
        "per_branch_influence_bars": {
            b["family"]: b["empirical_temporal_influence"][
                "bars_with_nonzero_influence"]
            for b in branch_sections},
        "tcn_theoretical_rf": next(
            b["theoretical_receptive_field"]
            for b in branch_sections
            if b["plugin"] == "tcn_branch"),
    }, indent=1))
    if args.output:
        Path(args.output).write_text(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
