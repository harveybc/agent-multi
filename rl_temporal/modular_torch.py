"""Torch implementation of the ``predictor.modular.v1`` temporal representation.

Same contract as the Keras engine at pin 3ecdb256 (predictor_plugins/
modular_temporal): column routing by fixed gather, one causal Conv1D branch per
declared feature group that keeps every input step, raw channel concatenation,
sinusoidal positional encoding right after fusion, a per-step projection, causal
full Transformer blocks, then residual Conv1D stages that reduce time by the
declared factors (default [2, 2, 1] for 24 -> 6) and channels down to
``output_channels``. The only reduction before a task head is the explicit
flatten of the ``(output_steps, output_channels)`` latent.

It plugs into Stable-Baselines3 as a ``BaseFeaturesExtractor`` so the installed
SAC and DQN consume it unchanged. Fidelity to the Keras engine is a TEST
(``rl_temporal.keras_import``), not an assumption; the Keras defaults that
matter for parity are reproduced here on purpose: exact (erf) GELU,
LayerNormalization epsilon 1e-3, attention scores scaled by 1/sqrt(key_dim),
a causal mask over keys.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from .observation_layout import ObservationLayout

CONFIG_SCHEMA = "predictor.modular.v1"
LN_EPS = 1e-3  # Keras LayerNormalization default; torch's is 1e-5


# ----------------------------------------------------------------------------
# configuration (mirrors predictor_plugins.modular_temporal.config._normalize
# for the fields this consumer depends on; the engine remains the authority)
# ----------------------------------------------------------------------------

def _default_time_factors(total_factor: int, stage_count: int) -> List[int]:
    factors = [1] * stage_count
    remainder = total_factor
    prime = 2
    while prime * prime <= remainder:
        while remainder % prime == 0:
            target = min(range(stage_count), key=factors.__getitem__)
            factors[target] *= prime
            remainder //= prime
        prime += 1
    if remainder > 1:
        target = min(range(stage_count), key=factors.__getitem__)
        factors[target] *= remainder
    return factors


def normalize_modular_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Canonical ENCODER architecture: the subset of ``predictor.modular.v1`` that
    determines branches, fusion and core. Head, horizons, targets, entry-point
    groups, alignment flags and the per-component regime/donor are dropped: a
    donor's compatibility is architectural, and regimes come from the run."""
    src = copy.deepcopy(config)
    # Unknown keys RAISE (they used to be dropped silently). `head`, `horizons`, `target_count`, `regime`,
    # `entry_point_groups` and `alignment_probe` are known engine keys this consumer deliberately ignores
    # (identity is architectural); `input_normalization` and anything else is not supported here.
    known = {"schema", "window", "sample_hours", "feature_names", "branches", "branch_steps", "core", "fusion",
             "head", "output_steps", "output_channels", "entry_point_groups", "horizons", "target_count",
             "regime", "alignment_probe"}
    unknown = sorted(set(src) - known)
    if unknown:
        raise ValueError(f"modular config keys not supported by the torch consumer: {unknown} "
                         "(NOT_SUPPORTED_BY_TORCH_CONSUMER; e.g. input_normalization)")
    for spec in src.get("branches") or []:
        bad = sorted(set(spec) - {"name", "features", "plugin", "params", "regime", "donor"})
        if bad:
            raise ValueError(f"branch spec keys not supported by the torch consumer: {bad}")
        pbad = sorted(set(spec.get("params") or {}) - {"channels", "kernel_size"})
        if pbad:
            raise ValueError(f"branch params not supported by the torch consumer: {pbad}")
    cbad = sorted(set((src.get("core") or {}).get("params") or {}) -
                  {"d_model", "heads", "blocks", "ff_dim", "dropout", "stage_channels", "time_factors", "kernel_size"})
    if cbad:
        raise ValueError(f"core params not supported by the torch consumer: {cbad}")
    if src.get("schema", CONFIG_SCHEMA) != CONFIG_SCHEMA:
        raise ValueError(f"unsupported modular config schema {src.get('schema')!r}")
    c: Dict[str, Any] = {"schema": CONFIG_SCHEMA}
    for key, default in (("window", 24), ("output_steps", 6), ("output_channels", 8)):
        c[key] = int(src.get(key, default))
    c["branch_steps"] = int(src.get("branch_steps", c["window"]))
    if c["branch_steps"] != c["window"]:
        raise ValueError("branch_steps must equal window: branches preserve the time axis")
    if "sample_hours" in src:
        c["sample_hours"] = src["sample_hours"]
    names = list(src.get("feature_names") or [])
    if not names or len(set(names)) != len(names):
        raise ValueError("feature_names must be unique ordered strings")
    c["feature_names"] = names
    if not src.get("branches"):
        raise ValueError("At least one branch is required")
    c["branches"] = []
    for spec in src["branches"]:
        if any(f not in names for f in spec["features"]):
            raise ValueError(f"branch {spec.get('name')} names a feature outside feature_names")
        params = dict(spec.get("params") or {})
        params.setdefault("channels", 16)
        params.setdefault("kernel_size", 3)
        c["branches"].append({"name": spec["name"], "features": list(spec["features"]),
                              "plugin": spec.get("plugin", "causal_conv1d"), "params": params})
    core = dict(src.get("core") or {})
    p = dict(core.get("params") or {})
    p.setdefault("d_model", 64)
    p.setdefault("heads", 4)
    p.setdefault("blocks", 2)
    p.setdefault("ff_dim", 128)
    p.setdefault("dropout", 0.0)
    p.setdefault("kernel_size", 3)
    p.setdefault("stage_channels", [32, 16, c["output_channels"]])
    p.setdefault("time_factors", _default_time_factors(c["window"] // c["output_steps"],
                                                       len(p["stage_channels"])))
    if p["stage_channels"][-1] != c["output_channels"]:
        raise ValueError("last stage channel must equal output_channels")
    if math.prod(p["time_factors"]) != c["window"] // c["output_steps"]:
        raise ValueError("time_factors must multiply to window // output_steps")
    if p["d_model"] % p["heads"]:
        raise ValueError("d_model must be divisible by heads")
    c["core"] = {"plugin": core.get("plugin", "transformer_conv"), "params": p}
    c["fusion"] = {"plugin": (src.get("fusion") or {}).get("plugin", "sequence_concat")}
    return c


def modular_config_sha256(config: Dict[str, Any]) -> str:
    c = normalize_modular_config(config)
    return hashlib.sha256(json.dumps(c, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# ----------------------------------------------------------------------------
# layers
# ----------------------------------------------------------------------------

class FeatureSelect(nn.Module):
    """Fixed column routing (tf.gather on the channel axis). No weights."""

    def __init__(self, indices: Sequence[int]):
        super().__init__()
        self.register_buffer("indices", torch.as_tensor(list(indices), dtype=torch.long), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, F) -> (B, T, k)
        return x.index_select(-1, self.indices)


class CausalConv1d(nn.Module):
    """Keras ``Conv1D(padding="causal")`` on ``(B, T, C)`` tensors."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.conv = nn.Conv1d(in_channels, out_channels, self.kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = nn.functional.pad(x.transpose(1, 2), (self.kernel_size - 1, 0))
        return self.conv(y).transpose(1, 2)


class StridedConv1d(nn.Module):
    """Keras ``Conv1D(kernel_size=f, strides=f, padding="valid")`` on ``(B, T, C)``."""

    def __init__(self, in_channels: int, out_channels: int, factor: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, int(factor), stride=int(factor))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


class PositionalEncoding(nn.Module):
    """Stateless sinusoidal positions, identical to the Keras layer."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length, width = x.shape[1], x.shape[2]
        position = torch.arange(length, dtype=torch.float32, device=x.device)[:, None]
        channel = torch.arange(width, device=x.device)[None, :]
        rates = torch.pow(torch.tensor(10000.0, device=x.device),
                          -2.0 * (channel // 2).to(torch.float32) / float(width))
        angle = position * rates
        encoding = torch.where(channel % 2 == 0, torch.sin(angle), torch.cos(angle))
        return x + encoding[None].to(x.dtype)


class KerasMultiHeadAttention(nn.Module):
    """Keras ``MultiHeadAttention`` parametrization: per-head Einsum denses."""

    def __init__(self, width: int, heads: int, key_dim: int):
        super().__init__()
        self.heads, self.key_dim = int(heads), int(key_dim)
        shape = (width, self.heads, self.key_dim)
        self.query_kernel = nn.Parameter(torch.empty(shape))
        self.query_bias = nn.Parameter(torch.zeros(self.heads, self.key_dim))
        self.key_kernel = nn.Parameter(torch.empty(shape))
        self.key_bias = nn.Parameter(torch.zeros(self.heads, self.key_dim))
        self.value_kernel = nn.Parameter(torch.empty(shape))
        self.value_bias = nn.Parameter(torch.zeros(self.heads, self.key_dim))
        self.output_kernel = nn.Parameter(torch.empty(self.heads, self.key_dim, width))
        self.output_bias = nn.Parameter(torch.zeros(width))
        for k in (self.query_kernel, self.key_kernel, self.value_kernel):
            nn.init.xavier_uniform_(k.view(width, -1))
        nn.init.xavier_uniform_(self.output_kernel.view(-1, width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = torch.einsum("btd,dhk->bthk", x, self.query_kernel) + self.query_bias
        k = torch.einsum("btd,dhk->bthk", x, self.key_kernel) + self.key_bias
        v = torch.einsum("btd,dhk->bthk", x, self.value_kernel) + self.value_bias
        scores = torch.einsum("bthk,bshk->bhts", q, k) / math.sqrt(self.key_dim)
        t = x.shape[1]
        causal = torch.tril(torch.ones(t, t, dtype=torch.bool, device=x.device))
        scores = scores.masked_fill(~causal[None, None], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        context = torch.einsum("bhts,bshk->bthk", weights, v)
        return torch.einsum("bthk,hkd->btd", context, self.output_kernel) + self.output_bias


class TransformerBlock(nn.Module):
    def __init__(self, width: int, heads: int, ff_dim: int):
        super().__init__()
        self.attention = KerasMultiHeadAttention(width, heads, width // heads)
        self.attention_norm = nn.LayerNorm(width, eps=LN_EPS)
        self.ffn_expand = nn.Linear(width, ff_dim)
        self.ffn_project = nn.Linear(ff_dim, width)
        self.ffn_norm = nn.LayerNorm(width, eps=LN_EPS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention_norm(x + self.attention(x))
        f = self.ffn_project(nn.functional.gelu(self.ffn_expand(x)))
        return self.ffn_norm(x + f)


class ResidualTemporalStage(nn.Module):
    """Downsample complete adjacent blocks, then a residual causal TCN."""

    def __init__(self, in_channels: int, channels: int, factor: int, kernel: int):
        super().__init__()
        self.skip_downsample = StridedConv1d(in_channels, channels, factor)
        self.block_projection = StridedConv1d(in_channels, channels, factor)
        self.downsample_norm = nn.LayerNorm(channels, eps=LN_EPS)
        self.temporal_conv = CausalConv1d(channels, channels, kernel)
        self.temporal_projection = CausalConv1d(channels, channels, 1)
        self.temporal_norm = nn.LayerNorm(channels, eps=LN_EPS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values = self.downsample_norm(self.skip_downsample(x) + self.block_projection(x))
        values = nn.functional.gelu(values)
        residual = self.temporal_projection(nn.functional.gelu(self.temporal_conv(values)))
        return nn.functional.gelu(self.temporal_norm(values + residual))


class Branch(nn.Module):
    def __init__(self, in_features: int, channels: int, kernel: int):
        super().__init__()
        self.conv = CausalConv1d(in_features, channels, kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.gelu(self.conv(x))


class TemporalCore(nn.Module):
    def __init__(self, in_channels: int, params: Dict[str, Any]):
        super().__init__()
        width = int(params["d_model"])
        self.positional_encoding = PositionalEncoding()
        self.model_projection = nn.Linear(in_channels, width)
        self.blocks = nn.ModuleList(
            TransformerBlock(width, int(params["heads"]), int(params["ff_dim"]))
            for _ in range(int(params["blocks"])))
        stages = []
        previous = width
        for channel, factor in zip(params["stage_channels"], params["time_factors"]):
            stages.append(ResidualTemporalStage(previous, int(channel), int(factor), int(params["kernel_size"])))
            previous = int(channel)
        self.stages = nn.ModuleList(stages)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model_projection(self.positional_encoding(x))
        for block in self.blocks:
            x = block(x)
        for stage in self.stages:
            if x.shape[1] % stage.skip_downsample.conv.stride[0]:
                raise ValueError("Temporal stage requires exact divisibility")
            x = stage(x)
        return x


class ModularTemporalEncoder(nn.Module):
    """Branches -> fusion -> core on a ``(B, window, n_features)`` tensor."""

    def __init__(self, modular_config: Dict[str, Any]):
        super().__init__()
        c = normalize_modular_config(modular_config)
        self.config = c
        names = list(c["feature_names"])
        self.selects = nn.ModuleDict()
        self.branches = nn.ModuleDict()
        fused_channels = 0
        for spec in c["branches"]:
            channels = int(spec["params"].get("channels", 16))
            kernel = int(spec["params"].get("kernel_size", 3))
            self.selects[spec["name"]] = FeatureSelect([names.index(f) for f in spec["features"]])
            self.branches[spec["name"]] = Branch(len(spec["features"]), channels, kernel)
            fused_channels += channels
        self.fused_channels = fused_channels
        self.core = TemporalCore(fused_channels, c["core"]["params"])
        self.output_steps, self.output_channels = int(c["output_steps"]), int(c["output_channels"])

    @property
    def branch_names(self) -> List[str]:
        return list(self.branches.keys())

    def stage_outputs(self, window: torch.Tensor) -> Dict[str, Any]:
        if window.dim() != 3 or window.shape[1] != self.config["window"] \
                or window.shape[2] != len(self.config["feature_names"]):
            raise ValueError(f"expected (B, {self.config['window']}, {len(self.config['feature_names'])}); "
                             f"got {tuple(window.shape)}")
        branches = {name: self.branches[name](self.selects[name](window)) for name in self.branches}
        fused = torch.cat(list(branches.values()), dim=-1) if len(branches) > 1 else next(iter(branches.values()))
        latent = self.core(fused)
        return {"branches": branches, "fused": fused, "latent": latent,
                "task_head_reduction": {"from": [self.output_steps, self.output_channels],
                                        "to": self.output_steps * self.output_channels, "op": "flatten"}}

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        latent = self.stage_outputs(window)["latent"]
        return latent.reshape(latent.shape[0], -1)

    @property
    def latent_dim(self) -> int:
        return self.output_steps * self.output_channels


# ----------------------------------------------------------------------------
# SB3 features extractors
# ----------------------------------------------------------------------------

def _sb3_base():
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
    return BaseFeaturesExtractor, FlattenExtractor


_Base, _Flatten = _sb3_base()


class ModularTemporalExtractor(_Base):
    """SB3 features extractor: modular latent (flattened) ++ agent-state extras.

    ``regimes`` maps every branch name and ``"core"`` to R0 (random init, trainable),
    R1 (donor weights, frozen) or R2 (donor weights, trainable). ``donor`` is the
    directory of a ``rl_temporal.donor_contract`` artifact; it is required by R1/R2
    and forbidden for an all-R0 extractor. Validation lives in ``donor_contract``.
    """

    def __init__(self, observation_space, *, layout: ObservationLayout, modular_config: Dict[str, Any],
                 regimes: Optional[Dict[str, str]] = None, donor: Optional[str] = None):
        from . import donor_contract

        layout = layout if isinstance(layout, ObservationLayout) else ObservationLayout(**layout)
        if int(np.prod(observation_space.shape)) != layout.total_dim:
            raise ValueError(f"observation space {observation_space.shape} does not match layout "
                             f"total_dim {layout.total_dim}")
        c = normalize_modular_config(modular_config)
        if tuple(c["feature_names"]) != tuple(layout.feature_order):
            raise ValueError("modular feature_names must equal the layout feature_order (same source information)")
        if int(c["window"]) != layout.window:
            raise ValueError(f"modular window {c['window']} != observation window {layout.window}")
        encoder = ModularTemporalEncoder(c)
        super().__init__(observation_space, features_dim=encoder.latent_dim + layout.extras_dim)
        self.layout = layout
        self.modular_config = c
        self.encoder = encoder
        self.regimes = donor_contract.apply_regimes(self.encoder, regimes, donor, c)
        self.donor = donor

    # the (window, features) tensor the baseline also sees
    def source_window(self, observations: torch.Tensor) -> torch.Tensor:
        return self.layout.split(observations)[0]

    def stage_outputs(self, observations: torch.Tensor) -> Dict[str, Any]:
        return self.encoder.stage_outputs(self.source_window(observations))

    def stage_outputs_from_window(self, window: torch.Tensor) -> Dict[str, Any]:
        return self.encoder.stage_outputs(window)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        window, extras = self.layout.split(observations)
        return torch.cat([self.encoder(window), extras], dim=1)


class NativeFlatExtractor(_Flatten):
    """The DECLARED native baseline representation: SB3's ``FlattenExtractor``.

    The whole ``(window, features)`` block plus the agent state are flattened
    into one vector and fed to the policy MLP (``net_arch``). Time is handled
    by position in the vector only: no convolution, no attention, no temporal
    bottleneck. This is the control for RL-S0/RL-D0; it is not our branching
    architecture and must never be labelled as such.
    """


def native_baseline_card(layout: ObservationLayout, *, net_arch: Sequence[int], action_dim: int) -> Dict[str, Any]:
    dims = [layout.total_dim, *[int(n) for n in net_arch], int(action_dim)]
    params = sum(dims[i] * dims[i + 1] + dims[i + 1] for i in range(len(dims) - 1))
    return {
        "name": "native_flat_mlp", "architecture": "flatten_mlp",
        "extractor": "stable_baselines3.common.torch_layers.FlattenExtractor (NativeFlatExtractor)",
        "time_handling": "flattened_window_no_temporal_structure",
        "source_information": {"layout_digest": layout.digest, "window": layout.window,
                               "n_features": layout.n_features, "extras_dim": layout.extras_dim,
                               "feature_order": list(layout.feature_order)},
        "net_arch": [int(n) for n in net_arch], "action_dim": int(action_dim),
        "parameter_count": int(params),
        "note": "control arm; one MLP head counted; SAC counts actor + n_critics copies, DQN online + target",
    }


def parameter_accounting(module: nn.Module) -> Dict[str, Any]:
    """Per-component parameter counts (total and trainable)."""
    encoder = getattr(module, "encoder", module)
    out: Dict[str, Any] = {"branches": {}, "core": {}, "total": 0, "trainable": 0}
    if isinstance(encoder, ModularTemporalEncoder):
        for name, branch in encoder.branches.items():
            t = sum(p.numel() for p in branch.parameters())
            tr = sum(p.numel() for p in branch.parameters() if p.requires_grad)
            out["branches"][name] = {"total": t, "trainable": tr}
        t = sum(p.numel() for p in encoder.core.parameters())
        tr = sum(p.numel() for p in encoder.core.parameters() if p.requires_grad)
        out["core"] = {"total": t, "trainable": tr}
    out["total"] = sum(p.numel() for p in module.parameters())
    out["trainable"] = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return out
