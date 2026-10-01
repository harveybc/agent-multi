"""R0/R1/R2 regimes, donor artifacts, encoder identity and optimizer ownership.

R0: random initialization, trainable, donor forbidden.
R1: donor weights, frozen (``requires_grad=False``), donor required.
R2: donor weights, trainable, donor required.

A donor artifact is a directory with ``donor.pt`` (state dict of a
``ModularTemporalEncoder``) and ``donor.json`` (schema, modular config sha,
per-component identity hashes, source, versions). ``random_donor`` builds a
fixture donor; real donors come from the predictor export (``keras_import``).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from torch import nn

DONOR_SCHEMA = "rl_temporal.donor.v1"
REGIMES = ("R0", "R1", "R2")


def _tensor_hash(tensors: Iterable[torch.Tensor]) -> str:
    h = hashlib.sha256()
    for t in tensors:
        h.update(t.detach().cpu().contiguous().to(torch.float32).numpy().tobytes())
    return h.hexdigest()


def component_identity(encoder) -> Dict[str, str]:
    """Hashes of the branch weights (all branches, in order) and of the core."""
    per_branch = {name: _tensor_hash(b.state_dict().values()) for name, b in encoder.branches.items()}
    return {"branches": _tensor_hash(b for br in encoder.branches.values() for b in br.state_dict().values()),
            "per_branch": per_branch,
            "core": _tensor_hash(encoder.core.state_dict().values())}


@dataclass(frozen=True)
class Donor:
    path: str
    identity: Dict[str, Any]
    modular_config_sha256: str

    @classmethod
    def load(cls, path) -> "Donor":
        p = Path(path)
        doc = json.loads((p / "donor.json").read_text(encoding="utf-8"))
        if doc.get("schema") != DONOR_SCHEMA:
            raise ValueError(f"unsupported donor schema {doc.get('schema')!r}")
        return cls(str(p), doc["identity"], doc["modular_config_sha256"])

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return torch.load(Path(self.path) / "donor.pt", map_location="cpu")


def save_donor(encoder, path, *, source: str, extra: Optional[Dict[str, Any]] = None) -> Donor:
    from .modular_torch import modular_config_sha256

    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in encoder.state_dict().items()}, p / "donor.pt")
    identity = component_identity(encoder)
    doc = {"schema": DONOR_SCHEMA, "source": source, "identity": identity,
           "modular_config_sha256": modular_config_sha256(encoder.config),
           "modular_config": encoder.config, "versions": {"torch": torch.__version__},
           "donor_pt_sha256": hashlib.sha256((p / "donor.pt").read_bytes()).hexdigest(),
           **(extra or {})}
    (p / "donor.json").write_text(json.dumps(doc, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return Donor(str(p), identity, doc["modular_config_sha256"])


def random_donor(modular_config: Dict[str, Any], *, seed: int, path) -> Donor:
    """A FIXTURE donor: random weights with a recorded identity. Never a pretrained one."""
    from .modular_torch import ModularTemporalEncoder

    torch.manual_seed(int(seed))
    encoder = ModularTemporalEncoder(modular_config)
    return save_donor(encoder, path, source="random_fixture_seed_%d" % int(seed),
                      extra={"pretrained": False, "label": "RANDOM_FIXTURE_NOT_PRETRAINED"})


def validate_regimes(regimes: Optional[Dict[str, str]], donor: Optional[str], modular_config: Dict[str, Any]) -> Dict[str, str]:
    from .modular_torch import normalize_modular_config, modular_config_sha256

    c = normalize_modular_config(modular_config)
    names = [b["name"] for b in c["branches"]] + ["core"]
    resolved = {n: "R0" for n in names}
    for k, v in (regimes or {}).items():
        if k not in resolved:
            raise ValueError(f"regime for unknown component {k!r}; known: {names}")
        if v not in REGIMES:
            raise ValueError(f"regime must be one of {REGIMES}; got {v!r} for {k}")
        resolved[k] = v
    needs_donor = [n for n, r in resolved.items() if r in ("R1", "R2")]
    if needs_donor and not donor:
        raise ValueError(f"regimes {sorted(set(resolved[n] for n in needs_donor))} ({needs_donor}) require a donor; "
                         "R1/R2 without a compatible donor is not runnable and must not be relabelled R0")
    if not needs_donor and donor:
        raise ValueError("an all-R0 representation forbids a donor (R0 = random initialization)")
    if donor:
        d = Donor.load(donor)
        if d.modular_config_sha256 != modular_config_sha256(c):
            raise ValueError("donor modular_config_sha256 does not match this run's modular config "
                             "(incompatible donor)")
    return resolved


def apply_regimes(encoder, regimes: Optional[Dict[str, str]], donor: Optional[str], modular_config: Dict[str, Any]) -> Dict[str, str]:
    """Load donor weights into R1/R2 components and freeze R1. Returns the resolved regimes."""
    resolved = validate_regimes(regimes, donor, modular_config)
    if donor:
        state = Donor.load(donor).state_dict()
        for name, regime in resolved.items():
            if regime == "R0":
                continue
            module = encoder.core if name == "core" else encoder.branches[name]
            prefix = "core." if name == "core" else f"branches.{name}."
            sub = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            missing, unexpected = module.load_state_dict(sub, strict=True), None
            del missing, unexpected
    for name, regime in resolved.items():
        module = encoder.core if name == "core" else encoder.branches[name]
        for p in module.parameters():
            p.requires_grad_(regime != "R1")
    return resolved


# ----------------------------------------------------------------------------
# identity across the consuming networks
# ----------------------------------------------------------------------------

def _networks(model) -> Dict[str, nn.Module]:
    name = type(model).__name__
    if name == "SAC":
        return {"actor": model.actor, "critic": model.critic, "critic_target": model.critic_target}
    if name == "DQN":
        return {"q_net": model.q_net, "q_net_target": model.q_net_target}
    raise ValueError(f"unsupported algorithm {name}")


def _extractor(net):
    ext = getattr(net, "features_extractor", None)
    if ext is None or not hasattr(ext, "encoder"):
        raise ValueError(f"{type(net).__name__} has no modular features extractor")
    return ext


def encoder_identity(model) -> Dict[str, Dict[str, Any]]:
    return {name: component_identity(_extractor(net).encoder) for name, net in _networks(model).items()}


def encoder_distance(net_a, net_b) -> Dict[str, float]:
    a, b = _extractor(net_a).encoder.state_dict(), _extractor(net_b).encoder.state_dict()
    per = {k: float((a[k].float() - b[k].float()).abs().sum()) for k in a}
    return {"total": float(sum(per.values())), "per_tensor": per}


def _optimizers(model) -> Dict[str, torch.optim.Optimizer]:
    name = type(model).__name__
    if name == "SAC":
        out = {"actor": model.actor.optimizer, "critic": model.critic.optimizer}
        if getattr(model, "ent_coef_optimizer", None) is not None:
            out["ent_coef"] = model.ent_coef_optimizer
        return out
    if name == "DQN":
        return {"q_net": model.policy.optimizer}
    raise ValueError(f"unsupported algorithm {name}")


def optimizer_ownership(model) -> Dict[str, Any]:
    """Which optimizer owns which extractor's parameters; targets own none."""
    nets = _networks(model)
    opts = _optimizers(model)
    opt_params = {n: {id(p) for g in o.param_groups for p in g["params"]} for n, o in opts.items()}
    owner: Dict[str, bool] = {}
    for net_name, net in nets.items():
        if net_name.endswith("_target"):
            continue
        ids = {id(p) for p in _extractor(net).parameters()}
        owner[net_name] = bool(ids & opt_params.get(net_name, set()))
    target_in_any = False
    for net_name, net in nets.items():
        if net_name.endswith("_target"):
            ids = {id(p) for p in _extractor(net).parameters()}
            target_in_any |= any(ids & s for s in opt_params.values())
    names: Dict[int, str] = {}
    for n, p in model.policy.named_parameters():
        names.setdefault(id(p), n)
    multi = sorted(names.get(pid, str(pid)) for pid in set().union(*opt_params.values())
                   if sum(pid in s for s in opt_params.values()) > 1)
    shared = None
    if type(model).__name__ == "SAC":
        shared = model.critic.features_extractor is model.actor.features_extractor
    return {"algorithm": type(model).__name__, "shared_features_extractor": shared,
            "extractor_owner": owner, "target_extractor_in_any_optimizer": target_in_any,
            "params_in_more_than_one_optimizer": multi,
            "optimizers": {n: len(s) for n, s in opt_params.items()}}


def count_parameter_updates(model, *, gradient_steps: int = 1, batch_size: int = 8) -> Dict[str, int]:
    """Run one ``model.train()`` and count, per extractor parameter, the optimizer
    steps that carried a gradient for it. A value above ``gradient_steps`` is a
    double update."""
    names: Dict[int, str] = {}
    for n, p in model.policy.named_parameters():
        names.setdefault(id(p), n)
    extractor_ids = set()
    for net_name, net in _networks(model).items():
        if not net_name.endswith("_target"):
            extractor_ids |= {id(p) for p in _extractor(net).parameters()}
    counts: Dict[str, int] = {names[i]: 0 for i in extractor_ids if i in names}
    originals = {}
    for opt_name, opt in _optimizers(model).items():
        originals[opt_name] = opt.step

        def stepper(*args, _opt=opt, _orig=opt.step, **kwargs):
            for g in _opt.param_groups:
                for p in g["params"]:
                    if id(p) in extractor_ids and p.grad is not None:
                        counts[names[id(p)]] += 1
            return _orig(*args, **kwargs)

        opt.step = stepper
    try:
        model.train(gradient_steps=int(gradient_steps), batch_size=int(batch_size))
    finally:
        for opt_name, opt in _optimizers(model).items():
            opt.step = originals[opt_name]
    return counts
