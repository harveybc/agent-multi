#!/usr/bin/env python3
"""Generate the exact ETH L1 system manifest (order §3 / repair §6).

Reads the REAL artifacts — pinned base config, nested split contract,
input CSV (rows + time bounds), the four D1 anchors (artifact sha +
canonical policy tensor sha + observation/action shapes) — and freezes
them, together with the split dates, observation contract, cost/margin
/SL-TP bindings, plugin surface and the source identities present at
generation time, into:

    examples/config/phase_3_eth_sac_dynamics/systems/
        ethusdt_4h_l1_system_v1.json

The generated manifest is then committed and FROZEN; regeneration that
changes any binding is a new manifest version, never an overwrite.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pipeline_plugins import _system_config as sysid  # noqa: E402
from tools import l1_factorial_screen as runner  # noqa: E402

# v1 was generated from a dirty obsolete tree (finding 194) and stays
# in the repo only as REJECTED evidence; v2 is generated exclusively
# from a clean commit (the generator refuses a dirty tree).
OUT_PATH = (REPO / "examples/config/phase_3_eth_sac_dynamics/systems/"
            "ethusdt_4h_l1_system_v2.json")

COST_KEYS = (
    "commission", "slippage", "leverage", "initial_cash", "k_sl", "k_tp",
    "position_size", "size_mode", "rel_volume", "max_order_volume",
    "min_order_volume", "continuous_action_threshold",
)

# Explicit normal cost/solvency contract (order §3.3, findings 192/193).
# Values adopted from the REVIEWED ETH-v2 environment contract
# (examples/config/phase_2_eth_anchored/optimization/
# phase_2_eth_anchored_full_v2.json): full spread 1e-4 carries the
# market friction, per-side slippage is DECLARED 0.0 exactly as in that
# reviewed environment (not silently absent), protected entries are
# enforced, and min-equity makes the gym-fx default (1% of initial
# cash) explicit instead of implicit. Any future deliberate difference
# is a named experiment factor, never a silent edit here.
NORMAL_CONTRACT = {
    "full_spread_rate": 0.0001,
    "slippage": 0.0,
    "require_protected_entries": True,
    "min_equity": 100.0,
}

# The executable plugin surface (order §3.1, finding 191): these names
# must equal the classes that execute; the materializer refuses drift
# and the runner takes its agent/pipeline from here. The curriculum
# wrapper is the intentionally varying element, bound explicitly.
EXECUTABLE_PLUGINS = {
    "agent_plugin": "sac_agent",
    "pipeline_plugin": "rl_pipeline_with_validation",
    "curriculum_pipeline_plugin": "rl_pipeline_with_solvency_curriculum",
}

BASE_PLUGIN_KEYS = (
    "env_plugin", "broker_plugin", "data_feed_plugin",
    "preprocessor_plugin", "reward_plugin", "strategy_plugin",
    "metrics_plugin",
)


def csv_facts(path: Path, date_column: str) -> dict:
    rows = 0
    first = last = None
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows += 1
            stamp = row.get(date_column)
            if first is None:
                first = stamp
            last = stamp
    return {"path": str(path), "sha256": sysid.sha_file(path),
            "rows": rows, "date_column": date_column,
            "time_bounds": {"first": first, "last": last}}


def anchor_facts(contract: dict) -> dict:
    from stable_baselines3 import SAC
    from agent_plugins.sac_agent import _policy_tensor_hash

    out = {}
    for seed, entry in sorted(contract["anchors"].items()):
        path = Path(entry["path"]).expanduser()
        actual = sysid.sha_file(path)
        if actual != entry["sha256"]:
            raise RuntimeError(f"anchor for seed {seed} drifted on disk")
        model = SAC.load(str(path), device="cpu")
        out[seed] = {
            "path": str(path),
            "sha256": actual,
            "policy_tensor_sha256": _policy_tensor_hash(model.policy),
            "observation_shape": list(model.observation_space.shape),
            "action_shape": list(model.action_space.shape),
            "n_updates_at_anchor": int(getattr(model, "_n_updates", 0)),
        }
        del model
    return out


def main() -> int:
    from tools import eth_curriculum_decision_experiment as d1

    # Finding 194: the frozen manifest must come from a CLEAN commit.
    for repo_name, root in (("agent-multi",
                             sysid.resolve_repo_root(Path(__file__))),
                            ("gym-fx", Path("/home/harveybc/Documents/"
                                            "GitHub/gym-fx"))):
        ident = sysid.source_tree_identity(root)
        if ident["dirty"]:
            raise RuntimeError(
                f"refusing to generate the frozen manifest from a DIRTY "
                f"{repo_name} tree (digest "
                f"{ident['dirty_untracked_digest']}); commit first")

    contract = runner.load_contract()
    base_path = Path(d1.ETH_BASE)
    base_sha = sysid.sha_file(base_path)
    if base_sha != d1.ETH_BASE_SHA256:
        raise RuntimeError("pinned base config drifted — refusing to "
                           "generate a manifest over a broken pin")
    base = json.loads(base_path.read_text())

    data = csv_facts(Path(d1.DATA_FILE), str(base.get("date_column",
                                                      "DATE_TIME")))
    nested_path = REPO / contract["nested_split_contract"]
    nested = json.loads(nested_path.read_text())
    if nested["source_sha256"] != data["sha256"]:
        raise RuntimeError("nested split contract pins a different CSV")

    normal_config = dict(base)
    normal_config.update(NORMAL_CONTRACT)
    obs = sysid.observation_manifest(normal_config)
    anchors = anchor_facts(contract)
    shapes = {tuple(a["observation_shape"]) for a in anchors.values()}
    if len(shapes) != 1:
        raise RuntimeError(f"anchor observation shapes disagree: {shapes}")
    obs["flattened_shape"] = sorted(shapes)[0] and list(sorted(shapes)[0])

    manifest = {
        "schema": sysid.MANIFEST_SCHEMA,
        "$doc": ("Exact ETH USD 4h system for the L1 matched factorial. "
                 "Every binding is a verified fact from the real "
                 "artifacts at generation time; the materializer refuses "
                 "any drift fail-closed."),
        "system": "ethusdt_4h_l1_v1",
        "asset": str(contract.get("asset")),
        "env_asset": str(contract.get("env_asset")),
        "data": data,
        "base_config": {
            "path": str(base_path.relative_to(REPO)),
            "sha256": base_sha,
        },
        "nested_split_contract": {
            "path": contract["nested_split_contract"],
            "sha256": sysid.sha_file(nested_path),
            "expected_rows": nested.get("expected_rows"),
        },
        "splits": {
            "dates": dict(d1.SPLITS),
            "evaluate_test_split": False,
        },
        "observation": obs,
        "costs": {
            "$doc": ("Normal cost/solvency contract: base identity "
                     "values plus the REVIEWED ETH-v2 environment "
                     "settings (spread 1e-4, declared slippage 0.0, "
                     "protected entries enforced, explicit min-equity "
                     "= gym-fx default 1% of initial cash)."),
            "config_bindings": {
                **{k: base.get(k) for k in COST_KEYS if k in base},
                **NORMAL_CONTRACT,
            },
        },
        "plugins": {
            **{k: base.get(k) for k in BASE_PLUGIN_KEYS if k in base},
            **EXECUTABLE_PLUGINS,
        },
        "anchors": anchors,
        "source_identity_at_manifest": {
            "agent-multi": sysid.source_tree_identity(
                sysid.resolve_repo_root(Path(__file__))),
            "gym-fx": sysid.source_tree_identity(
                Path("/home/harveybc/Documents/GitHub/gym-fx")),
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=1, sort_keys=True,
                         default=str) + "\n"
    if OUT_PATH.exists() and OUT_PATH.read_text() != payload:
        raise RuntimeError(
            f"{OUT_PATH} exists with different content — a changed "
            "binding is a NEW manifest version, never an overwrite")
    OUT_PATH.write_text(payload)
    print(json.dumps({"manifest": str(OUT_PATH),
                      "sha256": sysid.sha_file(OUT_PATH),
                      "rows": data["rows"],
                      "anchors": len(anchors)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
