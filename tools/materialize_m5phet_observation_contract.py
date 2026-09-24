#!/usr/bin/env python
"""Write the observation contract a policy bundle needs to accept market data.

A bundle's manifest declares `observation_size` and nothing else about what the observation MEANS.
A length is not a contract: 2724 floats in the wrong order are still 2724 floats, and the policy
answers them without complaint. So the M5PHET policy provider keeps the market-data path closed
until an `observation_contract.json` sits next to the manifest, declaring which columns, in which
order, over which window, with which normalization the checkpoint was fitted.

This tool derives that file from the campaign configuration that produced the checkpoint -- the
single place where those facts are already recorded -- rather than having an operator retype 83
column names into a second file, which is how the two would come to disagree.

It refuses to write a contract that does not rebuild the manifest's declared length. That check is
the whole point of the exercise: if the campaign configuration and the retained checkpoint do not
agree on the shape of the observation, one of them does not describe the other, and a file written
anyway would make the disagreement invisible.

    python tools/materialize_m5phet_observation_contract.py \\
        --config examples/config/phase_2_eth_anchored/optimization/phase_2_eth_anchored_full_v2.json \\
        --bundle ~/.local/state/m5phet/policy-eth4h-dev-20260924

Run it with an interpreter that can import gym-fx (or pass --gym-fx <checkout>); the length is
verified with the environment's own builder, never with arithmetic repeated here.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

SCHEMA = "m5phet_observation_contract.v1"

#: Only keys that can move an element of the observation. Everything else in a campaign's
#: environment block -- broker, commission, reward plugin -- is irrelevant to the vector, and
#: copying it would invite a reader to think changing it changes the observation.
OBSERVATION_KEYS = (
    "window_size", "price_column", "feature_columns", "feature_binary_columns",
    "feature_scaling", "feature_scaling_window", "feature_clip",
    "include_price_window", "include_agent_state", "agent_state_contract", "position_size",
)


def _digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(config_path, bundle_path, gym_fx=None):
    if gym_fx:
        sys.path.insert(0, str(gym_fx))
    from gym_fx.observation_builder import ObservationContract

    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    environment = config.get("environment")
    if not isinstance(environment, dict) or not environment.get("feature_columns"):
        raise SystemExit(f"{config_path} has no environment.feature_columns; it did not fit a "
                         "feature-aware policy and there is no contract to derive")
    manifest = json.loads((Path(bundle_path) / "manifest.json").read_text(encoding="utf-8"))

    declared = {key: environment[key] for key in OBSERVATION_KEYS if key in environment}
    contract = ObservationContract.from_config(declared)
    if contract.expected_length != int(manifest["observation_size"]):
        raise SystemExit(
            f"REFUSED: this configuration builds {contract.expected_length} elements and the "
            f"bundle declares {manifest['observation_size']} for {manifest['policy_id']!r}. "
            "One of them does not describe this checkpoint; writing the contract anyway would "
            "hide that")

    initial_cash = float(
        environment.get("initial_cash")
        or (config.get("risk") or {}).get("initial_cash")
        or 0.0)
    if initial_cash == 0.0:
        raise SystemExit("REFUSED: no initial_cash in the campaign config, and equity_norm and "
                         "unrealized_pnl_norm are expressed against it")

    return {
        "schema": SCHEMA,
        "policy_id": manifest["policy_id"],
        "observation_size": contract.expected_length,
        "required_rows": contract.required_rows,
        "asset": (config.get("data") or {}).get("asset"),
        "timeframe": (config.get("data") or {}).get("timeframe"),
        "source_config": str(Path(config_path).resolve()),
        "source_config_sha256": _digest_file(config_path),
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "contract_sha256": contract.digest,
        "environment": declared,
        # The episode-side state a price series does not contain. It is DECLARED here, by the
        # operator, rather than assumed inside the provider, and every answer built from it says
        # so -- a default nobody sees is indistinguishable from a fact. bar_index 0 of total_bars 1
        # is the start of an episode, which is what `steps_remaining_norm` reads as 1.0.
        "agent_state_default": {
            "position": 0,
            "equity": initial_cash,
            "initial_cash": initial_cash,
            "entry_price": 0.0,
            "holding_bars": 0,
            "bar_index": 0,
            "total_bars": 1,
        },
        "agent_state_default_reading": (
            "flat, equity equal to the cash the episode started with, at the start of the episode. "
            "A caller who is holding a position must say so, or the answer is about a different "
            "situation from theirs"),
        "provenance": (
            f"derived from {Path(config_path).name}, the campaign configuration that fitted "
            f"{manifest['policy_id']}. DEVELOPMENT: this declares the observation's shape and "
            "meaning, not that the policy's trading quality was established"),
    }, contract


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="campaign config that fitted the checkpoint")
    parser.add_argument("--bundle", required=True, help="policy bundle directory holding manifest.json")
    parser.add_argument("--gym-fx", default=None, help="gym-fx checkout, if it is not importable")
    parser.add_argument("--print-only", action="store_true", help="show the contract and write nothing")
    args = parser.parse_args(argv)

    bundle = Path(args.bundle).expanduser()
    declared, contract = build(Path(args.config).expanduser(), bundle, args.gym_fx)
    target = bundle / "observation_contract.json"
    body = json.dumps(declared, indent=1, sort_keys=True) + "\n"
    if args.print_only:
        print(body, end="")
        return 0
    target.write_text(body, encoding="utf-8")
    print(f"wrote {target}")
    print(f"  observation length {contract.expected_length} "
          f"({contract.window_size} x {len(contract.feature_columns)} features"
          f"{' + prices + returns' if contract.include_price_window else ''}"
          f"{' + agent state' if contract.include_agent_state else ''})")
    print(f"  rows required per question: {contract.required_rows}")
    print(f"  contract sha256: {contract.digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
