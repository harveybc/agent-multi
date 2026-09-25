"""An independent reference run of the gym-fx environment, for the sweep's tests only.

It shares no code with `agent_multi_m5phet.sweep_worker` on purpose. A harness checked against a
helper that calls the harness would prove that the code runs, not that the numbers it reports are
the environment's. So this file builds the env the way gym-fx's own CLI builds it and steps it
with a literal list of action values, and the test asserts the sweep reproduces what comes back.

Reads {"env_config": {...}, "actions": [float, ...]} on stdin, writes the rewards and positions.
Run with the gym-fx checkout as the working directory, in an interpreter that has gymnasium and
backtrader.
"""

import json
import sys
from importlib.metadata import entry_points


def main():
    payload = json.loads(sys.stdin.read())
    config = payload["env_config"]

    from gym_fx import build_environment

    def plugin(group, name):
        entry = next(e for e in entry_points().select(group=group) if e.name == name)
        made = entry.load()(config)
        made.set_params(**config)
        return made

    env = build_environment(
        config=config,
        data_feed_plugin=plugin("data_feed.plugins", config["data_feed_plugin"]),
        broker_plugin=plugin("broker.plugins", config["broker_plugin"]),
        strategy_plugin=plugin("strategy.plugins", config["strategy_plugin"]),
        preprocessor_plugin=plugin("preprocessor.plugins", config["preprocessor_plugin"]),
        reward_plugin=plugin("reward.plugins", config["reward_plugin"]),
        metrics_plugin=plugin("metrics.plugins", config["metrics_plugin"]))
    try:
        _observation, info = env.reset()
        rewards, positions, bars = [], [], []
        for value in payload["actions"]:
            bars.append(int(info["bar_index"]))
            _observation, reward, terminated, truncated, info = env.step([float(value)])
            rewards.append(float(reward))
            positions.append(int(info["position"]))
            if terminated or truncated:
                break
        print(json.dumps({"rewards": rewards, "positions": positions, "bars": bars,
                          "total": sum(rewards)}))
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
