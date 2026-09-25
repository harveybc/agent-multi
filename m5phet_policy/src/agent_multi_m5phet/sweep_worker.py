"""The simulation's side of the sweep boundary: one env, one action series, the env's own numbers.

Everything here runs in the operator's policy interpreter, inside the gym-fx checkout, because the
environment needs gymnasium and backtrader and because gym-fx and agent-multi both carry a
top-level `app` package -- importing one from the other's working directory resolves the wrong
repository, and the symptom is not an error but a different simulation.

It reads one JSON object on stdin and writes one on stdout. It receives no prompt and no path a
person typed: the bars, the environment configuration and the checkpoint all come from the caller,
which read them from the operator's own declarations and verified the checkpoint by digest.

Nothing is computed here that the environment does not compute. The reward is the env's, under the
env's configured reward plugin; the position is the env's; the bar a decision is applied to is the
bar the env says it is at. A sweep that derived its own return would be measuring this file.
"""

import json
import os
import sys
import time
from datetime import datetime


def _key(stamp):
    """One text form for a bar's instant, so the caller's decisions and the env's bars can meet.

    `datetime.fromisoformat` reads both spellings a CSV and a DataFrame produce ("... 08:00:00"
    and "...T08:00:00"), and `isoformat` writes one. Matching the raw strings instead would make a
    decision series silently miss every bar over a separator nobody chose deliberately.
    """
    text = str(stamp).strip()
    try:
        return datetime.fromisoformat(text).isoformat()
    except ValueError:
        return text


def _flatten_if_dict(env):
    """Exactly what agent-multi's SAC plugin does before training: a Dict space becomes a vector.

    Transcribed from `agent_plugins/sac_agent.py::_flatten_if_dict`. It is applied to EVERY run in
    a sweep, not only the fitted one, so the three runs differ in their actions and in nothing
    else; the wrapper changes the observation and no part of the dynamics or the reward.
    """
    from gymnasium import spaces
    from gymnasium.wrappers import FlattenObservation

    if isinstance(getattr(env, "observation_space", None), spaces.Dict):
        return FlattenObservation(env)
    return env


def _build(config):
    from importlib.metadata import entry_points

    from gym_fx import build_environment

    def instance(group, name):
        found = next((entry for entry in entry_points().select(group=group) if entry.name == name), None)
        if found is None:
            raise LookupError(f"PLUGIN_NOT_INSTALLED: {name!r} is not registered in {group!r}")
        made = found.load()(config)
        made.set_params(**config)
        return made

    return build_environment(
        config=config,
        data_feed_plugin=instance("data_feed.plugins", config["data_feed_plugin"]),
        broker_plugin=instance("broker.plugins", config["broker_plugin"]),
        strategy_plugin=instance("strategy.plugins", config["strategy_plugin"]),
        preprocessor_plugin=instance("preprocessor.plugins", config["preprocessor_plugin"]),
        reward_plugin=instance("reward.plugins", config["reward_plugin"]),
        metrics_plugin=instance("metrics.plugins", config["metrics_plugin"]))


def _actor(run):
    """(a function bar-key -> action value, a description of where the actions came from)."""
    kind = run["kind"]
    if kind == "constant":
        value = float(run["value"])
        return (lambda key, observation: value), None
    if kind == "series":
        actions = dict(run["actions"])

        def chosen(key, observation):
            if key not in actions:
                raise KeyError(key)
            return float(actions[key])

        return chosen, None
    if kind == "fitted":
        import numpy as np
        from stable_baselines3 import SAC

        model = SAC.load(run["checkpoint"], device="cpu")

        def predicted(key, observation):
            action, _state = model.predict(observation, deterministic=True)
            return float(np.atleast_1d(action)[0])

        return predicted, {"loaded": "stable_baselines3.SAC", "device": "cpu",
                           "deterministic": True,
                           "observation_shape": list(model.observation_space.shape)}
    raise ValueError(f"UNKNOWN_RUN_KIND: {kind!r}")


def replay(run, config, max_steps):
    env = _build(config)
    position_size = float(config.get("position_size", 1.0) or 1.0)
    try:
        index = [_key(stamp) for stamp in env.dataframe.index]
        wrapped = _flatten_if_dict(env)
        act, engine = _actor(run)
        observation, info = wrapped.reset()
        steps = []
        total = 0.0
        stopped = "MAX_STEPS"
        while len(steps) < max_steps:
            where = int(info["bar_index"])
            key = index[where] if 0 <= where < len(index) else index[-1]
            try:
                value = act(key, observation)
            except KeyError:
                return {"refusal": ("DECISION_MISSING_FOR_BAR: the replay reached the bar at "
                                    f"{key} after {len(steps)} step(s) and the decision series "
                                    "names no action for it; a default here would be an action "
                                    "nobody decided"), "run": run["name"]}
            observation, reward, terminated, truncated, info = wrapped.step([value])
            total += float(reward)
            steps.append({"t": key, "bar_index": where, "action": float(value),
                          "environment_return_training_reward": float(reward),
                          "position": int(info["position"]),
                          "position_units": float(info.get("position_units") or 0.0),
                          "position_fraction": int(info["position"]) * position_size})
            if terminated or truncated:
                stopped = "TERMINATED" if terminated else "TRUNCATED"
                break
        return {"run": run["name"], "steps": steps,
                "environment_return_training_reward_total": total,
                "steps_taken": len(steps), "stopped_because": stopped,
                "termination_cause": info.get("termination_cause"),
                "trades": info.get("trades"), "engine": engine}
    finally:
        env.close()


def main():
    payload = json.loads(sys.stdin.read())
    started = time.perf_counter()
    root = payload["gym_fx"]
    sys.path.insert(0, root)
    os.chdir(root)

    config = dict(payload["env_config"])
    max_steps = int(payload["max_steps"])
    answer = {"runs": {}, "gym_fx": root}
    for run in payload["runs"]:
        result = replay(run, config, max_steps)
        if "refusal" in result:
            print(json.dumps({"error": result["refusal"]}))
            return 2
        answer["runs"][run["name"]] = result
    answer["seconds"] = time.perf_counter() - started
    print(json.dumps(answer))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
