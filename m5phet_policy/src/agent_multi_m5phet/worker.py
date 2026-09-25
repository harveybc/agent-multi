"""The policy engine's side of the subprocess boundary: read one observation, return one action.

It reads a JSON object on stdin and writes one on stdout. It receives no prompt, no path chosen by a user and no shell.
The checkpoint path comes from the operator's bundle, which the caller verified by digest before invoking this.

Besides the action, the caller may ask for two readings the checkpoint itself holds and nothing here invents:

    actor_distribution   the SAC actor is a Gaussian over the PRE-SQUASH action; its mean and log-std for this
                         observation are read from the network. They describe the actor's spread, and nothing about
                         whether the action is right.
    critic               the twin Q critics evaluated at the actor's deterministic action for this observation. They
                         estimate discounted return under the training reward; they are not a realised profit.

Both are named in `evaluate` from a fixed vocabulary. Anything else in that list is refused, not guessed.
"""

import json
import sys
import time

EVALUATIONS = ("actor_distribution", "critic")


def main():
    request = json.loads(sys.stdin.read())
    started = time.perf_counter()
    import numpy as np
    from stable_baselines3 import SAC

    wanted = list(request.get("evaluate") or [])
    unknown = sorted(set(wanted) - set(EVALUATIONS))
    if unknown:
        print(json.dumps({"error": f"evaluate names {unknown}; this worker evaluates {list(EVALUATIONS)}"}))
        return 2
    model = SAC.load(request["checkpoint"], device="cpu")
    observation = np.asarray(request["observation"], dtype=np.float32)
    expected = model.observation_space.shape
    if observation.shape != expected:
        print(json.dumps({"error": f"observation shape {observation.shape} is not {expected}"}))
        return 2
    action, _state = model.predict(observation, deterministic=bool(request.get("deterministic", True)))
    answer = {"action": [float(v) for v in np.atleast_1d(action)], "observation_shape": list(expected)}
    if wanted:
        import torch

        with torch.no_grad():
            batch = torch.as_tensor(observation[None])
            scaled = model.actor(batch, deterministic=True)
            if "actor_distribution" in wanted:
                mean, log_std, _kwargs = model.actor.get_action_dist_params(batch)
                answer["actor_distribution"] = {
                    "family": "gaussian_pre_squash",
                    "squash": "tanh" if model.actor.squash_output else None,
                    "mean": [float(v) for v in mean[0]],
                    "log_std": [float(v) for v in log_std[0]],
                    "std": [float(v) for v in log_std[0].exp()],
                    "state_dependent_exploration": bool(model.use_sde)}
            if "critic" in wanted:
                values = [float(q[0]) for q in model.critic(batch, scaled)]
                answer["critic"] = {"q_values": values,
                                    "n_critics": len(values),
                                    "action_scaled": [float(v) for v in scaled[0]],
                                    "gamma": float(model.gamma)}
    answer["seconds"] = time.perf_counter() - started
    print(json.dumps(answer))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
