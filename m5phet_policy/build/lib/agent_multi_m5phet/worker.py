"""The policy engine's side of the subprocess boundary: read one observation, return one action.

It reads a JSON object on stdin and writes one on stdout. It receives no prompt, no path chosen by a user and no shell.
The checkpoint path comes from the operator's bundle, which the caller verified by digest before invoking this.
"""

import json
import sys
import time


def main():
    request = json.loads(sys.stdin.read())
    started = time.perf_counter()
    import numpy as np
    from stable_baselines3 import SAC

    model = SAC.load(request["checkpoint"], device="cpu")
    observation = np.asarray(request["observation"], dtype=np.float32)
    expected = model.observation_space.shape
    if observation.shape != expected:
        print(json.dumps({"error": f"observation shape {observation.shape} is not {expected}"}))
        return 2
    action, _state = model.predict(observation, deterministic=bool(request.get("deterministic", True)))
    print(json.dumps({"action": [float(v) for v in np.atleast_1d(action)],
                      "seconds": time.perf_counter() - started,
                      "observation_shape": list(expected)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
