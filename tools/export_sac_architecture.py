#!/usr/bin/env python3
"""Export readable component diagrams and a summary for an SB3 SAC checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _params(module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _dot(text: str) -> str:
    return "digraph G {\nrankdir=LR;\ngraph [bgcolor=white, pad=0.2];\n" + text + "\n}\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    from stable_baselines3 import SAC

    model = SAC.load(args.checkpoint, device="cpu")
    observation = int(model.observation_space.shape[0])
    action = int(model.action_space.shape[0])
    actor = model.actor
    critic = model.critic
    target = model.critic_target
    args.output_dir.mkdir(parents=True, exist_ok=True)

    actor_dot = _dot(f'''
node [shape=box, style="rounded,filled", fillcolor="#eef5ff", color="#315b7d", fontname="DejaVu Sans"];
obs [label="Observation\\n{observation} float32 values"];
flat [label="FlattenExtractor\\nidentity flatten"];
d1 [label="Linear {observation} -> 256\\nReLU"];
d2 [label="Linear 256 -> 256\\nReLU"];
mu [label="Mean head\\nLinear 256 -> {action}\\nHardtanh [-2, 2]"];
noise [label="gSDE log_std\\n256 x {action}", fillcolor="#fff4dc"];
squash [label="Squashed stochastic action\\nBox [-1, 1] ({action},)", fillcolor="#eaf8ea"];
obs -> flat -> d1 -> d2 -> mu -> squash;
d2 -> noise -> squash;
''')
    critic_dot = _dot(f'''
node [shape=box, style="rounded,filled", fillcolor="#f3f1ff", color="#554486", fontname="DejaVu Sans"];
obs [label="Observation\\n{observation}"];
act [label="Action\\n{action}"];
cat [label="Concatenate\\n{observation + action}"];
q1 [label="Q1\\nLinear {observation + action} -> 256, ReLU\\nLinear 256 -> 256, ReLU\\nLinear 256 -> 1"];
q2 [label="Q2\\nLinear {observation + action} -> 256, ReLU\\nLinear 256 -> 256, ReLU\\nLinear 256 -> 1"];
minimum [label="min(Q1, Q2)\\nreduces positive bias", fillcolor="#eaf8ea"];
target [label="Target critics\\nPolyak update tau={model.tau}", fillcolor="#fff4dc"];
obs -> cat; act -> cat; cat -> q1 -> minimum; cat -> q2 -> minimum; minimum -> target;
''')
    assembly_dot = _dot(f'''
node [shape=box, style="rounded,filled", fillcolor="#f7f7f7", color="#444444", fontname="DejaVu Sans"];
features [label="83 causal market features x 32 bars\\n2,656 normalized values"];
state [label="Agent/account state\\n36 values"];
guard [label="Observation contract\\nraw price window forbidden", fillcolor="#ffe8e8"];
obs [label="Flattened observation\\n{observation} values", fillcolor="#eef5ff"];
actor [label="SAC actor\\n755,713 parameters"];
action [label="Continuous action\\n[-1, 1]"];
execution [label="gym-fx execution mapping\\nNOP / open / early close\\nnative SL and TP"];
env [label="Equity, position, reward\\nnext observation"];
replay [label="Replay buffer\\n200,000 transitions"];
critics [label="Twin critics\\n1,511,426 parameters"];
features -> guard -> obs; state -> obs; obs -> actor -> action -> execution -> env -> obs;
env -> replay; action -> replay; replay -> critics; critics -> actor [label="policy gradient"];
''')

    for name, content in {
        "eth_sac_actor.dot": actor_dot,
        "eth_sac_critics.dot": critic_dot,
        "eth_sac_assembly.dot": assembly_dot,
    }.items():
        (args.output_dir / name).write_text(content, encoding="utf-8")

    summary = {
        "schema": "agent_multi.sac_architecture_summary.v1",
        "checkpoint": args.checkpoint.name,
        "observation_shape": list(model.observation_space.shape),
        "action_shape": list(model.action_space.shape),
        "actor_parameters": _params(actor),
        "critic_parameters": _params(critic),
        "target_critic_parameters": _params(target),
        "learning_rate": float(model.lr_schedule(1.0)),
        "buffer_size": int(model.buffer_size),
        "learning_starts": int(model.learning_starts),
        "batch_size": int(model.batch_size),
        "gamma": float(model.gamma),
        "tau": float(model.tau),
        "train_freq": str(model.train_freq),
        "gradient_steps": int(model.gradient_steps),
        "entropy_coefficient": model.ent_coef,
        "use_sde": bool(model.use_sde),
        "policy": str(model.policy),
    }
    (args.output_dir / "eth_sac_model_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items()
                      if key != "policy"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
