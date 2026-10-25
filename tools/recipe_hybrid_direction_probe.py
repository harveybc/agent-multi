"""Read-only directional-collapse probe on authenticated checkpoints; no fit."""
import argparse
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from app.frozen_recipe_inputs import file_sha, require
from tools.recipe_hybrid_queue import Market, authenticated, verify
from tools.recipe_sac_queue import read, write, now


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    require(not args.output.exists(), 'NEW_DIAGNOSTIC_OUTPUT_REQUIRED')
    torch.set_num_threads(1)
    spec = verify(args.root)
    sha = file_sha(args.root / 'SPEC.private.json')
    market = Market(spec, 'CAT_FIXED')
    env = market.env('INNER_A')
    obs, _ = env.reset()
    observations = []
    done = False
    while not done:
        if obs['action_mask'][1] and obs['action_mask'][2]:
            observations.append({k: v.copy() for k, v in obs.items()})
        obs, _, done, _, _ = env.step([0, .5, .5, .5])
    require(len(observations) > 1, 'NO_SHARED_FLAT_SUPPORT')
    indices = np.unique(np.linspace(0, len(observations)-1, 256).astype(int))
    batch = {k: np.stack([observations[i][k] for i in indices]) for k in observations[0]}
    reports = []
    status = read(args.root / 'STATUS.json')
    for unit in ('CAT_FIXED_s7', 'CAT_FIXED_s19', 'HYBRID_DD400_s7'):
        require(unit in status['completed'], 'CHECKPOINT_NOT_COMPLETE')
        result = args.root / status['completed'][unit]
        authenticated(result, sha)
        archive = result.parent / 'model-102400.zip'
        model = PPO.load(archive, device='cpu')
        model.policy.set_training_mode(False)
        tensor, _ = model.policy.obs_to_tensor(batch)
        with torch.no_grad():
            features = model.policy.features_extractor(tensor)
            head = model.policy.head(features)
            logits = model.policy.commands(head)
            dist = model.policy.get_distribution(tensor)
            probabilities = dist.cat.probs.cpu().numpy()
            commands = dist.get_actions(True)[:, 0].cpu().numpy().astype(int)
        reports.append(dict(
            unit=unit, archive_sha256=file_sha(archive), result_sha256=file_sha(result),
            probability_mean=probabilities.mean(0).tolist(),
            probability_min=probabilities.min(0).tolist(),
            probability_max=probabilities.max(0).tolist(),
            argmax_counts=np.bincount(commands, minlength=5).tolist(),
            logits_std_by_command=logits.std(0).cpu().tolist(),
            encoder_std_mean=float(features.std(0).mean()),
            head_std_mean=float(head.std(0).mean()),
            head_saturated_fraction=float((head.abs() > .99).float().mean()),
        ))
    write(args.output, dict(
        schema='hybrid_direction_probe.v1', spec_sha256=sha, finished_at=now(),
        scope='READ_ONLY_INNER_A_COMMON_FLAT_SUPPORT; NOT_TRADING_EVALUATION',
        policy_device='CPU; not a bitwise GPU replay claim', fits=0, test_read=False,
        observations=len(indices), eligible_flat_observations=len(observations),
        input_std_mean=float(batch['features'].std(0).mean()), reports=reports))


if __name__ == '__main__':
    main()
