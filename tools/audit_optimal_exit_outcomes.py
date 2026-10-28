"""Read sealed synthetic policies; compare sampled-policy expectation to argmax.

No training, financial data, checkpoint modification or release authorization.
The exact expectation assumes the existing memoryless close/hold policy.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from app.frozen_recipe_inputs import file_sha, require
from tools.recipe_sac_queue import read, write


def exit_distribution(close_probability):
    p = np.asarray(close_probability, dtype=float)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('EXIT_PROBABILITY')
    survival = np.r_[1., np.cumprod(1. - p)]
    return p * survival[:-1], float(survival[-1])


def expectation(case, probability):
    exits, passive = exit_distribution(probability)
    net = np.asarray(case['close_net'], dtype=float)
    if exits.shape != net.shape:
        raise ValueError('EXIT_PROBABILITY_SHAPE')
    expected = float(exits @ net + passive * case['terminal_net'])
    optimum = max(float(net.max()), case['terminal_net'])
    return dict(mean_net_usd=expected, regret_risk=(optimum - expected) / case['risk_usd'],
                early_close_probability=1. - passive)


def run(root):
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    manifest = read(root / 'CLOSE.json')
    require(manifest['learned_controls_passed'] == 0, 'EXPECTED_FAILED_STUDY')

    def checked(path):
        require(file_sha(path) == manifest['artifacts'][str(path.relative_to(root))], 'SEALED_ARTIFACT')
        return path

    datasets = {}
    for split in ('train', 'confirmation'):
        cases, arrays, masks = [], [], []
        for path in sorted((root / 'corpus').glob(split + '_*.json')):
            case = read(checked(path))
            with np.load(checked(path.with_suffix('.npz')), allow_pickle=False) as f:
                x = f['observations']
            cases.append(case)
            arrays.append(x)
            masks.append(np.tile([1, 0, 0, int(case['direction'] == 1), int(case['direction'] == -1)], (len(x), 1)))
        datasets[split] = (cases, arrays, masks)

    rows = []
    for path in sorted((root / 'fits').glob('*/RESULT.json')):
        result = read(checked(path))
        model = PPO.load(checked(path.parent / 'diagnostic_model.zip'), device='cpu')
        model.policy.set_training_mode(False)
        constant_probability = {}
        for split, (cases, arrays, masks) in datasets.items():
            x = dict(features=torch.from_numpy(np.concatenate(arrays)),
                     action_mask=torch.from_numpy(np.concatenate(masks).astype(np.float32)))
            with torch.no_grad():
                z = model.policy.features_extractor(x)
                pi, vf = model.policy.mlp_extractor(z)
                dist = model.policy.get_distribution(x)
                p = (dist.cat.probs[:, 3] + dist.cat.probs[:, 4]).numpy()
                commands = dist.get_actions(deterministic=True)[:, 0].numpy().astype(int)
            expected = []
            constant = []
            if split == 'train':
                for direction, column in ((1, 3), (-1, 4)):
                    selected = x['action_mask'][:, column].numpy().astype(bool)
                    constant_probability[direction] = float(p[selected].mean())
            at = 0
            for case, a in zip(cases, arrays):
                expected.append(expectation(case, p[at:at + len(a)]))
                constant.append(expectation(case, np.full(len(a), constant_probability[case['direction']])))
                at += len(a)
            rows.append(dict(unit=result['unit'], split=split, cases=len(cases), decision_rows=len(p),
                close_probability_min=float(p.min()), close_probability_max=float(p.max()),
                command_counts=np.bincount(commands, minlength=5).tolist(),
                actor_tanh_saturation_fraction=float((pi.abs() > .99).float().mean()),
                critic_tanh_saturation_fraction=float((vf.abs() > .99).float().mean()),
                stochastic_expectation={k: float(np.mean([r[k] for r in expected])) for k in expected[0]},
                constant_probability_from_train=constant_probability,
                constant_hazard_expectation={k: float(np.mean([r[k] for r in constant])) for k in constant[0]},
                deterministic=result['reports'][split]['summary']))
        del model
    return dict(schema='optimal_exit.policy_outcome_audit.v1', source_close_sha256=file_sha(root / 'CLOSE.json'),
        source_identity=manifest['identity'], training=False, financial_admission=False,
        scope='Exact memoryless sampled-policy expectations, not a change of evaluation policy or a promotion', rows=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.resolve().is_relative_to(args.root.resolve()), 'AUDIT_OUTPUT_OUTSIDE_SEALED_STUDY')
    require(not args.output.exists(), 'AUDIT_OUTPUT_EXISTS')
    report = run(args.root)
    write(args.output, report)
    print(json.dumps(report, indent=2))
