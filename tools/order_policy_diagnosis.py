"""Read-only frozen SAC probes; outputs are separate, no training or TEST scoring."""
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np

NAMES = ['CLOSE_SHORT', 'OPEN_SHORT', 'HOLD', 'OPEN_LONG', 'CLOSE_LONG']


def command_bins(values):
    x = np.asarray(values)
    if not np.isfinite(x).all() or (np.abs(x) > 1.000001).any():
        raise ValueError('INVALID_ACTION')
    bounds = np.asarray([-.6, -.2, .2, .6], dtype=x.dtype)
    return ((x >= bounds[0]).astype(int) + (x >= bounds[1])
            + (x > bounds[2]) + (x > bounds[3]))


def sample_indices(n, maximum=256):
    if n <= 0 or maximum < 2:
        raise ValueError('EMPTY_OR_INVALID_SAMPLE')
    return np.unique(np.linspace(0, n - 1, min(n, maximum), dtype=int))


def counterfactual_actions(actions):
    a = np.asarray(actions)
    grid = np.repeat(a[:, None, :], 8, axis=1)
    grid[:, :7, 0] = np.asarray([-.8, -.4, 0., .4, .8, -.99, .99], a.dtype)
    return grid, NAMES + ['EDGE_NEGATIVE', 'EDGE_POSITIVE', 'POLICY']


def output_root(parent, output):
    parent, output = Path(parent).resolve(), Path(output).resolve()
    if (output.is_relative_to(parent) or parent.is_relative_to(output)
            or (output.exists() and any(output.iterdir()))):
        raise ValueError('OUTPUT_MUST_BE_NEW_AND_OUTSIDE_PARENT')
    output.mkdir(parents=True, exist_ok=True)
    return output


def distribution(x):
    x = np.asarray(x)
    return dict(mean=x.mean(axis=0).tolist(), std=x.std(axis=0).tolist(),
                quantiles=np.quantile(x, [0, .05, .5, .95, 1], axis=0).tolist())


def occupancy(actions):
    bins = command_bins(np.asarray(actions)[..., 0]).ravel()
    return dict(zip(NAMES, np.bincount(bins, minlength=5).tolist()))


def policy_probe(model, observations, seed=7007):
    import torch
    from tools.recipe_sac_queue import preserve_rng, tensor_hash
    original_modes = [(module, module.training) for module in model.policy.modules()]
    before = tensor_hash(model)
    try:
        with preserve_rng(), torch.no_grad():
            torch.manual_seed(seed)
            model.policy.set_training_mode(False)
            obs, _ = model.policy.obs_to_tensor(observations)
            mu, log_std, _ = model.actor.get_action_dist_params(obs)
            action = torch.tanh(mu)
            sampled = torch.tanh(mu[None] + log_std.exp()[None] *
                                 torch.randn((32, *mu.shape), device=mu.device))
            embedding = model.actor.extract_features(obs, model.actor.features_extractor)
            grid, names = counterfactual_actions(action.cpu().numpy())
            q = []
            for j in range(grid.shape[1]):
                a = torch.as_tensor(grid[:, j], device=mu.device)
                q.append(torch.cat(model.critic(obs, a), dim=1).cpu().numpy())
            q = np.stack(q, axis=1)  # observation, action probe, twin critic
            qmin = q.min(axis=2)
            alpha = (float(model.log_ent_coef.exp().item()) if model.log_ent_coef is not None
                     else float(model.ent_coef_tensor.item()))
            report = dict(
                sampled_rows=len(mu), gaussian_samples_per_row=32, seed=seed,
                mu=distribution(mu.cpu().numpy()), std=distribution(log_std.exp().cpu().numpy()),
                deterministic_action=distribution(action.cpu().numpy()),
                deterministic_commands=occupancy(action.cpu().numpy()),
                stochastic_commands=occupancy(sampled.cpu().numpy()),
                stochastic_scope='DISTRIBUTION_ONLY_NOT_REALIZED_TRADING',
                actor_embedding=distribution(embedding.cpu().numpy()), alpha=alpha,
                target_entropy=float(model.target_entropy),
                q_actions=names, q_min_mean=qmin.mean(axis=0).tolist(),
                q_twin_abs_disagreement=np.abs(q[:, :, 0]-q[:, :, 1]).mean(axis=0).tolist(),
                q_argmax_counts=dict(zip(names, np.bincount(qmin.argmax(axis=1), minlength=8).tolist())),
                q_scope='FIXED_OBSERVATION_AND_POLICY_TP_SL; ESTIMATES_NOT_RETURNS',
                close_long_minus_hold=distribution(qmin[:, 4]-qmin[:, 2]))
            # Training-mode sensitivity at fixed weights, not a fitted contrast.
            modes = []
            model.policy.set_training_mode(True)
            for _ in range(16):
                m, _, _ = model.actor.get_action_dist_params(obs)
                qa = torch.cat(model.critic(obs, action), dim=1).min(dim=1).values
                modes.append((torch.tanh(m).cpu().numpy(), qa.cpu().numpy()))
            train_actions = np.stack([p[0] for p in modes])
            train_q = np.stack([p[1] for p in modes])
            report['dropout_mode_probe'] = dict(
                repeats=16, train_commands=occupancy(train_actions),
                command_change_fraction=float((command_bins(train_actions[..., 0]) !=
                    command_bins(action.cpu().numpy()[:, 0])[None]).mean()),
                action_abs_delta_mean=np.abs(train_actions-action.cpu().numpy()[None]).mean(axis=(0, 1)).tolist(),
                q_abs_delta_mean=float(np.abs(train_q-qmin[:, -1][None]).mean()),
                scope='MODE_SENSITIVITY_NOT_CAUSAL_ATTRIBUTION')
            return report
    finally:
        for module, mode in original_modes:
            module.training = mode
        if tensor_hash(model) != before:
            raise ValueError('PROBE_CHANGED_WEIGHTS')


def run(parent, output):
    import torch
    from stable_baselines3 import SAC
    from app.frozen_recipe_inputs import file_sha, require
    from app.recipe_runtime import RecipeMarket
    from tools.recipe_sac_queue import (verify, read, write, now, clean, tensor_hash,
                                        authenticated_result, preserve_rng)
    parent = Path(parent).resolve()
    spec = verify(parent)
    spec_sha = file_sha(parent/'SPEC.private.json')
    closed = read(parent/'CLOSE.json')
    require(closed['state'] == 'COMPLETE' and closed['spec_sha256'] == spec_sha, 'PARENT_NOT_CLOSED')
    sources = {a: parent/closed['completed'][a] for a in ('TARGET', 'COMMANDS5')}
    for path in sources.values():
        authenticated_result(path, spec_sha)
    output = output_root(parent, output)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    records = []
    started = time.monotonic()
    try:
        market = RecipeMarket(spec)
        from app.short_horizon.expert_guided.env import summary
        for arm, steps in [('TARGET', [100000]), ('COMMANDS5', [25000, 50000, 75000, 100000])]:
            market.order_contract = spec['order_contracts'][arm]
            for step in steps:
                folder = sources[arm].parent/f'step-{step}'
                model = SAC.load(folder/'model.zip', device=spec['device'])
                digest = tensor_hash(model)
                for period in ('TRAIN', 'INNER_A'):
                    progress = dict(state='REPLAYING', arm=arm, step=step, period=period,
                                    completed=len(records), total=10, heartbeat_at=now())
                    write(output/'STATUS.json', progress)
                    source = read(folder/(period+'.json'))
                    require(source['policy_tensor_sha256'] == digest, 'POLICY_PIN')
                    env = market.env(period)
                    obs, _ = env.reset()
                    bars, positions, actions = [], [], []
                    with preserve_rng():
                        while True:
                            bars.append(env.bar)
                            positions.append(0 if env.engine.pos is None else env.engine.pos.direction)
                            a, _ = model.predict(obs, deterministic=True)
                            actions.append(a.copy())
                            obs, _, done, _, _ = env.step(a)
                            if done:
                                break
                    native = clean(summary(env.engine.result()))
                    differences = {k: [source[k], value] for k, value in native.items()
                                   if k in source and source[k] != value}
                    original_counts = read(folder/(period+'.orders.json'))['counts']
                    replay = dict(exact=not differences and env.order_counts == original_counts,
                                  differences=differences, order_counts=env.order_counts)
                    name = f'{arm}-{step}-{period}'
                    write(output/(name+'-REPLAY.json'), replay)
                    require(replay['exact'], 'REPLAY_NOT_EXACT')
                    idx = sample_indices(len(bars))
                    selected = [market.inputs.observation(int(env.rows[bars[i]])) for i in idx]
                    observations = {k: np.stack([o[k] for o in selected]) for k in selected[0]}
                    write(output/'STATUS.json', dict(progress, state='PROBING', heartbeat_at=now()))
                    probe = policy_probe(model, observations)
                    actions = np.asarray(actions)
                    record = dict(arm=arm, step=step, period=period, policy_tensor_sha256=digest,
                                  replay=replay, metrics=native, policy_probe=probe,
                                  all_actions=distribution(actions),
                                  sampled_bar_sha256=hashlib.sha256(np.asarray(bars, np.int64)[idx].tobytes()).hexdigest(),
                                  command_names_scope='COMMANDS5 bins only; TARGET uses its own decoder',
                                  actual_decoder_counts=env.order_counts)
                    require(all(r['sampled_bar_sha256'] == record['sampled_bar_sha256']
                                for r in records if r['period'] == period), 'UNPAIRED_SAMPLE')
                    np.savez_compressed(output/(name+'-actions.npz'), bars=np.asarray(bars),
                                        positions=np.asarray(positions), actions=actions, sample_indices=idx)
                    require(tensor_hash(model) == digest, 'REPLAY_CHANGED_WEIGHTS')
                    write(output/(name+'.json'), clean(record))
                    records.append(clean(record))
                    env.close()
                del model
        # Reauthenticate all parent files after inference; no parent artifact was modified.
        for path in sources.values():
            authenticated_result(path, spec_sha)
        write(output/'REPORT.json', dict(state='COMPLETE', parent_spec_sha256=spec_sha,
              producer_sha256=file_sha(__file__), records=records, fits=0, test_scored=False,
              selection='NONE', elapsed_seconds=time.monotonic()-started,
              parent_result_sha256={a: file_sha(p) for a, p in sources.items()}, finished_at=now(),
              limits=['single seed', 'no saved replay buffer; no Bellman calibration',
                      'critic probes are not counterfactual realized returns',
                      'train-mode differences do not establish cause']))
        write(output/'STATUS.json', dict(state='COMPLETE', completed=10, total=10, heartbeat_at=now()))
        write(output/'MANIFEST.json', {p.name: file_sha(p) for p in output.iterdir() if p.is_file()})
    except Exception as exc:
        write(output/'STATUS.json', dict(state='FAILED', error_type=type(exc).__name__,
                                        error=str(exc), completed=len(records), heartbeat_at=now()))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    run(args.parent, args.output)


if __name__ == '__main__':
    main()
