"""CPU-only authenticated checkpoint ranks and one-component low-rank probes."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from agent_plugins.three_signal_sac import ThreeLogits
from app.capacity_measurements import spectrum, activation_spectrum, low_rank, signal_change
from app.frozen_recipe_inputs import file_sha
from app.rl_subset import support_hash
from app.selected_input_market import market_for
from app.structural_refit import load_bundle, verify_files
from app.training_forensics import tree_digest
from tools.run_discrete_curve_probe import read, write, now
from tools.run_long_sac_continuation import SUPPORT


def components(net):
    return {**{f'branch.{i}': b for i, b in enumerate(net.encoder.branches)},
            'fusion': net.encoder.fusion, 'attention': net.encoder.post_fusion,
            'core': net.encoder.core, 'head': net.head}


def probe_rows(inputs):
    years = pd.to_datetime(inputs.row_ts_ns, utc=True).year.to_numpy()
    selected = []
    for year in range(2020, 2024):
        indices = np.flatnonzero(inputs.valid & (inputs.period == 'TRAIN') & (years == year))
        if len(indices) < 128:
            raise ValueError('CAPACITY_YEAR_SUPPORT')
        selected.extend(indices[np.linspace(0, len(indices)-1, 128, dtype=int)])
    rows = np.asarray(selected)
    return rows, years[rows]


def point_epochs(config, seed, available):
    epochs = [0, 1, max(available)] if config is None else config[str(seed)]
    if (len(epochs) != 3 or epochs != sorted(set(epochs))
            or any(type(e) is not int or e < 0 or (e and e not in available) for e in epochs)):
        raise ValueError('CAPACITY_POINT_EPOCHS')
    return epochs


def extract_net(model, state, role):
    regularization = None if role == 'actor' else model.effective['critic_regularization']
    net = ThreeLogits(model.recipe, model.effective['architecture'], regularization)
    prefix = 'actor_network_params.module.0.module.' if role == 'actor' else 'qvalue_network_params.module.'
    subset = {k[len(prefix):]: v if role == 'actor' else v[int(role[-1])]
              for k, v in state.items() if k.startswith(prefix)}
    net.load_state_dict(subset, strict=True)
    return net.eval()


def profiles(net, x, years):
    activations = {}
    handles = []
    modules = components(net) | {'flattened': net.encoder, 'hidden': net.head[1]}
    def capture(name):
        def hook(module, args, output):
            activations[name] = output.detach().clone()
        return hook
    try:
        for name, module in modules.items():
            handles.append(module.register_forward_hook(capture(name)))
        with torch.inference_mode():
            logits = net(x)
    finally:
        for handle in handles:
            handle.remove()
    result = {}
    for name, values in activations.items():
        masks = {'pooled': np.ones(len(years), bool)} | {str(y): years == y for y in sorted(set(years))}
        result[name] = {}
        for label, mask in masks.items():
            a = values[mask]
            if a.ndim == 3:
                a = a.transpose(1, 2).reshape(-1, a.shape[1])
            result[name][label] = activation_spectrum(a)
    weights = {name: spectrum(p.reshape(p.shape[0], -1)) for name, p in net.named_parameters() if p.ndim >= 2}
    return logits, dict(activations=result, weights=weights,
                       parameters=sum(p.numel() for p in net.parameters()))


def interventions(net, x, years, baseline):
    before = tree_digest(net.state_dict())
    results = []
    for name, module in components(net).items():
        original = {k: v.detach().clone() for k, v in module.state_dict().items()}
        for fraction in (1., .75, .5):
            try:
                changed2 = original2 = 0.
                with torch.no_grad():
                    for key, p in module.named_parameters():
                        if p.ndim < 2:
                            continue
                        p.copy_(low_rank(original[key], fraction))
                        changed2 += float((p.double()-original[key].double()).square().sum())
                        original2 += float(original[key].double().square().sum())
                with torch.inference_mode():
                    candidate = net(x)
                if fraction == 1. and not torch.equal(candidate, baseline):
                    raise ValueError('CAPACITY_IDENTITY_OUTPUT')
                masks = {'pooled': np.ones(len(years), bool)} | {str(y): years == y for y in sorted(set(years))}
                results.append(dict(component=name, fraction=fraction,
                    relative_weight_l2=(changed2/max(original2, 1e-30))**.5,
                    function={label: signal_change(baseline[mask], candidate[mask]) for label, mask in masks.items()}))
            finally:
                module.load_state_dict(original)
            with torch.inference_mode():
                if not torch.equal(net(x), baseline) or tree_digest(net.state_dict()) != before:
                    raise ValueError('CAPACITY_RESTORE')
    return results


def run(a):
    if torch.cuda.is_available():
        raise ValueError('CAPACITY_CPU_REQUIRED')
    torch.set_num_threads(1)
    if a.output.exists():
        raise ValueError('CAPACITY_NEW_OUTPUT_REQUIRED')
    a.output.mkdir(parents=True)
    spec = read(a.parent/'SPEC.json')
    verify_files(a.parent, {k:v for k,v in spec['files'].items()
                           if k.startswith(('data/', 'engine/')) or k == 'SUPPORT_RECIPE.json'})
    summary = []
    shared = None
    point_config = None if a.epochs_json is None else read(a.epochs_json)
    if point_config is not None and set(point_config) != {'7', '19', '43'}:
        raise ValueError('CAPACITY_POINT_SEEDS')
    for seed in (7, 19, 43):
        root = a.snapshots/f'seed_{seed}'
        history = read(root/'HISTORY.json')
        model, _ = load_bundle(root/'initial_bundle', 'cpu')
        market = market_for(a.parent, spec, model.recipe)
        if support_hash(market.inputs) != SUPPORT:
            raise ValueError('CAPACITY_SUPPORT')
        rows, years = probe_rows(market.inputs)
        x = torch.as_tensor(market.inputs.windows[rows].copy())
        sample = tree_digest(dict(rows=rows, timestamps=market.inputs.row_ts_ns[rows], values=x))
        if shared is not None and sample != shared:
            raise ValueError('CAPACITY_UNPAIRED_PROBES')
        shared = sample
        epochs = sorted(int(p.stem.split('_')[-1]) for p in (root/'weights').glob('epoch_*.pt'))
        initial_state = {k:v.clone() for k,v in model.loss.state_dict().items()}
        with torch.inference_mode():
            reference = extract_net(model, initial_state, 'actor')(x).clone()
        for epoch in point_epochs(point_config, seed, epochs):
            path = root/'initial_bundle/model.pt' if epoch == 0 else root/'weights'/f'epoch_{epoch:04d}.pt'
            checksum = file_sha(path)
            if epoch:
                record = next(h for h in history if h['epoch'] == epoch)
                if checksum != record['weight_snapshot']['sha256']:
                    raise ValueError('CAPACITY_SNAPSHOT_HASH')
                state = torch.load(path, map_location='cpu', weights_only=True)
            else:
                state = initial_state
            model.loss.load_state_dict(state)
            result = dict(seed=seed, epoch=epoch, checkpoint_sha256=checksum,
                          sample_sha256=sample, roles={}, interventions=[], started=now())
            for role in ('actor', 'critic0', 'critic1'):
                net = extract_net(model, state, role)
                logits, profile = profiles(net, x, years)
                with torch.inference_mode():
                    native = model.surfaces(x) if role == 'actor' else model.q_surfaces(x)[int(role[-1])]
                profile['extraction_max_error'] = float((native-logits).abs().max())
                if not torch.allclose(native, logits, rtol=1e-5, atol=1e-6):
                    raise ValueError('CAPACITY_NET_EXTRACTION')
                result['roles'][role] = profile
                if role == 'actor':
                    result['change_from_initial'] = signal_change(reference, logits)
                    result['interventions'] = interventions(net, x, years, logits)
            if file_sha(path) != checksum:
                raise ValueError('CAPACITY_SOURCE_CHANGED')
            result.update(finished=now(), optimizer_steps=0, promoted=False, restoration_exact=True)
            write(a.output/f'SEED_{seed}_EPOCH_{epoch}.json', result)
            summary.append(dict(seed=seed, epoch=epoch, artifact=f'SEED_{seed}_EPOCH_{epoch}.json'))
            write(a.output/'PROGRESS.json', dict(at=now(), completed=len(summary), total=9, points=summary))
            print(f'capacity seed={seed} epoch={epoch} completed={len(summary)}/9', flush=True)
    write(a.output/'RESULT.json', dict(schema='capacity_diagnostic.v1', points=summary,
          sample_sha256=shared, rows=512, period='TRAIN2020-2023', optimizer_steps=0,
          promoted=False, optimal_width_inferred=False, data_years_required_inferred=False,
          files={r['artifact']:file_sha(a.output/r['artifact']) for r in summary}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('parent', 'snapshots', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--epochs-json', type=Path,
                        help='Optional three predeclared epochs per seed, e.g. before/at/after LR reduction')
    run(parser.parse_args())
