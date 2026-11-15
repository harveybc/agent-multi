"""Pinned, resumable three-arm online SAC continuation with epoch weight evidence."""
import argparse
from copy import deepcopy
import fcntl
from pathlib import Path

import numpy as np
import torch

from app.frozen_recipe_inputs import file_sha
from app.g3_replay import JointReplay
from app.rl_subset import digest, support_hash
from app.sac_training_reliability import split_optimizer, weight_diagnostics
from app.selected_input_market import market_for
from app.structural_refit import verify_files, export_bundle, load_bundle
from app.three_signal_learning import environment
from app.training_diagnostics import render_diagnostics
from app.training_forensics import rng_state, tree_digest
from tools.run_discrete_curve_probe import configure, read, write, now
from tools.run_sac_training_reliability import ARMS, fresh, reseed, code_identity, atomic_save
from tools.run_structural_sac_refit import fit_epochs, assess_full


ORDERS = {7: ('actor_lr01', 'all_lr01', 'control'),
          19: ('control', 'actor_lr01', 'all_lr01'),
          43: ('all_lr01', 'control', 'actor_lr01')}
SOURCE = '69b5254489d224d4b20ac358956e4dd3b9a67a2f214a9aff98045c5def25a536'
SUPPORT = '28d6088a92173592f39e221c1c2ab1efc9094e614eeccb11c9ae23cf14878c98'


def long_design(parent):
    d = deepcopy(parent)
    d.update(head_only=False, initialize_epoch_zero=True, monitor_mode='validation',
             monitor='validation_full_accounts_return_per_week',
             monitor_start_epoch=20, select_during_warmup=True, require_activity=True,
             matched_budget=False, minimum_epochs=100, max_epochs=2000, patience=20,
             min_delta=1e-8, diagnostics=True, replay_backend='memmap')
    return d


def checked_baseline(metrics):
    expected = {'TRAIN': (16.05822355000215, 745), 'VALIDATION': (18.369945750000117, 207)}
    for period, (value, trades) in expected.items():
        m = metrics[period]
        if abs(m['return_pct']-value) > 1e-9 or m['trades'] != trades or m['orders_per_week'] <= 2:
            raise ValueError('LONG_BASELINE_PARITY:' + period)


def render_history(root, history, design, best_epoch=None):
    initial = read(root/'EPOCH_ZERO.json')
    zero = dict(epoch=0, metrics=initial['metrics'], steps=0, updates=0,
                monitor=dict(improved=True, bad=0), last_update={})
    render_diagnostics([zero, *history], root/'diagnostics', design, best_epoch=best_epoch)


def arm(a, name, market, risk, design, identity, probe):
    root = a.output/name
    root.mkdir(exist_ok=True)
    armid = digest(dict(parent=identity, arm=name, factors=ARMS[name]))
    if (root/'RESULT.json').exists():
        r = read(root/'RESULT.json')
        if r['identity'] != armid:
            raise ValueError('LONG_ARM_IDENTITY')
        verify_files(root, r['files'])
        return r
    model = fresh(a)
    rates = split_optimizer(model, ARMS[name])
    initial = {n: p.detach().clone() for n, p in model.loss.named_parameters()}
    x = torch.as_tensor(probe, device='cuda')
    with torch.no_grad():
        initial_actions = model.surfaces(x).argmax(-1)
    write(root/'DESIGN.json', design)
    write(root/'EFFECTIVE.json', dict(identity=armid, seed=a.seed, arm=name,
        base_effective=model.effective, optimizer_groups=rates,
        trainability={n: p.requires_grad for n, p in model.loss.named_parameters()},
        features=model.recipe['list']['n'], groups=len(model.recipe['representation']['resolved']['names']),
        input_shape=list(probe.shape[1:]), gpu=torch.cuda.get_device_name()))
    # A real cold load must preserve both parameter grouping and inherited Adam moments.
    gate = root/'initial_bundle'
    if not gate.exists():
        export_bundle(gate, model, probe, dict(epoch=0, run_identity=armid, seed=a.seed))
    cold, manifest = load_bundle(gate, 'cuda')
    if (manifest['checkpoint']['run_identity'] != armid
            or tree_digest(cold.loss.state_dict()) != tree_digest(model.loss.state_dict())
            or tree_digest(cold.optimizer.state_dict()) != tree_digest(model.optimizer.state_dict())):
        raise ValueError('LONG_COLD_OPTIMIZER_PARITY')
    del cold
    write(root/'QUALIFICATION.json', dict(identity=armid, cold_optimizer_exact=True,
                                         rates=[g['lr'] for g in model.optimizer.param_groups]))
    reseed(a.seed)
    interval = model.recipe['split']['periods']['TRAIN']
    def assess(m):
        metrics = assess_full(m, market, risk, design)
        if not (root/'EPOCH_ZERO.json').exists():
            checked_baseline(metrics)
        return metrics
    def hook(m, row):
        before = tree_digest(m.loss.state_dict())
        row['weight_diagnostics'] = weight_diagnostics(m, initial, x, initial_actions)
        weights = root/'weights'
        weights.mkdir(exist_ok=True)
        target = weights/('epoch_%04d.pt' % row['epoch'])
        atomic_save(target, {n: v.detach().cpu() for n, v in m.loss.state_dict().items()})
        row['weight_snapshot'] = dict(path=str(target.relative_to(root)), sha256=file_sha(target))
        if before != tree_digest(m.loss.state_dict()):
            raise ValueError('LONG_HOOK_MUTATED_WEIGHTS')
    write(a.output/'PROGRESS.json', dict(stage='ONLINE_CONTINUATION', arm=name, seed=a.seed,
        completed_arms=sum((a.output/n/'RESULT.json').exists() for n in ORDERS[a.seed]), at=now()))
    state = fit_epochs(model,
        lambda: environment(market, risk, design['horizon'], interval, model.effective['gamma']),
        assess, design, a.seed, root, armid, probe,
        replay_factory=lambda real: JointReplay(real, None, design['batch_size'], a.seed+200, None),
        epoch_hook=hook)
    selected = int(state['best_bundle']['path'].split('_')[-1])
    render_history(root, state['history'], design, selected)
    before = rng_state()
    metrics = assess_full(model, market, risk, design, save=root/'final', diagnose=True)
    if tree_digest(before) != tree_digest(rng_state()):
        raise ValueError('LONG_FINAL_ASSESS_MUTATED_RNG')
    bundle = root/state['best_bundle']['relative_path']
    cold, _ = load_bundle(bundle, 'cuda')
    if not torch.equal(model.surfaces(x), cold.surfaces(x)):
        raise ValueError('LONG_FINAL_COLD_LOGITS')
    del cold
    files = {str(p.relative_to(root)): file_sha(p) for p in bundle.iterdir() if p.is_file()}
    files.update({n: file_sha(root/n) for n in ('HISTORY.json', 'TRAINING.json', 'EFFECTIVE.json', 'QUALIFICATION.json')})
    r = dict(identity=armid, parent_identity=identity, seed=a.seed, arm=name, factors=ARMS[name],
        epochs=state['epoch'], max_epochs=design['max_epochs'], minimum_epochs=design['minimum_epochs'],
        steps=state['steps'], updates=state['updates'], selected_epoch=selected, selected_metrics=metrics,
        last_metrics=state['history'][-1]['metrics'], stopping=state['stopping'],
        stop_reason=read(root/'TRAINING.json')['stop_reason'], cold_exact=True, files=files,
        promoted=False, phase2_closed=False, deap_released=False)
    write(root/'RESULT.json', r)
    return r


def run(a):
    a.output.mkdir(parents=True, exist_ok=True)
    with (a.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        configure(a.seed)
        torch.backends.cudnn.allow_tf32 = False
        if not torch.cuda.is_available():
            raise ValueError('LONG_CUDA_REQUIRED')
        parent = read(a.parent/'SPEC.json')
        verify_files(a.parent, {k: v for k, v in parent['files'].items()
            if k.startswith(('data/', 'engine/')) or k in ('SUPPORT_RECIPE.json', 'RISK.json')})
        if file_sha(a.bundle/'model.pt') != SOURCE:
            raise ValueError('LONG_SOURCE_CHANGED')
        model = fresh(a)
        recipe = model.recipe
        if recipe['list']['n'] != 71 or set(recipe['split']['periods']) != {'TRAIN', 'VALIDATION'}:
            raise ValueError('LONG_RECIPE_CHANGED')
        design = long_design(parent['design'])
        binding = dict(schema='long_online_sac359.v1', seed=a.seed, source=SOURCE,
            manifest=file_sha(a.bundle/'MANIFEST.json'), parent=file_sha(a.parent/'SPEC.json'),
            code=code_identity(), design=design, effective=model.effective,
            arms={n: ARMS[n] for n in ('control', 'actor_lr01', 'all_lr01')},
            order=list(ORDERS[a.seed]), hardware=torch.cuda.get_device_name(), torch=torch.__version__)
        binding['identity'] = digest(binding)
        if (a.output/'SPEC.json').exists() and read(a.output/'SPEC.json') != binding:
            raise ValueError('LONG_RESUME_IDENTITY')
        write(a.output/'SPEC.json', binding)
        market = market_for(a.parent, parent, recipe)
        if support_hash(market.inputs) != SUPPORT:
            raise ValueError('LONG_SUPPORT_CHANGED')
        risk = read(a.parent/'RISK.json')
        ix = np.flatnonzero(market.inputs.valid & (market.inputs.period == 'TRAIN'))
        probe = market.inputs.windows[ix[np.linspace(0, len(ix)-1, 128, dtype=int)]].copy()
        del model
        results = [arm(a, n, market, risk, design, binding['identity'], probe) for n in ORDERS[a.seed]]
        write(a.output/'RESULT.json', dict(identity=binding['identity'], seed=a.seed,
            status='LONG_ONLINE_COMPLETE', arms=results, promoted=False, phase2_closed=False, deap_released=False))
        write(a.output/'PROGRESS.json', dict(stage='COMPLETE', seed=a.seed, at=now()))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('parent', 'bundle', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--seed', type=int, choices=tuple(ORDERS), required=True)
    a = p.parse_args()
    try:
        run(a)
    except Exception as exc:
        a.output.mkdir(parents=True, exist_ok=True)
        write(a.output/'FAILURE.json', dict(at=now(), error=repr(exc)))
        raise
