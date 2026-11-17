"""Resumable selected299 branch A/D/control pilot, no live or automatic promotion."""
import argparse
from copy import deepcopy
import fcntl
import math
from pathlib import Path

import numpy as np
import torch

from app.branch_capacity_contrast import reduce_model
from app.best_plateau_rollback import ROLLBACK_PLATEAU
from app.frozen_recipe_inputs import file_sha
from app.g3_replay import JointReplay
from app.rl_subset import digest, support_hash
from app.sac_training_reliability import weight_diagnostics, split_optimizer
from app.selected_input_market import market_for
from app.structural_refit import verify_files, export_bundle, load_bundle
from app.three_signal_learning import environment
from app.training_forensics import rng_state, tree_digest
from tools.run_discrete_curve_probe import configure, read, write, now
from tools.run_long_sac_continuation import SOURCE, SUPPORT, long_design, checked_baseline
from tools.run_sac_training_reliability import reseed, code_identity, atomic_save
from tools.run_structural_sac_refit import fit_epochs, assess_full


def refinement_design(base, original_lr, rates):
    if (len(rates) != 3 or not math.isfinite(original_lr) or original_lr <= 0
            or any(not math.isclose(lr, original_lr, rel_tol=1e-12, abs_tol=0.)
                   for lr in rates)):
        raise ValueError('REFINEMENT_START_LR')
    d = deepcopy(base)
    d.update(patience=80, minimum_epochs=100, max_epochs=2000, min_delta=1e-8,
        monitor_start_epoch=20, select_during_warmup=True,
        monitor_mode='balanced_min', shared_monitor_decision=True,
        monitor='balanced_min_full_accounts_return_per_week',
        actor_plateau=dict(factor=.2, threshold=1e-8, cooldown=0,
            min_actor_lr=original_lr/625, post_reduction_epochs=20,
            exact_patience_divisor=4, restore_best=True, stop_before_reduce=True,
            scope='all'))
    return d


def run(a):
    monitor_policy = getattr(a, 'monitor_policy', 'validation')
    refinement = getattr(a, 'refinement_profile', 'none')
    if (refinement not in ('none', 'champion80') or
            (refinement == 'champion80' and (a.arm != 'control'
             or a.lr_policy != 'restore_best_halves' or monitor_policy != 'balanced_min'))):
        raise ValueError('REFINEMENT_PROFILE_CONFIGURATION')
    if (monitor_policy not in ('validation', 'balanced_min', 'champion_nonregression')
            or (monitor_policy != 'validation' and
                (a.arm != 'control' or a.lr_policy != 'restore_best_halves'))
            or (monitor_policy == 'validation' and a.seed != 7)):
        raise ValueError('MONITOR_CONTRAST_CONFIGURATION')
    a.output.mkdir(parents=True, exist_ok=True)
    with (a.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        configure(a.seed)
        torch.backends.cudnn.allow_tf32 = False
        if not torch.cuda.is_available():
            raise ValueError('CAPACITY_CUDA_REQUIRED')
        parent = read(a.parent/'SPEC.json')
        verify_files(a.parent, {k: v for k, v in parent['files'].items()
            if k.startswith(('data/', 'engine/')) or k in ('SUPPORT_RECIPE.json', 'RISK.json')})
        if file_sha(a.bundle/'model.pt') != SOURCE:
            raise ValueError('CAPACITY_SOURCE_CHANGED')
        source, _ = load_bundle(a.bundle, 'cuda')
        original_lr = source.effective['learning_rate']
        model, transfer = reduce_model(source, a.arm, a.seed,
            learning_rate=original_lr if refinement == 'champion80' else 3e-6)
        del source
        lr_policy = getattr(a, 'lr_policy', 'fixed')
        if lr_policy == 'restore_best_halves':
            split_optimizer(model, dict(actor=1., critic=1., temperature=1.))
        elif lr_policy != 'fixed':
            raise ValueError('CAPACITY_LR_POLICY')
        recipe = model.recipe
        if recipe['list']['n'] != 71 or set(recipe['split']['periods']) != {'TRAIN', 'VALIDATION'}:
            raise ValueError('CAPACITY_RECIPE_CHANGED')
        design = long_design(parent['design'])
        design.update(allow_ineligible_epoch_zero=True)
        if lr_policy == 'restore_best_halves':
            design.update(allow_ineligible_epoch_zero=False, fork_epoch=0,
                          actor_plateau=ROLLBACK_PLATEAU)
        if monitor_policy != 'validation':
            design.update(monitor_mode=monitor_policy, shared_monitor_decision=True,
                          monitor=monitor_policy + '_full_accounts_return_per_week')
        if refinement == 'champion80':
            design = refinement_design(design, original_lr,
                                       [g['lr'] for g in model.optimizer.param_groups])
        binding = dict(schema='branch_capacity364.v1' if lr_policy == 'fixed' else 'branch_rollback365.v1',
            lr_policy=lr_policy, arm=a.arm, seed=a.seed,
            source=SOURCE, support=SUPPORT, manifest=file_sha(a.bundle/'MANIFEST.json'),
            parent=file_sha(a.parent/'SPEC.json'), code=code_identity(), design=design,
            effective=model.effective, transfer=transfer,
            hardware=torch.cuda.get_device_name(), torch=torch.__version__)
        if monitor_policy != 'validation':
            binding.update(schema='monitor_contrast367.v1', monitor_policy=monitor_policy)
        if refinement == 'champion80':
            binding.update(schema='champion_refinement368.v1', refinement_profile=refinement,
                           source_learning_rate=original_lr, initialization='preserve_all')
        binding['identity'] = digest(binding)
        identity = binding['identity']
        if (a.output/'SPEC.json').exists() and read(a.output/'SPEC.json') != binding:
            raise ValueError('CAPACITY_RESUME_IDENTITY')
        write(a.output/'SPEC.json', binding)
        if (a.output/'RESULT.json').exists():
            result = read(a.output/'RESULT.json')
            if result['identity'] != identity:
                raise ValueError('CAPACITY_RESULT_IDENTITY')
            verify_files(a.output, result['files'])
            return
        write(a.output/'PROGRESS.json', dict(stage='INPUT_QUALIFICATION', arm=a.arm, at=now()))
        market = market_for(a.parent, parent, recipe)
        if support_hash(market.inputs) != SUPPORT:
            raise ValueError('CAPACITY_SUPPORT_CHANGED')
        risk = read(a.parent/'RISK.json')
        ix = np.flatnonzero(market.inputs.valid & (market.inputs.period == 'TRAIN'))
        probe = market.inputs.windows[ix[np.linspace(0, len(ix)-1, 128, dtype=int)]].copy()
        initial = {n: p.detach().clone() for n, p in model.loss.named_parameters()}
        x = torch.as_tensor(probe, device='cuda')
        with torch.no_grad():
            initial_actions = model.surfaces(x).argmax(-1)
        write(a.output/'DESIGN.json', design)
        write(a.output/'EFFECTIVE.json', dict(identity=identity, seed=a.seed, arm=a.arm,
            base_effective=model.effective, transfer=transfer,
            trainability={n: p.requires_grad for n, p in model.loss.named_parameters()},
            features=recipe['list']['n'], groups=len(recipe['representation']['resolved']['names']),
            input_shape=list(probe.shape[1:])))
        gate = a.output/'initial_bundle'
        if not gate.exists():
            export_bundle(gate, model, probe, dict(epoch=0, run_identity=identity, seed=a.seed))
        cold, manifest = load_bundle(gate, 'cuda')
        if (manifest['checkpoint']['run_identity'] != identity
                or tree_digest(cold.loss.state_dict()) != tree_digest(model.loss.state_dict())
                or tree_digest(cold.optimizer.state_dict()) != tree_digest(model.optimizer.state_dict())):
            raise ValueError('CAPACITY_COLD_OPTIMIZER_PARITY')
        del cold
        write(a.output/'QUALIFICATION.json', dict(identity=identity, cold_optimizer_exact=True,
            source=SOURCE, support=SUPPORT, rates=[g['lr'] for g in model.optimizer.param_groups],
            transfer=transfer, input_shape=list(probe.shape[1:])))
        reseed(a.seed)
        interval = recipe['split']['periods']['TRAIN']

        def assess(m):
            metrics = assess_full(m, market, risk, design)
            if a.arm == 'control' and not (a.output/'EPOCH_ZERO.json').exists():
                checked_baseline(metrics)
            return metrics

        def hook(m, row):
            before = tree_digest(m.loss.state_dict())
            row['weight_diagnostics'] = weight_diagnostics(m, initial, x, initial_actions)
            weights = a.output/'weights'
            weights.mkdir(exist_ok=True)
            target = weights/('epoch_%04d.pt' % row['epoch'])
            atomic_save(target, {n: v.detach().cpu() for n, v in m.loss.state_dict().items()})
            row['weight_snapshot'] = dict(path=str(target.relative_to(a.output)), sha256=file_sha(target))
            if before != tree_digest(m.loss.state_dict()):
                raise ValueError('CAPACITY_HOOK_MUTATED_WEIGHTS')

        state = fit_epochs(model,
            lambda: environment(market, risk, design['horizon'], interval, model.effective['gamma']),
            assess, design, a.seed, a.output, identity, probe,
            replay_factory=lambda real: JointReplay(real, None, design['batch_size'], a.seed+200, None),
            epoch_hook=hook)
        selected, metrics = None, None
        files = {n: file_sha(a.output/n) for n in
                 ('SPEC.json', 'HISTORY.json', 'TRAINING.json', 'EFFECTIVE.json', 'QUALIFICATION.json')}
        if state['best_bundle'] is not None:
            selected = int(state['best_bundle']['path'].split('_')[-1])
            before = rng_state()
            metrics = assess_full(model, market, risk, design, save=a.output/'final', diagnose=True)
            if tree_digest(before) != tree_digest(rng_state()):
                raise ValueError('CAPACITY_FINAL_ASSESS_MUTATED_RNG')
            bundle = a.output/state['best_bundle']['relative_path']
            cold, _ = load_bundle(bundle, 'cuda')
            if not torch.equal(model.surfaces(x), cold.surfaces(x)):
                raise ValueError('CAPACITY_FINAL_COLD_LOGITS')
            files.update({str(p.relative_to(a.output)): file_sha(p) for p in bundle.iterdir() if p.is_file()})
        write(a.output/'RESULT.json', dict(identity=identity, arm=a.arm, seed=a.seed,
            status='COMPLETE' if selected is not None else 'COMPLETE_NO_ELIGIBLE_CHECKPOINT',
            epochs=state['epoch'], steps=state['steps'], updates=state['updates'],
            selected_epoch=selected, selected_metrics=metrics,
            last_metrics=state['history'][-1]['metrics'], stopping=state['stopping'],
            stop_reason=read(a.output/'TRAINING.json')['stop_reason'], files=files,
            promoted=False, phase2_closed=False, deap_released=False))
        write(a.output/'PROGRESS.json', dict(stage='COMPLETE', arm=a.arm, at=now()))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('parent', 'bundle', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--seed', type=int, choices=(7,19), default=7)
    p.add_argument('--arm', choices=('A', 'D', 'control'), required=True)
    p.add_argument('--lr-policy', choices=('fixed','restore_best_halves'), default='fixed')
    p.add_argument('--monitor-policy', choices=('validation','balanced_min','champion_nonregression'),
                   default='validation')
    p.add_argument('--refinement-profile', choices=('none','champion80'), default='none')
    a = p.parse_args()
    try:
        run(a)
    except Exception as exc:
        a.output.mkdir(parents=True, exist_ok=True)
        write(a.output/'FAILURE.json', dict(at=now(), error=repr(exc)))
        raise
