"""Replay authenticated actor gradients without fitting or promoting a candidate."""
import argparse
import fcntl
from pathlib import Path
import time

import numpy as np
import torch

from app.actor_gradient_coupling import decompose, first_adamw_delta, classify_pressure
from app.actor_learning_diagnostics import actor_terms, clear_optimizer_memory
from app.frozen_recipe_inputs import file_sha
from app.rl_subset import digest, project_inputs, support_hash
from app.selected_warm_start import reset_learning_rng
from app.structural_refit import load_bundle, verify_files
from tools.diagnose_warmstart_drift import set_arm, frozen_state, verify_frozen_state
from tools.run_bayesian_head_contrast import seal, verify_done
from tools.run_discrete_curve_probe import configure, read, write, now
from tools.run_structural_sac_refit import market_for

ROOT = Path(__file__).resolve().parents[1]


def run(a):
    started = time.monotonic(); configure(283); torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available() or torch.cuda.get_device_name() != a.expected_device_name:
        raise ValueError('COUPLING_DEVICE')
    source = read(a.source/'CONFIG.json'); verify_done(a.source, source['identity'])
    if source['schema'] != 'opening_boundary_refinement.v1': raise ValueError('COUPLING_SOURCE')
    parent = read(a.parent/'SPEC.json'); verify_files(a.parent, parent['files'])
    model, _ = load_bundle(a.bundle, 'cuda')
    if file_sha(a.bundle/'model.pt') != source['source_sha256']: raise ValueError('COUPLING_WEIGHTS')
    if parent['identity'] != source['parent_identity']: raise ValueError('COUPLING_PARENT')
    market = market_for(a.parent, parent)
    market.inputs = project_inputs(market.inputs, model.recipe)
    if support_hash(market.inputs) != source['support_sha256']: raise ValueError('COUPLING_SUPPORT')
    records = read(a.source/'panel/RECORDS.json'); rows = np.array([r['row'] for r in records])
    if (not market.inputs.valid[rows].all() or not (market.inputs.period[rows] == 'TRAIN').all()
            or not np.array_equal(market.inputs.row_ts_ns[rows], [r['timestamp'] for r in records])):
        raise ValueError('COUPLING_TRAIN_PANEL')
    cfg = dict(schema='actor_gradient_coupling.v1', source_identity=source['identity'],
        source_receipt_sha256=file_sha(a.source/'DONE.json'), source_sha256=source['source_sha256'],
        recipe_identity=model.recipe['identity'], support_sha256=source['support_sha256'],
        model_promoted=False, test_read=False, new_trained_checkpoint=False,
        sources={str(p.relative_to(ROOT)):file_sha(p) for folder in ('app','agent_plugins','tools')
                 for p in (ROOT/folder).rglob('*.py')})
    cfg['identity'] = digest(cfg)
    if (a.output/'CONFIG.json').exists() and read(a.output/'CONFIG.json') != cfg:
        raise ValueError('COUPLING_CONFIG')
    write(a.output/'CONFIG.json', cfg)
    if (a.output/'DONE.json').exists(): verify_done(a.output, cfg['identity']); return
    def progress(stage, **kw):
        d = dict(stage=stage, at=now(), seconds=time.monotonic()-started, **kw)
        write(a.output/'PROGRESS.json', d); print(d, flush=True)
    set_arm(model, 'actor_only')
    named = [(n,p) for n,p in model.loss.named_parameters() if p.requires_grad]
    if len(named) != 4 or any(not n.startswith('actor_') or '.head.' not in n for n,_ in named):
        raise ValueError('COUPLING_SCOPE')
    params = [p for _,p in named]
    x = torch.tensor(market.inputs.windows[rows], device='cuda')
    z = model.surfaces(x)
    flat = lambda values: torch.cat([v.detach().flatten().double() for v in values])
    progress('JACOBIAN', rows=len(rows), parameters=sum(p.numel() for p in params))
    jac = torch.stack([flat(torch.autograd.grad(z[i,k], params, retain_graph=True))
                       for i in range(len(rows)) for k in range(3)]).reshape(len(rows),3,-1)
    z0 = z.detach().double(); del z
    native = torch.tensor([[o['reward_return'] for o in r['outcomes']] for r in records],
                          dtype=torch.double, device='cuda')
    masks = {}
    for role in ('fit','monitor'):
        for state in ('all','flat_allowed','flat_restricted','long','short'):
            take = []
            for r in records:
                kind = ('long' if r['position']==1 else 'short') if r['position'] else (
                    'flat_allowed' if r['allowed'] else 'flat_restricted')
                take.append(r['role']==role and state in ('all',kind))
            masks[f'{role}:{state}'] = torch.tensor(take, device='cuda')
    layer_slices = {}; offset = 0
    for name,p in named:
        layer_slices[name] = slice(offset, offset+p.numel()); offset += p.numel()
    summary = dict(identity=cfg['identity'], source_identity=source['identity'],
        rows=len(rows), head_parameters=offset, arms={}, model_promoted=False,
        source_unchanged=True, test_read=False, new_trained_checkpoint=False)
    all_arrays = dict(jacobian=jac.cpu().numpy(), rows=rows, logits=z0.cpu().numpy())
    for arm in read(a.source/'RESULT.json')['arms']:
        key = arm['arm']; progress('ARM', arm=key)
        saved = read(a.source/key/'GRADIENT.json'); effective = read(a.source/key/'EFFECTIVE.json')
        if saved['rows'] != rows.tolist(): raise ValueError('COUPLING_ROWS')
        torch.testing.assert_close(z0, torch.tensor(saved['baseline_logits'], device='cuda', dtype=torch.double), rtol=0, atol=0)
        ix = torch.tensor(saved['fit_batch_indices'], device='cuda')
        if any(records[i]['role'] != 'fit' for i in ix.tolist()): raise ValueError('COUPLING_MONITOR_GRADIENT')
        q = torch.tensor(saved['q'], device='cuda', dtype=torch.double)
        alpha = effective['alpha']; result = dict(seed=arm['seed'], critic=arm['critic'], controls={})
        for control, qq in [('recorded_q', q), ('native_return_oracle', native)]:
            parts, weight, jm = decompose(z0, qq, alpha, jac, ix)
            checks = {}
            # Autograd uses the same full-panel forward and sampled loss as282's witness.
            zz = model.surfaces(x)
            v,e,_ = actor_terms(zz[ix], qq.float()[ix], torch.tensor(alpha,device='cuda'))
            for term, loss in [('value',v),('entropy',e),('total',v+e)]:
                direct = flat(torch.autograd.grad(loss,params,retain_graph=True))
                computed = parts[term]['batch_gradient']
                torch.testing.assert_close(computed,direct,atol=3e-7,rtol=3e-5)
                checks[term] = float((computed-direct).abs().max())
                if control == 'recorded_q':
                    torch.testing.assert_close(parts[term]['total'], torch.tensor(saved[term],device='cuda',dtype=torch.double), atol=3e-6,rtol=3e-4)
            del zz,v,e
            projections = {f'{term}:{stage}':part[stage] for term,part in parts.items()
                for stage in ('local_logit','self_alone','batch_self','cross','total')}
            layer_summary = {}
            for name,sl in layer_slices.items():
                contribution = -torch.einsum('ndp,p->nd', jm[:,:,sl],parts['total']['batch_gradient'][sl])
                layer_summary[name] = {role:dict(mean=contribution[mask].mean(0).tolist(),
                    mean_absolute=contribution[mask].abs().mean(0).tolist()) for role,mask in masks.items()}
                all_arrays[f'{key}:{control}:layer:{name}'] = contribution.cpu().numpy()
            result['controls'][control] = dict(autograd_max_errors=checks,
                pressure={role:classify_pressure(native,qq,projections,mask) for role,mask in masks.items()},
                layers=layer_summary)
            for name,value in projections.items(): all_arrays[f'{key}:{control}:{name}'] = value.cpu().numpy()
        # Reproduce the historical first step on a disposable copy, with actual batch forward.
        replay,_ = load_bundle(a.bundle,'cuda'); set_arm(replay,'actor_only')
        clear_optimizer_memory(replay.optimizer); reset_learning_rng(arm['seed'])
        frozen = frozen_state(replay)
        rp = [p for n,p in replay.loss.named_parameters() if p.requires_grad]
        theta = flat(rp); initial_logits = replay.surfaces(x).detach().double()
        replay.optimizer.zero_grad(set_to_none=True)
        v,e,_ = actor_terms(replay.surfaces(x[ix]),q.float()[ix],torch.tensor(alpha,device='cuda'))
        (v+e).backward(); norm = torch.nn.utils.clip_grad_norm_(rp,10.,error_if_nonfinite=True)
        grad = flat([p.grad for p in rp])
        groups = replay.optimizer.param_groups
        if len(groups)!=1 or groups[0].get('maximize',False): raise ValueError('COUPLING_OPTIMIZER')
        group = groups[0]
        adaptive,decay = first_adamw_delta(theta,grad,group['lr'],group['weight_decay'],group['eps'])
        replay.optimizer.step(); verify_frozen_state(replay,frozen)
        delta = flat(rp)-theta
        torch.testing.assert_close(delta,adaptive+decay,rtol=1e-3,atol=2e-7)
        cold,_ = load_bundle(a.bundle,'cuda'); cold.load(a.source/key/'ACTOR_0001.pt');set_arm(cold,'actor_only')
        cp = [p for _,p in cold.loss.named_parameters() if p.requires_grad]
        if any(not torch.equal(p,c) for p,c in zip(rp,cp)): raise ValueError('COUPLING_STEP_REPLAY')
        actual_logits = replay.surfaces(x).detach().double()
        actual = (actual_logits[:,1:]-actual_logits[:,:1])-(initial_logits[:,1:]-initial_logits[:,:1])
        adam = dict(adaptive=torch.einsum('ndp,p->nd',jm,adaptive),
                    decay=torch.einsum('ndp,p->nd',jm,decay),
                    linear_actual=torch.einsum('ndp,p->nd',jm,delta),actual=actual)
        result['adamw'] = dict(checkpoint_exact=True,gradient_norm=float(norm),
            parameter_formula_max_error=float((delta-adaptive-decay).abs().max()),
            nonlinear_margin_max_error=float((actual-adam['linear_actual']).abs().max()),
            pressure={role:classify_pressure(native,q,adam,mask) for role,mask in masks.items()})
        for name,value in adam.items():all_arrays[f'{key}:adamw:{name}'] = value.cpu().numpy()
        summary['arms'][key] = result
        write(a.output/'SUMMARY.json',summary)
        del replay,cold
    if file_sha(a.bundle/'model.pt') != source['source_sha256']:raise ValueError('COUPLING_SOURCE_CHANGED')
    verify_done(a.source,source['identity'])
    np.savez_compressed(a.output/'DECOMPOSITION.npz',**all_arrays)
    summary.update(seconds=time.monotonic()-started,finished_at=now())
    write(a.output/'SUMMARY.json',summary);seal(a.output,cfg['identity']);progress('COMPLETE')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('parent','bundle','source','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--expected-device-name',required=True)
    a = p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(a)
        except Exception as exc:
            write(a.output/'FAILED.json',dict(error=repr(exc),at=now()));raise
