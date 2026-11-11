"""Diagnose sealed selected-input refinement without training another candidate."""
import argparse
import fcntl
from pathlib import Path
import time

import numpy as np
import torch

from app.actor_gradient_coupling import decompose, probability_derivative
from app.actor_learning_diagnostics import actor_terms, clear_optimizer_memory
from app.all_state_refinement import validate_panel
from app.frozen_recipe_inputs import file_sha
from app.rl_subset import digest, support_hash
from app.selected_input_market import market_for
from app.selected_refinement_diagnosis import adaptive_parts, action_quality
from app.structural_refit import load_bundle, verify_files
from tools.diagnose_warmstart_drift import set_arm
from tools.run_bayesian_head_contrast import seal, verify_done
from tools.run_discrete_curve_probe import configure, read, write, now


def flat(parameters):
    return torch.cat([p.detach().flatten().double() for p in parameters])


def head(model):
    set_arm(model, 'actor_only')
    named = [(n, p) for n, p in model.loss.named_parameters() if p.requires_grad]
    if len(named) != 4 or any('.head.' not in n or not n.startswith('actor_') for n, _ in named):
        raise ValueError('SELECTED_DIAG_SCOPE')
    return named


def run(a):
    started = time.monotonic()
    configure(291)
    source = read(a.source/'CONFIG.json')
    if source['schema'] != 'selected_input_bounded_refinement.v1':
        raise ValueError('SELECTED_DIAG_SOURCE')
    verify_done(a.source, source['identity'])
    export = read(a.bundle.parent/'RESULT.json')
    verify_done(a.bundle.parent, export['identity'])
    parent = read(a.parent/'SPEC.json')
    verify_files(a.parent, parent['files'])
    sha = file_sha(a.bundle/'model.pt')
    if sha != source['source_sha256'] or parent['identity'] != source['parent_identity']:
        raise ValueError('SELECTED_DIAG_LINEAGE')
    model, _ = load_bundle(a.bundle, 'cpu')
    market = market_for(a.parent, parent, model.recipe)
    if support_hash(market.inputs) != source['support_sha256']:
        raise ValueError('SELECTED_DIAG_SUPPORT')
    records = read(a.source/'panel/RECORDS.json')
    validate_panel(records)
    rows = [r['row'] for r in records]
    if (not market.inputs.valid[rows].all()
            or not (market.inputs.period[rows] == 'TRAIN').all()
            or market.inputs.row_ts_ns[rows].tolist() != [r['timestamp'] for r in records]):
        raise ValueError('SELECTED_DIAG_PANEL')
    report_cfg = read(a.report/'CONFIG.json')
    report_identity = digest(report_cfg)
    verify_done(a.report, report_identity)
    if (report_cfg['source'] != source['identity']
            or report_cfg['receipt_sha256'] != file_sha(a.source/'DONE.json')):
        raise ValueError('SELECTED_DIAG_REPORT_LINEAGE')
    report = read(a.report/'SUMMARY.json')
    events = sorted({e['timestamp'] for account in report['accounts'] if account['period']=='VALIDATION'
                     for e in account['effective_events']})
    event_rows = []
    for stamp in events:
        ix = np.flatnonzero(market.inputs.row_ts_ns == stamp)
        if len(ix) != 1 or not market.inputs.valid[ix[0]] or market.inputs.period[ix[0]] != 'VALIDATION':
            raise ValueError('SELECTED_DIAG_POSTHOC_ROW')
        event_rows.append(int(ix[0]))
    all_rows = rows + event_rows
    x = torch.tensor(market.inputs.windows[all_rows])
    n = len(rows)
    native = torch.tensor([[o['reward_return'] for o in r['outcomes']] for r in records], dtype=torch.double)
    fit = torch.tensor([i for i, r in enumerate(records) if r['role']=='fit'])
    masks = {}
    for role in ('fit', 'monitor'):
        masks[role] = torch.tensor([r['role']==role for r in records])
        masks[role+':flat_allowed'] = torch.tensor([r['role']==role and r['position']==0 and r['allowed'] for r in records])
    cfg = dict(schema='selected_refinement_cause.v1', source=source['identity'],
               export=export['identity'], report=report_identity,
               support=source['support_sha256'], rows=rows, posthoc_timestamps=events,
               source_sha256=sha, new_candidate=False, test_read=False,
               code={str(p.relative_to(Path(__file__).resolve().parents[1])):file_sha(p)
                     for folder in ('app','tools','agent_plugins')
                     for p in (Path(__file__).resolve().parents[1]/folder).rglob('*.py')})
    cfg['identity'] = digest(cfg)
    if (a.output/'CONFIG.json').exists() and read(a.output/'CONFIG.json') != cfg:
        raise ValueError('SELECTED_DIAG_CONFIG')
    write(a.output/'CONFIG.json', cfg)
    if (a.output/'DONE.json').exists():
        verify_done(a.output, cfg['identity']); return
    named = head(model); params = [p for _, p in named]; theta = flat(params)
    z = model.surfaces(x)
    jac = torch.stack([flat(torch.autograd.grad(z[i,k], params, retain_graph=True))
                       for i in range(len(x)) for k in range(3)]).reshape(len(x),3,-1)
    z0 = z.detach().double(); del z
    jm = jac[:,1:]-jac[:,:1]
    summary = dict(identity=cfg['identity'], arms=[], rows=n, fit=len(fit),
                   posthoc_rows=len(events), new_candidate=False, test_read=False)
    arrays = dict(jacobian=jac.numpy(), logits=z0.numpy(), rows=np.array(all_rows),
                  native_returns=native.numpy(), fit=fit.numpy())
    layer_slices = {}; offset = 0
    for name, parameter in named:
        layer_slices[name] = slice(offset, offset+parameter.numel())
        offset += parameter.numel()
    for seed in source['seeds']:
        key = f'calibrated_adamw_{seed}'
        print(dict(stage='DIAGNOSE', seed=seed, seconds=time.monotonic()-started), flush=True)
        effective = read(a.source/key/'EFFECTIVE.json')
        targets = read(a.source/key/'TARGETS.json')
        model.load(a.source/f'critic_{seed}/MODEL.pt'); params = [p for _,p in head(model)]
        if not torch.equal(flat(params), theta): raise ValueError('SELECTED_DIAG_ACTOR')
        with torch.no_grad(): measured_q = model.q_surfaces(x).min(0).values.double()
        q = torch.tensor(targets['effective'], dtype=torch.double)
        torch.testing.assert_close(measured_q[:n], q, atol=1e-5, rtol=1e-5)
        lookup = {r:i for i,r in enumerate(rows)}
        indices = torch.tensor([lookup[r] for r in effective['sampled_rows'][0]])
        if any(records[i]['role'] != 'fit' for i in indices.tolist()):
            raise ValueError('SELECTED_DIAG_FIT_ONLY')
        if file_sha(a.source/f'critic_{seed}/MODEL.pt') != effective['critic_sha256']:
            raise ValueError('SELECTED_DIAG_CRITIC')
        lr, wd = effective['learning_rate'], effective['weight_decay']
        alpha = effective['alpha']
        q_all = torch.cat([q, measured_q[n:]], dim=0)
        arrays[key+':q'] = q_all.numpy()
        native_fit = q_all.clone(); native_fit[fit] = native[fit]
        controls = {}; projections = {}
        for name, target, entropy in [('critic',q_all,alpha),('native_fit',native_fit,alpha),('critic_no_entropy',q_all,0.)]:
            parts, weights, _ = decompose(z0, target, entropy, jac, indices)
            probe = model.surfaces(x)
            v,e,_ = actor_terms(probe[indices], target[indices].float(), torch.tensor(entropy))
            direct = flat(torch.autograd.grad(v+e, params))
            error = float((direct-parts['total']['batch_gradient']).abs().max())
            torch.testing.assert_close(direct,parts['total']['batch_gradient'],atol=3e-7,rtol=3e-5)
            adam = adaptive_parts(parts['total'], weights, jm, theta, lr, wd, model.optimizer.param_groups[0]['eps'])
            stages = {f'{term}:{stage}':parts[term][stage]
                      for term in ('value','entropy','total')
                      for stage in ('local_logit','self_alone','batch_self','cross','total')}
            stages.update({'adam:'+k:v for k,v in adam.items()})
            detail = {}
            best = native.argmax(-1)
            for group, mask in masks.items():
                group_result = {}
                for stage, margin in stages.items():
                    dp = probability_derivative(z0.softmax(-1),margin)[:n]
                    selected = dp.gather(1,best[:,None]).squeeze(1)
                    gaps = native[:,1:]-native[:,:1]
                    positive = (gaps > 1e-8) & mask[:,None]
                    group_result[stage] = dict(
                        lowers_native_optimal_probability=int((mask & (selected < -1e-10)).sum()),
                        raises_native_optimal_probability=int((mask & (selected > 1e-10)).sum()),
                        suppresses_beneficial_gaps=int((positive & (margin[:n]<-1e-10)).sum()),
                        mean_native_optimal_probability_derivative=float(selected[mask].mean()))
                    arrays[f'{key}:{name}:{stage}'] = margin.numpy()
                detail[group] = group_result
            controls[name] = dict(autograd_max_error=error,groups=detail)
            projections[name] = stages
        # Replay only the already-saved first update on disposable in-memory weights.
        clear_optimizer_memory(model.optimizer)
        before = [p.detach().clone() for p in params]
        model.optimizer.zero_grad(set_to_none=True)
        v,e,_ = actor_terms(model.surfaces(x[indices]), q_all[indices].float(), torch.tensor(alpha))
        (v+e).backward(); torch.nn.utils.clip_grad_norm_(params,10.,error_if_nonfinite=True)
        model.optimizer.step()
        fraction = read(a.source/key/'STEPS.json')[0]['fraction']
        with torch.no_grad():
            for p,b in zip(params,before): p.copy_((b.double()+fraction*(p.double()-b.double())).to(p))
        replay_theta = flat(params); replay_logits = model.surfaces(x).detach().double()
        endpoints = {}
        for step in (1,8):
            model.load(a.source/key/f'ACTOR_{step:04d}.pt'); params = [p for _,p in head(model)]
            after_theta = flat(params); after_logits = model.surfaces(x).detach().double()
            actual = (after_logits[:,1:]-after_logits[:,:1])-(z0[:,1:]-z0[:,:1])
            linear = jm @ (after_theta-theta)
            if step==1:
                torch.testing.assert_close(replay_theta, after_theta, atol=3e-6, rtol=0)
                torch.testing.assert_close(replay_logits,after_logits,atol=2e-5,rtol=0)
                if not torch.equal(replay_logits.argmax(-1),after_logits.argmax(-1)):
                    raise ValueError('SELECTED_DIAG_REPLAY_ACTION')
                replay = dict(parameter_max_error=float((replay_theta-after_theta).abs().max()),
                              logit_max_error=float((replay_logits-after_logits).abs().max()),actions_equal=True)
            arrays[f'{key}:endpoint{step}:actual'] = actual.numpy()
            arrays[f'{key}:endpoint{step}:linear'] = linear.numpy()
            layer_effects = {}
            for layer, sl in layer_slices.items():
                component = jm[:,:,sl] @ (after_theta-theta)[sl]
                arrays[f'{key}:endpoint{step}:layer:{layer}'] = component.numpy()
                layer_effects[layer] = dict(
                    posthoc_sell_margin=component[n:,1].tolist(),
                    fit_sell_margin_mean=float(component[:n][masks['fit:flat_allowed'],1].mean()))
            endpoints[str(step)] = dict(nonlinear_margin_max_error=float((actual-linear).abs().max()),
                layers=layer_effects,
                groups={group:dict(changed_actions=int(((after_logits[:n].argmax(-1)!=z0[:n].argmax(-1))&mask).sum()),
                    lowers_optimal_probability=int((((after_logits[:n].softmax(-1)-z0[:n].softmax(-1))
                        .gather(1,native.argmax(-1)[:,None]).squeeze(1)<-1e-10)&mask).sum())) for group,mask in masks.items()},
                posthoc=[dict(timestamp=stamp,baseline_logits=z0[n+i].tolist(),after_logits=after_logits[n+i].tolist(),
                    critic_q=q_all[n+i].tolist(),actual_margin_delta=actual[n+i].tolist(),
                    linear_margin_delta=linear[n+i].tolist(),
                    local_value=projections['critic']['value:local_logit'][n+i].tolist(),
                    local_entropy=projections['critic']['entropy:local_logit'][n+i].tolist(),
                    shared_gradient=projections['critic']['total:total'][n+i].tolist(),
                    shared_adam=projections['critic']['adam:total'][n+i].tolist(),
                    native_fit_shared_adam=projections['native_fit']['adam:total'][n+i].tolist())
                    for i,stamp in enumerate(events)])
        summary['arms'].append(dict(arm=key,critic_quality={g:action_quality(native[m],q[m]) for g,m in masks.items()},
            controls=controls,replay=replay,endpoints=endpoints,alpha=alpha,
            q_cpu_max_error=float((measured_q[:n]-q).abs().max())))
        write(a.output/'SUMMARY.json',summary)
    np.savez_compressed(a.output/'PROJECTIONS.npz',**arrays)
    if file_sha(a.bundle/'model.pt')!=sha:raise ValueError('SELECTED_DIAG_SOURCE_CHANGED')
    verify_done(a.source,source['identity']);verify_done(a.bundle.parent,export['identity'])
    verify_done(a.report,report_identity)
    summary.update(seconds=time.monotonic()-started,source_unchanged=True,finished_at=now())
    write(a.output/'SUMMARY.json',summary);seal(a.output,cfg['identity'])
    print(dict(stage='COMPLETE',seconds=summary['seconds'],identity=cfg['identity']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('source','bundle','parent','report','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(args)
        except Exception as exc:
            write(args.output/'FAILED.json',dict(error=repr(exc),at=now()));raise
