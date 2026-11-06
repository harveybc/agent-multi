"""Authenticated CPU diagnosis of completed SAC runs, sequential and resumable."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from tensordict import TensorDict

from app.generalization_diagnostics import (summary, ring_timeline, verify_replay_windows,
    quarter_metrics, context_coverage, frozen_probe, soft_td_target)
from app.structural_refit import load_bundle, verify_files
from app.frozen_recipe_inputs import file_sha
from app.three_signal_learning import environment
from tools.run_structural_sac_refit import authenticate, market_for
from tools.run_discrete_curve_probe import read, write
from agent_plugins.sac_critic_regularization import module_mode
from app.training_diagnostics import render_diagnostics


def sample_surfaces(model, x):
    with torch.inference_mode():
        logits = model.surfaces(x)
        q = model.q_surfaces(x)
        p = logits.softmax(-1)
    return dict(logits=logits, q=q, p=p)


def model_probe(model, inputs):
    out = {}
    state = {k:v.detach().clone() for k,v in model.loss.state_dict().items()}
    with frozen_probe(model.loss):
        for period in ('TRAIN', 'VALIDATION'):
            candidates = np.flatnonzero(inputs.valid & (inputs.period == period))
            rows = candidates[np.linspace(0, len(candidates)-1, min(512, len(candidates)), dtype=int)]
            x = torch.tensor(inputs.windows[rows], device='cpu')
            s = sample_surfaces(model, x)
            entropy = -(s['p'] * s['p'].clamp_min(1e-30).log()).sum(-1)
            changed = x.flip(1)
            alt = sample_surfaces(model, changed)
            out[period] = dict(rows=len(rows), logit=summary(s['logits'].numpy()),
                q=summary(s['q'].numpy()), twin_gap=summary((s['q'][0]-s['q'][1]).abs().numpy()),
                entropy=summary(entropy.numpy()),
                q_std_across_observations=s['q'].std(dim=1,unbiased=False).numpy().tolist(),
                greedy_share={str(a):float((s['logits'].argmax(-1)==a).float().mean()) for a in range(3)},
                time_reversal_probability_change=summary((s['p']-alt['p']).abs().numpy()),
                time_reversal_action_change=float((s['logits'].argmax(-1)!=alt['logits'].argmax(-1)).float().mean()))
            worst = s['q'].abs().amax((0,2)).topk(min(3,len(rows))).indices.numpy()
            out[period]['largest_q_rows'] = [dict(timestamp=int(inputs.row_ts_ns[rows[i]]),
                q=s['q'][:,i,:].numpy().tolist(), max_abs_input=float(x[i].abs().max()),
                largest_input_column=model.recipe['list']['columns'][int(x[i].abs().amax(0).argmax())])
                for i in worst]
            bn = [m for m in model.q_net.modules() if isinstance(m, torch.nn.BatchNorm1d)]
            if bn:
                discrepancies = []
                def collect(module, args):
                    z = args[0]
                    discrepancies.append(dict(mean_in_running_std=summary(((z.mean(0)-module.running_mean) /
                        (module.running_var+module.eps).sqrt()).abs().numpy()),
                        variance_ratio=summary((z.var(0, unbiased=False)/(module.running_var+module.eps)).numpy())))
                hooks = [m.register_forward_pre_hook(collect) for m in bn]
                try:
                    with torch.no_grad():
                        model.q_surfaces(x)
                finally:
                    for hook in hooks: hook.remove()
                with frozen_probe(model.loss), torch.no_grad(), module_mode(model.q_net, True):
                    batch_q = []
                    for i in range(2):
                        with model.loss.qvalue_network_params[i].to_module(model.loss.qvalue_network):
                            batch_q.append(model.q_net.surface(x).detach().clone())
                    batch_q = torch.stack(batch_q)
                out[period]['batchnorm'] = dict(statistics=discrepancies,
                    q_batch_vs_running_abs=summary((batch_q-s['q']).abs().numpy()))
    if any(not torch.equal(v, state[k]) for k,v in model.loss.state_dict().items()):
        raise ValueError('PROBE_MUTATED_MODEL')
    return out


def td_probe(model, storage, slots):
    selected = slots[np.linspace(0,len(slots)-1,min(1024,len(slots)),dtype=int)]
    with frozen_probe(model.loss), torch.no_grad(), module_mode(model.loss.qvalue_network, False):
        x, nxt = storage['features'][selected], storage['next.features'][selected]
        q = model.q_surfaces(x)
        p = model.surfaces(nxt).softmax(-1)
        target_q = []
        for i in range(2):
            with model.loss.target_qvalue_network_params[i].to_module(model.loss.qvalue_network):
                target_q.append(model.q_net.surface(nxt).detach().clone())
        target_q = torch.stack(target_q)
        alpha = model.loss._alpha.detach()
        y = soft_td_target(storage['next.reward'][selected],storage['next.terminated'][selected],
            storage['steps_to_next_obs'][selected],p,target_q,alpha,model.effective['gamma'])
        td = TensorDict({tuple(k.split('.')):v[selected] for k,v in storage.items() if k!='index'},
                        batch_size=[len(selected)])
        native = model.loss._compute_target(td)
        if not torch.allclose(y, native, rtol=2e-5, atol=2e-5):
            raise ValueError('TD_TARGET_NATIVE_PARITY')
        action = storage['action'][selected].long().reshape(-1)
        qa = q.gather(-1,action[None,:,None].expand(2,-1,1)).squeeze(-1)
        return dict(rows=len(selected), reward=summary(storage['next.reward'][selected].numpy()),
            target=summary(y.numpy()), abs_td_error=summary((qa-y).abs().numpy()), alpha=float(alpha),
            native_target_max_abs_error=float((y-native).abs().max()),
            scope='frozen eval-mode TD0 on final retained TRAIN replay; not epoch loss or contemporaneous best-checkpoint replay')


def input_shift(inputs, columns):
    x = inputs.windows[:,-1,:]
    train = x[inputs.valid & (inputs.period=='TRAIN')]
    validation = x[inputs.valid & (inputs.period=='VALIDATION')]
    out = []
    for i,column in enumerate(columns):
        a,b = train[:,i],validation[:,i]
        lo,hi = np.quantile(a,[.01,.99])
        sd = float(a.std())
        out.append(dict(column=column, train_mean=float(a.mean()), validation_mean=float(b.mean()),
            train_std=sd, validation_std=float(b.std()),
            mean_shift_in_train_std=float((b.mean()-a.mean())/sd) if sd>0 else None,
            validation_outside_train_p01_p99=float(((b<lo)|(b>hi)).mean()),
            train_max_abs=float(np.abs(a).max()),validation_max_abs=float(np.abs(b).max())))
    return out


def diagnose(root, runtime, output):
    started = time.monotonic()
    spec = authenticate(root, runtime)
    run = root/'runs/seed_7'
    result = read(run/'RESULT.json')
    if result['identity'] != spec['identity']:
        raise ValueError('RESULT_IDENTITY')
    tracked = {p: file_sha(run/p) for p in ('RESULT.json','HISTORY.json','RESUME.json')}
    pointer = read(run/'RESUME.json')
    recovery = run/'recovery'/pointer['directory']
    if recovery.parent.resolve() != (run/'recovery').resolve():
        raise ValueError('RECOVERY_PATH')
    if file_sha(recovery/'MANIFEST.json') != pointer['manifest_sha256']:
        raise ValueError('RECOVERY_HASH')
    manifest = read(recovery/'MANIFEST.json')
    if manifest['identity'] != spec['identity']:
        raise ValueError('RECOVERY_IDENTITY')
    verify_files(recovery, manifest['files'])
    market = market_for(root, spec)
    inputs = market.inputs
    risk = read(root/'RISK.json')
    env = environment(market, risk, spec['design']['horizon'],
        market.recipe['split']['periods']['TRAIN'], spec['effective']['gamma'])
    bars = np.flatnonzero(env.valid & (np.arange(len(env.valid)) >= env.engine.first)
                          & (np.arange(len(env.valid)) < env.engine.last))
    rows = env.rows[bars]
    replay = torch.load(recovery/'learning.pt', map_location='cpu', weights_only=False)['replay']
    storage = replay['_storage']['_storage']
    capacity = storage['features'].shape[0]
    writer = replay['_writer']
    timeline = ring_timeline(writer['_write_count'], capacity, len(rows), writer['_cursor'])
    if int(replay['_storage']['_len']) != len(timeline['slots']):
        raise ValueError('REPLAY_LENGTH')
    verified = verify_replay_windows(storage['features'], inputs.windows, rows, timeline)
    next_rows = np.r_[rows[1:],rows[-1]]
    verify_replay_windows(storage['next.features'],inputs.windows,next_rows,timeline)
    expected_terminal = timeline['decision'] == len(rows)-1
    if not np.array_equal(storage['next.terminated'][timeline['slots']].numpy().ravel(),expected_terminal):
        raise ValueError('REPLAY_TERMINALS')
    stamps = pd.to_datetime(inputs.row_ts_ns[rows[timeline['decision']]], utc=True)
    quarters = stamps.tz_localize(None).to_period('Q').astype(str)
    slots = timeline['slots']
    rewards = storage['next.reward'][slots].numpy().ravel()
    replay_periods = []
    for quarter in sorted(set(quarters)):
        chosen = quarters == quarter
        replay_periods.append(dict(quarter=quarter, transitions=int(chosen.sum()),
            reward=summary(rewards[chosen]),
            epoch_min=int(timeline['epoch'][chosen].min()),epoch_max=int(timeline['epoch'][chosen].max())))
    report = dict(schema='sac.generalization_diagnosis.v1', identity=spec['identity'],
        test_read=False, training_launched=False, selected=result, replay=dict(
            verified_windows=verified, write_count=writer['_write_count'], capacity=capacity,
            train_decisions=len(rows), age=summary(timeline['age']), calendar=replay_periods,
            actual_sampled_batch_history_available=False,
            sampler='uniform; retained composition is not historical sampled-batch composition',
            reward=summary(rewards)), quarters={}, context={}, models={},
            input_shift=input_shift(inputs,market.recipe['list']['columns']))
    for period in ('TRAIN','VALIDATION'):
        folder = run/'final'/period
        with np.load(folder/'EQUITY.npz') as trace:
            report['quarters'][period] = quarter_metrics(trace['timestamp'], trace['equity'], read(folder/'TRADES.json'))
        base = inputs.valid & (inputs.period == period)
        report['context'][period] = context_coverage(inputs.row_ts_ns, inputs.windows[:,-1,:], base, 13, 6)
    selected_path = run/result['best_bundle']['relative_path']
    if file_sha(selected_path/'MANIFEST.json') != result['best_bundle']['manifest_sha256']:
        raise ValueError('SELECTED_MANIFEST')
    model, _ = load_bundle(selected_path)
    report['models']['selected'] = model_probe(model, inputs)
    report['models']['selected']['TD'] = td_probe(model,storage,slots)
    model.load(recovery/'model.pt')
    report['models']['last'] = model_probe(model, inputs)
    report['models']['last']['TD'] = td_probe(model,storage,slots)
    history = read(run/'HISTORY.json')
    report['history'] = history
    report['interpretation_limits'] = [
        'Single seed diagnostics, not causal attribution or hyperparameter selection.',
        'Time reversal is a frozen-model sensitivity probe, not a feasible market or feature importance.',
        'Q values are discounted reward units, not predicted account returns.',
        'Quarter equity uses entering account equity, without resetting positions or capital.',
        'Realized trades attributed by exit date can differ from mark-to-market quarterly returns.',
        'Long-context coverage is measured without dropping baseline rows or filling missing values.']
    if any(file_sha(run/p) != sha for p,sha in tracked.items()):
        raise ValueError('SOURCE_RUN_CHANGED')
    report['source_artifacts_unchanged'] = True
    report['diagnostic_sources'] = diagnostic_sources()
    report['seconds'] = time.monotonic()-started
    output.mkdir(parents=True, exist_ok=True)
    render_diagnostics(history,output/'training',spec['design'])
    write(output/'REPORT.json', report)
    pd.DataFrame([dict(period=p,**r) for p,rs in report['quarters'].items() for r in rs]).to_csv(output/'QUARTERS.csv',index=False)
    pd.DataFrame(report['input_shift']).to_csv(output/'INPUT_SHIFT.csv',index=False)
    write(output/'DONE.json', dict(identity=spec['identity'], report_sha256=file_sha(output/'REPORT.json'),
                                  source_artifacts=tracked, diagnostic_sources=diagnostic_sources(), seconds=report['seconds']))
    print(json.dumps(dict(run=root.name, status='COMPLETE',seconds=report['seconds'])),flush=True)


def diagnostic_sources():
    return {name:file_sha(Path(__file__).resolve().parents[1]/name) for name in
            ('tools/run_generalization_diagnostics.py','app/generalization_diagnostics.py','app/training_diagnostics.py')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', nargs=2, action='append', required=True, metavar=('ROOT','RUNTIME'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU_ONLY_REQUIRED')
    torch.set_num_threads(2)
    args.output.mkdir(parents=True,exist_ok=True)
    lock = (args.output/'queue.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    names = [Path(root).name for root,_ in args.run]
    if len(names) != len(set(names)):
        raise ValueError('UNIQUE_RUN_NAMES_REQUIRED')
    for root, runtime in args.run:
        root, runtime = Path(root).resolve(), Path(runtime).resolve()
        output = args.output.resolve()/root.name
        if output.is_relative_to(root):
            raise ValueError('OUTPUT_MUST_NOT_MODIFY_INPUT')
        done = output/'DONE.json'
        if done.exists():
            receipt = read(done)
            authenticate(root,runtime)
            if (receipt['identity'] == read(root/'SPEC.json')['identity']
                    and receipt.get('diagnostic_sources') == diagnostic_sources()
                    and file_sha(output/'REPORT.json') == receipt['report_sha256']
                    and all(file_sha(root/'runs/seed_7'/p)==sha for p,sha in receipt['source_artifacts'].items())):
                print(json.dumps(dict(run=root.name,status='ALREADY_COMPLETE')),flush=True)
                continue
            raise ValueError('DIAGNOSTIC_RESUME_IDENTITY')
        write(args.output/'QUEUE.json',dict(active=root.name,status='RUNNING'))
        try:
            diagnose(root, runtime, output)
        except Exception as exc:
            write(args.output/'QUEUE.json',dict(active=root.name,status='FAILED',error=str(exc)))
            raise
    write(args.output/'QUEUE.json',dict(status='COMPLETE',runs=names))


if __name__ == '__main__':
    main()
