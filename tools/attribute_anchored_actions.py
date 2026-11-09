"""Frozen anchored-actor account attribution and critic-margin inspection."""
import argparse
import csv
import fcntl
from pathlib import Path
import time

import numpy as np
import torch

from app.anchored_action_attribution import event_metrics, summarize_events
from app.frozen_recipe_inputs import file_sha
from app.frozen_signal_masks import prepared_signals
from app.rl_subset import digest, support_hash
from app.structural_refit import load_bundle
from app.three_signal_learning import emit, environment
from tools.attribute_ranking_improvement import attribute_pair
from tools.audit_fixed_actor_critics import verify_collection
from tools.run_action_ranking_contrast import seal, ensure_config
from tools.run_discrete_curve_probe import read, write, configure, now
from tools.run_structural_sac_refit import authenticate, market_for

ROOT=Path(__file__).resolve().parents[1]


def check_trace(inputs,logits,trace):
    rows=np.searchsorted(inputs.row_ts_ns,[r['timestamp'] for r in trace])
    if ((rows>=len(inputs.row_ts_ns)).any() or not inputs.valid[rows].all()
            or not np.array_equal(inputs.row_ts_ns[rows],[r['timestamp'] for r in trace])
            or not np.array_equal(prepared_signals(logits)[0][rows],[r['signal'] for r in trace])):
        raise ValueError('ANCHORED_TRACE_PARITY')


def export(root,accounts,events):
    keys=['replica','period','utc','old_signal','new_signal','position','effect','outcome','contribution_usd',
        'isolated_cash_gap','isolated_reward_gap','prefix_reward_gap','parent_pair_margin',
        'candidate_pair_margin','parent_probability_margin','candidate_probability_margin',
        'kl','q_gap','entropy_adjusted_parent_score_gap','q_reward_sign_conflict','cash_reward_sign_conflict']
    with (root/'EVENTS.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        writer.writerows({k:r[k] for k in keys} for r in events)
    write(root/'EVENTS.json',events)
    lines=['# Anchored actor attribution','',
        'Retrospective diagnostics; no training, fitted thresholds or policy promotion.','',
        '| Seed | Period | Delta USD | Changed signals | Helpful | Harmful | Cash reconciliation error |',
        '|---|---|---:|---:|---:|---:|---:|']
    for a in accounts:
        lines.append(f"| {a['replica']} | {a['period']} | {a['cash_delta_usd']:.6f} | {a['changed_signals']} | {a['helpful']} | {a['harmful']} | {a['reconciliation']['error_usd']} |")
    lines+=['','All changed decisions, including neutral ones, are in EVENTS.csv and EVENTS.json.',
        'Cash attribution is chronological and includes downstream opportunity and size effects.',
        'Q ordering is compared with discounted parent-policy continuations in the original same state.',
        'Validation is diagnostic only; do not fit intervention thresholds to these dates.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


def run(args):
    started=time.monotonic();configure(7);torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available() or torch.cuda.get_device_name()!='NVIDIA GeForce RTX 5090':
        raise ValueError('ANCHORED_SURFACE_DEVICE')
    spec=authenticate(args.source,args.source_runtime);market=market_for(args.source,spec)
    receipt=verify_collection(args.preservation);pc=read(args.preservation/'CONFIG.json')
    if (pc['source']!=spec['identity'] or pc['identity']!=receipt['identity']
            or support_hash(market.inputs)!=read(args.source/'PREFLIGHT.json')['support_sha256']
            or set(market.recipe['split']['periods'])!={'TRAIN','VALIDATION'}):
        raise ValueError('ANCHORED_SOURCE_IDENTITY')
    bundle=args.source/'runs/seed_7/bundles/epoch_0000'
    hashes={'parent':file_sha(bundle/'model.pt')}
    if hashes['parent']!=pc['source_sha256']:raise ValueError('ANCHORED_PARENT_WEIGHT')
    for replica in ('r19','r43'):
        r=verify_collection(args.ranking/replica/'fit')
        if r['identity']!=pc['parents'][replica]:raise ValueError('ANCHORED_CRITIC_PARENT')
        hashes[replica]=file_sha(args.preservation/replica/'kl_parent/ACTOR_0256.pt')
    config=dict(source=spec['identity'],preservation=receipt['identity'],hashes=hashes,
        replicas=['r19','r43'],periods=['TRAIN','VALIDATION'],endpoint=256,
        sources={n:file_sha(ROOT/n) for n in ('tools/attribute_anchored_actions.py',
            'app/anchored_action_attribution.py','tools/attribute_ranking_improvement.py',
            'app/decision_attribution.py','app/three_signal_learning.py','app/frozen_signal_masks.py')})
    config['identity']=digest(config);ensure_config(args.output,config)
    if (args.output/'DONE.json').exists():verify_collection(args.output);return
    reference,_=load_bundle(bundle,'cuda');parent,_=emit(reference,market.inputs);del reference
    risk=read(args.source/'RISK.json');accounts=[];events=[]
    for replica in config['replicas']:
        model,_=load_bundle(bundle,'cuda');path=args.preservation/replica/'kl_parent/ACTOR_0256.pt';model.load(path)
        calibration=torch.load(args.ranking/replica/'fit/paired_greedy/CALIBRATED.pt',map_location='cpu',weights_only=False)['loss']
        current=model.loss.state_dict()
        names=[n for n in current if n.startswith('qvalue_network_params.')]
        if not names or any(not torch.equal(current[n].detach().cpu(),calibration[n]) for n in names):
            raise ValueError('ANCHORED_CRITIC_MUTATED')
        candidate,twins=emit(model,market.inputs,twins=True)
        alpha=float(model.loss.log_alpha.detach().exp());del model,calibration,current
        np.savez(args.output/f'{replica}_SURFACES.npz',timestamp=market.inputs.row_ts_ns,
            valid=market.inputs.valid,parent=parent,candidate=candidate,twins=twins,alpha=alpha)
        for period in config['periods']:
            baseline=args.preservation/replica/'reference'/period
            changed=args.preservation/replica/'kl_parent/actor_0256'/period
            for logits,folder in ((parent,baseline),(candidate,changed)):
                check_trace(market.inputs,logits,read(folder/'ACTIONS.json'))
            interval=market.recipe['split']['periods'][period]
            def make():return environment(market,risk,spec['design']['horizon'],interval,spec['effective']['gamma'])
            def progress(**kw):write(args.output/'PROGRESS.json',dict(replica=replica,period=period,at=now(),seconds=time.monotonic()-started,**kw))
            result=attribute_pair(make,market.inputs.row_ts_ns,baseline,changed,progress)
            write(args.output/f'{replica}_{period}_ATTRIBUTION.json',result)
            details=[]
            for e in result['events']:
                row=int(np.searchsorted(market.inputs.row_ts_ns,e['timestamp']))
                details.append(dict(replica=replica,period=period,utc=e['utc'],timestamp=e['timestamp'],
                    **event_metrics(e,parent[row],candidate[row],twins[:,row],alpha)))
            summary=summarize_events(details)
            if abs(summary['cash_delta_usd']-result['delta_usd'])>1e-6:raise ValueError('ANCHORED_SUMMARY_CASH')
            accounts.append(dict(replica=replica,period=period,**summary,reconciliation=result['reconciliation']))
            events.extend(details)
    if hashes['parent']!=file_sha(bundle/'model.pt') or any(hashes[s]!=file_sha(args.preservation/s/'kl_parent/ACTOR_0256.pt') for s in config['replicas']):
        raise ValueError('ANCHORED_SOURCE_MUTATION')
    export(args.output,accounts,events)
    write(args.output/'RESULT.json',dict(identity=config['identity'],accounts=accounts,
        optimizer_steps=0,model_promoted=False,source_weights_unchanged=True))
    seal(args.output,config['identity'],started)
    write(args.output/'PROGRESS.json',dict(stage='COMPLETE',at=now(),seconds=time.monotonic()-started))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source','source-runtime','preservation','ranking','output'):p.add_argument('--'+n,type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(args)
        except Exception as exc:write(args.output/'FAILED.json',dict(at=now(),error=repr(exc)));raise


if __name__=='__main__':main()
