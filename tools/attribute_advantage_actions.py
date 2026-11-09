"""Frozen paired action-gap versus absolute-control account and Q attribution."""
import argparse
import fcntl
from pathlib import Path
import time

import numpy as np
import torch

from app.anchored_action_attribution import paired_event_metrics, summarize_events
from app.frozen_recipe_inputs import file_sha
from app.frozen_signal_masks import prepared_signals
from app.rl_subset import digest, support_hash
from app.structural_refit import load_bundle
from app.three_signal_learning import emit, environment
from tools.attribute_anchored_actions import check_trace, export
from tools.attribute_ranking_improvement import attribute_pair
from tools.audit_fixed_actor_critics import verify_collection
from tools.run_action_ranking_contrast import seal, ensure_config
from tools.run_discrete_curve_probe import read, write, configure, now
from tools.run_structural_sac_refit import authenticate, market_for

ROOT=Path(__file__).resolve().parents[1]
EXTRA=('baseline_q_gap','calibration_reward_gap','isolated_position',
       'baseline_calibration_conflict','candidate_calibration_conflict',
       'baseline_gap_error','candidate_gap_error','actor_change_against_candidate_q')


def run(args):
    started=time.monotonic();configure(7);torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available() or torch.cuda.get_device_name()!='NVIDIA GeForce RTX 5090':
        raise ValueError('PAIRED_ATTRIBUTION_GPU')
    spec=authenticate(args.source,args.source_runtime);market=market_for(args.source,spec)
    receipt=verify_collection(args.contrast);pc=read(args.contrast/'CONFIG.json')
    if (pc['source']!=spec['identity'] or pc['identity']!=receipt['identity']
            or pc['arms']!=['absolute','advantage'] or pc['actor_updates']!=256
            or support_hash(market.inputs)!=read(args.source/'PREFLIGHT.json')['support_sha256']
            or set(market.recipe['split']['periods'])!={'TRAIN','VALIDATION'}):
        raise ValueError('PAIRED_ATTRIBUTION_SOURCE')
    bundle=args.source/'runs/seed_7/bundles/epoch_0000'
    weights={'parent':bundle/'model.pt'}
    for replica in ('r19','r43'):
        for arm in pc['arms']:
            lane=args.contrast/replica/arm
            if verify_collection(lane)['identity']!=receipt['identity']:raise ValueError('PAIRED_ARM_IDENTITY')
            weights[f'{replica}_{arm}']=lane/'ACTOR_0256.pt'
    hashes={k:file_sha(p) for k,p in weights.items()}
    if hashes['parent']!=pc['source_sha256']:raise ValueError('PAIRED_PARENT_WEIGHT')
    config=dict(source=spec['identity'],contrast=receipt['identity'],hashes=hashes,
        replicas=['r19','r43'],periods=['TRAIN','VALIDATION'],endpoint=256,
        calibration_continuation='unchanged parent70 greedy',
        account_continuation='same-replica absolute256 greedy',
        sources={n:file_sha(ROOT/n) for n in ('tools/attribute_advantage_actions.py',
            'app/anchored_action_attribution.py','tools/attribute_anchored_actions.py',
            'tools/attribute_ranking_improvement.py','app/decision_attribution.py',
            'app/three_signal_learning.py','app/frozen_signal_masks.py')})
    config['identity']=digest(config);ensure_config(args.output,config)
    if (args.output/'DONE.json').exists():verify_collection(args.output);return
    reference,_=load_bundle(bundle,'cuda');parent,_=emit(reference,market.inputs);del reference
    calibration_signals=prepared_signals(parent)[0]
    risk=read(args.source/'RISK.json');accounts=[];events=[]
    for replica in config['replicas']:
        surfaces={};alphas={}
        for arm in pc['arms']:
            model,_=load_bundle(bundle,'cuda');model.load(weights[f'{replica}_{arm}'])
            calibrated=torch.load(args.contrast/replica/arm/'CALIBRATED.pt',map_location='cpu',weights_only=False)['loss']
            current=model.loss.state_dict();names=[n for n in current if n.startswith('qvalue_network_params.')]
            if not names or any(not torch.equal(current[n].detach().cpu(),calibrated[n]) for n in names):
                raise ValueError('PAIRED_CRITIC_MUTATED')
            surfaces[arm]=emit(model,market.inputs,twins=True)
            alphas[arm]=float(model.loss.log_alpha.detach().exp());del model,calibrated,current
        if alphas['absolute']!=alphas['advantage']:raise ValueError('PAIRED_ALPHA')
        old,oldq=surfaces['absolute'];new,newq=surfaces['advantage'];alpha=alphas['absolute']
        np.savez(args.output/f'{replica}_SURFACES.npz',timestamp=market.inputs.row_ts_ns,
            valid=market.inputs.valid,parent=parent,absolute=old,advantage=new,
            absolute_twins=oldq,advantage_twins=newq,alpha=alpha)
        for period in config['periods']:
            baseline=args.contrast/replica/'absolute/actor_0256'/period
            changed=args.contrast/replica/'advantage/actor_0256'/period
            for logits,folder in ((old,baseline),(new,changed),(parent,args.contrast/'reference'/period)):
                check_trace(market.inputs,logits,read(folder/'ACTIONS.json'))
            interval=market.recipe['split']['periods'][period]
            def make():return environment(market,risk,spec['design']['horizon'],interval,spec['effective']['gamma'])
            def progress(**kw):write(args.output/'PROGRESS.json',dict(replica=replica,period=period,
                at=now(),seconds=time.monotonic()-started,**kw))
            result=attribute_pair(make,market.inputs.row_ts_ns,baseline,changed,progress,
                calibration_signals=calibration_signals)
            write(args.output/f'{replica}_{period}_ATTRIBUTION.json',result)
            details=[]
            for e in result['events']:
                row=int(np.searchsorted(market.inputs.row_ts_ns,e['timestamp']))
                details.append(dict(replica=replica,period=period,utc=e['utc'],timestamp=e['timestamp'],
                    **paired_event_metrics(e,old[row],new[row],oldq[:,row],newq[:,row],alpha)))
            summary=summarize_events(details)
            if abs(summary['cash_delta_usd']-result['delta_usd'])>1e-6:raise ValueError('PAIRED_SUMMARY_CASH')
            active=[e for e in details if e['outcome']!='neutral']
            diagnostics={k:sum(e[k] for e in active) for k in
                ('baseline_calibration_conflict','candidate_calibration_conflict','actor_change_against_candidate_q')}
            accounts.append(dict(replica=replica,period=period,**summary,
                paired_critic_diagnostics=diagnostics,reconciliation=result['reconciliation']))
            events.extend(details)
    if any(hashes[k]!=file_sha(p) for k,p in weights.items()):raise ValueError('PAIRED_WEIGHT_MUTATION')
    export(args.output,accounts,events,extra_keys=EXTRA,title='Paired action-advantage attribution')
    with (args.output/'REPORT.md').open('a') as f:
        f.write('\nHere parent/candidate actor fields mean absolute/advantage. The additional calibration_reward_gap\n'
                'uses unchanged parent70 continuation from the same isolated control state, not the absolute suffix.\n'
                'Online critic errors are measured at these states; they can differ from calibration anchor states.\n')
    write(args.output/'RESULT.json',dict(identity=config['identity'],accounts=accounts,
        suffix_simulations=6*len(events),optimizer_steps=0,model_promoted=False,source_weights_unchanged=True))
    seal(args.output,config['identity'],started)
    write(args.output/'PROGRESS.json',dict(stage='COMPLETE',at=now(),seconds=time.monotonic()-started))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for n in ('source','source-runtime','contrast','output'):parser.add_argument('--'+n,type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(args)
        except Exception as exc:write(args.output/'FAILED.json',dict(at=now(),error=repr(exc)));raise


if __name__=='__main__':main()
