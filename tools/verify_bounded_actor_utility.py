"""Independent CPU native-account verification of the completed350 actor trial."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from app.bounded_actor_utility import qualify,winner
from app.bounded_functional_actor import BUDGET
from app.frozen_recipe_inputs import file_sha
from app.frozen_signal_masks import prepared_signals
from app.operational_opening_critic import emit_actor_reference
from app.rl_subset import support_hash
from app.selected_input_market import market_for
from app.structural_refit import load_bundle,verify_files
from app.three_signal_learning import environment,evaluate
from app.training_forensics import tree_digest
from tools.run_bayesian_head_contrast import seal,verify_done
from tools.run_discrete_curve_probe import read,write
from tools.run_refit_mask_pass import account_equal


def run(a):
    started=time.monotonic();torch.set_num_threads(1);torch.backends.cudnn.allow_tf32=False
    cfg=read(a.run/'CONFIG.json');verify_done(a.run,cfg['identity']);result=read(a.run/'RESULT.json')
    if (a.output/'DONE.json').exists():verify_done(a.output,cfg['identity']);return
    if file_sha(a.bundle/'model.pt')!=cfg['source_actor_sha256']:raise ValueError('UTILITY_VERIFY_SOURCE')
    spec=read(a.parent/'SPEC.json');verify_files(a.parent,spec['files'])
    model,_=load_bundle(a.bundle,'cpu');market=market_for(a.parent,spec,model.recipe);risk=read(a.parent/'RISK.json')
    if support_hash(market.inputs)!=cfg['support_sha256']:raise ValueError('UTILITY_VERIFY_SUPPORT')
    source_state={k:v.detach().clone() for k,v in model.loss.state_dict().items()}
    def measure(m,out):
        flags={n:p.requires_grad for n,p in m.loss.named_parameters()};z=emit_actor_reference(m,market.inputs,None,flags)[0]
        prep=prepared_signals(z)
        return {p:evaluate(environment(market,risk,spec['design']['horizon'],interval,m.effective['gamma']),z,
            prepared=prep,save=out/p,diagnose=True) for p,interval in m.recipe['split']['periods'].items()}
    reference=measure(model,a.output/'reference')
    for p in reference:account_equal(reference[p],result['reference'][p])
    checks=[];decisions=[]
    for u in result['units']:
        root=a.run/f"seed_{u['seed']}";verify_done(root,cfg['identity'])
        recovered=torch.load(root/'RECOVERY.pt',map_location='cpu',weights_only=False)
        if recovered['identity']!=cfg['identity'] or recovered['trace']!=u['trace']:raise ValueError('UTILITY_VERIFY_RECOVERY')
        count=0
        for attempt,r in enumerate(u['trace'],1):
            count+=int(r['accepted'])
            if r['attempt']!=attempt or r['accepted_total']!=count or attempt>64 or count>8:raise ValueError('UTILITY_VERIFY_BUDGET')
            if r['accepted']:
                if r['loss_after']>r['loss_before']+1e-10 or r['metrics']['step_margin_rms']<=1e-12:raise ValueError('UTILITY_VERIFY_OBJECTIVE')
                if any(r['metrics'][k]>BUDGET[k]+1e-12 for k in ('step_margin_rms','mean_kl','max_kl','relative_parameter_drift')):raise ValueError('UTILITY_VERIFY_TRUST')
        if count!=u['accepted'] or count!=recovered['accepted'] or len(u['trace'])!=u['attempts']:raise ValueError('UTILITY_VERIFY_COUNTS')
        states=recovered['model']['optimizer']['state'].values()
        if len(recovered['model']['optimizer']['state'])!=(4 if count else 0) or any(int(v['step'])!=count for v in states):raise ValueError('UTILITY_VERIFY_OPTIMIZER')
        before=torch.load(root/'FIRST_BEFORE.pt',map_location='cpu',weights_only=False)
        primary=None
        for p in u['points']:
            m,_=load_bundle(a.bundle,'cpu');m.load(root/f"MODEL_{p['accepted']}.pt")
            cp=torch.load(root/f"ACCEPTED_{p['accepted']}.pt",map_location='cpu',weights_only=False)
            if tree_digest(m.loss.state_dict())!=tree_digest(cp['model']['loss']):raise ValueError('UTILITY_VERIFY_MODEL')
            for k,v in m.loss.state_dict().items():
                actor=k.startswith('actor_') and '.head.' in k
                if not actor and not torch.equal(v,before['loss'][k]):raise ValueError('UTILITY_VERIFY_FROZEN')
                if not actor and not (k.startswith('qvalue_network_params.') and '.head.' in k) and not torch.equal(v,source_state[k]):raise ValueError('UTILITY_VERIFY_IMPORT')
            metrics=measure(m,a.output/f"seed_{u['seed']}_point_{p['accepted']}")
            for period in metrics:account_equal(metrics[period],p['metrics'][period])
            if p['accepted']==8:primary=metrics
            checks.append(dict(seed=u['seed'],accepted=p['accepted'],account_exact=True,frozen_exact=True,
                model_sha256=file_sha(root/f"MODEL_{p['accepted']}.pt")))
        if u['primary_endpoint_available']!=(count==8):raise ValueError('UTILITY_VERIFY_PRIMARY')
        if count==8:
            if primary is None:raise ValueError('UTILITY_VERIFY_MISSING_PRIMARY')
            for period in primary:account_equal(u['metrics'][period],primary[period])
        expected=qualify(u['metrics'],reference,u['accepted'])
        if expected!=u['qualification']:raise ValueError('UTILITY_VERIFY_VERDICT')
        decisions.append(dict(seed=u['seed'],**expected))
    if winner(result['units'])!=result['selected_seed']:raise ValueError('UTILITY_VERIFY_SELECTION')
    verify_done(a.run,cfg['identity'])
    if file_sha(a.bundle/'model.pt')!=cfg['source_actor_sha256']:raise ValueError('UTILITY_VERIFY_SOURCE_CHANGED')
    write(a.output/'CONFIG.json',cfg)
    write(a.output/'VERIFICATION.json',dict(identity=cfg['identity'],seconds=time.monotonic()-started,
        cpu_full_accounts=2+2*len(checks),checkpoints=checks,decisions=decisions,source_unchanged=True,
        selected_seed=result['selected_seed'],terminal=result['terminal'],test_read=False,
        verifier_sha256=file_sha(Path(__file__))))
    seal(a.output,cfg['identity'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('run','parent','bundle','output'):p.add_argument('--'+k,type=Path,required=True)
    run(p.parse_args())
