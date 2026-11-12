import argparse
import hashlib
import json
from pathlib import Path
import torch
from app.structural_refit import load_bundle
from app.selected_input_market import market_for
from app.bounded_functional_actor import BUDGET
from tools.diagnose_warmstart_drift import set_arm
from tools.run_discrete_curve_probe import configure

parser=argparse.ArgumentParser()
for name in ('run','source','bundle','parent','output'):parser.add_argument('--'+name,type=Path,required=True)
a=parser.parse_args()
read=lambda p:json.loads(p.read_text())
cfg=read(a.run/'CONFIG.json');result=read(a.run/'RESULT.json');guard=read(a.run/'GUARD.json')
receipts=0
for root in (a.run,a.source,a.bundle.parent):
    done=read(root/'DONE.json')
    for name,sha in done['files'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha
        receipts+=1
configure(292);torch.backends.cudnn.allow_tf32=False
model,_=load_bundle(a.bundle,'cuda');set_arm(model,'actor_only')
market=market_for(a.parent,read(a.parent/'SPEC.json'),model.recipe)
records=read(a.source/'panel/RECORDS.json'); rows=[r['row'] for r in records]
x=torch.tensor(market.inputs.windows[rows],device='cuda')
with torch.no_grad():initial=model.surfaces(x).detach().double()
native=torch.tensor([[o['reward_return'] for o in r['outcomes']] for r in records],device='cuda',dtype=torch.double)
chosen=initial.argmax(-1);order=native.sort(dim=-1,descending=True).values
indices=[i for i,r in enumerate(records) if r['role']=='fit' and chosen[i]==native[i].argmax()
         and order[i,0]-order[i,1]>1e-8]
assert indices==guard['indices']
actions=chosen[indices]; assert actions.cpu().tolist()==guard['actions']
other=torch.tensor([[b for b in range(3) if b!=int(c)] for c in actions],device='cuda')
ix=torch.tensor(indices,device='cuda')
initial_margins=initial[ix,actions,None]-initial[ix[:,None],other]
assert torch.equal(initial_margins,torch.tensor(guard['source_margins'],device='cuda',dtype=torch.double))
# Reconstruct targets independently of the training helper.
labels=[];weights=[]
for i,r in enumerate(records):
    best=int(native[i].argmax());old=int(chosen[i])
    is_error=r['role']=='fit' and best!=old and float(order[i,0]-order[i,1])>1e-8
    labels.append(best if is_error else -1)
    weights.append(float(native[i,best]-native[i,old]) if is_error else 0.)
assert labels==cfg['error_targets'] and weights==cfg['error_weights']
assert not set(indices)&{i for i,v in enumerate(weights) if v>0}
assert all(v==0 for v,r in zip(weights,records) if r['role']!='fit')
audits=[]
for arm in result['arms']:
    key=arm['arm'];baseline,_=load_bundle(a.bundle,'cuda');baseline.load(a.source/f"critic_{arm['seed']}/MODEL.pt")
    fixed={k:v.detach().clone() for k,v in baseline.loss.state_dict().items()
           if not(k.startswith('actor_') and '.head.' in k)}
    base_head={k:v.detach().clone() for k,v in baseline.loss.state_dict().items()
               if k.startswith('actor_') and '.head.' in k}
    for step in arm['steps']:
        assert step['final_guard']['passed'] and step['final_guard']['minimum_margin_change']>=-1e-7
        if step['accepted']:
            assert step['metrics']['step_margin_rms']>=1e-6
            assert step['loss_after']<=step['loss_before']+1e-10
            assert all(step['metrics'][k]<=BUDGET[k]+1e-12 for k in BUDGET if k in step['metrics'])
    for point in (1,8):
        model.load(a.run/key/f'ACTOR_{point:04d}.pt');set_arm(model,'actor_only')
        state=model.loss.state_dict()
        assert all(torch.equal(v,state[k]) for k,v in fixed.items())
        with torch.no_grad():z=model.surfaces(x).detach().double()
        change=z[ix,actions,None]-z[ix[:,None],other]-initial_margins
        assert torch.min(change)>=-1e-7
        assert torch.equal(z[ix].argmax(-1),actions)
        count=sum(s['accepted'] for s in arm['steps'][:point])
        optimizer=model.optimizer.state_dict()['state']
        if count==0:
            assert not optimizer
            assert all(torch.equal(v,state[k]) for k,v in base_head.items())
        else:
            assert len(optimizer)==4 and all(int(v['step'])==count for v in optimizer.values())
        # Check the endpoint's reported probability/decision progress independently.
        for role in ('fit','monitor'):
            ids=[i for i,r in enumerate(records) if r['role']==role and
                 int(chosen[i])!=int(native[i].argmax()) and float(order[i,0]-order[i,1])>1e-8]
            b=native[ids].argmax(-1)
            gain=(z[ids].softmax(-1).gather(1,b[:,None])-initial[ids].softmax(-1).gather(1,b[:,None])).squeeze(1)
            reported=arm['steps'][point-1]['corrective']['progress'][role]
            assert reported['rows']==len(ids)
            assert reported['corrected']==int((z[ids].argmax(-1)==b).sum())
            assert abs(reported['mean_preferred_probability_change']-float(gain.mean()))<1e-12
        actual_margin_min=float(change.min())
        assert abs(actual_margin_min-arm['steps'][point-1]['final_guard']['minimum_margin_change'])<1e-12
        audits.append(dict(arm=key,point=point,accepted=count,
            minimum_margin_change=actual_margin_min,optimizer_step_count_exact=True,frozen_components_exact=True))
output=dict(identity=cfg['identity'],verified_files=receipts,protected_fit_rows=len(indices),target_fit_errors=sum(v>0 for v in weights),target_weight_sum=sum(weights),
            checkpoint_checks=audits,new_source_weights_unchanged=True,test_read=False)
a.output.write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output,indent=2))
