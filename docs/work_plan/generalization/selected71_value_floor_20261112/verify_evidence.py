"""Independent mask, checkpoint, constraint and account audit for experiment303."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from app.frozen_recipe_inputs import file_sha
from app.selected_input_market import market_for
from app.structural_refit import load_bundle
from tools.diagnose_warmstart_drift import set_arm
from tools.run_bayesian_head_contrast import verify_done
from tools.run_discrete_curve_probe import configure


def read(path):return json.loads(path.read_text())


def verify(a):
    configure(292);torch.backends.cudnn.allow_tf32=False
    cfg=read(a.run/'CONFIG.json');result=read(a.run/'RESULT.json')
    report=read(a.report/'SUMMARY.json');export=read(a.bundle.parent/'RESULT.json')
    count=0
    for root in (a.run,a.report,a.calibration,a.diagnosis,a.bundle.parent):
        receipt=read(root/'DONE.json');verify_done(root,receipt['identity']);count+=len(receipt['files'])
    assert cfg['schema']=='source_value_floor_interior_correction.v1'
    assert result['identity']==cfg['identity']==report['source_identity']
    assert file_sha(a.bundle/'model.pt')==cfg['source_sha256']
    assert cfg['source_identity']==read(a.calibration/'CONFIG.json')['identity']
    assert cfg['diagnosis_identity']==read(a.diagnosis/'CONFIG.json')['identity']
    entropy_deltas={}
    for period,actual in result['reference'].items():
        expected=export['metrics'][period]
        assert actual.keys()==expected.keys()
        for key,value in actual.items():
            if key in ('mean_signal_entropy','mean_requested_command_entropy'):
                delta=abs(value-expected[key]);assert delta<1e-6
                entropy_deltas[f'{period}:{key}']=delta
            else:assert value==expected[key],(period,key)
    records=read(a.run/'panel/RECORDS.json')
    assert records==read(a.calibration/'panel/RECORDS.json')
    native=np.array([[o['reward_return'] for o in r['outcomes']] for r in records])
    fit=np.array([r['role']=='fit' for r in records]);rows=np.array([r['row'] for r in records])
    model,_=load_bundle(a.bundle,a.device);set_arm(model,'actor_only')
    market=market_for(a.parent,read(a.parent/'SPEC.json'),model.recipe)
    assert (market.inputs.period[rows]=='TRAIN').all() and market.inputs.valid[rows].all()
    x=torch.tensor(market.inputs.windows[rows],device=a.device)
    with torch.no_grad():z=model.surfaces(x).cpu().numpy().astype(float)
    optimal=native>=native.max(-1,keepdims=True)-1e-8
    chosen=z.argmax(-1);good=optimal[np.arange(len(z)),chosen]
    source_values=native[np.arange(len(z)),chosen]
    allowed=native>=source_values[:,None]-1e-8
    protected=np.flatnonzero(fit&~allowed.all(-1))
    errors=fit&~good;w=np.where(errors,native.max(-1)-native[np.arange(len(z)),chosen],0.)
    guard=cfg['guard']
    assert protected.tolist()==guard['indices'] and chosen[protected].tolist()==guard['actions']
    assert optimal.tolist()==cfg['error_targets'] and w.tolist()==cfg['error_weights']
    assert optimal[protected].tolist()==guard['optimal_actions']
    assert allowed[protected].tolist()==guard['admissible_actions']
    assert np.flatnonzero(fit).tolist()==guard['fit_indices']
    assert source_values[fit].tolist()==guard['fit_source_values']
    cr=[];cp=[];cb=[]
    for row in protected:
        for bad in np.flatnonzero(~allowed[row]):cr.append(row);cp.append(chosen[row]);cb.append(bad)
    cr=np.array(cr);cp=np.array(cp);cb=np.array(cb)
    assert cr.tolist()==guard['constraint_indices'] and cp.tolist()==guard['constraint_preferred']
    assert cb.tolist()==guard['constraint_suboptimal']
    floor=z[cr,cp]-z[cr,cb]
    assert np.array_equal(floor,guard['source_margins'])
    buffer=8*np.finfo(np.float32).eps*(1+abs(z[cr,cp])+abs(z[cr,cb]))
    assert np.array_equal(buffer,cfg['interior_buffer'])
    checks=[];end_states=[];full_losses=[];error_switches=[]
    for arm in result['arms']:
        name=arm['arm'];critic=a.calibration/f"critic_{arm['seed']}/MODEL.pt"
        start=torch.load(critic,map_location='cpu',weights_only=False)
        for point in (1,8):
            checkpoint=a.run/name/f'ACTOR_{point:04d}.pt'
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            changed=[k for k,v in saved['loss'].items() if not torch.equal(v,start['loss'][k])]
            assert all(k.startswith('actor_') and '.head.' in k for k in changed)
            accepted=sum(s['accepted'] for s in arm['steps'][:point])
            optimizer_steps=[int(v['step']) for v in saved['optimizer']['state'].values() if 'step' in v]
            assert len(changed)==(4 if accepted else 0)
            assert len(optimizer_steps)==(4 if accepted else 0) and all(s==accepted for s in optimizer_steps)
            candidate,_=load_bundle(a.bundle,a.device);candidate.load(checkpoint);set_arm(candidate,'actor_only')
            with torch.no_grad():after=candidate.surfaces(x).cpu().numpy().astype(float)
            delta=(after[cr,cp]-after[cr,cb])-floor
            assert delta.min()>=-1e-7
            values_after=native[np.arange(len(z)),after.argmax(-1)]
            assert ((values_after-source_values)[fit]>=-1e-8).all()
            assert optimal[fit&good,after[fit&good].argmax(-1)].all()
            before_lse=np.logaddexp.reduce(z,axis=-1)-np.logaddexp.reduce(np.where(optimal,z,-np.inf),axis=-1)
            after_lse=np.logaddexp.reduce(after,axis=-1)-np.logaddexp.reduce(np.where(optimal,after,-np.inf),axis=-1)
            loss0=float((w*before_lse).sum()/w.sum());loss1=float((w*after_lse).sum()/w.sum())
            checks.append(dict(arm=name,point=point,accepted=accepted,changed_head_tensors=changed,
                frozen_tensors_exact=len(saved['loss'])-len(changed),optimizer_steps_exact=True,
                minimum_margin_change=float(delta.min()),source_value_regressions=0,source_correct_preserved=True,
                full_fit_error_loss_before=loss0,full_fit_error_loss_after=loss1,
                corrected_fit_errors=int((errors&optimal[np.arange(len(z)),after.argmax(-1)]).sum())))
            if point==8:
                end_states.append(saved['loss']);full_losses.append([loss0,loss1])
                for i in np.flatnonzero(chosen!=after.argmax(-1)):
                    before_p=np.exp(z[i]-np.logaddexp.reduce(z[i]))
                    after_p=np.exp(after[i]-np.logaddexp.reduce(after[i]))
                    error_switches.append(dict(arm=name,panel_index=int(i),role=records[i]['role'],
                        timestamp=records[i]['timestamp'],old=int(chosen[i]),new=int(after[i].argmax()),
                        native_returns=native[i].tolist(),source_optimal=bool(good[i]),
                        optimal_mass_before=float(before_p[optimal[i]].sum()),
                        optimal_mass_after=float(after_p[optimal[i]].sum())))
            del candidate
        for step in arm['steps']:
            assert step['final_guard']['passed'] and step['final_guard']['source_value_regressions']==0
            if step['accepted']:
                assert step['projection']['minimum_absolute_slack']>=-1e-10
                assert step['loss_after']<step['loss_before']
    different=sum(not torch.equal(end_states[0][k],end_states[1][k]) for k in end_states[0])
    different_heads=sum(not torch.equal(end_states[0][k],end_states[1][k]) for k in end_states[0]
                        if k.startswith('actor_') and '.head.' in k)
    accounts=[]
    timestamps={r['timestamp'] for r in records}
    for entry in report['accounts']:
        period=entry['period'];endpoint=next(r for r in result['arms'] if r['arm']==entry['arm'])['points'][-1]
        delta=endpoint['metrics'][period]['net_profit_usd']-result['reference'][period]['net_profit_usd']
        assert abs(delta-entry['delta_usd'])<1e-8 and abs(entry['reconciliation']['error_usd'])<1e-8
        events=read(a.report/'attribution'/f"{entry['arm']}_{period}"/'EVENTS.json')
        assert abs(sum(e['contribution_usd'] for e in events)-delta)<1e-8
        assert sum(abs(e['contribution_usd'])>1e-6 for e in events)==len(entry['effective_events'])
        before_root=a.run/'reference'/period
        after_root=a.run/entry['arm']/'actor_0008'/period
        same_trades=read(before_root/'TRADES.json')==read(after_root/'TRADES.json')
        with np.load(before_root/'EQUITY.npz') as left,np.load(after_root/'EQUITY.npz') as right:
            same_equity=left.files==right.files and all(np.array_equal(left[k],right[k]) for k in left.files)
        blocked=[]
        for event in events:
            old=event['prefix_old']['first_effect'];new=event['prefix_new']['first_effect']
            if (event['state']['position'] is None and old['entries']==new['entries']==0
                    and old['closes']==new['closes']==0
                    and (old['count_changes'].get('refused_ENTRY_CUTOFF',0)
                         or new['count_changes'].get('refused_ENTRY_CUTOFF',0))):
                blocked.append(dict(timestamp=event['timestamp'],old=event['old_signal'],new=event['new_signal']))
        accounts.append(dict(arm=entry['arm'],period=period,delta_usd=delta,
            trades_exact=same_trades,equity_exact=same_equity,entry_cutoff_signal_changes=blocked,
            effective_event_count=len(entry['effective_events']),
            effective_events_outside_panel=sum(e['timestamp'] not in timestamps for e in entry['effective_events']),
            reconciliation_error_usd=entry['reconciliation']['error_usd']))
    value=dict(identity=cfg['identity'],verified_files=count,protected_fit_rows=len(protected),
        corrective_fit_rows=int(errors.sum()),constraint_count=len(cr),native_checked_fit_rows=int(fit.sum()),
        new_hold_guards=int(sum(optimal[i].sum()>1 for i in protected)),
        tied_error_targets=int(sum(errors& (optimal.sum(-1)>1))),
        source_unchanged=True,checkpoints=checks,accounts=accounts,
        reference_account_exact=True,reference_cross_runtime_entropy_deltas=entropy_deltas,
        differing_endpoint_tensors_between_seeds=different,
        differing_actor_head_tensors_between_seeds=different_heads,panel_action_switches=error_switches,
        full_fit_error_losses=full_losses,test_read=False,model_promoted=False)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('parent','run','report','bundle','calibration','diagnosis','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--device',default='cpu');verify(p.parse_args())
