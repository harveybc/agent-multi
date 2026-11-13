"""Four-arm frozen-actor coverage calibration diagnostic; no STL or promotion."""
import argparse
import copy
import fcntl
from pathlib import Path
import time
import numpy as np
import torch

from app.critic_coverage_split import split_new_panel
from app.critic_fit_probe import fit_metrics
from app.frozen_recipe_inputs import file_sha
from app.operational_opening_critic import emit_actor_reference
from app.rl_subset import digest,support_hash
from app.selected_critic_calibration import calibrate
from app.selected_input_market import market_for
from app.structural_refit import load_bundle,verify_files
from tools.run_bayesian_head_contrast import seal,verify_done
from tools.run_discrete_curve_probe import configure,read,write,now


def run(a):
    configure(314);torch.backends.cudnn.allow_tf32=False;started=time.monotonic()
    if not torch.cuda.is_available() or torch.cuda.get_device_name()!=a.expected_device_name:raise ValueError('COVERAGE_DEVICE')
    parent=read(a.parent/'SPEC.json');verify_files(a.parent,parent['files'])
    cc=read(a.calibration/'CONFIG.json');verify_done(a.calibration,cc['identity'])
    dc=read(a.diagnostic/'CONFIG.json');verify_done(a.diagnostic,dc['identity'])
    source,_=load_bundle(a.bundle,'cuda');sha=file_sha(a.bundle/'model.pt')
    if sha!=cc['source_sha256'] or sha!=dc['source_sha256'] or dc['calibration']!=cc['identity']:raise ValueError('COVERAGE_LINEAGE')
    market=market_for(a.parent,parent,source.recipe)
    if source.recipe['list']['n']!=71 or support_hash(market.inputs)!=cc['support_sha256']:raise ValueError('COVERAGE_SOURCE')
    old=read(a.calibration/'panel/RECORDS.json');anchors=read(a.diagnostic/'design/PANEL.json')['anchors']
    nf,held=split_new_panel(anchors,[r['timestamp'] for r in old]);new=[]
    for i,anchor in enumerate(anchors):
        folder=a.diagnostic/'anchors'/str(anchor['row']);verify_done(folder,dc['identity']);result=read(folder/'RESULT.json')
        if result['anchor']!=anchor or not result['parent_state_unchanged']:raise ValueError('COVERAGE_ANCHOR')
        outcomes=result['outcomes']['source']
        if any(not o['parent_unchanged'] or abs(o['soft_return']-o['reward_return'])>1e-8 for o in outcomes):raise ValueError('COVERAGE_TARGET')
        new.append(dict(anchor,outcomes=outcomes,role='fit' if i in nf else 'monitor'))
    records=old+new;rows=np.array([r['row'] for r in records]);trainrows=np.flatnonzero(market.inputs.valid&(market.inputs.period=='TRAIN'))
    for r in records:
        if not market.inputs.valid[r['row']] or market.inputs.period[r['row']]!='TRAIN' or int(market.inputs.row_ts_ns[r['row']])!=r['timestamp']:
            raise ValueError('COVERAGE_ROW')
        if r['role']=='fit' and r['year']>=2023:raise ValueError('COVERAGE_FIT_YEAR')
    cfg=dict(schema='critic_coverage_contrast.v1',source_sha256=sha,calibration=cc['identity'],diagnostic=dc['identity'],
        support=cc['support_sha256'],seeds=[19,43],updates=4096,arms=['old113','expanded146'],new_fit=nf,held=held,
        records_sha256=digest(records),historical_mixed_cot=True,model_promoted=False,validation_read=False,
        code={n:file_sha(Path(__file__).resolve().parents[1]/n) for n in (
            'tools/run_critic_coverage_contrast.py','app/critic_coverage_split.py','app/selected_critic_calibration.py')})
    cfg['identity']=digest(cfg)
    if (a.output/'CONFIG.json').exists() and read(a.output/'CONFIG.json')!=cfg:raise ValueError('COVERAGE_RESUME')
    write(a.output/'CONFIG.json',cfg);write(a.output/'PARTITION.json',dict(records=records,new_fit=nf,held=held))
    if (a.output/'DONE.json').exists():verify_done(a.output,cfg['identity']);return
    flags={n:p.requires_grad for n,p in source.loss.named_parameters()}
    baseline=emit_actor_reference(source,market.inputs,trainrows,flags)[0][trainrows]
    original={k:v.detach().cpu().clone() for k,v in source.loss.state_dict().items()}
    x=torch.tensor(market.inputs.windows[rows],device='cuda');targets=np.array([[o['reward_return'] for o in r['outcomes']] for r in records])
    masks={'old_fit':np.array([i for i,r in enumerate(old) if r['role']=='fit']),
        'old_monitor':np.array([i for i,r in enumerate(old) if r['role']=='monitor']),
        'new_fit':len(old)+np.array(nf),'held':len(old)+np.array(held)}
    def measures(q):
        result={key:fit_metrics(q[:,ix],targets[ix]) for key,ix in masks.items()}
        for kind in sorted({r['kind'] for r in new}):
            ix=np.array([len(old)+i for i in held if new[i]['kind']==kind])
            if len(ix):result['held_'+kind]=fit_metrics(q[:,ix],targets[ix])
        for year in range(2020,2024):
            ix=np.array([len(old)+i for i in held if new[i]['year']==year])
            if len(ix):result['held_'+str(year)]=fit_metrics(q[:,ix],targets[ix])
        return result
    results=[]
    for seed in cfg['seeds']:
        for arm in cfg['arms']:
            out=a.output/f'{arm}_{seed}'
            if (out/'DONE.json').exists():verify_done(out,cfg['identity']);results.append(read(out/'RESULT.json'));continue
            out.mkdir(parents=True,exist_ok=True);model,_=load_bundle(a.bundle,'cuda')
            rr=copy.deepcopy(records)
            if arm=='old113':
                for r in rr[len(old):]:r['role']='monitor'
            curve=[]
            def progress(point):
                with torch.no_grad():q=model.q_surfaces(x).cpu().numpy()
                curve.append(dict(updates=point['updates'],metrics=measures(q)))
                write(out/'CURVE.json',curve)
                write(a.output/'PROGRESS.json',dict(stage='CALIBRATION',arm=arm,seed=seed,updates=point['updates'],
                    arms_complete=len(results),arms_total=4,at=now(),seconds=time.monotonic()-started))
            receipt=calibrate(model,x,rr,seed,progress)
            after=emit_actor_reference(model,market.inputs,trainrows,flags)[0][trainrows]
            if not np.array_equal(after,baseline):raise ValueError('COVERAGE_ACTOR_CHANGED')
            changed=[k for k,v in model.loss.state_dict().items() if not torch.equal(v.detach().cpu(),original[k])]
            if any(not (k.startswith(('qvalue_network_params.','target_qvalue_network_params.')) and '.head.' in k) for k in changed):
                raise ValueError('COVERAGE_FROZEN_CHANGED')
            for name,p in model.loss.named_parameters():p.requires_grad_(flags[name])
            model.save(out/'MODEL.pt')
            cold,_=load_bundle(a.bundle,'cuda');cold.load(out/'MODEL.pt')
            if not np.array_equal(emit_actor_reference(cold,market.inputs,trainrows,flags)[0][trainrows],baseline):raise ValueError('COVERAGE_COLD_ACTOR')
            with torch.no_grad():
                if not torch.equal(cold.q_surfaces(x),model.q_surfaces(x)):raise ValueError('COVERAGE_COLD_Q')
            result=dict(identity=cfg['identity'],arm=arm,seed=seed,fit_count=len(receipt['fit_rows']),
                held_count=len(held),metrics=curve[-1]['metrics'],changed_tensors=changed,actor_exact=True,cold_exact=True,
                model_sha256=file_sha(out/'MODEL.pt'),seconds=time.monotonic()-started)
            write(out/'CALIBRATION.json',receipt);write(out/'RESULT.json',result);seal(out,cfg['identity']);results.append(result)
            del model,cold
    if file_sha(a.bundle/'model.pt')!=sha:raise ValueError('COVERAGE_SOURCE_MUTATED')
    write(a.output/'RESULT.json',dict(identity=cfg['identity'],arms=results,seconds=time.monotonic()-started,
        actor_changed=False,model_promoted=False,stl_included=False,validation_read=False,test_read=False,
        historical_mixed_cot=True,scope='out-of-fit diagnostic on previously inspected correlated TRAIN panel'))
    seal(a.output,cfg['identity']);write(a.output/'PROGRESS.json',dict(stage='COMPLETE',at=now(),seconds=time.monotonic()-started))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','bundle','calibration','diagnostic','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--expected-device-name',required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(a)
        except Exception as exc:write(a.output/'FAILED.json',dict(error=repr(exc),at=now()));raise
