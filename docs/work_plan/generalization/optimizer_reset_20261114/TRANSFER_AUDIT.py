"""Read-only TRAIN-panel transfer audit; never selects an actor or checkpoint."""
import argparse
from pathlib import Path
import numpy as np
import torch
from app.frozen_recipe_inputs import file_sha
from app.native_fit_coverage import validate_record
from app.role_routed_critic import RoleHeads
from app.robust_adamw_direction import numpy_heads
from app.rl_subset import digest
from tools.run_bayesian_head_contrast import seal,verify_done
from tools.run_discrete_curve_probe import configure,read,write
from tools.run_training_forensics import template


def main(a):
    configure(318);torch.backends.cudnn.allow_tf32=False
    if torch.cuda.get_device_name()!=a.expected_device_name:raise ValueError('TRANSFER_DEVICE')
    parents={name:read(getattr(a,name)/'CONFIG.json') for name in ('probe','coverage','inherited','fresh')}
    for name,cfg in parents.items():verify_done(getattr(a,name),cfg['identity'])
    if (parents['fresh']['inherited_loss_control']!=parents['inherited']['identity']
            or parents['fresh']['probe']!=parents['probe']['identity']
            or parents['fresh']['coverage']!=parents['coverage']['identity']):raise ValueError('TRANSFER_LINEAGE')
    cfg=dict(schema='decision_cost_transfer_audit.v1',parents={k:v['identity'] for k,v in parents.items()},
        script_sha256=file_sha(Path(__file__)),scope='TRAIN-only stop/audit labels; pretrained encoder saw whole TRAIN',
        selection=False,optimizer_updates=0,validation_read=False,test_read=False)
    cfg['identity']=digest(cfg)
    if (a.output/'CONFIG.json').exists() and read(a.output/'CONFIG.json')!=cfg:raise ValueError('TRANSFER_RESUME_IDENTITY')
    if (a.output/'DONE.json').exists():verify_done(a.output,cfg['identity']);return
    write(a.output/'CONFIG.json',cfg)
    support=read(a.coverage/'prepared/SUPPORT.json');records=read(a.coverage/'PANEL.json')['records']
    y=np.stack([validate_record(r) for r in records])
    z=torch.load(a.coverage/'prepared/DATA.pt',weights_only=True,map_location='cpu')['z']
    zg=z.to('cuda');zn=z.double().numpy();out=[];masks=[]
    for fold in ('A','B'):
        info=support['folds'][fold]
        for seed in (19,43):
            initial=torch.load(a.probe/'units'/f'{fold}_{seed}_8192_0'/'PROPOSAL.pt',map_location='cpu',weights_only=False)['initial']
            variants=[('start',0,initial)]
            for name,root in (('inherited',a.inherited),('fresh',a.fresh)):
                for loss in ('mse','cost','hybrid'):
                    unit=root/'units'/f'{fold}_{seed}_{loss}'
                    for path in sorted(unit.glob('ENDPOINT_*.pt')):
                        snap=torch.load(path,map_location='cpu',weights_only=False)
                        if snap['step'] in (256,1024):variants.append((f'{name}_{loss}',snap['step'],snap['state']))
            head=RoleHeads(template('cuda'),1,seed);cpu=RoleHeads(template('cpu'),1,seed)
            def views(state,ix):
                head.load_state_dict(state);cpu.load_state_dict(state)
                with torch.no_grad():
                    g=head(zg[:,ix]).mean(2).cpu().numpy();c=cpu(z[:,ix]).mean(2).numpy()
                    gs=np.concatenate([head(zg[:,i:i+1]).mean(2).cpu().numpy() for i in ix],axis=1)
                    cs=np.concatenate([cpu(z[:,i:i+1]).mean(2).numpy() for i in ix],axis=1)
                return [g,c,gs,cs,numpy_heads(state,zn[:,ix])]
            for panel in ('stop','audit'):
                ix=np.array(info['masks'][panel])
                if not len(ix) or set(ix)&set(info['expanded']):raise ValueError('TRANSFER_FIT_OVERLAP')
                base=views(initial,ix);base_actions=np.stack([q.min(0).argmax(1) for q in base])
                stable=(base_actions==base_actions[0]).all(0)
                if not stable.any():raise ValueError('TRANSFER_NO_STABLE_ROWS')
                masks.append(dict(fold=fold,seed=seed,panel=panel,total=len(ix),stable=int(stable.sum()),indices=ix[stable].tolist()))
                truth=y[ix];costs=truth.max(1,keepdims=True)-truth;before=costs[np.arange(len(ix)),base_actions[0]]
                for name,step,state in variants:
                    qviews=base if name=='start' else views(state,ix)
                    actions=np.stack([q.min(0).argmax(1) for q in qviews]);regrets=[]
                    for q,act in zip(qviews,actions):
                        r=costs[np.arange(len(ix)),act]
                        regrets.append(dict(mse=float(np.mean((q-truth[None])**2)),regret=float(r.mean()),stable_regret=float(r[stable].mean())))
                    after=costs[np.arange(len(ix)),actions[0]];delta=after-before
                    kinds={k:np.array([r['kind']==k for r in np.array(records,dtype=object)[ix]])&stable for k in sorted({records[i]['kind'] for i in ix})}
                    out.append(dict(fold=fold,seed=seed,panel=panel,variant=name,step=step,rows=len(ix),stable_rows=int(stable.sum()),
                        metrics=dict(zip(('gpu','cpu','gpu_single','cpu_single','numpy'),regrets)),
                        stable_geometry_disagreements=int(((actions!=actions[0]).any(0)&stable).sum()),
                        helpful=int(((delta< -1e-8)&stable).sum()),harmful=int(((delta>1e-8)&stable).sum()),
                        by_kind={k:dict(rows=int(m.sum()),helpful=int(((delta< -1e-8)&m).sum()),harmful=int(((delta>1e-8)&m).sum())) for k,m in kinds.items()}))
    for name,cfg_parent in parents.items():verify_done(getattr(a,name),cfg_parent['identity'])
    result=dict(identity=cfg['identity'],scopes=out,starting_masks=masks,model_promoted=False,optimizer_updates=0)
    write(a.output/'RESULT.json',result);seal(a.output,cfg['identity'])
    print(dict(scopes=len(out),masks=len(masks),disagreements=sum(r['stable_geometry_disagreements'] for r in out)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('probe','coverage','inherited','fresh','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--expected-device-name',required=True)
    main(p.parse_args())
