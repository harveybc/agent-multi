"""Read authenticated coupling evidence; distinguish margins from probabilities."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

from app.actor_gradient_coupling import probability_derivative
from app.frozen_recipe_inputs import file_sha
from app.rl_subset import digest
from tools.run_bayesian_head_contrast import seal, verify_done
from tools.run_discrete_curve_probe import read, write


def run(a):
    start=time.monotonic();torch.set_num_threads(1)
    config=read(a.source/'CONFIG.json');verify_done(a.source,config['identity'])
    panel_config=read(a.panel/'CONFIG.json');verify_done(a.panel,panel_config['identity'])
    if (config['source_identity']!=panel_config['identity'] or
            config['source_receipt_sha256']!=file_sha(a.panel/'DONE.json')):
        raise ValueError('COUPLING_REPORT_SOURCE')
    cfg=dict(schema='coupling_probability_report.v1',source_identity=config['identity'],
        source_receipt_sha256=file_sha(a.source/'DONE.json'),
        code_sha256=file_sha(Path(__file__)),helper_sha256=file_sha(
            Path(__file__).resolve().parents[1]/'app/actor_gradient_coupling.py'))
    cfg['identity']=digest(cfg)
    if (a.output/'CONFIG.json').exists() and read(a.output/'CONFIG.json')!=cfg:
        raise ValueError('COUPLING_REPORT_CONFIG')
    write(a.output/'CONFIG.json',cfg)
    if (a.output/'DONE.json').exists():verify_done(a.output,cfg['identity']);return
    records=read(a.panel/'panel/RECORDS.json');summary=read(a.source/'SUMMARY.json')
    data=np.load(a.source/'DECOMPOSITION.npz',allow_pickle=False)
    if not np.array_equal(data['rows'],[r['row'] for r in records]):raise ValueError('COUPLING_REPORT_ROWS')
    y=np.array([[o['reward_return'] for o in r['outcomes']] for r in records]);best=y.argmax(1)
    z=torch.tensor(data['logits']);p=z.softmax(-1)
    out=dict(identity=cfg['identity'],source_identity=config['identity'],arms={},
        meaning='Native-optimal opening among three symbols on the same frozen-policy continuation; diagnostic only')
    for key,arm in summary['arms'].items():
        result={}
        for role in ('fit','monitor'):
            ix=np.array([i for i,r in enumerate(records) if r['role']==role and r['position']==0 and r['allowed']])
            pick=ix[(best[ix]!=0)&((y[ix].max(1)-y[ix,0])>1e-8)]
            control_result={}
            for control in ('recorded_q','native_return_oracle'):
                stages={}
                for term in ('value','entropy','total'):
                    for stage in ('local_logit','self_alone','batch_self','cross','total'):
                        m=torch.tensor(data[f'{key}:{control}:{term}:{stage}'])
                        dp=probability_derivative(p,m).numpy()
                        stages[f'{term}:{stage}']=dict(
                            optimal_open_probability_down=int((dp[pick,best[pick]] < -1e-10).sum()),
                            optimal_open_probability_up=int((dp[pick,best[pick]] > 1e-10).sum()),
                            optimal_open_probability_negligible=int((np.abs(dp[pick,best[pick]]) <= 1e-10).sum()),
                            mean_native_expected_return_derivative=float((dp[ix]*y[ix]).sum(1).mean()))
                contributions={name:float(np.abs(data[f'{key}:{control}:layer:{name}'][pick,best[pick]-1]).sum())
                    for name in arm['controls'][control]['layers']}
                denom=sum(contributions.values())
                control_result[control]=dict(stages=stages,
                    layer_absolute_projection_shares={k:v/denom if denom else None for k,v in contributions.items()})
            dm=torch.tensor(data[f'{key}:adamw:actual'])
            zz=z+torch.cat([torch.zeros_like(z[:,:1]),dm],dim=-1)
            dp=(zz.softmax(-1)-p).numpy()
            adaptive=data[f'{key}:adamw:adaptive'][ix];decay=data[f'{key}:adamw:decay'][ix]
            result[role]=dict(flat_allowed_rows=len(ix),native_optimal_openings=len(pick),
                controls=control_result,
                actual_adam_optimal_probability_down=int((dp[pick,best[pick]] < -1e-10).sum()),
                decay_to_adaptive_projection_l2=float(np.linalg.norm(decay)/np.linalg.norm(adaptive)))
        out['arms'][key]=result
    # Local linear sensitivity rank is not an information-capacity estimate.
    fit=np.array([i for i,r in enumerate(records) if r['role']=='fit'])
    j=data['jacobian'];jm=(j[:,1:]-j[:,:1])[fit].reshape(2*len(fit),-1)
    singular=np.linalg.svd(jm,compute_uv=False)
    out['fit_jacobian']=dict(rows=jm.shape[0],parameters=jm.shape[1],
        singular_values=singular.tolist(),rank_relative_1e_6=int((singular>singular[0]*1e-6).sum()),
        top_squared_energy_fraction=float(singular[0]**2/(singular**2).sum()),
        top5_squared_energy_fraction=float((singular[:5]**2).sum()/(singular**2).sum()),
        caution='Sensitivity spectrum at this checkpoint/panel only; not Shannon capacity or absent market signal')
    out['seconds']=time.monotonic()-start
    write(a.output/'SUMMARY.json',out);seal(a.output,cfg['identity'])
    print(dict(identity=cfg['identity'],arms=len(out['arms']),seconds=out['seconds']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','panel','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);run(a)
