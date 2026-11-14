"""Exact saved-block margin attribution after verified347; no model promotion."""
import argparse
from pathlib import Path
import numpy as np
import torch
from app.frozen_recipe_inputs import file_sha
from app.rl_subset import digest
from app.weight_decision_forensics import BLOCKS,hybrid_state,native_forward,exact_shapley
from tools.run_bayesian_head_contrast import seal,verify_done
from tools.run_discrete_curve_probe import read,write


def run(a):
    cfg=read(a.run/'CONFIG.json');vcfg=read(a.verified/'CONFIG.json')
    if cfg!=vcfg:raise ValueError('BLOCK_REPORT_VERIFIED_ID')
    verify_done(a.run,cfg['identity']);verify_done(a.verified,cfg['identity']);verify_done(a.coverage,cfg['parents']['coverage'])
    evidence=read(a.verified/'VERIFICATION.json')
    if evidence['transactions_exact']!=256 or evidence['factorial_cells']<8:raise ValueError('BLOCK_REPORT_GATE')
    rcfg=dict(schema='matched_divergence_blocks.v1',parent=cfg['identity'],code_sha256=file_sha(Path(__file__)),
        helper_sha256=file_sha(Path(__file__).resolve().parents[1]/'app/weight_decision_forensics.py'),blocks=list(BLOCKS))
    rcfg['identity']=digest(rcfg)
    if (a.output/'DONE.json').exists():verify_done(a.output,rcfg['identity']);return
    z=torch.load(a.coverage/'prepared/DATA.pt',weights_only=True,map_location='cpu')['z'].double().numpy();rows=[];max_error=0.
    for case in read(a.run/'RESULT.json')['cases']:
        for entry in case['factorials']:
            path=a.run/'cases'/case['case']/entry['file']
            if file_sha(path)!=entry['sha256']:raise ValueError('BLOCK_REPORT_WITNESS')
            w=torch.load(path,weights_only=False,map_location='cpu');c=w['origins']['C']['before'];t=w['origins']['T']['before']
            row=entry['row'];p=entry['control_action'];other=entry['treatment_action'];coalitions=[]
            for mask in range(16):
                state=hybrid_state(c,t,mask);x=z[:,row:row+1];qn=native_forward(state,x)[0];qt=[]
                for twin in range(2):
                    prefix=f'twins.{twin}.0.'
                    h=torch.nn.functional.silu(torch.nn.functional.linear(torch.from_numpy(x[twin]),state[prefix+'0.weight'].double(),state[prefix+'0.bias'].double()))
                    qt.append(torch.nn.functional.linear(h,state[prefix+'2.weight'].double(),state[prefix+'2.bias'].double()).numpy())
                max_error=max(max_error,float(np.max(abs(np.stack(qt)-qn))))
                if max_error>1e-11:raise ValueError('BLOCK_REPORT_FORWARD')
                q=qn.min(0)[0];coalitions.append(dict(mask=mask,margin=float(q[p]-q[other]),action=int(q.argmax())))
            phi=exact_shapley(np.array([[r['margin']] for r in coalitions]))[:,0]
            delta=coalitions[-1]['margin']-coalitions[0]['margin']
            if not np.isclose(phi.sum(),delta,atol=1e-12,rtol=1e-12):raise ValueError('BLOCK_REPORT_SUM')
            rows.append(dict(case=case['case'],row=row,kind=entry['kind'],offset=entry['control_acquisition'],
                margin_control=coalitions[0]['margin'],margin_treatment=coalitions[-1]['margin'],delta=delta,
                contributions=dict(zip(BLOCKS,map(float,phi))),coalitions=coalitions))
    write(a.output/'CONFIG.json',rcfg);write(a.output/'RESULT.json',dict(identity=rcfg['identity'],rows=rows,
        forward_max_error=max_error,coalitions=len(rows)*16,actor_updates=0,model_promoted=False));seal(a.output,rcfg['identity'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('run','verified','coverage','output'):p.add_argument('--'+k,type=Path,required=True)
    torch.set_num_threads(1);run(p.parse_args())
