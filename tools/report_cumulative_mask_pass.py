"""Independently replay frozen selection decisions and audit retained weights."""
import argparse
import csv
from pathlib import Path

import torch

from app.cumulative_readmission import cumulative_pass
from app.frozen_actor_compaction import effective_actor
from app.frozen_recipe_inputs import file_sha
from app.rl_subset import digest
from app.structural_refit import verify_files
from app.weight_space_diagnostics import tensor_summary
from tools.run_cumulative_mask_pass import authenticate, make_expanded
from tools.run_discrete_curve_probe import read, write


def report(root,output):
    if output.exists(): raise ValueError('NEW_REPORT_OUTPUT_REQUIRED')
    s=authenticate(root);done=read(root/'DONE.json');verify_files(root,done['files'])
    if done['identity']!=s['identity']:raise ValueError('REPORT_IDENTITY')
    result=read(root/'RESULT.json');rows={}
    for path in (root/'cache').glob('*.json'):
        row=read(path)
        if row['sha256']!=digest(row['result']): raise ValueError('REPORT_CACHE')
        rows[tuple(row['columns'])]=row['result']
    replay=cumulative_pass(s['universe'],s['blocks'],s['initial'],lambda cols:rows[tuple(cols)])
    if replay!=result['result']:raise ValueError('REPORT_SELECTOR_REPLAY')
    s=dict(s,device='cpu');torch.set_num_threads(2)
    base,expanded,restoration,_=make_expanded(s)
    old=effective_actor(base).state_dict();new=expanded.state_dict()
    groups=base.recipe['representation']['resolved'];active=base.recipe['list']['columns']
    universe=read(Path(s['parent'])/'SUPPORT_RECIPE.json')
    ug=universe['representation']['resolved'];names=universe['list']['columns']
    group_columns={g:[names[j] for j in ix] for g,ix in zip(ug['names'],ug['indices'])}
    checked=[];index_buffers=[]
    for key,value in old.items():
        if key.startswith('encoder.indices_'):
            i=int(key.rsplit('_',1)[1]);expected=[names.index(c) for c in group_columns[groups['names'][i]]]
            if new[key].tolist()!=expected:raise ValueError('REPORT_GROUP_INDEX')
            index_buffers.append(key);continue
        actual=new[key]
        if key.startswith('encoder.branches.') and key.endswith('.0.conv.weight'):
            i=int(key.split('.')[2]);g=groups['names'][i]
            original=[active[j] for j in groups['indices'][i]]
            actual=actual[:,[group_columns[g].index(c) for c in original],:]
        elif key=='encoder.fusion.conv.weight':
            actual=actual[:,:value.shape[1],:]
        if not torch.equal(value,actual):raise ValueError('REPORT_COMMON_WEIGHT:'+key)
        checked.append(key)
    output.mkdir(parents=True)
    write(output/'WEIGHT_SPACE.json',dict(identity=s['identity'],retained_tensors_exact=True,
        checked_tensor_count=len(checked),checked_tensors=checked,remapped_buffers=index_buffers,
        restored=restoration,head={k:tensor_summary(v) for k,v in old.items() if k.startswith('head.')},
        scope='all current actor tensors exact; archived learned connections only; no optimization or causal importance claim'))
    trace=replay['trace'];deleted=[r for r in trace if r['phase']=='DELETE'];added=[r for r in trace if r['phase']=='REINSERT']
    summary=dict(identity=s['identity'],counts=replay['counts'],evaluations=replay['evaluations'],
        delete_blocks=len(deleted),return_blocks=len(added),historical_return_features=sum(len(r['block']) for r in added),
        accepted=sum(r['accepted'] for r in trace),ties=sum(r['candidate_score']==r['before_score'] for r in trace),
        ineligible=sum(not r['candidate_eligible'] for r in trace),seconds=result['seconds'],
        metrics=result['final']['metrics'],baseline_parity=result['baseline_parity'],
        reload_exact=result['reload_exact'],reload_accounts_exact=result['reload_accounts_exact'],
        source_unchanged=result['source_unchanged'],verified_files=len(done['files']),
        decision_replay_exact=True,mode=result['mode'],test_read=False,optimizer_steps=0,model_promoted=False)
    write(output/'SUMMARY.json',summary)
    write(output/'SELECTION_TRACE.json',replay)
    donor={r['column']:r for r in restoration['records']}
    decisions={c:r for r in trace for c in r['block']}
    with (output/'FEATURE_LEDGER.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['feature','initially_active','selected','phase','block',
            'candidate_score','baseline_score','eligible','accepted','donor','std_ratio'])
        writer.writeheader()
        for c in names:
            row=decisions[c];d=donor.get(c,{})
            writer.writerow(dict(feature=c,initially_active=c in active,selected=c in replay['selected'],
                phase=row['phase'],block=' | '.join(row['block']),candidate_score=row['candidate_score'],
                baseline_score=row['before_score'],eligible=row['candidate_eligible'],accepted=row['accepted'],
                donor=d.get('donor','current69'),std_ratio=d.get('std_ratio',1.)))
    write(output/'MANIFEST.json',dict(identity=s['identity'],files={p.name:file_sha(p) for p in output.iterdir()}))
    print(summary['counts'],len(checked),len(done['files']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report(a.root,a.output)
