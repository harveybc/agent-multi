"""Authenticate TRAIN-only boundary diagnostics and independently check rankings."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from app.rl_subset import digest
from app.selected_refinement_diagnosis import decision_strata, decision_behavior
from app.fit_decision_preservation import protected_actions, margins, preservation
from app.targeted_fit_errors import error_targets


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(root, source, output):
    for directory in (root,source):
        for name,expected in read(directory/'DONE.json')['files'].items():
            p=(directory/name).resolve()
            if not p.is_relative_to(directory.resolve()) or sha(p)!=expected:
                raise ValueError('DIAGNOSTIC_FILE_CHANGED:'+name)
    cfg=read(root/'CONFIG.json');summary=read(root/'SUMMARY.json')
    if (cfg['identity'] != digest({k:v for k,v in cfg.items() if k!='identity'})
            or summary['identity'] != cfg['identity']
            or read(root/'DONE.json')['identity'] != cfg['identity']
            or cfg['source']!=read(source/'CONFIG.json')['identity']):
        raise ValueError('DIAGNOSTIC_SOURCE')
    assert cfg['schema']=='selected_calibrated_boundary_diagnosis.v1'
    assert cfg['posthoc_timestamps']==[] and not cfg['validation_read']
    assert summary['actor_updates']==0 and summary['source_unchanged']
    records=read(source/'panel/RECORDS.json');arrays=np.load(root/'PROJECTIONS.npz')
    np.testing.assert_array_equal(arrays['rows'],[r['row'] for r in records])
    native=np.array([[o['reward_return'] for o in r['outcomes']] for r in records])
    np.testing.assert_allclose(arrays['native_returns'],native,atol=2e-7,rtol=0)
    masks={role:np.array([r['role']==role for r in records]) for role in ('fit','monitor')}
    masks.update({role+':flat_allowed':np.array([r['role']==role and r['position']==0 and r['allowed']
                    for r in records]) for role in ('fit','monitor')})
    checks=0
    for arm in summary['arms']:
        assert all(records[i]['role']=='fit' for i in arm['sampled_fit_indices'])
        q=arrays[f"calibrated_adamw_{arm['seed']}:q"]
        for name,mask in masks.items():
            for key,values in [('critic_quality',q),('actor_quality',arrays['logits'])]:
                chosen=values[mask].argmax(-1); y=arrays['native_returns'][mask]
                regret=y.max(-1)-y[np.arange(len(y)),chosen]
                got=arm[key][name]
                assert got['optimal_count']==int((regret<=1e-8).sum())
                np.testing.assert_allclose(got['mean_regret'],regret.mean(),atol=1e-12,rtol=0)
                checks+=2
        assert max(x['autograd_max_error'] for x in arm['controls'].values())<3e-7
    # Independently reconstruct projected optimal probability mass using numpy.
    p=np.exp(arrays['logits']-arrays['logits'].max(-1,keepdims=True))
    p/=p.sum(-1,keepdims=True)
    best=arrays['native_returns'].max(-1)
    optimal=arrays['native_returns']>=best[:,None]-1e-8
    actor_correct=optimal[np.arange(len(records)),arrays['logits'].argmax(-1)]
    strata={k:v.numpy() for k,v in decision_strata(records).items()}
    projected_checks=0;groups={}
    for group,mask in strata.items():
        chosen=[r for r,m in zip(records,mask) if m]
        groups[group]=dict(rows=int(mask.sum()),actor=decision_behavior(chosen,arrays['logits'][mask]),
            critics={},derivatives={})
    for arm in summary['arms']:
        seed=str(arm['seed']);prefix=f"calibrated_adamw_{seed}"
        q=arrays[prefix+':q']
        for group,mask in strata.items():
            chosen=[r for r,m in zip(records,mask) if m]
            groups[group]['critics'][seed]=decision_behavior(chosen,q[mask])
            groups[group]['derivatives'][seed]={}
        for control,detail in arm['controls'].items():
            for stage in detail['groups']['fit']:
                margin=arrays[f'{prefix}:{control}:{stage}']
                dz=np.column_stack((np.zeros(len(records)),margin))
                dp=p*(dz-(p*dz).sum(-1,keepdims=True))
                mass=(dp*optimal).sum(-1)
                for group,mask in masks.items():
                    got=detail['groups'][group][stage]
                    assert got['lowers_native_optimal_probability']==int((mask&(mass < -1e-10)).sum())
                    assert got['raises_native_optimal_probability']==int((mask&(mass > 1e-10)).sum())
                    np.testing.assert_allclose(got['mean_native_optimal_probability_derivative'],mass[mask].mean(),atol=1e-12,rtol=1e-10)
                    projected_checks+=3
                if stage not in ('total:local_logit','total:cross','adam:total'):continue
                for group,mask in strata.items():
                    target=groups[group]['derivatives'][seed].setdefault(control,{})
                    target[stage]=dict(
                        lowers_optimal_mass=int((mask&(mass < -1e-10)).sum()),
                        raises_optimal_mass=int((mask&(mass > 1e-10)).sum()),
                        lowers_on_currently_correct=int((mask&actor_correct&(mass < -1e-10)).sum()),
                        raises_on_current_errors=int((mask&~actor_correct&(mass > 1e-10)).sum()),
                        mean_derivative=float(mass[mask].mean()) if mask.any() else None)
    import torch
    z=torch.tensor(arrays['logits']);y=torch.tensor(arrays['native_returns'])
    guard_rows,_=protected_actions(z,y,records)
    kept=np.zeros(len(records),dtype=bool);kept[guard_rows.numpy()]=True
    coverage={name:dict(rows=int(mask.sum()),correct=int((mask&actor_correct).sum()),
        protected=int((mask&kept).sum()),correct_unprotected=int((mask&actor_correct&~kept).sum()))
        for name,mask in strata.items() if name.startswith('fit:')}
    labels,weights=error_targets(z,y,records)
    targeted=(labels>=0).numpy()
    error_coverage={name:dict(errors=int((mask&~actor_correct).sum()),
        targeted=int((mask&targeted).sum()),omitted_rows=np.flatnonzero(mask&~actor_correct&~targeted).tolist())
        for name,mask in strata.items() if name.startswith('fit:')}
    witness_z=torch.tensor([[0.,3.,0.],[2.,1.,0.]])
    witness_y=torch.tensor([[0.,2.,1.],[1.,1.,0.]])
    wr,wa=protected_actions(witness_z,witness_y,[{'role':'fit'},{'role':'fit'}])
    changed=witness_z.clone();changed[1]=torch.tensor([0.,0.,3.])
    witness=preservation(changed,margins(witness_z,wr,wa),wr,wa)
    assert wr.tolist()==[0] and witness['passed'] and changed[1].argmax()==2
    omitted,omitted_weight=error_targets(changed[1:],witness_y[1:],[{'role':'fit'}])
    assert omitted.tolist()==[-1] and omitted_weight.tolist()==[0.]
    output.mkdir(parents=True,exist_ok=True)
    (output/'GUARD_COVERAGE.json').write_text(json.dumps(dict(identity=cfg['identity'],
        source_guard_sha256=sha(Path(__file__).resolve().parents[4]/'app/fit_decision_preservation.py'),
        source_target_sha256=sha(Path(__file__).resolve().parents[4]/'app/targeted_fit_errors.py'),
        strict_guard_rows=guard_rows.tolist(),fit_strata=coverage,
        fit_error_targets=error_coverage,
        synthetic_witness=dict(legacy_guard_passed=witness['passed'],unprotected_hold_changed_to_bad_close=True,
                               bad_close_omitted_by_error_target=True,real_actor_changed=False),
        remedy='Before refinement protect correct optimal action sets and target joint optimal probability mass for errors, including HOLD ties; no need to freeze all-equivalent actions',
        scope='coverage omission and reproducible counterexample, not attribution of historical losses'),indent=2)+'\n')
    (output/'STATE_DECISIONS.json').write_text(json.dumps(dict(identity=cfg['identity'],
        groups=groups,scope='native full-policy continuation; not isolated early-close trade profit; local derivatives only'),
        indent=2,sort_keys=True)+'\n')
    (output/'SUMMARY.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    audit=dict(source_identity=cfg['source'],identity=cfg['identity'],runtime='57497834',
        source_receipt_sha256=sha(source/'DONE.json'),receipt_sha256=sha(root/'DONE.json'),
        diagnostic_files=len(read(root/'DONE.json')['files']),calibration_files=len(read(source/'DONE.json')['files']),
        independent_ranking_checks=checks,independent_projection_checks=projected_checks,
        report_code_sha256=sha(Path(__file__)),strata_code_sha256=sha(Path(__file__).resolve().parents[4]/'app/selected_refinement_diagnosis.py'),
        actor_updates=0,validation_read=False,new_candidate=False)
    (output/'INTEGRITY.json').write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n')
    (output/'MANIFEST.json').write_text(json.dumps(dict(files={n:sha(output/n)
        for n in ('SUMMARY.json','INTEGRITY.json','STATE_DECISIONS.json','GUARD_COVERAGE.json')}),indent=2)+'\n')
    print(json.dumps(audit))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('root','source','output'):parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args();collect(a.root,a.source,a.output)
