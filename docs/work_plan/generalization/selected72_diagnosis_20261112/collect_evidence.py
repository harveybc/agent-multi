"""Authenticate TRAIN-only boundary diagnostics and independently check rankings."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


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
    if cfg['source']!=read(source/'CONFIG.json')['identity']:
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
    output.mkdir(parents=True,exist_ok=True)
    (output/'SUMMARY.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    audit=dict(source_identity=cfg['source'],identity=cfg['identity'],runtime='142a0b8c',
        source_receipt_sha256=sha(source/'DONE.json'),receipt_sha256=sha(root/'DONE.json'),
        diagnostic_files=len(read(root/'DONE.json')['files']),calibration_files=len(read(source/'DONE.json')['files']),
        independent_ranking_checks=checks,actor_updates=0,validation_read=False,new_candidate=False)
    (output/'INTEGRITY.json').write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n')
    (output/'MANIFEST.json').write_text(json.dumps(dict(files={n:sha(output/n)
        for n in ('SUMMARY.json','INTEGRITY.json')}),indent=2)+'\n')
    print(json.dumps(audit))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('root','source','output'):parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args();collect(a.root,a.source,a.output)
