"""Publish authenticated current-source refinement and account attribution."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(run,report,audit,output):
    for root in (run,report):
        for name,expected in read(root/'DONE.json')['files'].items():
            p=(root/name).resolve()
            if not p.is_relative_to(root.resolve()) or sha(p)!=expected:
                raise ValueError('COLLECTION_IDENTITY:'+name)
    result=read(run/'RESULT.json');cfg=read(run/'CONFIG.json');check=read(audit)
    assert result['identity']==cfg['identity']==check['identity']
    assert cfg['schema']=='source_value_floor_interior_correction.v1'
    summary=read(report/'SUMMARY.json')
    assert summary['source_identity']==cfg['identity'] and len(summary['accounts'])==4
    output.mkdir(parents=True,exist_ok=True)
    values={'AUDIT.json':check,'GUARD.json':read(run/'GUARD.json'),
        'TARGETS.json':{k:cfg[k] for k in ('error_targets','error_weights','source_identity',
            'diagnosis_identity','guard_origin','interior_buffer','interior_formula','minimum_linear_slack','optimal_action_tolerance','guard_semantics','corrective_margin_semantics')},
        'SUMMARY.json':summary,'STEPS.json':{arm['arm']:arm['steps'] for arm in result['arms']}}
    for name,value in values.items():
        (output/name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    shutil.copy2(run/'RETURNS.png',output/'RETURNS.png')
    names=list(values)+['RETURNS.png']
    manifest=dict(identity=cfg['identity'],runtime='b18e84c5',
        source_receipt_sha256=sha(run/'DONE.json'),report_receipt_sha256=sha(report/'DONE.json'),
        files={name:sha(output/name) for name in names},source_unchanged=True,test_read=False,model_promoted=False)
    (output/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(identity=cfg['identity'],published_files=len(names))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('run','report','audit','output'):parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args();collect(a.run,a.report,a.audit,a.output)
