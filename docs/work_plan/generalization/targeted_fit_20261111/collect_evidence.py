"""Build small public evidence from sealed local results; no weights or credentials."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

parser=argparse.ArgumentParser()
for key in ('run','report','audit','rejection','output'):parser.add_argument('--'+key,type=Path,required=True)
a=parser.parse_args()
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,value):
    (a.output/name).write_text(json.dumps(value,indent=2)+'\n')
for root in (a.run,a.report):
    for name,expected in read(root/'DONE.json')['files'].items():
        assert sha(root/name)==expected
r=read(a.run/'RESULT.json');cfg=read(a.run/'CONFIG.json')
audit=read(a.audit);rejection=read(a.rejection)
assert r['identity']==cfg['identity']==audit['identity']==rejection['identity']
a.output.mkdir(parents=True,exist_ok=True)
write('AUDIT.json',audit);write('REJECTION.json',rejection)
write('GUARD.json',read(a.run/'GUARD.json'))
write('TARGETS.json',dict(labels=cfg['error_targets'],weights=cfg['error_weights'],
    source_identity=cfg['source_identity'],prior_guard_identity=cfg['prior_guard_identity']))
write('SUMMARY.json',read(a.report/'SUMMARY.json'))
write('STEPS.json',{arm['arm']:arm['steps'] for arm in r['arms']})
shutil.copy2(a.run/'RETURNS.png',a.output/'RETURNS.png')
names=['AUDIT.json','REJECTION.json','GUARD.json','TARGETS.json','SUMMARY.json','STEPS.json','RETURNS.png']
write('MANIFEST.json',dict(identity=cfg['identity'],runtime='183255f5',
    source_receipt_sha256=sha(a.run/'DONE.json'),report_receipt_sha256=sha(a.report/'DONE.json'),
    files={name:sha(a.output/name) for name in names},source_unchanged=True,test_read=False,model_promoted=False))
print(json.dumps(dict(identity=cfg['identity'],published_files=len(names)),indent=2))
