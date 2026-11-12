"""Publish compact authenticated selection evidence without private runtime paths."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

parser=argparse.ArgumentParser()
for key in ('qualification','selection','report','output'):parser.add_argument('--'+key,type=Path,required=True)
parser.add_argument('--export',type=Path)
a=parser.parse_args();read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,value):(a.output/name).write_text(json.dumps(value,indent=2)+'\n')
counts={}
for key,root,name in [('qualification',a.qualification,'DONE.json'),('selection',a.selection,'DONE.json'),('report',a.report,'MANIFEST.json')]:
    receipt=read(root/name)
    for item,expected in receipt['files'].items():
        path=(root/item).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path)!=expected:raise ValueError('EVIDENCE_HASH:'+item)
    counts[key]=len(receipt['files'])
summary=read(a.report/'SUMMARY.json');qualified=read(a.qualification/'RESULT.json')
spec=read(a.selection/'SPEC.private.json');result=read(a.selection/'RESULT.json')
if not qualified['qualified'] or summary['identity']!=result['identity']:raise ValueError('EVIDENCE_IDENTITY')
trace=read(a.report/'SELECTION_TRACE.json')['trace'];inventory=read(a.selection/'INVENTORY.json')
historical=set(inventory['historical_phase2_absent'])
returns={c for r in trace if r['phase']=='REINSERT' for c in r['block']}
if not historical<=returns:raise ValueError('INCOMPLETE_HISTORICAL_RETURN')
if spec['source_files'][spec['baseline']+'/MANIFEST.json']!=sha(Path(spec['baseline'])/'MANIFEST.json'):
    raise ValueError('BASELINE_CHANGED')
analysis={}
for phase in ('DELETE','REINSERT'):
    rows=[r for r in trace if r['phase']==phase]
    deltas=[r['candidate_score']-r['before_score'] for r in rows]
    analysis[phase]=dict(blocks=len(rows),features=sum(len(r['block']) for r in rows),
        accepted=[r['block'] for r in rows if r['accepted']],
        improved=sum(d>0 for d in deltas),equal=sum(d==0 for d in deltas),worse=sum(d<0 for d in deltas),
        ineligible=sum(not r['candidate_eligible'] for r in rows),
        min_score_delta=min(deltas),max_score_delta=max(deltas))
a.output.mkdir(parents=True,exist_ok=True)
names=['SUMMARY.json','WEIGHT_SPACE.json','SELECTION_TRACE.json','FEATURE_LEDGER.csv']
for name in names:shutil.copy2(a.report/name,a.output/name)
write('QUALIFICATION.json',qualified)
write('SENSITIVITY.json',dict(identity=summary['identity'],phases=analysis,
    historical_features=len(historical),historical_return_complete=True,
    initial_features=spec['initial'],selected=result['result']['selected'],
    source_metrics=result['control']['metrics'],final_metrics=result['final']['metrics'],
    conclusion_scope='ordered frozen two-pass screen conditional on current weights, donors, support and score; not global feature optimum'))
names+=['QUALIFICATION.json','SENSITIVITY.json']
if a.export:
    receipt=read(a.export/'DONE.json')
    for item,expected in receipt['files'].items():
        path=(a.export/item).resolve()
        if not path.is_relative_to(a.export.resolve()) or sha(path)!=expected:raise ValueError('EXPORT_HASH:'+item)
    exported=read(a.export/'RESULT.json');config=read(a.export/'CONFIG.json')
    assert exported['qualified'] and exported['cold_exact'] and config['selection']==summary['identity']
    assert exported['identity']==receipt['identity']
    entropy={'mean_signal_entropy','mean_requested_command_entropy'}
    entropy_deltas={}
    for period,actual in exported['metrics'].items():
        expected=summary['metrics'][period]
        assert {k:v for k,v in actual.items() if k not in entropy}=={k:v for k,v in expected.items() if k not in entropy}
        entropy_deltas[period]={k:actual[k]-expected[k] for k in entropy}
        assert all(abs(v)<=exported['parity']['atol'] for v in entropy_deltas[period].values())
    write('EXPORT_ENTROPY_DELTA.json',entropy_deltas);names+=['EXPORT_ENTROPY_DELTA.json']
    write('EXPORT.json',exported);names+=['EXPORT.json'];counts['export']=len(receipt['files'])
write('MANIFEST.json',dict(identity=summary['identity'],qualification_identity=qualified['identity'],runtime='2cd155be',
    source_receipts={k:sha(root/name) for k,root,name in [
        ('qualification',a.qualification,'DONE.json'),('selection',a.selection,'DONE.json'),('report',a.report,'MANIFEST.json')]},
    export_receipt_sha256=sha(a.export/'DONE.json') if a.export else None,
    verified_files=counts,files={n:sha(a.output/n) for n in names},
    source_unchanged=True,test_read=False,model_promoted=False))
print(json.dumps(dict(counts=summary['counts'],phases=analysis,verified_files=counts),indent=2))
