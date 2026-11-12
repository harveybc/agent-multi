"""Publish compact authenticated selection evidence without private runtime paths."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

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
    account_checks={}
    for period in ('TRAIN','VALIDATION'):
        before=a.qualification/'interior_error_project_guard_19/native'/period
        after=a.export/'native'/period
        rows=[];equities=[];trades=[]
        for root in (before,after):
            m=read(root/'METRICS.json');t=read(root/'TRADES.json');trades.append(t)
            with np.load(root/'EQUITY.npz') as z:
                stamp=z['timestamp'].copy();equity=z['equity'].copy()
            equities.append((stamp,equity))
            daily=pd.Series(equity,index=pd.to_datetime(stamp,utc=True)).resample('1D').last().ffill()
            r=daily.pct_change().dropna().to_numpy()
            sharpe=float(np.sqrt(365)*r.mean()/r.std(ddof=1))
            cash_error=abs(sum(v['net_usd'] for v in t)-m['net_profit_usd'])
            assert abs(sharpe-m['sharpe'])<1e-12 and cash_error<1e-6
            rows.append(dict(sharpe=sharpe,cash_error=cash_error,
                trades_sha256=sha(root/'TRADES.json'),equity_sha256=sha(root/'EQUITY.npz')))
        assert np.array_equal(equities[0][0],equities[1][0])
        delta=equities[1][1]-equities[0][1]
        account_checks[period]=dict(before=rows[0],after=rows[1],
            changed_equity_rows=int(np.count_nonzero(delta)),max_equity_delta_usd=float(np.max(np.abs(delta))),
            final_equity_delta_usd=float(delta[-1]))
        if period=='VALIDATION':
            assert len(trades[0])==len(trades[1])
            account_checks[period].update(changed_trade_records=sum(x!=y for x,y in zip(*trades)),
                net_per_trade_exact=all(x['net_usd']==y['net_usd'] for x,y in zip(*trades)),
                entry_shift_bars=[y['entry_bar_abs']-x['entry_bar_abs'] for x,y in zip(*trades) if x!=y])
    write('ACCOUNT_CHECKS.json',account_checks);names+=['ACCOUNT_CHECKS.json']
write('MANIFEST.json',dict(identity=summary['identity'],qualification_identity=qualified['identity'],runtime='096fbc7c',
    source_receipts={k:sha(root/name) for k,root,name in [
        ('qualification',a.qualification,'DONE.json'),('selection',a.selection,'DONE.json'),('report',a.report,'MANIFEST.json')]},
    export_receipt_sha256=sha(a.export/'DONE.json') if a.export else None,
    verified_files=counts,files={n:sha(a.output/n) for n in names},
    source_unchanged=True,test_read=False,model_promoted=False,
    legacy_current_donor_label='current69 in raw ledger denotes the unchanged current qualified72 actor, not an old donor'))
print(json.dumps(dict(counts=summary['counts'],phases=analysis,verified_files=counts),indent=2))
