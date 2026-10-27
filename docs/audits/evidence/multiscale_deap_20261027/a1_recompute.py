"""Audit 1: recompute reports/fitness from sealed cache ledgers (read-only)."""
import json,hashlib,math,sys
from collections import Counter
from pathlib import Path
import numpy as np
D=Path(sys.argv[1])
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
identity=json.loads((D/'IDENTITY.json').read_text())
IDH=digest(identity)
def load(genes,split='calibration',mode='ideal'):
    k=digest(dict(genes=genes,split=split,mode=mode));p=D/'cache'/(k+'.json')
    r=json.loads(p.read_text())
    assert r['identity_sha256']==IDH,'IDENTITY'
    assert r['payload_sha256']==digest(r['payload']),'PAYLOAD'
    return r['payload']
def aggregate(rows):
    returns=np.asarray([v for r in rows for v in r['daily_returns']],float)
    std=float(np.std(returns,ddof=1));sharpe=float(np.mean(returns)/std*np.sqrt(365)) if std>0 else None
    trades=[t for r in rows for t in r['trades']]
    sides={str(d):[t for t in trades if t['direction']==d] for d in (1,-1)}
    reasons=[]
    if sharpe is None:reasons.append('UNDEFINED_SHARPE')
    if len(trades)/sum(r['weeks'] for r in rows)<2:reasons.append('ACTIVITY_BELOW_TWO')
    for d in (1,-1):
        if not sides[str(d)]:reasons.append('NO_SIDE_'+str(d))
        if not any(t['close_reason']=='AGENT_CLOSE' for t in sides[str(d)]):reasons.append('NO_CLOSE_'+str(d))
    return dict(components=dict(sharpe=sharpe,return_fraction=float(np.mean([r['return_fraction'] for r in rows])),
        orders_per_day=len(trades)/sum(r['business_days'] for r in rows)),reasons=reasons,trades=len(trades),
        nreturns=len(returns))
scales=json.loads((D/'SCALES.json').read_text())
def score(name,c):
    keys={'sharpe':('sharpe',),'sharpe_return':('sharpe','return_fraction'),'sharpe_return_activity':('sharpe','return_fraction','orders_per_day')}[name]
    return sum(c[k]/scales[k] for k in keys)
out={}
# scales from controls
controls=json.loads((D/'CONTROLS.json').read_text())
rec={k:float(np.median([abs(load(c['genes'])['report']['components'][k]) for c in controls])) for k in scales}
out['scales_recomputed']=rec;out['scales_reported']=scales
searches=json.loads((D/'SEARCH_RESULTS.json').read_text())
rows=[]
for s in searches:
    p=load(s['genes']);a=aggregate(p['episodes'])
    f=score(s['objective'],a['components'])
    st=json.loads((D/'searches'/s['id']/'STATE.json').read_text())['payload']
    rows.append(dict(id=s['id'],reported_fitness=s['fitness'],recomputed_fitness=f,diff=f-s['fitness'],
        sharpe_rep=s['report']['components']['sharpe'],sharpe_re=a['components']['sharpe'],
        ret_rep=s['report']['components']['return_fraction'],ret_re=a['components']['return_fraction'],
        reasons=a['reasons'],nreturns=a['nreturns'],state_best_equal=st['best']==s['genes'],state_score=st['best_score'],
        evaluations=st['evaluations'],generation=st['generation'],stop=st['stop_reason']))
out['searches']=rows
# confirmation recompute
conf=json.loads((D/'CONFIRMATION.json').read_text())
crows=[]
for c in conf:
    mode='ideal'
    if c['id'].startswith('midpoint_'):mode=c['id'].split('_')[1]
    p=load(c['genes'],'confirmation',mode);a=aggregate(p['episodes'])
    crows.append(dict(id=c['id'],sharpe_rep=c['report']['components']['sharpe'],sharpe_re=a['components']['sharpe'],
        ret_rep=c['report']['components']['return_fraction'],ret_re=a['components']['return_fraction'],
        trades_rep=c['report']['trades'],trades_re=a['trades'],reasons_rep=c['report']['reasons'],reasons_re=a['reasons']))
out['confirmation']=crows
print(json.dumps(out,indent=1))
