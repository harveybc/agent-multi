import json,hashlib,sys
from pathlib import Path
import numpy as np
D=Path(sys.argv[1])
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def load(genes,split,mode='ideal'):
    return json.loads((D/'cache'/(digest(dict(genes=genes,split=split,mode=mode))+'.json')).read_text())['payload']
S=json.loads((D/'SEARCH_RESULTS.json').read_text());C=json.loads((D/'CONTROLS.json').read_text())
arms={r['id']:r['genes'] for r in S+C}
for split in ('calibration','confirmation'):
  for a in ('sharpe_s7','sharpe_return_s7','sharpe_return_activity_s43','control_1','control_2'):
    p=load(arms[a],split);tr=[t for e in p['episodes'] for t in e['trades']]
    rr=[];be=[]
    for e in p['episodes']:rr+=e['effective_rr'];be+=e['barrier_breakeven']
    # recompute
    rr2=[abs(t['take_profit']-t['entry_price'])/abs(t['entry_price']-t['stop_loss']) for t in tr]
    out={}
    for d in (1,-1):
        T=[t for t in tr if t['direction']==d]
        sl=np.array([abs(t['entry_price']-t['stop_loss'])/1e-4 for t in T]);tp=np.array([abs(t['take_profit']-t['entry_price'])/1e-4 for t in T])
        fin=sum(t['financing_usd'] for t in T);units=sum(t['units'] for t in T)
        out[d]=dict(n=len(T),sl_pips_med=float(np.median(sl)),sl_pips_min=float(sl.min()),sl_lt2pips=int((sl<2).sum()),tp_pips_med=float(np.median(tp)),
            rr_agg=float(tp.sum()/sl.sum()),fin_usd=round(fin,2),net=round(sum(t['net_usd'] for t in T),2),
            sl_hits=sum(t['close_reason']=='SL' for t in T))
    print(split,a,'rr_report_mean=%.3f recomputed=%.3f median_ratio=%.3f breakeven_mean=%.3f'%(p['report']['mean_effective_rr'],np.mean(rr2),np.median(rr2),p['report']['mean_barrier_breakeven']),json.dumps(out))
