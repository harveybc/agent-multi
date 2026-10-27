"""A5: exact-path counterfactual of every AGENT_CLOSE (hold until SL/TP/forced flat, engine semantics incl.
limit_through and stop-first), and open-time risk vs oracle gain for every entry."""
import json,hashlib,sys
from collections import Counter
from pathlib import Path
import numpy as np
D=Path(sys.argv[1]);sys.path.insert(0,sys.argv[2])
from app.strategy_scenario_matrix import market_frame,WARMUP
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def load(genes,split,mode='ideal'):
    return json.loads((D/'cache'/(digest(dict(genes=genes,split=split,mode=mode))+'.json')).read_text())['payload']
S=json.loads((D/'SEARCH_RESULTS.json').read_text());C=json.loads((D/'CONTROLS.json').read_text())
arms={r['id']:r['genes'] for r in S+C}
use=['sharpe_s7','sharpe_return_s7','sharpe_return_activity_s43','control_0','control_1','control_2']
THROUGH=1e-4;RT=None
out={}
for split in ('calibration','confirmation'):
  for a in use+(['midpoint_inverted'] if split=='confirmation' else []):
    p=load(arms['control_0'],split,'inverted') if a=='midpoint_inverted' else load(arms[a],split)
    ec=Counter();gain=0.;n=0;risk=Counter();risk_net=Counter()
    for ep in p['episodes']:
        cell=ep['cell'];f=market_frame(cell);o,h,l,c=(f[k].to_numpy() for k in ('OPEN','HIGH','LOW','CLOSE'))
        hz=cell['horizon']
        for t in ep['trades']:
            d=t['direction'];sl,tp=t['stop_loss'],t['take_profit']
            # risk vs oracle gain at entry: decision bar b=entry-1, k=entry
            k=t['entry_bar_abs'];fut=min(k+hz-1,len(c)-1)
            gainpx=d*(c[fut]-o[k]);riskpx=abs(t['entry_price']-sl);rewardpx=abs(tp-t['entry_price'])
            cls='gain<risk' if gainpx<riskpx else 'gain>=risk'
            risk[cls]+=1;risk_net[cls]+=t['net_usd']
            if t['close_reason']!='AGENT_CLOSE':continue
            n+=1;kx=t['exit_bar_abs'];wk_end=(kx//120)*120+107;cf=None
            for j in range(kx,min(wk_end,len(c)-1)+1):
                if d>0:
                    if o[j]<=sl:cf=('SL',o[j]);break
                    if o[j]>=tp:cf=('TP',o[j]);break
                    if l[j]<=sl:cf=('SL',sl);break
                    if h[j]>=tp+THROUGH:cf=('TP',tp);break
                else:
                    if o[j]>=sl:cf=('SL',o[j]);break
                    if o[j]<=tp:cf=('TP',o[j]);break
                    if h[j]>=sl:cf=('SL',sl);break
                    if l[j]<=tp-THROUGH:cf=('TP',tp);break
            if cf is None:cf=('FORCED_FLAT',c[min(wk_end,len(c)-1)])
            g=d*(t['exit_price']-cf[1])/1e-4
            ec[(cf[0],'close_better' if g>0 else 'hold_better')]+=1;gain+=g
    out[f'{split}/{a}']=dict(agent_closes=n,outcomes={f'{k[0]}|{k[1]}':v for k,v in ec.items()},close_minus_hold_pips=gain,
        opens_by_gain_vs_risk=dict(risk),net_usd_by_gain_vs_risk={k:round(v,2) for k,v in risk_net.items()})
print(json.dumps(out,indent=1))
