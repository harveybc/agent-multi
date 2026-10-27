"""Audit 2: per-tape behaviour, sides vs trend, SL/TP presence, sizing, early-close counterfactual."""
import json,hashlib,math,sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
D=Path(sys.argv[1]);SRC=sys.argv[2]
sys.path.insert(0,SRC)
from app.strategy_scenario_matrix import market_frame,WARMUP
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
identity=json.loads((D/'IDENTITY.json').read_text());IDH=digest(identity)
def load(genes,split='calibration',mode='ideal'):
    k=digest(dict(genes=genes,split=split,mode=mode));r=json.loads((D/'cache'/(k+'.json')).read_text())
    assert r['identity_sha256']==IDH and r['payload_sha256']==digest(r['payload']);return r['payload']
fin=json.loads((D/'FINALISTS_BEFORE_CONFIRMATION.json').read_text())
ctr=json.loads((D/'CONTROLS.json').read_text())
arms=[(f['id'],f['genes']) for f in fin]+[(c['id'],c['genes']) for c in ctr]
out=defaultdict(dict)
viol=Counter()
for split in ('calibration','confirmation'):
  modes=['ideal']+(['inverted','nop'] if split=='confirmation' else [])
  for mode in modes:
    for aid,genes in (arms if mode=='ideal' else [('midpoint',ctr[0]['genes'])]):
        p=load(genes,split,mode)
        tapes=[]
        for ep in p['episodes']:
            cell=ep['cell'];bars=market_frame(cell);c=bars['CLOSE'].to_numpy();o=bars['OPEN'].to_numpy()
            h=bars['HIGH'].to_numpy();l=bars['LOW'].to_numpy()
            seg=c[WARMUP:];move=float(seg[-1]-seg[0])
            pre=float(c[WARMUP+cell['switch']]-c[WARMUP]);post=float(seg[-1]-c[WARMUP+cell['switch']])
            gaps=int(np.sum(np.abs(o[1:]-c[:-1])>1e-15))
            tr=ep['trades'];L=[t for t in tr if t['direction']==1];S=[t for t in tr if t['direction']==-1]
            early=[];
            for t in tr:
                # contract checks
                if t['stop_loss'] is None or t['take_profit'] is None:viol['NO_PROTECTION']+=1
                d=t['direction']
                if not (d*(t['take_profit']-t['entry_price'])>0 and d*(t['entry_price']-t['stop_loss'])>0):viol['PROTECTION_SIDE']+=1
                if t['rel_volume']!=0.1:viol['REL_VOLUME']+=1
                if int(t['units'])!=t['units']:viol['FRACTIONAL_UNITS']+=1
                lev=t['units']*t['entry_price']/ (t['margin_usd']) if t['margin_usd'] else None
                if lev is None or abs(lev-50)>1e-6:viol['LEVERAGE_NOT_50']+=1
                if t['close_reason']=='AGENT_CLOSE':
                    # counterfactual hold: from exit bar walk until SL/TP (mid OHLC, stop first) or tape end close
                    k=t['exit_bar_abs'];cf=None
                    for j in range(k,len(c)):
                        if d>0:
                            if l[j]<=t['stop_loss']:cf=('SL',t['stop_loss']);break
                            if h[j]>=t['take_profit']:cf=('TP',t['take_profit']);break
                        else:
                            if h[j]>=t['stop_loss']:cf=('SL',t['stop_loss']);break
                            if l[j]<=t['take_profit']:cf=('TP',t['take_profit']);break
                    if cf is None:cf=('END',c[-1])
                    gain=d*(t['exit_price']-cf[1])  # >0: closing early was better than holding (gross, per unit)
                    early.append(dict(cf=cf[0],better=gain>0,gain_pips=gain/1e-4,net=t['net_usd']))
            tapes.append(dict(db=cell['drift_before'],da=cell['drift_after'],vb=cell['vol_before'],va=cell['vol_after'],
                horizon=cell['horizon'],period=cell['period'],move_pips=move/1e-4,pre_pips=pre/1e-4,post_pips=post/1e-4,gaps=gaps,
                longs=len(L),shorts=len(S),net=ep['net_profit_usd'],
                long_net=sum(t['net_usd'] for t in L),short_net=sum(t['net_usd'] for t in S),
                reasons=dict(Counter(t['close_reason'] for t in tr)),
                early_n=len(early),early_better=sum(e['better'] for e in early),
                early_cf=dict(Counter(e['cf'] for e in early)),early_gain_pips=float(sum(e['gain_pips'] for e in early)),
                max_dd=ep['max_drawdown_pct']))
        out[split][aid+('' if mode=='ideal' else '_'+mode)]=tapes
print(json.dumps(dict(violations=dict(viol),tapes=out),default=float))
