"""R5: adverse intra-week gaps unknown to the oracle (tape computed on the ungapped path), disposable runtime patch."""
import json,sys,os
os.environ['CUDA_VISIBLE_DEVICES']=''
SRC,ENGINE,RECIPE,D=sys.argv[1:5]
sys.path.insert(0,SRC)
from pathlib import Path
import numpy as np,pandas as pd
import app.multiscale_strategy_search as M
import app.strategy_scenario_matrix as SM
import tools.systematic_strategy_matrix as T
D=Path(D);orig=SM.market_frame
def base(cell):return {k:v for k,v in cell.items() if k not in ('gap_pips','gap_seed')}
def frame(cell):
    f=orig(base(cell))
    if not cell.get('gap_pips'):return f
    c=f['CLOSE'].to_numpy().copy();n=len(c);rng=np.random.default_rng(cell['gap_seed'])
    off=np.arange(n)%120
    cand=np.flatnonzero((np.arange(n)>=SM.WARMUP+cell['switch'])&(off>14)&(off<100))
    bars=np.sort(rng.choice(cand,3,replace=False))
    sign=-np.sign(cell['drift_after']) if cell['drift_after']!=0 else rng.choice([-1,1])
    shift=np.zeros(n)
    for b in bars:shift[b:]+=sign*cell['gap_pips']*1e-4
    c=c+shift;o=np.r_[1.1,c[:-1]]
    for b in bars:o[b]=c[b-1]+sign*cell['gap_pips']*1e-4
    amp=np.where(np.arange(-SM.WARMUP,n-SM.WARMUP)<cell['switch'],cell['vol_before'],cell['vol_after']);w=amp*.25+1e-6
    return pd.DataFrame(dict(OPEN=o,HIGH=np.maximum(o,c)+w,LOW=np.minimum(o,c)-w,CLOSE=c),index=f.index)
SM.market_frame=frame;T.market_frame=frame
cur={'cell':None}
ot=M.output_tape
def tape(engine,horizon,mode):
    cell=cur['cell']
    if not cell or not cell.get('gap_pips'):return ot(engine,horizon,mode)
    g=orig(base(cell));c,o=engine.c,engine.o
    engine.c=g['CLOSE'].to_numpy();engine.o=g['OPEN'].to_numpy()
    try:return ot(engine,horizon,mode)
    finally:engine.c,engine.o=c,o
M.output_tape=tape
oe=M.episode
def episode(task):cur['cell']=task[0];return oe(task)
M.initialize(RECIPE,ENGINE)
S=json.loads((D/'SEARCH_RESULTS.json').read_text());C=json.loads((D/'CONTROLS.json').read_text())
arms={r['id']:r['genes'] for r in S+C}
out={}
for split in ('confirmation',):
  P=M.panel(split)
  for a in ('sharpe_s7','sharpe_return_s7','sharpe_return_activity_s43','control_0','control_1','control_2'):
    res={}
    for G in (0,10,30):
        for s in ((1,2,3) if G else (0,)):
            rows=[episode((dict(c,gap_pips=G,gap_seed=100*s+i) if G else c,arms[a],'ideal')) for i,c in enumerate(P)]
            r=M.aggregate(rows);tr=[t for x in rows for t in x['trades']]
            res[f'gap{G}_s{s}']=dict(S=r['components']['sharpe'],R=r['components']['return_fraction'],dd=r['worst_drawdown_pct'],
                sl=r['close_reasons'].get('SL',0),sl_at_open=sum(t['close_reason']=='SL' and t['exit_clock']=='OPEN' for t in tr),
                worst_trade=min(t['net_usd'] for t in tr))
    out[a]=res;print(a,json.dumps({k:[round(v['S'],2),round(v['R']*100,2),round(v['dd'],2),v['sl'],v['sl_at_open'],round(v['worst_trade'],1)] for k,v in res.items()}),file=sys.stderr,flush=True)
print(json.dumps(out))
