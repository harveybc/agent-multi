"""R3: counterexamples on disposable copies: mirrored panels, stochastic paths, adverse weekly gaps, and an
oracle-quality ladder. Musashi's code is imported read-only; only this process's module attributes are patched."""
import json,hashlib,sys,os
os.environ['CUDA_VISIBLE_DEVICES']=''
SRC,ENGINE,RECIPE,D=sys.argv[1:5]
sys.path.insert(0,SRC)
from pathlib import Path
import numpy as np,pandas as pd
import app.multiscale_strategy_search as M
import app.strategy_scenario_matrix as SM
import tools.systematic_strategy_matrix as T
D=Path(D)
orig_frame=SM.market_frame
def frame(cell):
    base={k:v for k,v in cell.items() if k not in ('mirror','noise_seed','noise_scale','gap_pips','gap_seed')}
    f=orig_frame(base)
    c=f['CLOSE'].to_numpy().copy()
    amp_b,amp_a=cell['vol_before'],cell['vol_after']
    n=len(c)
    if cell.get('noise_seed') is not None:
        rng=np.random.default_rng(cell['noise_seed'])
        inc=np.diff(np.r_[1.1,c])
        amp=np.where(np.arange(-SM.WARMUP,n-SM.WARMUP)<cell['switch'],amp_b,amp_a)
        inc=inc+cell.get('noise_scale',1.0)*amp*rng.standard_normal(n)
        c=1.1+np.cumsum(inc)
    o=np.r_[1.1,c[:-1]]
    if cell.get('gap_pips'):
        rng=np.random.default_rng(cell.get('gap_seed',0))
        idx=f.index
        newweek=np.r_[False,(idx[1:]-idx[:-1])>pd.Timedelta(hours=1)]
        shift=np.cumsum(np.where(newweek,rng.choice([-1,1],n)*cell['gap_pips']*1e-4,0.))
        c=c+shift
        o=np.r_[1.1,c[:-1]]
        # true open gap: the first bar of each week opens at the gapped level
        o[newweek]=c[np.flatnonzero(newweek)-1]+(shift[newweek]-shift[np.flatnonzero(newweek)-1])
    if cell.get('mirror'):
        c=2.2-c;o=2.2-o
    amp=np.where(np.arange(-SM.WARMUP,n-SM.WARMUP)<cell['switch'],amp_b,amp_a)
    wick=amp*.25+1e-6
    return pd.DataFrame(dict(OPEN=o,HIGH=np.maximum(o,c)+wick,LOW=np.minimum(o,c)-wick,CLOSE=c),index=f.index)
SM.market_frame=frame;T.market_frame=frame
ladder={'eta':0.0,'seed':0}
orig_tape=M.output_tape
def tape(engine,horizon,mode):
    if ladder['eta']==0:return orig_tape(engine,horizon,mode)
    # degrade the oracle: replace engine.c used for the future target by a noisy copy, then restore
    c=engine.c.copy();rng=np.random.default_rng([ladder['seed'],int(abs(engine.c[0]*1e9))%2**31,len(c)])
    k=np.minimum(np.arange(len(c))+1,len(c)-1);fut=np.minimum(k+horizon-1,len(c)-1)
    sd=np.std(c[fut]-engine.o[k])
    engine.c=c+ladder['eta']*sd*rng.standard_normal(len(c))
    try:return orig_tape(engine,horizon,mode)
    finally:engine.c=c
M.output_tape=tape
M.initialize(RECIPE,ENGINE)
S=json.loads((D/'SEARCH_RESULTS.json').read_text());C=json.loads((D/'CONTROLS.json').read_text())
arms={r['id']:r['genes'] for r in S+C}
use=['sharpe_s7','sharpe_return_s7','sharpe_return_activity_s43','control_0','control_1','control_2']
def swap(g):return g[3:]+g[:3]
def run(genes,cells):
    rows=[M.episode((c,genes,'ideal')) for c in cells];r=M.aggregate(rows)
    return dict(S=r['components']['sharpe'],R=r['components']['return_fraction'],L=r['sides']['1']['trades'],Sh=r['sides']['-1']['trades'],
        eligible=r['eligible'],reasons=r['reasons'],close=r['close_reasons'],wr=r['win_rate'],dd=r['worst_drawdown_pct'])
out={}
for split in ('calibration','confirmation'):
    P=M.panel(split)
    for a in use:
        g=arms[a];res={}
        res['original']=run(g,P)
        res['mirror']=run(g,[dict(c,mirror=True) for c in P])
        res['mirror_swapped_genes']=run(swap(g),[dict(c,mirror=True) for c in P])
        for s in (1,2,3):
            res[f'noise1x_s{s}']=run(g,[dict(c,noise_seed=1000*s+i,noise_scale=1.0) for i,c in enumerate(P)])
        for s in (1,2):
            res[f'gap20_s{s}']=run(g,[dict(c,gap_pips=20,gap_seed=500*s+i) for i,c in enumerate(P)])
        if split=='confirmation':
            for eta in (0.25,0.5,1.0,2.0):
                for s in (1,2,3):
                    ladder.update(eta=eta,seed=77*s)
                    M.RUNTIME['templates'].clear()
                    res[f'oracle_eta{eta}_s{s}']=run(g,[dict(c,ladder=eta,ladder_seed=s) for c in P])
                ladder.update(eta=0.0)
            M.RUNTIME['templates'].clear()
        out.setdefault(split,{})[a]=res
        print(split,a,json.dumps({k:(round(v['S'],3) if v['S'] is not None else None,round(v['R'],4),v['L'],v['Sh']) for k,v in res.items()}),flush=True,file=sys.stderr)
print(json.dumps(out,default=float))
