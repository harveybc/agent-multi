"""A4: paired tape-block bootstrap CIs for Sharpe/return differences between finalists and controls."""
import json,hashlib,sys
from pathlib import Path
import numpy as np
D=Path(sys.argv[1])
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def load(genes,split):
    return json.loads((D/'cache'/(digest(dict(genes=genes,split=split,mode='ideal'))+'.json')).read_text())['payload']
S=json.loads((D/'SEARCH_RESULTS.json').read_text());C=json.loads((D/'CONTROLS.json').read_text())
arms={r['id']:r['genes'] for r in S+C}
rng=np.random.default_rng(20261027)
def sharpe(mat,idx):
    x=mat[idx].ravel();return x.mean()/x.std(ddof=1)*np.sqrt(365)
out={}
for split in ('calibration','confirmation'):
    R={a:np.array([e['daily_returns'] for e in load(g,split)['episodes']]) for a,g in arms.items()}
    F={a:np.array([e['return_fraction'] for e in load(g,split)['episodes']]) for a,g in arms.items()}
    n=24;B=10000;idx=rng.integers(0,n,(B,n))
    base={a:sharpe(R[a],np.arange(n)) for a in R}
    boots={a:np.array([sharpe(R[a],i) for i in idx]) for a in R}
    res={}
    for a in R:
        lo,hi=np.percentile(boots[a],[2.5,97.5]);res[a]=dict(sharpe=base[a],ci=[lo,hi],ret=F[a].mean())
    pairs=[('sharpe_s7','sharpe_return_s7'),('sharpe_s7','sharpe_return_activity_s43'),('sharpe_return_s7','sharpe_return_activity_s43'),
           ('sharpe_s7','control_1'),('sharpe_s7','control_2'),('sharpe_s7','control_0'),('sharpe_return_s7','control_2'),
           ('sharpe_return_activity_s43','control_2'),('sharpe_s7','sharpe_s19'),('sharpe_s7','sharpe_s43'),('control_1','control_2')]
    pr={}
    for a,b in pairs:
        diff=boots[a]-boots[b];lo,hi=np.percentile(diff,[2.5,97.5])
        rd=np.array([F[a][i].mean()-F[b][i].mean() for i in idx]);rlo,rhi=np.percentile(rd,[2.5,97.5])
        pr[a+' - '+b]=dict(dS=base[a]-base[b],ci=[lo,hi],p_le0=float((diff<=0).mean()),dR=F[a].mean()-F[b].mean(),dR_ci=[rlo,rhi],
            tapes_a_better=int(sum(F[a]>F[b])))
    # rank of finalists among all three arms per tape? per-tape sharpe not defined well; give per-tape return wins
    out[split]=dict(arms=res,pairs=pr)
    # seed spread for sharpe-only finalists
    out[split]['seed_spread_sharpe_only']=[base['sharpe_s7'],base['sharpe_s19'],base['sharpe_s43']]
print(json.dumps(out,indent=1))
