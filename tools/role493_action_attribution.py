"""Read-only accounting attribution of fixed 492 policy disagreements."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from role492_fixed_vintage import HASH, SEEDS, sha


def run(panel, models, refs, out):
    assert sha(panel)==HASH
    rows=[]; details=[]; seedstats=[]
    with np.load(panel) as d:
        for role in (0,-1,1):
            manifest=json.loads((refs/f'manifest_{role}.json').read_text())
            if role==1:
                assert all('checkpoint' not in m and m['n']<16 for m in manifest)
                continue
            ref=pd.read_csv(refs/f'weekly_{role}.csv')
            names=['NOP','BUY','SELL'] if role==0 else ['HOLD','CLOSE']
            for half in (1,2):
                mask=(d['side']==role)&(d['w']>='2023-01')&(d['w']<'2024')&((d['w']<'2023-07')==(half==1))
                w=d['w'][mask]; x=d['x'][mask];cash=d['cash'][mask];actor=d['actor'][mask]
                if role==-1:
                    np.testing.assert_allclose(cash[:,0],cash[:,2],atol=1e-6)
                    x=np.c_[x,d['ctx'][mask]];cash=cash[:,:2];actor=(actor==1).astype(int)
                ix=np.arange(len(w));frames=[]
                for seed in SEEDS:
                    act=[]
                    for year in (2021,2022):
                        m,=[v for v in manifest if v['seed']==seed and v['year']==year]
                        path=models/m['checkpoint']
                        assert sha(path)==m['sha256']
                        act.append(joblib.load(path).predict(x).argmax(1))
                    old,new=act
                    f=pd.DataFrame(dict(week=w,idx=ix,seed=seed,old=old,new=new,
                        diff=cash[ix,new]-cash[ix,old],vs_actor=cash[ix,new]-cash[ix,actor]))
                    weekly=f.groupby('week')[['diff','vs_actor']].mean()
                    orig=ref[(ref.seed==seed)&(ref.half==half)].set_index('week').sort_index()
                    assert weekly.index.equals(orig.index)
                    np.testing.assert_allclose(weekly,orig[['recent_old','recent_actor']],rtol=0,atol=1e-8)
                    frames.append(f)
                f=pd.concat(frames,ignore_index=True)
                weeks=f.week.nunique()
                wt=1/(weeks*f.groupby('week').idx.transform('size'))
                assert np.isclose(wt.sum(),1)
                dis=f.old!=f.new
                rows.append(dict(role=role,half=half,states=len(ix),weeks=weeks,
                    disagree_unique_any=int(f.loc[dis,'idx'].nunique()),
                    disagree_seed_states=int(dis.sum()),seed_states=len(f),
                    disagree_weight=float(wt[dis].sum()),net=float((wt*f['diff']).sum()),
                    new_vs_actor=float((wt*f.vs_actor).sum())))
                for i,first in enumerate(names):
                    for j,last in enumerate(names):
                        pick=(f.old==i)&(f.new==j)
                        g=f.loc[pick]; mass=wt[pick].sum()
                        val=float((wt[pick]*g['diff']).sum())
                        details.append(dict(role=role,half=half,old=first,new=last,
                            seed_states=len(g),unique_states=g.idx.nunique(),weeks=g.week.nunique(),
                            better=(g['diff']>1e-8).sum(),worse=(g['diff']< -1e-8).sum(),
                            tied=(g['diff'].abs()<=1e-8).sum(),weight=mass,
                            conditional_usd=val/mass if mass else None,contribution_usd=val,
                            contribution_vs_actor=float((wt[pick]*g.vs_actor).sum())))
                        for seed in SEEDS:
                            s=pick&(f.seed==seed)
                            seedstats.append(dict(role=role,half=half,old=first,new=last,seed=seed,
                                contribution_usd=3*(wt[s]*f.loc[s,'diff']).sum()))
    out.mkdir(parents=True,exist_ok=True)
    parts=pd.DataFrame(details); summary=pd.DataFrame(rows)
    totals=parts.groupby(['role','half'])[['contribution_usd','contribution_vs_actor']].sum()
    np.testing.assert_allclose(totals,summary.set_index(['role','half']).sort_index()[['net','new_vs_actor']],atol=1e-8)
    for name,df in [('transitions',parts),('summary',summary),('seeds',pd.DataFrame(seedstats))]:
        df.to_csv(out/(name+'.csv'),index=False,float_format='%.8f')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('panel','models','refs','out'):p.add_argument(k,type=Path)
    run(**vars(p.parse_args()))
