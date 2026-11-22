"""Summarize paired 411 TRAIN interventions, not a new sequential account."""
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path('/var/tmp/role411-train-attribution')
PARTS = ['barrier_gross','close_gross','spread','slippage','commission','financing','continuation']


def main():
    rows=[]
    for seed in (7,19,43):
        f=pd.read_csv(OUT/f'offers-{seed}.csv.gz')
        assert not f.duplicated(['origin_ns','side','h']).any()
        assert np.max(np.abs(f[PARTS].sum(axis=1)-f.target))<1e-7
        f['managed_minus_barrier']=f.target-f.barrier_target
        f['cost']=f[['spread','slippage','commission','financing']].sum(axis=1)
        masks=dict(all=np.ones(len(f),bool),chosen=f.chosen==1,top=f.decile==10)
        for name,mask in masks.items():
            q=f[mask]
            for (side,h),g in [(('ALL',0),q),*q.groupby(['side','h'])]:
                cols=['target','pred','barrier_target','managed_minus_barrier','cost',*PARTS]
                row=dict(seed=seed,population=name,side=side,h=h,n=len(g),**g[cols].mean().to_dict(),
                         changed=int(g.exit_changed.sum()),max_error=g.error.abs().max())
                t=g[g.exit_changed]
                row.update(changed_helped=int((t.managed_minus_barrier>1e-7).sum()),
                           changed_hurt=int((t.managed_minus_barrier< -1e-7).sum()))
                rows.append(row)
    result=pd.DataFrame(rows)
    result.to_csv(OUT/'SUMMARY.csv',index=False)
    print(result[(result.side=='ALL') & (result.population=='chosen')].to_string(index=False))
    print(result[(result.side=='OPEN_BUY') & (result.h==6) & (result.population=='top')].to_string(index=False))


if __name__=='__main__':
    main()
