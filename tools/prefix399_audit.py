"""Independent read-only checks on immutable chunks of the in-progress 395 calendar."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def check(src, shared, limit, shard):
    clock=pd.DatetimeIndex(pd.read_parquet(shared/'CLOCK.parquet').timestamp)
    script=pd.read_parquet(shared/'SCRIPT.parquet')
    recipe=json.loads((shared/'RECIPE.json').read_text())
    binding=json.loads((src/'BINDING.json').read_text())
    for f in ('RECIPE.json','CLOCK.parquet','SCRIPT.parquet'):
        if sha(shared/f) not in [v for k,v in binding.items() if k.endswith('/'+f)]:
            raise ValueError(('unbound',f))
    frames=[]
    n=blocks=rows=0
    for j,part in enumerate(sorted(src.glob('part-[0-9]*'))):
        if n==limit: break
        st=json.loads((part/'STATE.json').read_text())
        b=pd.read_parquet(part/'OHLC.parquet')
        if (st['start']!=n or st['end']!=n+len(b) or st['end']>limit or
                not b.index.equals(clock[n:st['end']].as_unit(b.index.unit))):
            raise ValueError(('coverage',part))
        n=st['end']
        if j%3==shard:
            if any(sha(part/f)!=st['hashes'][f] for f in ('FEATURES.parquet','OHLC.parquet')):
                raise ValueError(('hash',part))
            x=pd.read_parquet(part/'FEATURES.parquet')
            if (list(x)!=recipe['list']['columns'] or not x.index.equals(b.index) or
                    not np.isfinite(x.to_numpy()).all() or not np.isfinite(b.to_numpy()).all() or
                    not ((b.LOW>0)&(b.LOW<=b[['OPEN','CLOSE']].min(axis=1))&
                         (b.HIGH>=b[['OPEN','CLOSE']].max(axis=1))).all() or
                    not 0<=st['max_delta']<=1e-9):
                raise ValueError(('invalid synthetic component',part))
            rows+=len(b);blocks+=1
        frames.append(b)
    if n!=limit: raise ValueError(('incomplete snapshot',n,limit))
    prices=pd.concat(frames)
    legs=[]
    for j,s in enumerate(script.itertuples()):
        if j%3!=shard or s.start_row==0 or s.censored or s.start_row+s.bars>limit: continue
        i=int(s.start_row)
        anchor=prices.iloc[i-1]
        b=prices.iloc[i:i+s.bars]
        extremes=b.HIGH if s.side>0 else b.LOW
        last=float(extremes.iloc[-1])
        got=10000*np.log(last/(anchor.LOW if s.side>0 else anchor.HIGH))
        loc=int(np.argmax(extremes) if s.side>0 else np.argmin(extremes))
        legs.append([int(j),int(s.regime),float(abs(got-10000*s.displacement)),len(b)-1-loc])
    return dict(rows=rows,blocks=blocks,legs=legs,limit=limit,shard=shard,release=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('src',type=Path)
    parser.add_argument('shared',type=Path)
    parser.add_argument('limit',type=int)
    parser.add_argument('shard',type=int,choices=range(3))
    args=parser.parse_args()
    print(json.dumps(check(args.src,args.shared,args.limit,args.shard)))
