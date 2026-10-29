#!/usr/bin/env python3
"""Independent scalar reconstruction of sampled oracle records and search receipts."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.zigzag_opportunities import HOUR, PIP, engine, load_tape, sha
from tools.zigzag_opportunity_deap import authenticated, identity, write


def audit_event_choices(tape, zigzag, genes, records, future_pivots, samples):
    """Scalar enumeration independent of the vectorized endpoint selector."""
    chunks=[]
    for a,b in tape.chunks:
        pivots=zigzag.hindsight_pivots(tape.high[a:b],tape.low[a:b],zigzag.Params(*genes))
        chunks.append((a,b,[(a+i,price) for i,_,price in pivots]))
    checked=0
    for j in samples:
        i=int(records['origin'][j])
        events=next(p for a,b,p in chunks if a<=i<b)
        expected=[p for p in events if p[0]>i][:future_pivots]
        assert len(expected)==future_pivots
        assert [int(e) for e in records['candidate_end'][j]]==[e for e,_ in expected]
        for side,sign in enumerate((1,-1)):
            scores=[]
            for end,target in expected:
                top=sign*tape.open[i]; draw=0.
                for b in range(i,end+1):
                    top=max(top,sign*(tape.high[b] if side==0 else tape.low[b]))
                    draw=max(draw,top-sign*(tape.low[b] if side==0 else tape.high[b]))
                cost=tape.fill_cost[i]+tape.fill_cost[end]
                net=sign*(target-tape.open[i])-cost-(tape.financing[side,end]-tape.financing[side,i])
                duration=(tape.times[end]-tape.times[i])/HOUR+1
                scores.append(net/(max(draw,cost,PIP)*duration))
                checked+=1
            best=max(range(len(scores)),key=lambda k:scores[k])
            assert records['end'][side,j]==expected[best][0]
            assert np.isclose(records['score'][side,j],scores[best],rtol=1e-12,atol=1e-12)
    return checked


def audit(root):
    spec=json.loads((root/'SPEC.private.json').read_text())
    ident=identity(spec)
    tape=load_tape(spec)
    zigzag,_,_=engine(spec['engine_root'])
    report=authenticated(root/'RESULTS.json',ident)
    assert len(report['rows'])==4
    checks=[]
    rng=np.random.default_rng(123)
    for row in report['rows']:
        h=row['horizon']
        for seed in spec['budget']['seeds']:
            chain=authenticated(root/f'h{h}_seed{seed}.json',ident)
            assert chain['population']>=15 and chain['generations']>=30
            assert [r['generation'] for r in chain['history']]==list(range(1,chain['generations']+1))
        assert sha(root/row['records_file'])==row['records_sha256']
        r=np.load(root/row['records_file'])
        assert np.array_equal(r['origin'],np.flatnonzero(tape.allowed))
        complete=r['pair_valid']
        assert np.isfinite(r['score'][:,complete]).all()
        assert np.isnan(r['score'][:,~complete]).all()
        assert int(complete.sum())==row['complete_origins']
        scored=np.where(complete[None,:],r['score'],0)
        assert np.isclose(np.maximum(0,scored.max(axis=0)).mean(),row['fitness'],rtol=0,atol=1e-14)
        supported=np.flatnonzero(complete)
        samples=rng.choice(supported,min(256,len(supported)),replace=False)
        params=row['params']
        candidates_checked=audit_event_choices(tape,zigzag,
            (params['depth'],params['deviation_pips'],params['backstep']),r,
            spec['future_pivots'],samples)
        worst=0.
        for j in samples:
            i=int(r['origin'][j])
            for side,sign in enumerate((1,-1)):
                end=int(r['end'][side,j]); pivot=bool(r['pivot_exit'][side,j])
                assert i < end < len(tape.open)
                assert end in r['candidate_end'][j]
                assert np.all(np.diff(tape.times[i:end+1])==HOUR)
                assert tape.session[i]==tape.session[end]
                last=end if pivot else end-1
                target=float(r['target'][side,j])
                assert pivot
                assert target in (tape.high[end],tape.low[end])
                top=sign*tape.open[i]; dd=0.; adverse=0.
                for b in range(i,last+1):
                    high=sign*(tape.high[b] if side==0 else tape.low[b])
                    low=sign*(tape.low[b] if side==0 else tape.high[b])
                    top=max(top,high)
                    dd=max(dd,top-low)
                    adverse=max(adverse,sign*tape.open[i]-low)
                dd=max(dd,top-sign*target)
                adverse=max(adverse,sign*(tape.open[i]-target))
                fees=tape.fill_cost[i]+tape.fill_cost[end]
                net=sign*(target-tape.open[i])-fees-(tape.financing[side,end]-tape.financing[side,i])
                duration=(tape.times[end]-tape.times[i])/HOUR+int(pivot)
                score=net/(max(dd,fees,PIP)*duration)
                for key,val in [('drawdown',dd),('mae',adverse),('net',net),('duration',duration),('score',score)]:
                    delta=abs(float(r[key][side,j])-val)
                    worst=max(worst,delta)
                    assert np.isclose(r[key][side,j],val,rtol=1e-12,atol=1e-12),(h,i,side,key,delta)
        sch=authenticated(root/f'h{h}_INTERVAL_SCHEDULE.json',ident)
        trades=sch['trades']
        assert all(a['release']<=b['origin'] for a,b in zip(trades,trades[1:]))
        assert np.isclose(sum(t['net_pips'] for t in trades),sch['net_pips'],rtol=1e-12,atol=1e-8)
        checks.append(dict(horizon=h,scalar_paths=2*len(samples),candidate_paths=candidates_checked,max_abs_error=worst,
                           generations_verified=3*spec['budget']['generations'],schedule_nonoverlap=True))
    return dict(status='PASS',identity=ident,checks=checks,
                scope='SCALAR_RECONSTRUCTION_AND_RECEIPTS_NOT_NATIVE_EXECUTION_QUALIFICATION',
                source_sha256=sha(__file__),results_sha256=sha(root/'RESULTS.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True)
    a=p.parse_args(); result=audit(a.root)
    write(a.root/'AUDIT.json',result)
    print(json.dumps(result,indent=2))
