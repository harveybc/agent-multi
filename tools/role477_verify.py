"""Independent calendar and seasonal-label checks plus matched no-signal control."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from app.causal_stl_inputs import fit_history
from tools.role477_accounts import evaluate
from tools.role476_restart_audit import digest


def verify(a):
    hour=3600*10**9
    cal=np.array([[pd.Timestamp(s).value for s in pair] for pair in
                  json.loads((a.history/'CLOSURES.json').read_text())])
    end=pd.Timestamp('2024-01-01',tz='UTC')
    close=pd.concat([pd.read_parquet(a.history/'OHLC.parquet').CLOSE,
        pd.read_parquet(a.market/'bars.parquet',columns=['timestamp','BidClose'],
        filters=[('timestamp','<',end)]).set_index('timestamp').BidClose])
    bt=close.index.as_unit('ns').asi8+hour
    cfg=json.loads((a.data/'dataset_meta.json').read_text())['stl']
    ar=np.load(a.data/'stl_series.npz'); ats=ar['avail_ns']; seasonal=ar['seasonal']
    assert np.all(np.diff(bt)>0) and np.all(np.diff(ats)>0)
    evidence={}
    for year in range(2020,2024):
        d=np.load(a.root/str(year)/'batch.npz')
        t=d['ts'];idx=np.searchsorted(bt,t);np.testing.assert_array_equal(bt[idx],t)
        j=np.searchsorted(cal[:,1],t,side='right')-1;op=cal[j,1];flat=cal[j+1,0]-12*hour
        assert (t>=op+12*hour).all() and (t<flat).all() and (bt[idx-24]>=op).all()
        label_checks={};max_stl=0.
        for task,hs in [('short',range(1,7)),('long',(24,48))]:
            ix=np.flatnonzero(d['ok_'+task]); ix=ix[t[ix]+max(hs)*hour<pd.Timestamp(str(year+1),tz='UTC').value]
            origin=np.searchsorted(ats,t[ix]);np.testing.assert_array_equal(ats[origin],t[ix])
            for col,h in enumerate(hs):
                np.testing.assert_array_equal(ats[origin+h],ats[origin]+h*hour)
                error=np.abs(d['y_'+task][ix,col]-(seasonal[origin+h]-seasonal[origin]))
                assert np.max(error)<1e-8
                label_checks[h]=dict(rows=len(ix),max_abs=float(error.max()))
                eligible=ix[(idx[ix]>cfg['history'])&(pd.to_datetime(t[ix],utc=True).year==year)]
                k=eligible[len(eligible)//2]
                for row in (idx[k],idx[k]+h):
                    pos=np.searchsorted(ats,bt[row]);assert ats[pos]==bt[row]
                    fresh=fit_history(close.iloc[row-cfg['history']+1:row+1],cfg).seasonal[-1]
                    max_stl=max(max_stl,abs(fresh-float(seasonal[pos])))
        assert max_stl<1e-7
        cut=pd.Timestamp(f'{year-1}-07-01',tz='UTC').value
        clock=t[t>=cut];stats,_,_=evaluate(clock,np.zeros((len(clock),8)),year,a.history,a.market,a.bundle)
        np.testing.assert_allclose(list(stats['context'].values()),list(stats['seasonal'].values()),atol=1e-8)
        evidence[year]=dict(business_origins=len(t),gate_violations=0,labels=label_checks,
                            stl_sample_max_abs=max_stl,null_context=stats['context'])
    spec=json.loads(a.spec.read_text()); warm=spec['producer_settings']
    anchors={Path(warm['predictor_root']):a.snapshot/'predictor',
             Path(warm['technical_package']):a.snapshot/'pandas_ta',
             Path(warm['compatibility_path']):a.snapshot/'stl-producer-compat-20261111'}
    count=0
    for f,h in spec['pins'].items():
        original=Path(f)
        for parent,base in anchors.items():
            if original.is_relative_to(parent) and original.suffix=='.py':
                local=base/original.relative_to(parent)
                if digest(local)!=h:raise ValueError(f'producer identity: {local.name}')
                count+=1
    for k,local in ((spec['core_producer'],'core_selection_bundle.py'),
                    (warm['technical_generator'],'normalize_phase1b.py')):
        if digest(a.snapshot/local)!=spec['pins'][k]:raise ValueError(local)
    legacy=json.loads((a.history/'RESULT.json').read_text())
    assert digest(a.spec)==legacy['source_hashes']['spec']
    for k,v in legacy['outputs'].items():
        assert digest(a.history/k)==v
    evidence['sources']=dict(producer_spec=digest(a.spec),verified_python_files=count+2,
        history_receipt=digest(a.history/'RESULT.json'),bars=digest(a.market/'bars.parquet'),
        labels=digest(a.data/'dataset_sdelta.npz'),stl=digest(a.data/'stl_series.npz'))
    a.output.write_text(json.dumps(evidence,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('root','history','market','bundle','data','output','snapshot','spec'):p.add_argument(k,type=Path)
    verify(p.parse_args())
