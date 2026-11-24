"""Development-only calibration of predicted seasonal increments to bid-close changes."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def year(y):
    return pd.Timestamp(str(y), tz='UTC').value


def export(a):
    import torch
    from app.structural_refit import load_bundle
    torch.set_num_threads(2)
    d=np.load(a.src/'dataset_sdelta.npz')
    w=np.load(a.src/'windows.npz')
    np.testing.assert_array_equal(w['ts'],d['ts'])
    mask=(d['ts']>=year(2019)) & (d['ts']<year(2024))
    ts=d['ts'][mask]
    x=torch.from_numpy(w['windows'][mask])
    heads=[]
    with torch.no_grad():
        for task,n in [('short',6),('long',2)]:
            model,_=load_bundle(a.bundle,'cpu')
            ck=torch.load(a.src/f'retsu-phase1-sdelta-tuned/jobs/tuned_{task}/champion.pt',
                          map_location='cpu',weights_only=False)
            names=dict(model.loss.named_parameters())
            for name,value in ck['encoder']:
                names[name].copy_(value)
            head=torch.nn.Linear(208,n)
            head.load_state_dict(ck['head'])
            rec=ck['record']
            enc=model.actor_net.encoder.eval()
            with model.loss.actor_network_params.to_module(model.actor):
                z=torch.cat([head(enc({'features':v})/rec['embedding_scale']) for v in x.split(256)]).numpy()
            hat=z*np.array(rec['target_std'])+np.array(rec['target_mean'])
            valid=(d['monitor'] & d['ok_'+task])[mask]
            ref=np.abs(hat[valid]-d['y_'+task][mask][valid]).mean(axis=0).mean()
            np.testing.assert_allclose(ref,rec['best_val_mae'],atol=1e-6,rtol=0)
            heads.append(hat)
    np.savez_compressed(a.signals,ts=ts,x=np.column_stack(heads))


def evaluate(a):
    from tools.role461_seasonal_account import H,NS,load,cost_arrays,simulate,entry
    d=np.load(a.signals)
    ts,x=d['ts'],d['x']
    assert np.isfinite(x).all() and (np.diff(ts)>0).all() and year(2019)<=ts.min()<ts.max()<year(2024)
    s=np.load(a.src/'stl_series.npz')
    times,close=s['avail_ns'],s['close'].astype(float)
    idx=np.searchsorted(times,ts)
    np.testing.assert_array_equal(times[idx],ts)
    train=ts<year(2023)
    y=np.full_like(x,np.nan)
    for j,h in enumerate(H):
        # Require an uninterrupted run of h hours and a destination inside its split.
        end=idx+h
        m=end<len(times)
        m[m]=((times[end[m]]==ts[m]+h*NS) &
              (times[end[m]]<np.where(train[m],year(2023),year(2024))))
        y[m,j]=close[end[m]]-close[idx[m]]
    fit=train[:,None] & np.isfinite(y)
    def slope(m):
        xx=x[m]
        assert xx.size and (xx@xx)>0
        return float((xx@y[m])/(xx@xx))
    global_beta=slope(fit)
    beta=np.zeros((2,len(H))); counts=np.zeros_like(beta,dtype=int)
    for j in range(len(H)):
        for side in (0,1):
            m=np.zeros_like(fit)
            m[:,j]=fit[:,j] & ((x[:,j]>=0)==bool(side))
            counts[side,j]=m.sum()
            beta[side,j]=slope(m)
    cases={'direct':x,'global_scale':global_beta*x,
           'by_sign_horizon':x*np.where(x>=0,beta[1],beta[0])}
    p=json.loads(a.policy.read_text())['policy']
    prof=json.loads((a.bars/'RECIPE.json').read_text())['business']['evaluation_and_policy']['profile']
    data=list(load(a.src,a.bars,p))
    t,atr=data[0],data[5]
    spr,roll=cost_arrays(t,json.loads((a.bars/'closures.json').read_text()),prof)
    v=~train
    oi=np.searchsorted(t+NS,ts[v])
    np.testing.assert_array_equal(t[oi]+NS,ts[v])
    report={}; errors=[]; opportunity=[]; books=[]; diagnostics=[]
    probe={**p,'min_rr':0.}  # Observe sub-threshold candidates; never execute them.
    a.out.mkdir(exist_ok=True,parents=True)
    for name,amount in cases.items():
        for split,m in [('train',train),('development',v)]:
            for j,h in enumerate(H):
                m0=m & np.isfinite(y[:,j])
                errors.append(dict(arm=name,split=split,h=h,n=int(m0.sum()),
                                   mae=float(np.abs(y[m0,j]-amount[m0,j]).mean()),
                                   persistence_mae=float(np.abs(y[m0,j]).mean())))
        data[6]=np.full_like(data[6],np.nan)
        paths=amount[v]
        origins=ts[v]
        data[6][oi]=paths
        metrics,book=simulate(data,(None,None),p,2023,'bridge','sl_only',(prof,spr,roll))
        report[name]=metrics
        for r in book:
            books.append(dict(arm=name,**r))
        # Same flat-state opportunity set independent of position occupancy and subsequent results.
        for j,row in enumerate(oi):
            if not data[8][row+1] or t[row+1]-t[row]!=NS: continue
            choice=entry(paths[j],paths[j],atr[row],None,p)
            opportunity.append(dict(arm=name,ts=int(origins[j]),action=choice[0] if choice else 0))
            proposed=entry(paths[j],paths[j],atr[row],None,probe)
            rr=proposed[1]/proposed[2] if proposed else 0.
            assert bool(choice)==(rr>p['min_rr'])
            diagnostics.append(dict(arm=name,ts=int(origins[j]),prethreshold_rr=rr))
    (a.out/'report.json').write_text(json.dumps(dict(global_beta=global_beta,beta=beta.tolist(),
        fit_counts=counts.tolist(),accounts=report),indent=2)+'\n')
    pd.DataFrame(errors).to_csv(a.out/'errors.csv',index=False)
    pd.DataFrame(books).to_csv(a.out/'books.csv.gz',index=False)
    pd.DataFrame(opportunity).pivot(index='ts',columns='arm',values='action').to_csv(a.out/'opportunities.csv.gz')
    pd.DataFrame(diagnostics).groupby('arm').prethreshold_rr.quantile([.5,.9,.99,1]).to_csv(a.out/'proposed_rr.csv')


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='op',required=True)
    for op,keys in [('export',('src','bundle','signals')),('evaluate',('src','bars','policy','signals','out'))]:
        parser=sub.add_parser(op)
        for k in keys: parser.add_argument(k,type=Path)
    a=p.parse_args()
    (export if a.op=='export' else evaluate)(a)
