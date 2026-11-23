"""Independent CPU replay. Seasonal skill, same-phase baseline and fixed price bridge."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from app.structural_refit import load_bundle
from app.causal_stl_inputs import fit_history

H=3600*10**9

@torch.no_grad()
def main(P, bundle):
    torch.set_num_threads(1)
    meta=json.loads((P/'dataset_meta.json').read_text())
    d=np.load(P/'dataset_sdelta.npz');w=np.load(P/'windows.npz');bar=np.load(P/'stl_series.npz')
    assert np.array_equal(w['ts'],d['ts'])
    t=d['ts'];win=w['windows'];bt=bar['avail_ns'];c=bar['close'].astype(float);s=bar['seasonal']
    cfg=meta['stl'];period=cfg['period'];hist=cfg['history'];cache={}
    assert np.all(np.diff(bt)>0)
    reports=[]
    for mode in ('frozen','tuned'):
        for task in ('short','long'):
            m,_=load_bundle(bundle,'cpu')
            f=P/f'retsu-phase1-sdelta-{mode}/jobs/{mode}_{task}/champion.pt'
            ck=torch.load(f,map_location='cpu',weights_only=False);rec=ck['record']
            params=dict(m.loss.named_parameters())
            for name,weights in ck['encoder']:params[name].copy_(weights)
            hs=np.array(meta['horizons_'+task]);hd=torch.nn.Linear(208,len(hs));hd.load_state_dict(ck['head'])
            enc=m.actor_net.encoder.eval();idx=np.flatnonzero(d['holdout']&d['ok_'+task])
            with m.loss.actor_network_params.to_module(m.actor):
                raw=torch.cat([hd(enc({'features':x})/rec['embedding_scale']) for x in torch.from_numpy(win[idx]).split(256)]).numpy()
            hat=raw*np.array(rec['target_std'])+np.array(rec['target_mean'])
            oi=np.searchsorted(bt,t[idx]);assert np.array_equal(bt[oi],t[idx])
            assert np.all(oi >= hist-1)
            for b in oi:
                if int(b) not in cache:
                    fit=fit_history(c[b-hist+1:b+1],cfg)
                    cache[int(b)]=fit.seasonal[-period:].copy()
            for j,h in enumerate(hs):
                fi=np.searchsorted(bt,t[idx]+h*H)
                assert np.array_equal(bt[fi],t[idx]+h*H)
                ds=s[fi]-s[oi];dc=c[fi]-c[oi]
                assert np.allclose(ds,d['y_'+task][idx,j],atol=1e-8)
                # Forecast h clock hours ahead using only the completed cycle at t.
                phase=np.array([cache[int(b)][(int(h)-1)%period] for b in oi])
                mae=lambda v:float(abs(v).mean())
                a=mae(hat[:,j]-ds);saved=rec['holdout_2024_mae_per_horizon'][str(h)]
                assert abs(a-saved)<1e-6,(mode,task,h,a,saved)
                reports.append(dict(mode=mode,h=int(h),n=len(oi),seasonal_model=a,replay_diff=a-saved,
                    seasonal_persistence=mae(ds),seasonal_phase=mae(phase-s[fi]),price_model=mae(hat[:,j]-dc),
                    price_persistence=mae(dc),clock_gaps=int(np.sum(fi-oi!=h))))
    (P/'BRIDGE.json').write_text(json.dumps(reports,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifact_dir',type=Path)
    parser.add_argument('bundle_dir',type=Path)
    args=parser.parse_args()
    main(args.artifact_dir,args.bundle_dir)
