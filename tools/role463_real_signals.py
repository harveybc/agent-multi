"""Export actual 2024 short/long champion increments, retaining UTC row keys."""
import argparse
from pathlib import Path
import numpy as np
import torch
from app.structural_refit import load_bundle


@torch.no_grad()
def export(src,bundle,out):
    torch.set_num_threads(1)
    w=np.load(src/'windows.npz');d=np.load(src/'dataset_sdelta.npz')
    assert np.array_equal(w['ts'],d['ts'])
    ix=np.flatnonzero(d['holdout']);windows=torch.from_numpy(w['windows'][ix])
    arrays={'ts':d['ts'][ix]}
    for mode in ('tuned','frozen'):
        heads=[]
        for task,hs in [('short',range(1,7)),('long',[24,48])]:
            m,_=load_bundle(bundle,'cpu')
            ck=torch.load(src/f'retsu-phase1-sdelta-{mode}/jobs/{mode}_{task}/champion.pt',
                          map_location='cpu',weights_only=False)
            rec=ck['record'];params=dict(m.loss.named_parameters())
            for n,v in ck['encoder']:params[n].copy_(v)
            head=torch.nn.Linear(208,len(hs));head.load_state_dict(ck['head']);enc=m.actor_net.encoder.eval()
            with m.loss.actor_network_params.to_module(m.actor):
                raw=torch.cat([head(enc({'features':x})/rec['embedding_scale']) for x in windows.split(256)]).numpy()
            pred=raw*np.array(rec['target_std'])+np.array(rec['target_mean'])
            ok=d['ok_'+task][ix];y=d['y_'+task][ix][ok]
            ref=[rec['holdout_2024_mae_per_horizon'][str(h)] for h in hs]
            np.testing.assert_allclose(abs(pred[ok]-y).mean(axis=0),ref,rtol=0,atol=1e-6)
            heads.append(pred)
        arrays[mode]=np.column_stack(heads)
    np.savez_compressed(out,**arrays)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for k in ('src','bundle','out'):ap.add_argument(k,type=Path)
    a=ap.parse_args();export(a.src,a.bundle,a.out)
