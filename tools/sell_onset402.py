"""Read-only TRAIN probe: pair actor/Q margins and actual collector counts per epoch."""
from pathlib import Path
import json
import numpy as np
import torch
from tools import pilot391 as p
from tools.account397 import assemble

def main():
    p.configure(7)
    root=Path('/var/tmp/sac399-series')
    bundle=root/'model/61/bundle'
    model,_=p.load_bundle(bundle,'cpu')
    train,_=assemble(Path('/var/tmp/sac70-warm-v3-20261108'),bundle,root/'history397-61-r2')
    valid=np.flatnonzero(train.inputs.valid)
    ix=valid[np.linspace(0,len(valid)-1,1024,dtype=int)]
    out=[]
    for arm in ('fixed','adaptive'):
        for path in sorted((Path('/var/tmp/alpha401')/arm/'snapshots').glob('epoch_*')):
            p.verify_files(path,p.read(path/'MANIFEST.json')['files'])
            state=p.read(path/'STATE.json')
            r=state['history'][-1]
            buf=torch.load(path/'learning.pt',map_location='cpu',weights_only=False)['replay']
            d=buf['_storage']
            n=d['_len']; cursor=buf['_writer']['_cursor']; steps=r['decisions']
            assert n>=steps and state['steps']==buf['_writer']['_write_count']
            # The final traversal is contiguous in logical time even when the ring wrapped.
            act=d['_storage']['action'][(cursor-steps+np.arange(steps))%n].reshape(-1)
            assert len(act)==48629
            model.load(path/'model.pt')
            logits,q=p.emit(model,train.inputs,ix,twins=True)
            q=q[:,ix].min(0); probs=torch.tensor(logits[ix]).softmax(-1).numpy()
            assert np.isfinite(q).all() and np.isfinite(probs).all()
            out.append(dict(arm=arm,epoch=state['epoch'],collector=act.bincount(minlength=3).tolist(),
                        stored=d['_storage']['action'][:n].flatten().bincount(minlength=3).tolist(),
                        actor_sell_mean=float(probs[:,2].mean()),critic_sell_top=int((q.argmax(-1)==2).sum()),
                        median_sell_adv=float(np.median(q[:,2]-q[:,:2].max(-1))),probe_rows=len(ix)))
    print(json.dumps(out))

if __name__=='__main__': main()
