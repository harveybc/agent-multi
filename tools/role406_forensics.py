"""Post-hoc, read-only 406 matched weight/role interventions; run on CPU."""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy_with_logits as bce

from tools.role404_account import run
from tools.role404_data import DEST, HS, ROLES, file_sha
from tools.role404_fit import Expert

ROOT = Path('/var/tmp/role406-cls4-paired')
SEEDS = (7,19,43)
read = lambda p: json.loads(p.read_text())


def monitor(role):
    parts=[]
    cut=int(np.datetime64('2023-07-01','ns').astype('int64'))
    for i,h in enumerate(HS):
        d=DEST/f'labels-2023-{h}'
        assert file_sha(d/'rows.npz')==read(d/'DONE.json')['sha']
        with np.load(d/'rows.npz') as f:
            ids=f[role+'_id'];keep=ids[:,1]>=cut;v=f[role][keep]
            parts.append((ids[keep,0],np.full(len(v),i),v[:,:3],v[:,3]>0))
    return tuple(torch.from_numpy(np.concatenate(a)) for a in zip(*parts))


def disp(weights,base,prefix):
    # Each layer includes its biases. No alignment across unrelated seeds.
    a=torch.cat([v.flatten().double() for k,v in base.items() if k.startswith(prefix)])
    b=torch.cat([weights[k].flatten().double() for k in base if k.startswith(prefix)])
    return float((b-a).norm()/a.norm())


@torch.inference_mode()
def probe(role,seed,data,x):
    key=f'cls-{role}-4-{seed}';a=ROOT/'champion'/key/'anchor.pt'
    w={};r={}
    for arm in ('fine','explore'):
        d=ROOT/arm/key;r[arm]=read(d/'RESULT.json')
        assert file_sha(a)==r[arm]['anchor'] and file_sha(d/'expert.pt')==r[arm]['weights']
        w[arm]=torch.load(d/'expert.pt',map_location='cpu',weights_only=True)
    base=torch.load(a,map_location='cpu',weights_only=True)['model']
    ids,h,g,y=data
    model=Expert(208,role.startswith('CLOSE'),'cls').eval()
    def scores(weights):
        model.load_state_dict(weights)
        return torch.cat([model(x[ids[k:k+8192]],h[k:k+8192],g[k:k+8192])[:,0]
                          for k in range(0,len(ids),8192)])
    def loss(v):return float(bce(v,y.float()))
    layers=('net.0.','net.2.')
    p,q=[scores(w[t]) for t in ('fine','explore')]
    lp,lq=loss(p),loss(q)
    for arm,l in [('fine',lp),('explore',lq)]:
        assert abs(l-r[arm]['metrics'][0])<2e-6
    assert int((((p>0)!=y)&((q>0)==y)).sum())==r['fine']['explorer_corrected']
    hybrid=[loss(scores({k:(w['explore'] if k.startswith(layer) else w['fine'])[k]
                         for k in base})) for layer in layers]
    return dict(role=role,seed=seed,n=len(ids),flips=float(((p>0)!=(q>0)).float().mean()),
        delta_loss=lq-lp,delta_loss_hidden=.5*((hybrid[0]-lp)+(lq-hybrid[1])),
        delta_loss_output=.5*((hybrid[1]-lp)+(lq-hybrid[0])),
        displacement={arm:[disp(w[arm],base,l) for l in layers] for arm in w})


def main(out):
    torch.set_num_threads(1)
    assert read(ROOT/'DONE.json')['complete_pairs']==3
    assert file_sha(DEST/'market.npz')==read(DEST/'CACHE.json')['sha']
    with np.load(DEST/'market.npz') as f:x=torch.from_numpy(f['x'])
    heads=[]
    for role in ROLES:
        m=monitor(role)
        heads.extend(probe(role,s,m,x) for s in SEEDS)
    del x
    original=read(ROOT/'COMPARISON.json'); accts={}
    for opening,closing in [('fine','fine'),('explore','fine'),('fine','explore'),('explore','explore')]:
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for seed in SEEDS:
                for role in ROLES:
                    name=f'cls-{role}-4-{seed}'
                    (root/name).symlink_to(ROOT/(opening if role.startswith('OPEN') else closing)/name)
            run(root,[('cls',4,s) for s in SEEDS])
            group=[read(root/'accounts'/f'cls-4-{s}.json') for s in SEEDS]
            if opening==closing:
                for s,a in zip(SEEDS,group):assert a==original[str(s)]['accounts'][opening]
            accts[f'{opening}/{closing}']=group
    out.write_text(json.dumps(dict(heads=heads,accounts=accts,promoted=False),indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('out',type=Path)
    main(parser.parse_args().out)
