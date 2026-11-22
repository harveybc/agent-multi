"""Two selectors, identical archived candidates, original GPU for prefix parity."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
from role429_worker import model,heads


def candidates(d):
    return [('champ',d['champs'][0]),('parent',d['parent'][0])]+[
        (f'rollback-{e}',st) for e,j,st in d['rollback'] if j==0]


def weekly(p,y,week):
    values=[]
    for q in np.unique(week):
        i=np.flatnonzero(week==q)
        ix=i[np.argsort(-p[i],axis=0,kind='stable')[:max(1,len(i)//10)]]
        values.append(y[ix,np.arange(4)])
    return np.concatenate(values).ravel()


@torch.inference_mode()
def run(src,mon,out,tag):
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    path=src/f'{tag}.npz'
    with path.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    with np.load(path) as z:
        x,v,t=[torch.from_numpy(z[k]).cuda() for k in ('x','v','t')]
        sem,week,raw=z['sem'],z['week'],z['raw']
    with np.load(mon) as z:mw,my=z['week'],z['raw']
    net=model().cuda().eval();results=[]
    for s in range(1,int(sem.max())+1):
        d=torch.load(path.with_suffix(f'.s{s}.pt'),map_location='cpu',weights_only=False)
        assert d['sha']==sha
        rows=[];seen=set()
        for key,st in candidates(d):
            sig=hashlib.sha256(b''.join(st[k].numpy().tobytes() for k in sorted(st))).hexdigest()
            if sig in seen:continue
            seen.add(sig);net.load_state_dict(st)
            vals=[]
            for h in range(s):
                i=np.flatnonzero(sem==h)
                vals.append(weekly(heads(net(x[i]))[-1].cpu().numpy(),raw[i],week[i]))
            floors=[float(np.concatenate(vals[:j+1]).mean(dtype=np.float64)) for j in range(s)]
            legal=bool(np.all(np.array(floors)>=np.array(d['entry'][0])-1.e-9))
            pred=heads(net(v))[-1]
            rows.append(dict(key=key,sha=sig,legal=legal,floors=floors,mse=float((pred-t).square().mean()),
                             economic=float(weekly(pred.cpu().numpy(),my,mw).mean(dtype=np.float64))))
        assert rows[0]['legal'] and np.allclose(rows[0]['floors'],d['exit'][0][:-1],atol=1e-9,rtol=0)
        eligible=[r for r in rows if r['legal']]
        # Keep archived champion on numerical ties in the original MSE criterion.
        a=next(r for r in eligible if r['mse']<=min(e['mse'] for e in eligible)+1e-8)
        b=max(eligible,key=lambda r:(r['economic'],-r['mse']))
        results.append(dict(stage=s,A=a['key'],B=b['key'],rows=rows))
    tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(results,allow_nan=False));tmp.replace(out)


if __name__=='__main__':run(*map(Path,sys.argv[1:4]),sys.argv[4])
