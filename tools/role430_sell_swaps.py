"""Attribute the 429 held-decile delta; run parallel CPU cells and a GPU parity lane."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from role429_campaign import OUT as SOURCE, base, common, selected, atomic, w, np, torch

OUT=Path('/var/tmp/role430-sell-swaps')


def cell(job,bank,device):
    year,seed=job
    tag=f'{year}-{seed}'
    ref=base.read(SOURCE/f'{tag}.json')
    pre=base.read(SOURCE/f'{tag}-preflight.json')
    d=torch.load(SOURCE/f'{tag}.pt',map_location='cpu',weights_only=False)
    assert d['sha']==pre['sha']
    x,t=bank
    ids,_,_,y,z=common(year,seed,t)
    ts=z['origin_ns']; ii=np.searchsorted(t,ts)
    dt=t[ids];ev=dt>=base.ns(f'{year}-07-01');dt=dt[ev];y=y[ev]
    ix=np.searchsorted(ts,dt)
    assert np.array_equal(t[ii],ts) and np.array_equal(ts[ix],dt) and dt.max()<base.ns(f'{year+1}-01-01')
    masks=[];scores=[]
    with torch.inference_mode():
        X=torch.from_numpy(x[ii]).to(device)
        for state in d['champs']:
            net=w.model().to(device);net.load_state_dict(state);net.eval()
            p=(w.heads(net(X))[-1]*pre['scale']).cpu().numpy()[ix]
            scores.append(p);masks.append(selected(p,dt))
    if device=='cuda':return scores,masks
    a,b=masks;enter=b&~a;leave=a&~b
    assert enter.sum()==leave.sum() and int(enter.sum()+leave.sum())==ref['top_changed']
    inc,out=y[enter],y[leave]
    delta=float((inc.sum()-out.sum())/a.sum())
    assert abs(delta-(ref['ranking']['bp']-ref['control']['bp']))<1e-5
    weekly=(dt//86400000000000+3)//7
    partner=w.partners([np.flatnonzero(weekly==v) for v in np.unique(weekly)],np.random.default_rng(429))
    diff=y-y[partner]
    good=[(p-p[partner])*diff>0 for p in scores]
    correct=~good[0]&good[1];damage=good[0]&~good[1]
    assert correct.sum()==ref['corrected_pairs'] and damage.sum()==ref['damaged_pairs']
    np.savez_compressed(OUT/f'{tag}-ledger.npz',origin_ns=dt,horizons=base.HS,managed_bp=y,
                        control=a,ranking=b,partner=partner,correct=correct,damage=damage)
    row=dict(n=int(enter.sum()),entrants_bp=float(inc.mean()),exits_bp=float(out.mean()),delta_bp=delta,
             corrected_abs_bp=float(abs(diff[correct]).mean()),damaged_abs_bp=float(abs(diff[damage]).mean()))
    return scores,masks,row


def main():
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    assert base.read(SOURCE/'DONE.json')['decision']=='reject_for_integration'
    assert base.file_sha(base.DEST/'market.npz')==base.read(base.DEST/'CACHE.json')['sha']
    OUT.mkdir(exist_ok=True)
    with np.load(base.DEST/'market.npz') as z:bank=z['x'],z['t']
    jobs=[(y,s) for y in (2022,2023) for s in (7,19,43)]
    rows={}
    with ThreadPoolExecutor(3) as cpu, ThreadPoolExecutor(1) as gpu:
        remote={j:gpu.submit(cell,j,bank,'cuda') for j in jobs}
        local={j:cpu.submit(cell,j,bank,'cpu') for j in jobs}
        for j in jobs:
            s,m,r=local[j].result();gs,gm=remote[j].result()
            r['gpu_mask_differences']=sum(int((a!=b).sum()) for a,b in zip(m,gm))
            r['max_device_delta_bp']=max(float(abs(a-b).max()) for a,b in zip(s,gs))
            rows[str(j)]=r
    atomic(OUT/'RESULTS.json',rows)
    print(rows)


if __name__=='__main__':main()
