"""Paired native role labels. TRAIN only; no learned action generation."""
from collections import Counter
from functools import lru_cache
from pathlib import Path
import json
import multiprocessing as mp
import sys
import numpy as np
from tools.specialist_exposure403 import ROOT,PARENT,HOUR,assemble,environment,clone,file_sha,save

DEST = Path('/var/tmp/role404')
ROLES = ('OPEN_BUY','OPEN_SELL','CLOSE_BUY','CLOSE_SELL')
HS = (3,6,12,24)

def setup():
    global MARKET,RISK
    b=ROOT/'model/61/bundle'; d=ROOT/'history397-61-r2'
    spec=json.loads((ROOT/'out/CONTRACT.json').read_text())
    for k,p in [('model',b/'model.pt'),('data',d/'RESULT.json'),('risk',PARENT/'RISK.json'),('spec',PARENT/'SPEC.json')]:
        if file_sha(p)!=spec[k]: raise ValueError(k)
    MARKET,_=assemble(PARENT,b,d)
    RISK=json.loads((PARENT/'RISK.json').read_text())

def label(task):
    y,h=task
    dest=DEST/f'labels-{y}-{h}'
    dest.mkdir(exist_ok=True,parents=True)
    if (dest/'DONE.json').exists():
        old=json.loads((dest/'DONE.json').read_text())
        if old['code']!=file_sha(__file__) or old['sha']!=file_sha(dest/'rows.npz'): raise ValueError('changed shard')
        return
    env=environment(MARKET,RISK,h,[f'{y}-01-01',f'{y+1}-01-01'],.99)
    e=env.engine;e.reset()
    counts=Counter(); anchors={}; base=e.cfg.capital
    rows=[[] for _ in ROLES]
    for k in range(e.first+3,e.last):
        seg=e.seg_of[k]
        if seg not in anchors and env.valid[k] and e.v[k]+HOUR>=e.op[seg]+36*HOUR and e.entry_gate(k)[0]:
            anchors[seg]=k

    @lru_cache(maxsize=8192)
    def extra(anchor,cash):
        # This backend's flat entry gate depends on session and available
        # balance, not past fills. Create a flat account at its only entry tick.
        a=clone(e);a.next_decision=anchor;a.balance=cash
        a.trades.clear();a.decisions.clear();a.entries_day.clear()
        direction=int(np.sign(e.c[anchor]-e.c[anchor-3]))
        if not direction:return 0.
        env.bar=anchor
        a.step(anchor,1 if direction==1 else 2,entry_plan=env.entry_plan(direction))
        j=anchor+1
        while a.pos is not None:a.step(j,0);j+=1
        return a.balance-cash

    def terminal(cash,when,anchor):
        return cash+(extra(anchor,cash) if anchor>=when else 0.)

    for b in range(e.first,e.last):
        if b%512==0:save(dest/'PROGRESS.json',dict(done=b-e.first,total=e.last-e.first,counts=counts))
        if env.valid[b] and e.entry_gate(b)[0]:
            anc=anchors.get(e.seg_of[b],-1)
            flat=terminal(base,b+1,anc)
            for r,direction in enumerate((1,-1)):
                env.bar=b;trial=clone(e)
                info=trial.step(b,r+1,entry_plan=env.entry_plan(direction))
                if not info['entry']:continue
                counts[ROLES[r]]+=1
                k=b+1; alternatives=[]
                while trial.pos is not None:
                    if env.valid[k]:
                        p=trial.pos
                        px=e.quote_price(k,e.c[k],-direction,int(e.v[k])+HOUR)
                        scale=abs(p.entry_price-p.sl)
                        left=(e.ff[e.seg_of[k]]-int(e.v[k])-HOUR)/HOUR
                        geom=(max(0.,direction*(px-p.sl)/scale),max(0.,direction*(p.tp-px)/scale),max(0.,left))
                        other=clone(trial);out=other.step(k,3)
                        assert other.pos is None and len(other.trades)==1
                        if out['close']=='AGENT_CLOSE':
                            alternatives.append((k,geom,other.balance))
                        else:counts['priority_'+ROLES[r+2]]+=1
                    trial.step(k,0);k+=1
                assert len(trial.trades)==1 and abs(trial.balance-base-trial.trades[0]['net_usd'])<1e-7
                if k==e.last or (anc>=b and e.seg_of[anc]==e.seg_of[e.last]):
                    counts['boundary_excluded']+=1
                    continue
                end=terminal(trial.balance,k,anc)
                # Purge at session end for time splits, including reentry labels.
                limit=int(e.ff[e.seg_of[b]])
                origin=int(e.v[b])+HOUR
                rows[r].append((int(env.rows[b]),origin,limit,0,0,0,(end-flat)/base*1e4,(trial.balance-base)/base*1e4))
                for j,g,cash in alternatives:
                    now=terminal(cash,j+1,anc)
                    rows[r+2].append((int(env.rows[j]),origin,limit,*g,(now-end)/base*1e4,(cash-trial.balance)/base*1e4))
        e.step(b,0);e.decisions.clear()
    ref=Path('/var/tmp/specialist-exposure403')/f'{y}-h{h}.json'
    proof=json.loads(ref.read_text())
    for role,side in zip(ROLES[:2],('buy','sell')):
        assert counts[role]==proof['counts'][side+'_filled']
    arrays={}
    for role, group in zip(ROLES,rows):
        # Keep integer times exact; never coerce timestamps through float64.
        arrays[role+'_id']=np.asarray([v[:3] for v in group],dtype='int64')
        a=np.asarray([v[3:] for v in group],dtype='float32');arrays[role]=a
        assert len(a) and np.isfinite(a).all() and a[:,-2].min()<0<a[:,-2].max()
        ids=arrays[role+'_id']
        assert len(np.unique(ids[:,:2],axis=0))==len(a)
    tmp=dest/'writing.npz';np.savez_compressed(tmp,**arrays);tmp.replace(dest/'rows.npz')
    save(dest/'DONE.json',dict(code=file_sha(__file__),sha=file_sha(dest/'rows.npz'),
         source=proof['identity'],year=y,horizon=h,counts=counts,n={r:len(rows[i]) for i,r in enumerate(ROLES)}))

if __name__=='__main__':
    setup()
    with mp.get_context('fork').Pool(4) as pool:
        pool.map(label,[(int(y),h) for y in sys.argv[1:] for h in HS],chunksize=1)
