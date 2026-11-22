"""Same 417 experiment, explicit one-hot dispatch and alternative fixed state codes."""
from pathlib import Path
import numpy as np
from role417_paired_close import base, atomic, SEEDS, checks, run

OUT=Path('/var/tmp/role418-state-encoding')
ARMS=('curve','direct','level','gray','onehot','thermometer')


def side(value):
    if value in ('CLOSE_BUY',1):return (1,0)
    if value in ('CLOSE_SELL',-1):return (0,1)
    raise ValueError('Expected an open BUY or SELL position')


def encode(arm,logit,g):
    if not np.isfinite(g).all() or (g<0).any() or (g>1).any():raise ValueError('invalid normalized state')
    q=np.minimum(15,(16*g).astype(np.int64))
    if arm=='direct':z=g
    elif arm=='level':z=q/15
    elif arm=='gray':z=((q^(q>>1))[:,:,None]>>np.arange(3,-1,-1))&1
    elif arm=='onehot':z=q[:,:,None]==np.arange(16)
    elif arm=='thermometer':z=q[:,:,None]>=np.arange(1,16)
    else:raise ValueError(arm)
    return np.column_stack((logit,z.reshape(len(g),-1)))


def tests():
    checks()
    assert side(1)==side('CLOSE_BUY')==(1,0) and side(-1)==side('CLOSE_SELL')==(0,1)
    try:side(0)
    except ValueError:pass
    else:raise AssertionError('flat routed to close')
    u=np.repeat(np.arange(16)[:,None]/16,3,axis=1);l=np.zeros((16,8))
    a=encode('gray',l,u)[:,8:12]
    assert len(np.unique(a,axis=0))==16 and np.all(np.abs(np.diff(a,axis=0)).sum(1)==1)
    for name,n in [('gray',20),('onehot',56),('thermometer',53),('level',11)]:
        z=encode(name,l,u)
        assert z.shape==(16,n) and np.array_equal(z[-1:],encode(name,l[:1],np.ones((1,3))))
    assert (encode('onehot',l,u)[:,8:].reshape(16,3,16).sum(2)==1).all()
    np.testing.assert_array_equal(encode('thermometer',l,u)[:,8:23].sum(1),np.arange(16))


def main():
    tests();base.torch.set_num_threads(1);OUT.mkdir(exist_ok=True)
    assert base.file_sha(base.DEST/'market.npz')==base.read(base.DEST/'CACHE.json')['sha']
    with np.load(base.DEST/'market.npz') as z:bank=z['x'],z['t'],z['valid']
    results=[]
    for seed in SEEDS:
        row=run(seed,bank,arms=ARMS,encode=encode,out=OUT,route=side)
        ref=base.read(Path('/var/tmp/role417-paired-close')/f'seed-{seed}.json')
        for arm in ('curve','direct'):
            assert row['account'][arm]==ref['account'][arm]
            assert abs(row['same_entry_bp'][arm]-ref['same_entry_bp'][arm])<1e-9
        atomic(OUT/f'seed-{seed}.json',row)
        results.append(row)
        print(seed,row['same_entry_bp'],flush=True)
    def passed(r,arm):
        cells=list(r['cells'].values())
        def loss(k):return np.average([c['regret'][k] for c in cells],weights=[c['conflict_ticks'] for c in cells])
        return (loss(arm)<min(loss('curve'),loss(arm+'_swap')) and
                r['same_entry_bp'][arm]>r['same_entry_bp']['curve'] and
                r['account'][arm]['ret']>r['account']['curve']['ret'])
    atomic(OUT/'DONE.json',dict(candidates=[arm for arm in ARMS[2:] if all(passed(r,arm) for r in results)],promoted=False))


if __name__=='__main__':main()
