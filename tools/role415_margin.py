"""Change only the ordinal target/clipping into a scaled signed payoff target."""
import numpy as np
from role413_sequence import evaluate, OUT, atomic, base, PAIRS, ridge
import role413_sequence as seq


def economic(x,y,u):
    diffs=np.stack([y[:,a]-y[:,b] for a,b in PAIRS],axis=1)
    # One shared TRAIN scale: pairwise amounts remain comparable, ties remain zero.
    scale=np.median(np.abs(diffs[diffs!=0]))
    margins=np.clip(ridge(x,np.clip(diffs/scale,-1,1),u),-1,1)
    votes=np.zeros((len(u),8))
    for k,(a,b) in enumerate(PAIRS):
        votes[:,a]+=margins[:,k];votes[:,b]-=margins[:,k]
    return votes/7


def main():
    base.torch.set_num_threads(1)
    assert base.file_sha(base.DEST/'market.npz')==base.read(base.DEST/'CACHE.json')['sha']
    with np.load(base.DEST/'market.npz') as z:bank=(z['x'],z['t'],z['valid'])
    seq.ranks=economic
    result={}
    for tag,years,year in [('historical',[2020,2021],2022),('extended',[2020,2021,2022],2023)]:
        for seed in (7,19,43):
            new=evaluate(years,year,seed,bank)['arms']
            old=base.read(OUT/f'{tag}-{seed}.json')['arms']
            assert old['raw']==new['raw'] and old['nop']==new['nop']
            result[f'{tag}-{seed}']=new
            print(tag,seed,[(k,round(v['net_pct'],2),v['trades']) for k,v in new.items()])
    atomic(OUT/'MARGIN_415.json',result)


if __name__=='__main__':
    main()
