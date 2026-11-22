"""Post-hoc activity-matched placebo; match counts, never use returns for matching."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import monotonic
from unittest.mock import patch

import numpy as np
from role441_shadow_reentry import OUT as PRIOR, SRC, atomic, base, prepare_replay

OUT = Path('/var/tmp/role442-activity-placebo')
RATES = (.15,.30,.45,.60,.75)
N = 32


def cell(key):
    year,seed = key
    tag = f'{year}-{seed}'
    ref = base.read(PRIOR/f'{tag}.json')
    pn,replay = prepare_replay(year,seed)
    assert replay()==ref['baseline']
    engine = pn.e
    step = engine.step
    controls = []
    clock = monotonic()
    for rep in range(N):
        # Common tape per calendar/replicate across seeds and thinning rates.
        u = np.random.default_rng([442,year,rep]).random(engine.last+1)
        block = []
        for rate in RATES:
            def guarded(bar,cmd,entry_plan=None):
                if cmd==2 and engine.pos is None and entry_plan is not None and u[bar]<rate:
                    cmd,entry_plan = 0,None
                return step(bar,cmd,entry_plan=entry_plan)
            with patch.object(engine,'step',guarded):
                ac = replay()
            block.append(dict(rate=rate,trades=ac['trades'],ret=ac['ret'],dd=ac['dd']))
        controls.append(block)
        atomic(OUT/f'{tag}-progress.json',dict(completed=rep+1,total=N,seconds=monotonic()-clock))
    assert replay()==ref['baseline']
    result = {}
    for arm in ('unconditional','overlap_only'):
        target = ref[arm]
        matched = [min(block,key=lambda a:abs(a['trades']-target['trades'])) for block in controls]
        good = [a for a in matched if abs(a['trades']-target['trades'])<=.05*target['trades']]
        result[arm] = dict(usable=len(good),target=target,median_random_ret=float(np.median([a['ret'] for a in good])) if good else None,
                           fraction_random_beaten=sum(target['ret']>a['ret'] for a in good)/len(good) if good else None)
    atomic(OUT/f'{tag}.json',dict(result=result,controls=controls))
    return dict(year=year,seed=seed,result=result)


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    assert len(base.read(PRIOR/'DONE.json'))==6
    assert base.file_sha(base.DEST/'market.npz')==base.read(SRC/'PROTOCOL.json')['market']
    with ProcessPoolExecutor(3) as pool:
        rows = list(pool.map(cell,[(y,s) for y in (2022,2023) for s in (7,19,43)]))
    atomic(OUT/'DONE.json',dict(cells=rows,replays=6*N*len(RATES),promoted=False))
