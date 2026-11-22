"""Veto only the first eligible SELL still overlapping the passive shadow."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

from role440_reentry import OUT as PRIOR, SRC, atomic, base, prepare_replay

OUT = Path('/var/tmp/role441-shadow-reentry')


def cell(key):
    year,seed = key
    tag = f'{year}-{seed}'
    ref = base.read(PRIOR/f'{tag}.json')
    pn,replay = prepare_replay(year,seed)
    assert replay()==ref['baseline']
    engine = pn.e
    step = engine.step
    native = type(engine).step
    shadow = None
    vetoes = []

    def guarded(bar,cmd,entry_plan=None):
        nonlocal shadow
        if shadow is not None:
            assert shadow.pos is not None and engine.pos is None
            if cmd==2 and entry_plan is not None:
                assert engine.entry_gate(bar)[0]
                vetoes.append(int(bar))
                cmd,entry_plan = 0,None
                shadow = None
            else:
                native(shadow,bar,0)
                if shadow.pos is None:
                    shadow = None
        alternate = base.clone(engine) if engine.pos is not None and cmd==3 else None
        info = step(bar,cmd,entry_plan=entry_plan)
        if info.get('close')=='AGENT_CLOSE':
            assert alternate is not None and engine.pos is None
            native(alternate,bar,0)
            shadow = alternate if alternate.pos is not None else None
        return info

    with patch.object(engine,'step',guarded):
        account = replay()
    assert vetoes and len(set(vetoes))==len(vetoes) and replay()==ref['baseline']
    row = dict(year=year,seed=seed,vetoes=len(vetoes),baseline=ref['baseline'],
               unconditional=ref['guarded'],overlap_only=account)
    atomic(OUT/f'{tag}.json',dict(**row,veto_bars=vetoes))
    return row


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    assert len(base.read(PRIOR/'DONE.json'))==6
    assert base.file_sha(base.DEST/'market.npz')==base.read(SRC/'PROTOCOL.json')['market']
    with ProcessPoolExecutor(3) as pool:
        rows = list(pool.map(cell,[(y,s) for y in (2022,2023) for s in (7,19,43)]))
    atomic(OUT/'DONE.json',rows)
