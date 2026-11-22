"""Attribute exclusive SELL entries, then replay a causal one-off reentry veto."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

from role438_close_control import OUT as CONTROL, SRC, atomic, base, prepare_replay
from role439_trade_overlap import OUT as LEDGER

OUT = Path('/var/tmp/role440-reentry')


def classify(rows):
    groups = {k: [] for k in ('first','later','other')}
    end = -1
    for row in rows:
        bar = row['entry_bar']
        if row['group']=='shared':
            m,b = row['managed'],row['barriers']
            end = b['exit_bar'] if m['reason']=='AGENT_CLOSE' and m['exit_bar']<b['exit_bar'] else -1
            count = 0
        elif row['group']=='managed_only':
            # Native fills are at the next bar. A new fill on the passive exit
            # bar was not available for entry at the preceding decision.
            if bar<=end:
                group = 'first' if count==0 else 'later'
                count += 1
            else:
                group = 'other'
            groups[group].append(row['managed']['net'])
    return {k:dict(n=len(v),net_usd=sum(v)) for k,v in groups.items()}


def cell(key):
    year,seed = key
    tag = f'{year}-{seed}'
    attribution = classify(base.read(LEDGER/f'{tag}.json'))
    pn,replay = prepare_replay(year,seed)
    baseline = replay()
    assert baseline==base.read(CONTROL/f'{tag}.json')['managed']
    ref, = [row['managed_only'] for row in base.read(LEDGER/'DONE.json')
            if (row['year'],row['seed'])==key]
    assert sum(row['n'] for row in attribution.values())==ref['n']
    cap = pn.e.cfg.capital
    assert abs(sum(row['net_usd'] for row in attribution.values())/cap*100-ref['delta_pp'])<1e-8

    step = pn.e.step
    armed = None
    vetoes = []

    def guarded(bar,cmd,entry_plan=None):
        nonlocal armed
        segment = int(pn.e.seg_of[bar])
        if armed is not None and armed[0]!=segment:
            armed = None
        if armed is not None and cmd==2 and pn.e.pos is None and entry_plan is not None:
            assert armed[1]<bar and pn.e.entry_gate(bar)[0]
            vetoes.append([armed[1],int(bar)])
            armed = None
            cmd,entry_plan = 0,None
        info = step(bar,cmd,entry_plan=entry_plan)
        if info.get('close')=='AGENT_CLOSE':
            assert pn.e.pos is None
            armed = segment,int(bar)
        return info

    with patch.object(pn.e,'step',guarded):
        guarded_account = replay()
    assert len(vetoes)>0 and len({c for c,b in vetoes})==len(vetoes)
    assert replay()==baseline
    out = dict(year=year,seed=seed,capital=cap,groups=attribution,vetoes=len(vetoes),
               baseline=baseline,guarded=guarded_account)
    atomic(OUT/f'{tag}.json',dict(**out,veto_bars=vetoes))
    print(tag,'verified',flush=True)
    return out


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    assert base.file_sha(base.DEST/'market.npz')==base.read(SRC/'PROTOCOL.json')['market']
    with ProcessPoolExecutor(3) as pool:
        rows = list(pool.map(cell,[(y,s) for y in (2022,2023) for s in (7,19,43)]))
    atomic(OUT/'DONE.json',rows)
