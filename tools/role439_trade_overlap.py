"""Partition the exact SELL account difference by matched entry opportunity."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from role438_close_control import base, atomic, replay_pair, SRC, OUT as PRIOR

OUT = Path('/var/tmp/role439-trade-overlap')


def indexed(trades):
    by_bar = {int(t['entry_bar_abs']): t for t in trades}
    assert len(by_bar) == len(trades) and all(t['direction'] == -1 for t in trades)
    return by_bar


def cell(key):
    year,seed = key
    tag = f'{year}-{seed}'
    ref,left,right,capital = replay_pair(year,seed)
    assert ref == base.read(PRIOR/f'{tag}.json')
    m,b = indexed(left),indexed(right)
    audit = []
    for bar in sorted(m.keys() | b.keys()):
        if bar in m and bar in b:
            assert all(m[bar][k]==b[bar][k] for k in ('entry_price','direction','stop_loss','take_profit'))
            group = 'shared'
        else:
            group = 'managed_only' if bar in m else 'barriers_only'
        record = dict(entry_bar=bar,group=group)
        for name,tr in [('managed',m.get(bar)),('barriers',b.get(bar))]:
            record[name] = None if tr is None else dict(net=float(tr['net_usd']),exit_bar=int(tr['exit_bar_abs']),
                                                       reason=str(tr['close_reason']),units=float(tr['units']))
        audit.append(record)
    def net(row,arm):
        return row[arm]['net'] if row[arm] else 0.
    out = dict(year=year,seed=seed)
    for group in ('shared','managed_only','barriers_only'):
        sub = [r for r in audit if r['group']==group]
        out[group] = dict(n=len(sub),delta_pp=100*sum(net(r,'managed')-net(r,'barriers') for r in sub)/capital)
    for arm in ('managed','barriers'):
        assert abs(sum(net(r,arm) for r in audit)-ref[arm]['net_usd']) < 1e-6
        assert sum(r[arm] is not None for r in audit)==ref[arm]['trades']
    out['total_pp'] = ref['managed']['ret']-ref['barriers']['ret']
    assert abs(sum(out[g]['delta_pp'] for g in ('shared','managed_only','barriers_only'))-out['total_pp']) < 1e-8
    atomic(OUT/f'{tag}.json',audit)
    return out


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    assert base.file_sha(base.DEST/'market.npz')==base.read(SRC/'PROTOCOL.json')['market']
    with ProcessPoolExecutor(3) as pool:
        rows = list(pool.map(cell,[(y,s) for y in (2022,2023) for s in (7,19,43)]))
    atomic(OUT/'DONE.json',rows)
