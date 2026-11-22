"""TRAIN-only native HOLD-to-settlement exposure census; no actor or learner."""
import argparse
from collections import Counter
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

from app.frozen_recipe_inputs import file_sha
from app.sequential_account_pareto import clone
from app.three_signal_learning import environment
from tools.account397 import assemble

ROOT = Path('/var/tmp/sac399-series')
PARENT = Path('/var/tmp/sac70-warm-v3-20261108')
OUT = Path('/var/tmp/specialist-exposure403')
HOUR = 3600 * 10**9


def save(path, data):
    tmp = path.with_suffix('.partial')
    tmp.write_text(json.dumps(data, sort_keys=True) + '\n')
    tmp.replace(path)


def census(task):
    year, horizon = task
    name = f'{year}-h{horizon}'
    env = environment(MARKET, RISK, horizon, [f'{year}-01-01', f'{year+1}-01-01'], .99)
    e = env.engine
    e.reset()
    total = Counter()
    start = time.monotonic()
    base = dict(year=year, horizon=horizon, identity=IDENTITY, reference='always_flat_then_native_hold',
                population='independent entry origins, one annual engine interval', training=False)
    for b in range(e.first, e.last):
        if b % 512 == 0:
            save(OUT / (name+'.progress'), dict(base, bars_done=b-e.first, bars_total=e.last-e.first,
                 counts=total, seconds=time.monotonic()-start))
        if env.valid[b]:
            total['market_ticks'] += 1
            if e.entry_gate(b)[0]:
                total['eligible_ticks'] += 1
                for side, cmd, direction in [('buy',1,1),('sell',2,-1)]:
                    env.bar = b
                    trial = clone(e)
                    info = trial.step(b, cmd, entry_plan=env.entry_plan(direction))
                    if not info.get('entry'):
                        total[side+'_refused'] += 1
                        continue
                    total[side+'_filled'] += 1
                    k = b+1
                    valid_count = 0
                    while trial.pos is not None and k < e.last:
                        if env.valid[k]:
                            valid_count += 1
                            age = (int(e.v[k])+HOUR-int(e.v[b+1])) / HOUR
                            band = 'lt6' if age < 6 else '6to11' if age < 12 else '12to23' if age < 24 else 'ge24'
                            total[side+'_position_ticks_'+band] += 1
                        info = trial.step(k, 0)
                        k += 1
                    assert trial.pos is None and len(trial.trades) == 1
                    total[side+'_settled_'+str(info['close'])] += 1
                    if k == e.last:
                        total[side+'_settled_at_interval_end'] += 1
                    if valid_count == 0:
                        total[side+'_no_close_decision'] += 1
        e.step(b, 0)
        # The flat prefix's old decision log is not state for fills or gates.
        e.decisions.clear()
        assert e.pos is None and not e.trades and e.balance == e.cfg.capital
    for side in ('buy','sell'):
        assert total[side+'_filled'] + total[side+'_refused'] == total['eligible_ticks']
    save(OUT / (name+'.json'), dict(base, counts=total, seconds=time.monotonic()-start))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('years', nargs='+', type=int, choices=range(2016,2024))
    args = p.parse_args()
    bundle = ROOT/'model/61/bundle'
    data = ROOT/'history397-61-r2'
    spec = json.loads((ROOT/'out/CONTRACT.json').read_text())
    for key, path in [('model', bundle/'model.pt'), ('risk',PARENT/'RISK.json'),
                      ('spec',PARENT/'SPEC.json'), ('data', data/'RESULT.json')]:
        assert file_sha(path) == spec[key]
    RISK = json.loads((PARENT/'RISK.json').read_text())
    MARKET, _ = assemble(PARENT, bundle, data)
    IDENTITY = {k:spec[k] for k in ('spec','risk','data','model')} | {'code':file_sha(__file__)}
    OUT.mkdir(exist_ok=True)
    tasks = []
    for y in args.years:
        for h in sorted({r['horizon'] for r in RISK['rows']}):
            dest = OUT/f'{y}-h{h}.json'
            if dest.exists():
                assert json.loads(dest.read_text())['identity'] == IDENTITY
            else:
                tasks.append((y,h))
    with mp.get_context('fork').Pool(min(4,len(os.sched_getaffinity(0)))) as workers:
        workers.map(census, tasks, chunksize=1)
