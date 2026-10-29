"""Archive-level witnesses: account-level preemption of requested CLOSE, cache parity, NOP-on-missing."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import *
out = {}; tot = dict(close_requests=0, preempted_SL=0, preempted_TP=0, accounts=0, max_cache_err=0., missing_entry_on_unavailable=0)
per = {}
from native_lib import CrossedWorld, execution_specs, CONFIG
avail = {}
for s in execution_specs('confirmation', 'missing', False):
    w = CrossedWorld(s, CONFIG); avail[cell_key(s)] = w.available.copy()
for q in QUALITIES:
    for n in names():
        r = replay(q, n); pre = 0; req = 0
        for wd in r['worlds']:
            tot['accounts'] += 1; tot['max_cache_err'] = max(tot['max_cache_err'], wd.get('cache_max_error', 0.))
            for t in wd['trace']:
                if t['action'] == 3:
                    req += 1
                    if t['close'] in ('SL', 'TP'): pre += 1; tot['preempted_'+t['close']] += 1
                if q == 'missing' and t['action'] in (1, 2, 3) and not avail[cell_key(wd['spec'])][t['bar']]:
                    tot['missing_entry_on_unavailable'] += 1
        tot['close_requests'] += req; per[f'{q}/{label(n)}'] = dict(close_requests=req, preempted=pre)
out['totals'] = tot; out['per_policy'] = per
dump('F2_archive_counts.json', out); print(json.dumps(tot)); print({k: v for k, v in per.items() if v['preempted']})
