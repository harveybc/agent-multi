"""Nested, outcome-blind expansion of current-policy FIT situations."""
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np

from app.all_state_refinement import START, BOUNDARY
from app.decision_attribution import fork_env, state_view


def expand_coverage(trace, anchors, multiplier=4, seed=288):
    if type(multiplier) is not int or multiplier<1:raise ValueError('COVERAGE_MULTIPLIER')
    by_time={};groups=defaultdict(list)
    fields=('timestamp','window_start','row','role','year','position','allowed','logits')
    def stratum(r):
        return (r['year'],r['position'],r['allowed'],int(np.argmax(r['logits'])) if not r['position'] else 0)
    for raw in trace:
        r={k:raw[k] for k in fields};t=r['timestamp'];z=np.asarray(r['logits'],dtype=float)
        if (t in by_time or r['role']!='fit' or not START<=t<BOUNDARY
                or datetime.fromtimestamp(t/1e9,timezone.utc).year!=r['year']
                or r['position'] not in (-1,0,1) or type(r['allowed']) is not bool
                or z.shape!=(3,) or not np.isfinite(z).all()):
            raise ValueError('COVERAGE_TRACE')
        by_time[t]=r;groups[stratum(r)].append(r)
    if len({r['timestamp'] for r in anchors})!=len(anchors):raise ValueError('COVERAGE_ANCHORS')
    selected={};counts=Counter();monitor=[]
    for r in anchors:
        if r['role']=='monitor':monitor.append(dict(r));continue
        if r['role']!='fit' or r['timestamp'] not in by_time:raise ValueError('COVERAGE_ANCHOR_MISSING')
        current=by_time[r['timestamp']]
        if any(current[k]!=r[k] for k in ('row','window_start')):raise ValueError('COVERAGE_SUPPORT')
        selected[r['timestamp']]=current;counts[stratum(current)]+=1
    if not counts or not monitor:raise ValueError('COVERAGE_ROLES')
    narrow=len(selected);rng=np.random.default_rng(seed);audit=[]
    for key,count in sorted(counts.items()):
        pool=sorted((r for r in groups[key] if r['timestamp'] not in selected),key=lambda r:r['timestamp'])
        requested=(multiplier-1)*count;take=min(requested,len(pool))
        for i in rng.permutation(len(pool))[:take]:
            row=pool[int(i)];selected[row['timestamp']]=row
        audit.append(dict(stratum=list(key),original=count,requested_additional=requested,
                          available_additional=len(pool),selected_additional=take))
    result=sorted(list(selected.values())+monitor,key=lambda r:r['timestamp'])
    return result,dict(seed=seed,multiplier=multiplier,narrow_fit=narrow,broad_fit=len(selected),
        monitor=len(monitor),shortfall=narrow*multiplier-len(selected),strata=audit,
        all_original_timestamps_retained=True,outcomes_used=False)


def current_fit_trace(make, inputs, signals, logits):
    env=make(['2020-01-01','2023-01-01']);env.reset();trace=[]
    while not env.finished:
        row=int(env.rows[env.bar]);ts=int(inputs.row_ts_ns[row])
        if inputs.valid[row]:
            pos=0 if env.engine.pos is None else env.engine.pos.direction
            opened=[];before=state_view(env)
            if not pos:
                for action in (1,2):
                    fork=fork_env(env);n=fork.engine.counts['entries'];fork.step(action)
                    opened.append(fork.engine.counts['entries']>n)
            if state_view(env)!=before:raise ValueError('COVERAGE_PARENT_MUTATED')
            trace.append(dict(row=row,timestamp=ts,window_start=int(inputs.slot_ts_ns[row,0]),
                role='fit',year=datetime.fromtimestamp(ts/1e9,timezone.utc).year,
                position=pos,allowed=bool(opened and all(opened)),logits=logits[row].tolist()))
        env.step(int(signals[row]))
    return trace
