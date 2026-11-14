"""Admission checks for nested real-label critic coverage contrasts."""
import numpy as np
import pandas as pd

from app.rl_subset import digest


def compact_support(support):
    n=support['old_count'];rows=support['rows'];all_old=set(range(n))
    if n<1 or len(set(rows))!=len(rows):raise ValueError('NATIVE_COVERAGE_ROWS')
    additions=set()
    for info in support['folds'].values():
        base=set(info['masks']['fit']);expanded=info['expanded']['4'];chosen=set(expanded)
        if (not base or not base<=all_old or not base<chosen or len(chosen)!=len(expanded)
                or len(chosen)!=4*len(base) or (chosen-base)&all_old
                or any(type(i) is not int or i<0 or i>=len(rows) for i in expanded)):
            raise ValueError('NATIVE_COVERAGE_SUPPORT')
        if chosen&(set(info['masks']['stop'])|set(info['masks']['audit'])):
            raise ValueError('NATIVE_COVERAGE_HELD')
        additions.update(chosen-all_old)
    indices=np.array(list(range(n))+sorted(additions),dtype=np.int64)
    remap={int(v):i for i,v in enumerate(indices)}
    folds={f:dict(masks=info['masks'],expanded=[remap[i] for i in info['expanded']['4']])
           for f,info in support['folds'].items()}
    return indices,folds


def validate_record(record):
    year=record['year'];start=pd.Timestamp(f'{year}-01-01',tz='UTC').value
    end=pd.Timestamp(f'{year+1}-01-01',tz='UTC').value
    outcomes=record['outcomes']
    if not start<=record['window_start']<=record['timestamp']<record['terminal']<end:
        raise ValueError('NATIVE_COVERAGE_TIME')
    if len(outcomes)!=3 or digest(outcomes)!=record['outcomes_sha256']:
        raise ValueError('NATIVE_COVERAGE_OUTCOMES')
    values=np.array([o['reward_return'] for o in outcomes],dtype=float)
    if not np.isfinite(values).all() or not all(o['parent_unchanged'] for o in outcomes):
        raise ValueError('NATIVE_COVERAGE_VALUES')
    return values
