"""Fixed target-blind subdivision of the already inspected312 panel."""
import numpy as np


def split_new_panel(anchors,old_timestamps,seed=314):
    stamps=[a['timestamp'] for a in anchors]
    if len(set(stamps))!=len(stamps) or set(stamps)&set(old_timestamps):raise ValueError('COVERAGE_SPLIT_OVERLAP')
    if any(a['year'] not in range(2020,2024) for a in anchors):raise ValueError('COVERAGE_SPLIT_YEAR')
    rng=np.random.default_rng(seed);fit=[];held=[]
    for year in range(2020,2024):
        for kind in sorted({a['kind'] for a in anchors}):
            ix=sorted((i for i,a in enumerate(anchors) if a['year']==year and a['kind']==kind),key=lambda i:stamps[i])
            perm=rng.permutation(ix).tolist();n=0 if year==2023 else len(ix)//2
            fit.extend(perm[:n]);held.extend(perm[n:])
    return sorted(fit),sorted(held)
