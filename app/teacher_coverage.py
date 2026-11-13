"""Label-blind nested FIT expansions with deterministic calendar coverage."""
import numpy as np


def nested_support(base,candidates,quarters,factors=(4,16),seed=323):
    base=np.asarray(base,dtype=np.int64);candidates=np.asarray(candidates,dtype=np.int64)
    quarters=np.asarray(quarters)
    if (base.ndim!=1 or candidates.ndim!=1 or quarters.shape!=candidates.shape or not len(base)
            or len(set(base.tolist()))!=len(base) or len(set(candidates.tolist()))!=len(candidates)
            or set(base.tolist())&set(candidates.tolist()) or (base<0).any() or (candidates<0).any()):
        raise ValueError('COVERAGE_POOL')
    if not factors or any(type(f) is not int or f<=1 for f in factors) or list(factors)!=sorted(set(factors)):
        raise ValueError('COVERAGE_FACTORS')
    required=(max(factors)-1)*len(base)
    if len(candidates)<required:raise ValueError('COVERAGE_INSUFFICIENT_POOL')
    rng=np.random.default_rng(seed)
    banks=[rng.permutation(candidates[quarters==q]).tolist() for q in sorted(set(quarters.tolist()))]
    order=[];i=0
    while len(order)<required:
        for bank in banks:
            if i<len(bank):order.append(bank[i])
        i+=1
    return {f:np.sort(np.r_[base,order[:(f-1)*len(base)]]) for f in factors}
