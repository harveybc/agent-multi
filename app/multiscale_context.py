"""Optional causal context transport without changing baseline eligibility."""
from dataclasses import replace

import numpy as np


def context_config(config, observations, stride):
    keys={'observations','stride_rows','branch_width','branch_blocks'}
    if (not isinstance(config,dict) or set(config)!=keys
            or any(type(config[k]) is not int or config[k]<1 for k in keys)
            or config['observations']!=observations or config['stride_rows']<=stride):
        raise ValueError('MULTISCALE_CONTEXT_CONFIG')
    return dict(config)


def pack_context(inputs, config):
    w=inputs.recipe['tensor']['window']
    c=context_config(config,w['observations'],w['stride_rows'])
    n,t,f=inputs.windows.shape
    if t!=w['observations']: raise ValueError('CONTEXT_ALREADY_PACKED')
    slots=np.arange(n)[:,None]-np.arange(c['observations']-1,-1,-1)*c['stride_rows']
    long=inputs.windows[np.maximum(slots,0),-1,:].copy()
    available=(slots[:,0]>=0)&np.isfinite(long).all(axis=(1,2))
    long[~available]=np.nan
    packed=np.concatenate((inputs.windows,long),axis=1)
    packed.flags.writeable=False
    return replace(inputs,windows=packed)
