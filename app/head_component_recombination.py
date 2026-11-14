"""Exact native critic-head component copies and a FIT-only diagnostic gate."""
import math

import torch


def combine_heads(hidden_source,output_source):
    shapes={f'twins.{t}.0.{layer}.{field}':shape for t in (0,1)
            for layer,fields in ((0,{'weight':(8,208),'bias':(8,)}),(2,{'weight':(3,8),'bias':(3,)}))
            for field,shape in fields.items()}
    if set(hidden_source)!=set(shapes) or set(output_source)!=set(shapes):
        raise ValueError('COMPONENT_SCHEMA')
    out={}
    for key,shape in shapes.items():
        a,b=hidden_source[key],output_source[key]
        if (not isinstance(a,torch.Tensor) or not isinstance(b,torch.Tensor)
                or tuple(a.shape)!=shape or tuple(b.shape)!=shape or a.device!=b.device
                or a.dtype!=torch.float32 or b.dtype!=torch.float32
                or not torch.isfinite(a).all() or not torch.isfinite(b).all()):
            raise ValueError('COMPONENT_TENSOR')
        out[key]=(a if key.split('.')[3]=='0' else b).detach().clone()
    return out


def fit_preservation_gate(control,candidate,tolerance=1e-8):
    values=[d.get(k,float('nan')) for d in (control,candidate) for k in ('fit','expanded_fit')]
    if not math.isfinite(tolerance) or tolerance<0 or any(not math.isfinite(v) or v<0 for v in values):
        raise ValueError('COMPONENT_FIT_METRICS')
    return (candidate['fit']<=control['fit']+tolerance
            and candidate['expanded_fit']<control['expanded_fit']-tolerance)
