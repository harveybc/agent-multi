"""Descriptive FIT-only geometry; never production preprocessing."""
import numpy as np
import torch

from app.head_spillover import spectrum


def encoder_identity(state):
    actor={k.split('.encoder.',1)[1]:v for k,v in state.items()
           if k.startswith('actor_network_params.') and '.encoder.' in k}
    critic={k.split('.encoder.',1)[1]:v for k,v in state.items()
            if k.startswith('qvalue_network_params.') and '.encoder.' in k}
    if not actor or actor.keys()!=critic.keys():raise ValueError('GEOMETRY_ENCODER_KEYS')
    result=[]
    for twin in (0,1):
        if any(critic[k].shape!=(2,*v.shape) for k,v in actor.items()):
            raise ValueError('GEOMETRY_ENCODER_SHAPE')
        result.append(dict(twin=twin,tensors=len(actor),
            identical_tensors=sum(torch.equal(v,critic[k][twin]) for k,v in actor.items()),
            maximum_float_difference=max(float((v-critic[k][twin]).abs().max())
                for k,v in actor.items() if v.is_floating_point()),
            storage_aliases=sum(v.untyped_storage().data_ptr()==critic[k].untyped_storage().data_ptr()
                for k,v in actor.items())))
    return result


def geometry(fit,query,standardize):
    x,q=np.asarray(fit,float),np.asarray(query,float)
    if (x.ndim!=2 or q.ndim!=2 or len(x)<2 or not len(q) or x.shape[1]!=q.shape[1]
            or not np.isfinite(x).all() or not np.isfinite(q).all()):
        raise ValueError('GEOMETRY_INPUT')
    center=x.mean(0);scale=x.std(0);active=np.ptp(x,axis=0)>0
    safe=np.where(active,scale,1.)
    a,b=x-center,q-center
    if standardize:a,b=a/safe,b/safe
    # Dimensions constant in FIT have no learned distance scale. Preserve and
    # report their query drift separately, not as a manufactured huge distance.
    a,b=a[:,active],b[:,active]
    if not a.shape[1]:a=np.zeros((len(x),1));b=np.zeros((len(q),1))
    return a,b,dict(center=center.tolist(),scale=scale.tolist(),active=active.tolist(),
        constant_fit_dimensions=int((~active).sum()),
        query_rows_with_constant_dimension_drift=int(np.any(q[:,~active]!=x[0,~active],axis=1).sum()),
        unique_fit_rows=int(len(np.unique(x,axis=0))),spectrum=spectrum(a))


def distances(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    # Small FIT matrices; bounded query batches avoid a full 3D broadcast.
    if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[1] or not a.shape[1]:
        raise ValueError('GEOMETRY_DISTANCE_SHAPE')
    out=[]
    for i in range(0,len(b),128):
        delta=b[i:i+128,None,:]-a[None,:,:]
        out.append(np.sqrt(np.mean(delta*delta,axis=-1)))
    return np.concatenate(out)


def nearest_values(distance,targets,k):
    d,y=np.asarray(distance,float),np.asarray(targets,float)
    if (d.ndim!=2 or y.shape!=(d.shape[1],3) or not 1<=k<=len(y)
            or not np.isfinite(d).all() or not np.isfinite(y).all() or (d<0).any()):
        raise ValueError('GEOMETRY_NEIGHBOR_INPUT')
    ids=np.argsort(d,axis=1,kind='stable')[:,:k]
    return y[ids].mean(1),ids


def label_energy(targets):
    y=np.asarray(targets,float)
    if y.ndim!=2 or y.shape[1]!=3 or len(y)<2 or not np.isfinite(y).all():
        raise ValueError('GEOMETRY_LABELS')
    levels=y.mean(1);gaps=y-levels[:,None]
    level_energy=float(3*np.sum((levels-levels.mean())**2))
    gap_energy=float(np.sum((gaps-gaps.mean(0))**2));total=level_energy+gap_energy
    optimal=np.abs(y-y.max(1)[:,None])<=1e-8
    return dict(common_value_energy_share=level_energy/total if total else None,
        action_difference_energy_share=gap_energy/total if total else None,
        all_actions_tied=int(optimal.all(1).sum()),optimal_including_ties=optimal.sum(0).tolist(),
        unambiguous_optimal_counts=[int((optimal[:,i]&(optimal.sum(1)==1)).sum()) for i in range(3)],
        total_centered_energy=total)
