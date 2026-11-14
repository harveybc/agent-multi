"""Nonmutating FIT gradient geometry with numerical directional witnesses."""
import numpy as np
import torch


def components(q,y):
    if (q.ndim!=3 or q.shape[0]!=2 or q.shape[1:]!=y.shape or y.shape[-1]!=3
            or not len(y) or not torch.isfinite(q).all() or not torch.isfinite(y).all()):
        raise ValueError('FIT_GRADIENT_SHAPE')
    error=q-y.unsqueeze(0);level=error.mean(-1,keepdim=True)
    return dict(total=error.square().mean(),level=level.square().mean(),gap=(error-level).square().mean())


def partition(original,expanded,n):
    a=np.asarray(original);b=np.asarray(expanded)
    if any(x.ndim!=1 or x.dtype.kind not in 'iu' or not len(x) or len(set(x))!=len(x)
           or (x<0).any() or (x>=n).any() for x in (a,b)) or not set(a)<set(b):
        raise ValueError('FIT_GRADIENT_PARTITION')
    return a,np.array([i for i in b if i not in set(a)]),b


def alignment(left,right):
    a,b=np.asarray(left,dtype=float),np.asarray(right,dtype=float)
    if a.ndim!=1 or a.shape!=b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('FIT_GRADIENT_VECTOR')
    an,bn=float(np.linalg.norm(a)),float(np.linalg.norm(b));dot=float(a@b)
    cosine=dot/(an*bn) if min(an,bn)>1e-14 else None
    return dict(left_norm=an,right_norm=bn,dot=dot,cosine=cosine,
        conflict=cosine is not None and cosine < -1e-8,
        left_derivative_descend_right_unit=-dot/bn if bn>1e-14 else None)


def inspect_gradients(head,z,y,original,added):
    old=np.asarray(original);new=np.asarray(added)
    old,_,expanded=partition(old,np.r_[old,new],len(y))
    if z.dtype!=torch.float64 or y.dtype!=torch.float64 or z.device.type!='cpu':
        raise ValueError('FIT_GRADIENT_PRECISION')
    if any(p.dtype!=torch.float64 or p.device.type!='cpu' for p in head.parameters()):
        raise ValueError('FIT_GRADIENT_MODEL_PRECISION')
    named=[(n,p) for n,p in head.named_parameters() if p.requires_grad]
    if not named:raise ValueError('FIT_GRADIENT_EMPTY_SCOPE')
    parameters=[p for _,p in named];before={k:v.detach().clone() for k,v in head.state_dict().items()}
    losses={};vectors={};indices=dict(original=old,added=new,expanded=expanded)
    for scope,ix in indices.items():
        values=components(head(z[:,ix]).mean(2),y[ix])
        losses[scope]={k:float(v.detach()) for k,v in values.items()}
        for key,value in values.items():
            gradient=torch.autograd.grad(value,parameters,retain_graph=True)
            vectors[scope+'_'+key]=torch.cat([v.flatten() for v in gradient]).detach().numpy()
    weight=len(old)/len(expanded)
    loss_error=max(abs(v['total']-v['level']-v['gap']) for v in losses.values())
    decomp=max(float(np.max(np.abs(vectors[s+'_total']-vectors[s+'_level']-vectors[s+'_gap']))) for s in indices)
    union=max(float(np.max(np.abs(vectors['expanded_'+c]-weight*vectors['original_'+c]-(1-weight)*vectors['added_'+c])))
              for c in ('total','level','gap'))
    if max(loss_error,decomp,union)>1e-10:raise ValueError('FIT_GRADIENT_IDENTITY')
    slices={};offset=0
    for name,p in named:slices[name]=[offset,offset+p.numel()];offset+=p.numel()
    groups={'all':np.arange(offset)}
    for group,layer in (('hidden','0'),('output','2')):
        ix=[i for name,(lo,hi) in slices.items() if name.split('.')[3]==layer for i in range(lo,hi)]
        if ix:groups[group]=np.array(ix)
    pairs=[('original_total','added_total'),('original_gap','added_gap'),('original_total','expanded_total'),
           ('original_gap','expanded_total'),('original_gap','expanded_level'),('expanded_gap','expanded_level')]
    geometry={g:{a+':'+b:alignment(vectors[a][ix],vectors[b][ix]) for a,b in pairs} for g,ix in groups.items()}
    witnesses=[]
    try:
        for direction in ('added_total','expanded_total','expanded_level'):
            norm=float(np.linalg.norm(vectors[direction]))
            if norm<=1e-14:continue
            unit=-vectors[direction]/norm
            for epsilon in (1e-4,1e-5):
                observed={}
                for sign in (-1,1):
                    with torch.no_grad():
                        for name,p in named:
                            lo,hi=slices[name]
                            p.copy_(before[name]+sign*epsilon*torch.from_numpy(unit[lo:hi]).reshape_as(p))
                        observed[sign]={s:{k:float(v) for k,v in components(head(z[:,ix]).mean(2),y[ix]).items()}
                                        for s,ix in indices.items() if s!='expanded'}
                for scope in ('original','added'):
                    for component in ('total','gap'):
                        analytical=float(vectors[scope+'_'+component]@unit)
                        numerical=(observed[1][scope][component]-observed[-1][scope][component])/(2*epsilon)
                        error=abs(analytical-numerical)
                        if not np.isclose(analytical,numerical,atol=1e-7,rtol=1e-4):raise ValueError('FIT_GRADIENT_DERIVATIVE')
                        witnesses.append(dict(direction=direction,scope=scope,component=component,epsilon=epsilon,
                            analytical=analytical,numerical=numerical,absolute_error=error))
    finally:
        head.load_state_dict(before)
    if any(not torch.equal(v,before[k]) for k,v in head.state_dict().items()):raise ValueError('FIT_GRADIENT_MUTATED')
    return dict(losses=losses,original_weight=weight,trainable_names=[n for n,_ in named],
        parameter_slices=slices,groups=geometry,loss_decomposition_error=loss_error,
        gradient_decomposition_error=decomp,gradient_union_error=union,finite_differences=witnesses,
        finite_difference_max_error=max((w['absolute_error'] for w in witnesses),default=0.),
        parameters_unchanged=True),vectors
