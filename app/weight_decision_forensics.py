"""Analytic native-head derivatives and exact finite component attribution."""
from itertools import permutations
import math
import numpy as np

BLOCKS=('twin0_hidden','twin0_output','twin1_hidden','twin1_output')


def parameters(state):
    result={k:v.detach().cpu().double().numpy() for k,v in state.items()}
    expected=[f'twins.{t}.0.{l}.{s}' for t in range(2) for l in (0,2) for s in ('weight','bias')]
    if list(result)!=expected or not all(np.isfinite(v).all() for v in result.values()):raise ValueError('FORENSICS_STATE')
    return result


def native_forward(state,z):
    arrays=parameters(state);z=np.asarray(z,dtype=float)
    if z.ndim!=3 or z.shape[0]!=2 or not np.isfinite(z).all():raise ValueError('FORENSICS_EMBEDDINGS')
    cache=[];q=[]
    for twin in range(2):
        p=f'twins.{twin}.0.'
        u=z[twin]@arrays[p+'0.weight'].T+arrays[p+'0.bias']
        sig=np.exp(-np.logaddexp(0.,-u));h=u*sig;derivative=sig*(1+u*(1-sig))
        q.append(h@arrays[p+'2.weight'].T+arrays[p+'2.bias'])
        cache.append(dict(z=z[twin],u=u,h=h,derivative=derivative,output_weight=arrays[p+'2.weight']))
    return np.stack(q),cache


def weighted_min_gradient(state,z,coefficients):
    q,cache=native_forward(state,z);c=np.asarray(coefficients,dtype=float)
    if c.shape!=q.shape[1:] or not np.isfinite(c).all():raise ValueError('FORENSICS_COEFFICIENTS')
    route=q.argmin(0);blocks=[]
    for twin,values in enumerate(cache):
        routed=c*(route==twin);du=(routed@values['output_weight'])*values['derivative']
        blocks.extend([(du[:,:,None]*values['z'][:,None,:]).reshape(len(c),-1),du,
            (routed[:,:,None]*values['h'][:,None,:]).reshape(len(c),-1),routed])
    return np.concatenate(blocks,axis=1)


def cost_parts(q,y,temperature):
    q=np.asarray(q,dtype=float);y=np.asarray(y,dtype=float)
    if (q.ndim!=3 or q.shape[0]!=2 or q.shape[2]!=3 or y.shape!=q.shape[1:]
            or not np.isfinite(q).all() or not np.isfinite(y).all()
            or not math.isfinite(temperature) or temperature<=0):raise ValueError('FORENSICS_COST')
    costs=y.max(1,keepdims=True)-y;logits=q.min(0)/temperature;logits-=logits.max(1,keepdims=True)
    p=np.exp(logits);p/=p.sum(1,keepdims=True);expected=(p*costs).sum(1)
    return dict(probabilities=p,row_cost=temperature*expected,
        logit_gradient=p*(costs-expected[:,None]),
        row_regret=costs[np.arange(len(y)),q.min(0).argmax(1)])


def cost_gradients(state,z,y,temperature):
    q,_=native_forward(state,z);parts=cost_parts(q,y,temperature)
    return weighted_min_gradient(state,z,parts['logit_gradient']),parts


def block_indices(state):
    arrays=parameters(state);out={b:[] for b in BLOCKS};offset=0
    for k,v in arrays.items():
        t=int(k.split('.')[1]);layer=int(k.split('.')[3]);name=BLOCKS[2*t+int(layer==2)]
        out[name].extend(range(offset,offset+v.size));offset+=v.size
    return {k:np.array(v,dtype=int) for k,v in out.items()}


def hybrid_state(before,after,mask):
    if type(mask) is not int or not 0<=mask<16:raise ValueError('FORENSICS_MASK')
    parameters(before);parameters(after)
    if any(before[k].shape!=after[k].shape or before[k].dtype!=after[k].dtype for k in before):raise ValueError('FORENSICS_PAIR')
    return {k:(after if mask&(1<<(2*int(k.split('.')[1])+int(k.split('.')[3]=='2'))) else before)[k].clone() for k in before}


def exact_shapley(values):
    values=np.asarray(values,dtype=float)
    if values.ndim!=2 or values.shape[0]!=16 or not np.isfinite(values).all():raise ValueError('FORENSICS_COALITIONS')
    phi=np.zeros((4,values.shape[1]))
    for b in range(4):
        for mask in range(16):
            if mask&(1<<b):continue
            n=mask.bit_count();weight=math.factorial(n)*math.factorial(3-n)/math.factorial(4)
            phi[b]+=weight*(values[mask|(1<<b)]-values[mask])
    # A second enumeration validates the combinatorial weighting independently.
    check=np.zeros_like(phi)
    for order in permutations(range(4)):
        mask=0
        for b in order:
            next_mask=mask|(1<<b);check[b]+=(values[next_mask]-values[mask])/24.;mask=next_mask
    if not np.allclose(phi,check,atol=1e-12,rtol=1e-12) or not np.allclose(phi.sum(0),values[-1]-values[0],atol=1e-12,rtol=1e-12):
        raise ValueError('FORENSICS_SHAPLEY_EFFICIENCY')
    return phi


def cosine(a,b):
    a=np.asarray(a);b=np.asarray(b);den=np.linalg.norm(a)*np.linalg.norm(b)
    return None if den==0 else float(a@b/den)


def gradient_contributions(margin_gradient,row_gradients,fit,own_row,kinds):
    g=np.asarray(row_gradients);fit=np.asarray(fit);m=np.asarray(margin_gradient)
    total=-float(m@g[fit].mean(0));own=-float(m@g[own_row])/len(fit) if own_row in set(fit) else 0.
    roles={str(k):-float(m@g[[i for i in fit if kinds[i]==k]].sum(0))/len(fit) for k in sorted(set(kinds[i] for i in fit))}
    if abs(sum(roles.values())-total)>1e-9*max(1,abs(total)):raise ValueError('FORENSICS_ROLE_SUM')
    return dict(total_descent=total,own_descent=own,others_descent=total-own,roles=roles)
