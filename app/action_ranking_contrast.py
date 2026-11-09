"""TRAIN-only paired-action diagnostic design and critic calibration metrics."""
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tensordict import TensorDict
from agent_plugins.sac_critic_regularization import module_mode


def replication_design(name='original'):
    base=dict(replica=name,anchor_seed=41,update_seed=7,chosen_action_seed=101,
        seeds=[7,19,43,73],shard_weights=[1,1],
        arms=['control','chosen_stochastic','paired_stochastic','paired_greedy'])
    if name=='original':return base
    variants={
        'r19':dict(anchor_seed=101,update_seed=19,chosen_action_seed=751,seeds=[101,103,107,109]),
        'r43':dict(anchor_seed=149,update_seed=43,chosen_action_seed=857,seeds=[211,223,227,229]),
    }
    if name not in variants:raise ValueError('RANKING_REPLICA')
    return base|variants[name]|dict(shard_weights=[5,2],arms=base['arms']+['paired_stochastic_reward'])


def shard_owner(index,weights):
    if (len(weights)!=2 or any(type(v) is not int or v<1 for v in weights)
            or type(index) is not int or index<0):raise ValueError('RANKING_SHARD_WEIGHTS')
    return int(index%sum(weights)>=weights[0])


def calibration_target(arm,greedy,stochastic,plain):
    if arm=='paired_greedy':return greedy
    if arm=='paired_stochastic_reward':return plain.mean(1)
    if arm in ('control','chosen_stochastic','paired_stochastic'):return stochastic.mean(1)
    raise ValueError('RANKING_ARM')


def differentiable_q(model,features):
    # The inspection q_surfaces API installs parameter views for inference.
    # Native functional execution keeps gradients attached to optimizer tensors.
    with module_mode(model.loss.qvalue_network,False):
        batch=TensorDict(dict(features=features),[len(features)])
        return model.loss._vmap_qnetworkN0(batch,model.loss.qvalue_network_params)['action_value']


def select_anchors(trace,per_stratum=32,fit_per_stratum=24,seed=41):
    if not 0<fit_per_stratum<per_stratum:raise ValueError('RANKING_ANCHOR_COUNTS')
    groups=defaultdict(list)
    timestamps=[int(r['timestamp']) for r in trace]
    if len(set(timestamps))!=len(timestamps):raise ValueError('RANKING_DUPLICATE_ROWS')
    for i,r in enumerate(trace):
        year=pd.Timestamp(r['timestamp'],tz='UTC').year
        if year not in (2020,2021,2022,2023):raise ValueError('RANKING_TRAIN_ONLY')
        if r['position'] not in (-1,0,1):raise ValueError('RANKING_POSITION')
        groups[year,r['position']].append(i)
    rng=np.random.default_rng(seed);out=[]
    for (year,position),indices in sorted(groups.items()):
        chosen=rng.permutation(indices)[:per_stratum]
        fit=min(fit_per_stratum,max(1,int(len(chosen)*fit_per_stratum/per_stratum)))
        for j,i in enumerate(chosen):
            out.append(dict(trace_index=int(i),timestamp=timestamps[i],year=year,position=position,
                role='fit' if j<fit else 'monitor'))
    return sorted(out,key=lambda x:x['trace_index'])


def target_loss(q,target,chosen=None):
    if (q.ndim!=3 or q.shape[0]!=2 or q.shape[-1]!=3 or target.shape!=q.shape[1:]
            or not torch.isfinite(q).all() or not torch.isfinite(target).all()):
        raise ValueError('RANKING_TARGET_SHAPE_OR_FINITE')
    if chosen is not None:
        if chosen.shape!=(q.shape[1],) or chosen.dtype!=torch.int64 or not ((chosen>=0)&(chosen<3)).all():
            raise ValueError('RANKING_CHOSEN_ACTION')
        ix=chosen.reshape(1,-1,1).expand(2,-1,1)
        q=q.gather(-1,ix);target=target.gather(-1,chosen[:,None])
    return F.smooth_l1_loss(q,target.unsqueeze(0).expand_as(q))


def ranking_metrics(q,target):
    q,target=np.asarray(q,dtype=float),np.asarray(target,dtype=float)
    if q.shape!=target.shape or q.ndim!=2 or q.shape[1]!=3 or not len(q) or not np.isfinite([q,target]).all():
        raise ValueError('RANKING_METRIC_INPUT')
    pairs=np.array([[0,1],[0,2],[1,2]])
    truth=target[:,pairs[:,0]]-target[:,pairs[:,1]]
    pred=q[:,pairs[:,0]]-q[:,pairs[:,1]]
    informative=np.abs(truth)>1e-8
    agreement=np.where(np.abs(pred)<=1e-8,.5,(np.sign(pred)==np.sign(truth)).astype(float))
    greedy=q.argmax(1)
    regret=target.max(1)-target[np.arange(len(q)),greedy]
    return dict(n=len(q),rmse=float(np.sqrt(np.mean((q-target)**2))),
        pairwise_comparisons=int(informative.sum()),target_ties=int((~informative).sum()),
        predicted_pair_ties=int((np.abs(pred)<=1e-8).sum()),
        pairwise_accuracy=float(agreement[informative].mean()) if informative.any() else None,
        mean_regret=float(regret.mean()),maximum_regret=float(regret.max()),
        q_greedy_counts=[int((greedy==a).sum()) for a in range(3)],
        per_action_rmse=np.sqrt(np.mean((q-target)**2,axis=0)).tolist())
