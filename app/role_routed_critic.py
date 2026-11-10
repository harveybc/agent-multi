"""Market-only value banks with explicit external native-role selection."""
from copy import deepcopy

import numpy as np
import torch
from torch import nn

from app.decision_attribution import fork_env, state_view


class RoleHeads(nn.Module):
    def __init__(self, template, count, seed):
        super().__init__()
        if count not in (1,3):raise ValueError('ROLE_HEAD_COUNT')
        device=next(template.parameters()).device
        devices=[device.index or 0] if device.type=='cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            self.twins=nn.ModuleList()
            for _ in range(2):
                heads=nn.ModuleList()
                for _ in range(count):
                    head=deepcopy(template)
                    for m in head.modules():
                        if isinstance(m,nn.Linear):m.reset_parameters()
                    head.requires_grad_(True);heads.append(head)
                self.twins.append(heads)

    def forward(self, embeddings):
        if embeddings.ndim!=3 or embeddings.shape[0]!=2:
            raise ValueError('ROLE_EMBEDDING_SHAPE')
        return torch.stack([torch.stack([h(embeddings[t]) for h in heads],dim=1)
                            for t,heads in enumerate(self.twins)])


def select_role(outputs, roles):
    if (outputs.ndim!=4 or outputs.shape[0]!=2 or outputs.shape[2:]!=(3,3)
            or roles.shape!=(outputs.shape[1],) or roles.dtype!=torch.long
            or not ((roles>=0)&(roles<3)).all()):
        raise ValueError('ROLE_CONTROLLER_INPUT')
    return outputs.gather(2,roles[None,:,None,None].expand(2,-1,1,3)).squeeze(2)


def matched_schedule(banks,updates,batch,seed):
    if not banks or any(not len(b) for b in banks):raise ValueError('ROLE_EMPTY_BANK')
    rng=np.random.default_rng(seed)
    count=updates*batch
    order=np.concatenate([rng.permutation(len(banks)) for _ in range((count+len(banks)-1)//len(banks))])[:count]
    return np.array([rng.choice(banks[int(b)]) for b in order],dtype=np.int64).reshape(updates,batch)


def delayed_histories(snapshot, lags=(4,12), delays=(0,1,3)):
    if snapshot.finished or snapshot.engine.pos is not None:
        raise ValueError('ROLE_HISTORY_ORIGIN')
    if not lags or not delays or min(delays)<0 or max(delays)>=min(lags):
        raise ValueError('ROLE_HISTORY_TIMING')
    before=state_view(snapshot)
    forks={f'{direction}:{delay}':fork_env(snapshot)
           for direction in ('long','short') for delay in delays}
    for step in range(1,max(lags)+1):
        for key,e in forks.items():
            direction,delay=key.split(':')
            e.step((1 if direction=='long' else 2) if step==int(delay)+1 else 0)
        if len({(e.bar,e.finished) for e in forks.values()})!=1:
            raise ValueError('ROLE_HISTORY_CLOCK')
        ref=next(iter(forks.values()))
        if ref.finished:break
        if step in lags:
            states={};histories={}
            for key,e in forks.items():
                if any(not np.array_equal(e.last_observation[k],ref.last_observation[k])
                       for k in ('features','mask')):raise ValueError('ROLE_HISTORY_OBSERVATION')
                histories[key]=state_view(e)
                direction=1 if key.startswith('long:') else -1
                if e.engine.pos is not None and e.engine.pos.direction==direction:states[key]=fork_env(e)
            yield step,states,dict(histories=histories,observation_equal=True,
                elapsed_hours=(state_view(ref)['timestamp']-before['timestamp'])/(3600*10**9))
    if state_view(snapshot)!=before:raise ValueError('ROLE_HISTORY_MUTATED_PARENT')
