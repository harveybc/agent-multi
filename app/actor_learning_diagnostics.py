"""Non-promoting actor-gradient and same-observation diagnostic primitives."""
import copy

import numpy as np
import torch

from agent_plugins.sac_critic_regularization import module_mode


def clear_optimizer_memory(optimizer):
    groups = [{k: copy.deepcopy(v) for k,v in g.items() if k != 'params'}
              for g in optimizer.param_groups]
    before = len(optimizer.state)
    optimizer.state.clear()
    after = [{k:v for k,v in g.items() if k != 'params'} for g in optimizer.param_groups]
    if groups != after or optimizer.state:
        raise ValueError('OPTIMIZER_RESET_CONTRACT')
    return dict(previous_states=before, remaining_states=0, hyperparameters_unchanged=True)


def initialize_critic_heads(model, seed):
    """Reset effective online twin heads only, preserving RNG and parameter objects."""
    params = model.loss.qvalue_network_params
    weights = [(k,v) for k,v in params.items(True,True) if 'head' in k and k[-1]=='weight']
    if len(weights)!=2 or any(v.ndim!=3 or v.shape[0]!=2 for _,v in weights):
        raise ValueError('CRITIC_HEAD_LAYOUT')
    devices = [model.device.index or 0] if model.device.type=='cuda' else []
    changed=[]
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        torch.manual_seed(seed)
        for key,weight in weights:
            bias_key=(*key[:-1],'bias');bias=params[bias_key]
            if bias.shape!=weight.shape[:2]:raise ValueError('CRITIC_HEAD_BIAS_LAYOUT')
            for twin in range(2):
                layer=torch.nn.Linear(weight.shape[-1],weight.shape[-2],
                                      device=weight.device,dtype=weight.dtype)
                weight[twin].copy_(layer.weight);bias[twin].copy_(layer.bias)
            changed.extend(['.'.join(key),'.'.join(bias_key)])
    return dict(seed=seed,parameters=changed,twins=2,layers_per_twin=2,
                initialization='native torch.nn.Linear',target_heads_reset=False)


def action_calibration(q, returns, actions):
    from app.policy_return_calibration import calibration
    q,g,a=map(np.asarray,(q,returns,actions))
    if q.ndim!=1 or q.shape!=g.shape or q.shape!=a.shape or not np.isin(a,[0,1,2]).all():
        raise ValueError('ACTION_CALIBRATION_INPUT')
    result={}
    for signal,name in enumerate(('NOP','BUY','SELL')):
        take=a==signal;n=int(take.sum())
        result[name]=dict(rows=n,critic=calibration(q[take],g[take]) if n else None,
            zero=calibration(np.zeros(n),g[take]) if n else None)
    return result


def actor_terms(logits, q, alpha):
    if logits.shape != q.shape or logits.ndim != 2 or logits.shape[1] != 3:
        raise ValueError('ACTOR_DIAGNOSTIC_SHAPE')
    if not torch.isfinite(logits).all() or not torch.isfinite(q).all():
        raise ValueError('ACTOR_DIAGNOSTIC_NONFINITE')
    logp = logits.log_softmax(-1)
    p = logp.exp()
    return -(p*q.detach()).sum(-1).mean(), (alpha.detach()*p*logp).sum(-1).mean(), p


def actor_gradient_report(model, batch):
    flags = {n:p.requires_grad for n,p in model.loss.named_parameters()}
    try:
        for n,p in model.loss.named_parameters():
            p.requires_grad_(n.startswith('actor_') and '.head.' in n)
        named = [(n,p) for n,p in model.loss.named_parameters() if p.requires_grad]
        params = [p for _,p in named]
        with module_mode(model.actor, False), module_mode(model.loss.qvalue_network, False):
            logits = model.surfaces(batch['features'])
            with torch.no_grad():
                q = model.q_surfaces(batch['features']).min(0).values
            value, entropy, probability = actor_terms(logits,q,model.loss.log_alpha.exp())
            native, _ = model.loss.actor_loss(batch.clone())
            torch.testing.assert_close(value+entropy,native,atol=2e-5,rtol=2e-5)
            def grad(loss):
                return torch.autograd.grad(loss,params,retain_graph=True)
            gv, ge, gn = grad(value), grad(entropy), grad(native)
            for v,e,n in zip(gv,ge,gn):
                torch.testing.assert_close(v+e,n,atol=2e-5,rtol=2e-5)
            centered, _, _ = actor_terms(logits,q-q.mean(-1,keepdim=True),model.loss.log_alpha.exp())
            gc = grad(centered)
            for c,v in zip(gc,gv):
                torch.testing.assert_close(c,v,atol=2e-5,rtol=2e-5)
            gp = grad(probability[:,0].mean())
            flat = lambda gs:torch.cat([g.detach().flatten().double() for g in gs])
            v,e,n,c,p = map(flat,(gv,ge,gn,gc,gp))
            norms = dict(value=float(v.norm()),entropy=float(e.norm()),native=float(n.norm()))
            cos = float(torch.dot(v,e)/(v.norm()*e.norm())) if v.norm()*e.norm()>0 else None
            layers = {name:dict(value_norm=float(a.norm()),entropy_norm=float(b.norm()),
                native_norm=float(z.norm())) for (name,_),a,b,z in zip(named,gv,ge,gn)}
            gaps = q[:,1:]-q[:,:1]
            return dict(loss=dict(value=float(value.detach()),entropy=float(entropy.detach()),
                native=float(native.detach())),norms=norms,value_entropy_cosine=cos,
                gradient_max_error=float((v+e-n).abs().max()),
                common_offset_gradient_max_error=float((c-v).abs().max()),
                nop_directional_derivative=dict(value=float(-torch.dot(p,v)),
                    entropy=float(-torch.dot(p,e)),total=float(-torch.dot(p,n)),
                    meaning='change in batch mean NOP probability per infinitesimal SGD step; not AdamW'),
                alpha=float(model.loss.log_alpha.detach().exp()),
                mean_probability=probability.detach().mean(0).tolist(),
                q_action_gap_mean=gaps.mean(0).tolist(),q_nop_argmax_share=float((q.argmax(-1)==0).float().mean()),
                layers=layers)
    finally:
        for n,p in model.loss.named_parameters():
            p.requires_grad_(flags[n])


def matched_position_summary(left, right):
    keys = ('timestamp','signal','position','command','reward','soft_return')
    a,b = [{k:np.asarray(x[k]) for k in keys} for x in (left,right)]
    if (any(v.ndim!=1 or len(v)!=len(a['timestamp']) for d in (a,b) for v in d.values())
            or not np.array_equal(a['timestamp'],b['timestamp'])
            or not (np.diff(a['timestamp'])>0).all()):
        raise ValueError('AMBIGUITY_SUPPORT')
    if not all(np.isfinite(d[k]).all() for d in (a,b) for k in keys):
        raise ValueError('AMBIGUITY_NONFINITE')
    same_action = a['signal']==b['signal']
    output = {}
    for name,match in [('same_position',a['position']==b['position']),
                       ('different_position',a['position']!=b['position'])]:
        take = same_action & match
        n = int(take.sum())
        dr = (a['reward']-b['reward'])[take]
        dg = (a['soft_return']-b['soft_return'])[take]
        output[name] = dict(rows=n,
            command_disagreements=int(((a['command']!=b['command']) & take).sum()),
            reward_rmse=float(np.sqrt(np.mean(dr**2))) if n else None,
            return_rmse=float(np.sqrt(np.mean(dg**2))) if n else None)
    return dict(total_rows=len(same_action),same_action_rows=int(same_action.sum()),
        strata=output,causal_position_effect_established=False)
