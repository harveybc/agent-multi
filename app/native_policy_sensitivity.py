"""Read-only categorical policy probes; no fitting and no trading claims."""
import numpy as np
import torch

from app.frozen_recipe_inputs import require

MASKS=dict(flat=[1,1,1,0,0],long=[1,0,0,1,0],short=[1,0,0,0,1])


def probe_policy(model, windows):
    require(windows.ndim==3 and len(windows)>1 and np.isfinite(windows).all(),'PROBE_INPUT')
    p=model.policy
    was_training=p.training
    p.set_training_mode(False)
    x=torch.as_tensor(np.array(windows,copy=True),device=p.device,dtype=torch.float32).requires_grad_(True)
    obs={'features':x,'action_mask':torch.tensor(MASKS['flat'],device=p.device).float().expand(len(x),-1)}
    try:
        enc=p.features_extractor(obs)
        actor,critic=p.mlp_extractor(enc)
        logits=p.commands(actor)
        pre_actor=p.mlp_extractor.policy_net[0](enc)
        pre_critic=p.mlp_extractor.value_net[0](enc)
        def std(t): return float(t.detach().double().std(0,unbiased=False).mean())
        result=dict(encoder_std_mean=std(enc),encoder_abs_max=float(enc.detach().abs().max()),
            actor_std_mean=std(actor),critic_std_mean=std(critic),
            actor_tanh_saturated_fraction=float((actor.detach().abs()>.99).float().mean()),
            actor_tanh_exact_one_fraction=float((actor.detach().abs()==1).float().mean()),
            critic_tanh_saturated_fraction=float((critic.detach().abs()>.99).float().mean()),
            actor_pre_abs_max=float(pre_actor.detach().abs().max()),
            critic_pre_abs_max=float(pre_critic.detach().abs().max()),routes={},perturbations={})
        base_flat=None
        for route,mask in MASKS.items():
            obs['action_mask']=torch.tensor(mask,device=p.device).float().expand(len(x),-1)
            with torch.no_grad():
                d=p.get_distribution(obs)
                probs=d.cat.probs.cpu().numpy()
                commands=probs.argmax(-1)
                entropy=float(d.cat.entropy().mean())
            if route=='flat': base_flat=probs
            a,b={'flat':(1,2),'long':(3,0),'short':(4,0)}[route]
            grad=torch.autograd.grad((logits[:,a]-logits[:,b]).sum(),x,retain_graph=True)[0]
            result['routes'][route]=dict(probability_mean=probs.astype(float).mean(0).tolist(),
                probability_span_max=float(np.ptp(probs,axis=0).max()),
                argmax_counts=np.bincount(commands,minlength=5).tolist(),entropy=entropy,
                margin_input_gradient_abs_max=float(grad.abs().max()),
                margin_input_gradient_abs_mean=float(grad.abs().mean()))
        changed=dict(zero=np.zeros_like(windows),repeat_current=np.repeat(windows[:,-1:,:],windows.shape[1],axis=1),
                     reverse_time=windows[:,::-1,:].copy(),reorder_windows=np.roll(windows,1,axis=0))
        for name,z in changed.items():
            with torch.no_grad():
                probe={'features':torch.as_tensor(z,device=p.device,dtype=torch.float32),
                       'action_mask':torch.tensor(MASKS['flat'],device=p.device).float().expand(len(x),-1)}
                probs=p.get_distribution(probe).cat.probs.cpu().numpy()
            result['perturbations'][name]=dict(flat_probability_abs_delta_max=float(np.abs(probs-base_flat).max()),
                flat_argmax_changed=int(np.count_nonzero(probs.argmax(-1)!=base_flat.argmax(-1))))
        return result
    finally:
        p.set_training_mode(was_training)


class FixedRule:
    """Negative control: ignores market features; only uses business action masks."""
    def __init__(self,model,rule):
        require(rule in ('LONG_HOLD','LONG_CLOSE'),'RULE')
        self.policy=model.policy
        self.rule=rule

    def predict(self,obs,deterministic=True):
        mask=obs['action_mask']
        command=3 if self.rule=='LONG_CLOSE' and mask[3] else 1 if mask[1] else 0
        return np.array([command,.5,.5,.5],np.float32),None
