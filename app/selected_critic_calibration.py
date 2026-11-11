"""Fresh-head calibration on current-policy FIT returns, actor held exact."""
import numpy as np
import torch
from app.critic_refinement import begin_stage,sync_targets
from app.critic_fit_probe import set_scope,probe_q,fit_metrics
from app.action_ranking_contrast import target_loss
from app.selected_warm_start import reset_learning_rng
from app.rl_subset import digest
from tools.audit_fixed_actor_critics import covering_schedule
from tools.diagnose_warmstart_drift import frozen_state,verify_frozen_state


def calibrate(model,x,records,seed,progress,updates=4096):
    fit=np.array([i for i,r in enumerate(records) if r['role']=='fit'])
    monitor=np.array([i for i,r in enumerate(records) if r['role']=='monitor'])
    if not len(fit) or not len(monitor):raise ValueError('CALIBRATION_ROLES')
    y=torch.tensor([[o['reward_return'] for o in r['outcomes']] for r in records],device=x.device,dtype=torch.float32)
    if not torch.isfinite(y).all():raise ValueError('CALIBRATION_TARGET')
    init=begin_stage(model,'selected_input_refinement',seed=seed);set_scope(model,'heads')
    frozen=frozen_state(model);buffers={n:v.detach().clone() for n,v in model.loss.named_buffers()}
    reset_learning_rng(seed);schedule=covering_schedule(len(fit),updates,128,seed);curve=[]
    for step in range(updates+1):
        if step:
            ix=fit[schedule[step-1]];model.optimizer.zero_grad(set_to_none=True)
            loss=target_loss(probe_q(model,x[ix]),y[ix]);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.loss.parameters(),10.,error_if_nonfinite=True);model.optimizer.step()
        if step in (0,64,256,1024,2048,updates):
            verify_frozen_state(model,frozen)
            if any(not torch.equal(v,dict(model.loss.named_buffers())[n]) for n,v in buffers.items()):
                raise ValueError('CALIBRATION_BUFFER_CHANGED')
            with torch.no_grad():q=model.q_surfaces(x).cpu().numpy()
            curve.append(dict(updates=step,metrics={k:fit_metrics(q[:,ix],y[ix].cpu().numpy())
                for k,ix in [('fit',fit),('monitor',monitor)]}))
            progress(curve[-1])
    # Synchronize only after verifying the online fit never moved frozen targets.
    sync=sync_targets(model)
    return dict(initialization=init,curve=curve,fit_rows=fit.tolist(),monitor_rows=monitor.tolist(),
        schedule_sha256=digest(schedule.tolist()),actor_frozen=True,target_sync=sync)
