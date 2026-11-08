"""Non-promoting critic diagnostics with explicit, single-factor interventions."""
import torch
from torch.nn import functional as F

from app.generalization_diagnostics import soft_td_target, summary
from app.selected_warm_start import target_surfaces
from tools.diagnose_warmstart_drift import set_arm

ARMS = ('no_update', 'production', 'no_core_dropout', 'no_weight_decay', 'fixed_target')


def backup_components(reward, terminated, duration, probability, target_q, online_taken, alpha, gamma):
    """Exact additive decomposition of the minimum-twin soft TD residual."""
    p = probability
    discount = torch.pow(gamma, duration.reshape(-1)) * (~terminated.reshape(-1))
    mean_value = (p * target_q.mean(0)).sum(-1)
    minimum_value = (p * target_q.min(0).values).sum(-1)
    entropy = -(p * p.clamp_min(1e-30).log()).sum(-1)
    parts = dict(reward=reward.reshape(-1), entropy_bonus=discount*alpha*entropy,
        mean_backup_minus_online=discount*mean_value-online_taken,
        minimum_twin_penalty=-discount*(mean_value-minimum_value))
    parts['residual'] = sum(parts.values())
    return parts


def configure_arm(model, arm):
    if arm not in (*ARMS, 'mean_target'):
        raise ValueError('CRITIC_AUDIT_ARM')
    set_arm(model, 'no_update' if arm == 'no_update' else 'critic_only')
    if arm == 'no_core_dropout':
        for module in model.q_net.encoder.core.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.
    if arm == 'no_weight_decay':
        for group in model.optimizer.param_groups:
            group['weight_decay'] = 0.
    if arm == 'mean_target':
        native_target = model.loss._compute_target
        def mean_target(batch):
            with torch.no_grad():
                native = native_target(batch)
                p = model.surfaces(batch['next','features']).softmax(-1)
                q = target_surfaces(model,batch['next','features'])
                discount = torch.pow(model.effective['gamma'],batch['steps_to_next_obs'].reshape(-1))
                discount *= ~batch['next','terminated'].reshape(-1)
                return native + discount*(p*(q.mean(0)-q.min(0).values)).sum(-1)
        model.loss._compute_target = mean_target
        model._audit_mean_target = True


def algebra_audit(model, batch):
    """Compare actual loss internals to independent arithmetic, without an update."""
    loss = model.loss
    with torch.random.fork_rng(), torch.no_grad():
        p = model.surfaces(batch['next', 'features']).softmax(-1)
        tq = target_surfaces(model, batch['next', 'features'])
        expected = soft_td_target(batch['next','reward'], batch['next','terminated'],
            batch['steps_to_next_obs'], p, tq, loss.log_alpha.exp(), model.effective['gamma'])
        actual = loss._compute_target(batch.clone())
        torch.testing.assert_close(expected, actual, atol=2e-5, rtol=2e-5)
        terminal = batch['next','terminated'].reshape(-1)
        terminal_error = (float((actual[terminal]-batch['next','reward'].reshape(-1)[terminal]).abs().max())
                          if terminal.any() else None)
    # Capture outside vmap, where the twin dimension is an ordinary tensor axis.
    with torch.random.fork_rng():
        seen = []
        original = loss._vmap_qnetworkN0
        def capture(td, params):
            result = original(td, params)
            seen.append(result['action_value'])
            return result
        loss._vmap_qnetworkN0 = capture
        try:
            native, _ = loss.qvalue_loss(batch.clone())
        finally:
            loss._vmap_qnetworkN0 = original
        q = seen[-1]
        chosen = q.gather(-1,batch['action'].reshape(1,-1,1).expand(2,-1,1)).squeeze(-1)
        target = expected.expand_as(chosen)
        functions = {'smooth_l1':F.smooth_l1_loss, 'l2':F.mse_loss, 'l1':F.l1_loss}
        manual = functions[loss.loss_function](chosen,target,reduction='none').sum(0).mean()
        torch.testing.assert_close(native,manual,atol=2e-5,rtol=2e-5)
    # Temporarily enable both roles to test loss separation, then restore trainability.
    flags = {n:p.requires_grad for n,p in loss.named_parameters()}
    try:
        for n,param in loss.named_parameters():
            param.requires_grad_(n.startswith(('actor_', 'qvalue_')) and '.head.' in n)
        actor = [p for n,p in loss.named_parameters() if n.startswith('actor_') and p.requires_grad]
        critic = [p for n,p in loss.named_parameters() if n.startswith('qvalue_') and p.requires_grad]
        with torch.random.fork_rng():
            actor_loss,_ = loss.actor_loss(batch.clone())
            ga = torch.autograd.grad(actor_loss,critic,allow_unused=True)
            critic_loss,_ = loss.qvalue_loss(batch.clone())
            gc = torch.autograd.grad(critic_loss,actor,allow_unused=True)
        leak_a = sum(float(g.abs().sum()) for g in ga if g is not None)
        leak_c = sum(float(g.abs().sum()) for g in gc if g is not None)
        if leak_a or leak_c:raise ValueError('CRITIC_AUDIT_GRADIENT_LEAK')
    finally:
        for n,param in loss.named_parameters():param.requires_grad_(flags[n])
    return dict(target_max_error=float((actual-expected).abs().max()),
        terminal_max_error=terminal_error, loss_error=float((native-manual).abs().detach()),
        actor_to_critic_gradient=leak_a,critic_to_actor_gradient=leak_c,
        target=summary(actual.detach().cpu().numpy()),
        online_train_q=summary(chosen.detach().cpu().numpy()),loss_function=loss.loss_function)


def checked_update(model, batch, arm):
    before = {k:v.clone() for k,v in model.loss.target_qvalue_network_params.items(True,True)}
    if arm == 'no_update':return {'polyak_max_error':0.}
    result = model.update(batch)
    maximum = 0.
    with torch.no_grad():
        for key,old in before.items():
            new = model.loss.target_qvalue_network_params[key]
            source = model.loss.qvalue_network_params[key]
            expected = old.lerp(source,model.effective['tau']) if old.is_floating_point() else old
            torch.testing.assert_close(new,expected,atol=1e-6,rtol=1e-6)
            if old.is_floating_point():maximum=max(maximum,float((new-expected).abs().max()))
            if arm == 'fixed_target':new.copy_(old)
    result['polyak_max_error']=maximum
    return result
