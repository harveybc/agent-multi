"""Optional real-checkpoint CUDA canary; no saved weights or account execution."""
import os
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.mark.parametrize('objective,scope',[('absolute','heads'),('advantage','heads'),
                                          ('advantage','critic_representation')])
def test_real_calibrated_head_gradients_and_frozen_actor(objective,scope):
    root=os.environ.get('ACTION_RANKING_SOURCE')
    probe=os.environ.get('ACTION_RANKING_PROBE')
    surfaces=os.environ.get('ACTION_RANKING_SURFACES')
    runtime=os.environ.get('ACTION_RANKING_SOURCE_RUNTIME')
    if not all((root,probe,surfaces,runtime)):
        pytest.skip('explicit real diagnostic artifacts required')
    from app.action_ranking_contrast import target_loss,advantage_loss
    from app.structural_refit import load_bundle
    from app.actor_learning_diagnostics import clear_optimizer_memory
    from app.frozen_recipe_inputs import file_sha
    from app.three_signal_learning import emit
    from tools.diagnose_warmstart_drift import set_arm
    from app.critic_fit_probe import set_scope,probe_q
    from tools.run_structural_sac_refit import authenticate,market_for
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available() and torch.cuda.get_device_name()=='NVIDIA GeForce RTX 5090'
    root=Path(root);spec=authenticate(root,Path(runtime));market=market_for(root,spec)
    bundle=root/'runs/seed_7/bundles/epoch_0000';sha=file_sha(bundle/'model.pt')
    model,_=load_bundle(bundle,'cuda');model.load(Path(probe)/'fresh/CALIBRATED.pt')
    indices=np.flatnonzero(market.inputs.valid & (market.inputs.period=='TRAIN'))[:64]
    x=torch.tensor(market.inputs.windows[indices],device='cuda')
    with torch.no_grad():
        actor=model.surfaces(x).clone();q0=model.q_surfaces(x).clone()
    np.testing.assert_allclose(actor.cpu().numpy(),np.load(surfaces)['reference'][indices],rtol=2e-5,atol=2e-5)
    before={n:p.detach().clone() for n,p in model.loss.named_parameters()}
    set_scope(model,scope);clear_optimizer_memory(model.optimizer)
    model.optimizer.zero_grad(set_to_none=True)
    q=probe_q(model,x)
    torch.testing.assert_close(q,q0,rtol=2e-5,atol=2e-5)
    # Deliberate test target, not a financial training label.
    target=q0.mean(0)+torch.tensor([.1,-.1,.2],device='cuda')
    (target_loss if objective=='absolute' else advantage_loss)(q,target).backward();model.optimizer.step()
    changed=[n for n,p in model.loss.named_parameters() if not torch.equal(p,before[n])]
    assert changed and all(n.startswith('qvalue_network_params.') for n in changed)
    if scope=='heads':assert all('.head.' in n for n in changed)
    else:assert any('.encoder.' in n for n in changed)
    with torch.no_grad():torch.testing.assert_close(model.surfaces(x),actor,rtol=0,atol=0)
    set_arm(model,'actor_only')
    full,_=emit(model,market.inputs)
    expected=np.load(surfaces)['reference'];valid=np.isfinite(expected).all(1)
    assert np.array_equal(full[valid],expected[valid])
    assert file_sha(bundle/'model.pt')==sha
