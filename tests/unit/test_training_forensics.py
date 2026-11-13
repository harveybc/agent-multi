from copy import deepcopy

import numpy as np
import torch
from torch import nn

from app.training_forensics import tree_digest,rng_state,restore_rng,mse_components,sinusoid_lags,balance_teacher
from app.role_routed_critic import RoleHeads


def test_mse_components_and_gradients_add():
    q=torch.randn(2,9,3,requires_grad=True);y=torch.randn(9,3)
    a,b=mse_components(q,y);m=(q-y).square().mean()
    assert torch.allclose(a+b,m)
    g=torch.autograd.grad(a+b,q,retain_graph=True)[0]
    assert torch.allclose(g,torch.autograd.grad(m,q)[0],atol=1e-7)


def test_digest_detects_weights_optimizer_and_rng():
    s={'w':torch.ones(3),'opt':{'exp_avg':torch.zeros(3)},'rng':torch.arange(4)}
    h=tree_digest(s)
    assert tree_digest(deepcopy(s))==h
    for k in ('w','opt','rng'):
        v=deepcopy(s)
        if k=='opt':v[k]['exp_avg'][0]=1
        else:v[k][0]+=1
        assert tree_digest(v)!=h


def test_adamw_restart_restores_exact_state_and_random_stream(tmp_path):
    torch.manual_seed(19);m=nn.Linear(5,3);o=torch.optim.AdamW(m.parameters(),lr=3e-5)
    def advance(model,opt,n):
        for _ in range(n):
            x=torch.randn(8,5);opt.zero_grad();model(x).square().mean().backward();opt.step()
    advance(m,o,4)
    p=tmp_path/'state.pt';torch.save(dict(model=m.state_dict(),opt=o.state_dict(),rng=rng_state()),p)
    advance(m,o,9);expected=tree_digest((m.state_dict(),o.state_dict(),rng_state()))
    r=nn.Linear(5,3);opt=torch.optim.AdamW(r.parameters(),lr=3e-5)
    s=torch.load(p,weights_only=False);r.load_state_dict(s['model']);opt.load_state_dict(s['opt']);restore_rng(s['rng'])
    advance(r,opt,9)
    assert tree_digest((r.state_dict(),opt.state_dict(),rng_state()))==expected


def test_multiscale_lags_are_prefix_invariant_and_teacher_representable():
    a=sinusoid_lags(128);b=sinusoid_lags(256)
    assert np.array_equal(a,b[:128])
    teacher=nn.Sequential(nn.Linear(208,8),nn.SiLU(),nn.Linear(8,3))
    student=deepcopy(teacher);x=torch.from_numpy(a)
    assert torch.equal(student(x),teacher(x))


def test_balanced_teacher_uses_only_fit_thresholds_and_all_actions():
    torch.manual_seed(321)
    x=torch.randn(150,208);z=torch.stack([x,x]);fit=np.arange(99)
    teacher=RoleHeads(nn.Sequential(nn.Linear(208,8),nn.SiLU(),nn.Linear(8,3)),1,271)
    other=deepcopy(teacher);changed=z.clone();changed[:,99:]+=1000
    a=balance_teacher(teacher,z,fit);b=balance_teacher(other,changed,fit)
    assert a==b and min(a['fit_counts'])>=32
    assert tree_digest(teacher.state_dict())==tree_digest(other.state_dict())
    assert torch.equal(teacher(z)[0],teacher(z)[1])
