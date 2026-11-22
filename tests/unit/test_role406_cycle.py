import pytest
import torch
from tools.role406_cycle import CYCLE, clone_pair, rate


def test_budget_and_rates():
    rates=[rate(i+1) for i in range(CYCLE)]
    assert rates[0]==pytest.approx(3e-4) and rates[-1]==pytest.approx(1.2e-5)
    assert all(a>b for a,b in zip(rates,rates[1:]))
    with pytest.raises(ValueError):rate(21)


def test_fork_preserves_moments_and_rng_without_sharing_storage():
    m=torch.nn.Linear(3,1); opt=torch.optim.AdamW(m.parameters())
    x=torch.ones(2,3); m(x).sum().backward();opt.step()
    rng=torch.get_rng_state(); b,bopt=clone_pair(m,opt)
    assert torch.equal(torch.get_rng_state(),rng)
    for a,c in zip(m.parameters(),b.parameters()):
        assert torch.equal(a,c) and a.data_ptr()!=c.data_ptr()
        for k,v in opt.state[a].items():
            assert torch.equal(v,bopt.state[c][k]) and v.data_ptr()!=bopt.state[c][k].data_ptr()
    before=m(x).detach().clone()
    bopt.zero_grad();b(x).sum().backward();bopt.step()
    assert torch.equal(m(x),before) and not torch.equal(b(x),before)


def test_paired_resume_keeps_endpoint_weights_and_permutations(tmp_path):
    torch.manual_seed(31)
    x=torch.randn(24,3);fine=torch.nn.Linear(3,1)
    opt=torch.optim.AdamW(fine.parameters(),lr=1.2e-5)
    opt.zero_grad();fine(x).square().mean().backward();opt.step()
    explore,eo=clone_pair(fine,opt)
    def run(a,ao,b,bo,steps):
        orders=[]
        for i in steps:
            bo.param_groups[0]['lr']=rate(i)
            perm=torch.randperm(len(x)); orders.append(perm)
            for batch in perm.split(6):
                for net,o in ((a,ao),(b,bo)):
                    o.zero_grad();net(x[batch]).square().mean().backward();o.step()
        return torch.stack(orders)
    run(fine,opt,explore,eo,range(1,8))
    p=tmp_path/'fork.pt'
    torch.save((fine.state_dict(),opt.state_dict(),explore.state_dict(),eo.state_dict(),torch.get_rng_state()),p)
    order=run(fine,opt,explore,eo,range(8,21))
    fa,oa=clone_pair(fine,opt);fb,ob=clone_pair(explore,eo)
    wa,sa,wb,sb,rng=torch.load(p,weights_only=True)
    fa.load_state_dict(wa);oa.load_state_dict(sa);fb.load_state_dict(wb);ob.load_state_dict(sb)
    torch.set_rng_state(rng)
    assert torch.equal(order,run(fa,oa,fb,ob,range(8,21)))
    for net,replay in ((fine,fa),(explore,fb)):
        assert all(torch.equal(a,b) for a,b in zip(net.parameters(),replay.parameters()))
