import torch
from tools.role458_full_train import lower,better
from role405_plateau import observe


def test_forks_preserve_independent_adam_moments_and_lr():
    def state(v):
        return dict(opt=dict(param_groups=[dict(lr=.0003)],state={1:{'exp_avg':torch.tensor(v)}}),
                    w={'w':torch.tensor(v)})
    champ,tail=state(1.),state(2.)
    a,b=lower(champ,.00006),lower(tail,.00006)
    b['opt']['state'][1]['exp_avg'].add_(7)
    assert a['opt']['state'][1]['exp_avg'].item()==1
    assert tail['opt']['state'][1]['exp_avg'].item()==2
    assert a['w']['w'].item()==1 and b['w']['w'].item()==2
    assert champ['opt']['param_groups'][0]['lr']==tail['opt']['param_groups'][0]['lr']==.0003
    assert a['opt']['param_groups'][0]['lr']==b['opt']['param_groups'][0]['lr']==.00006


def test_activity_and_plateau_boundaries():
    assert better((True,-1),(False,100))
    assert not better((False,100),(True,-1))
    assert not better((True,0.5+1e-9),(True,0.5))
    miss=0
    for ep in range(1,41):
        miss,lr=observe(ep,False,miss,.0003,False)
        assert lr==(None if ep<40 else .00006)
    assert observe(41,True,12,.00006,False)==(0,None)
