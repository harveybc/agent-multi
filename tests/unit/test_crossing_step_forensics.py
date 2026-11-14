from copy import deepcopy
import numpy as np
import pytest
import torch
from app.crossing_step_forensics import adam_parts,flat,harmful_rows,attribute
from app.role_routed_critic import RoleHeads
from app.decision_cost_objective import cost_objective
from app.weight_decision_forensics import native_forward,cost_gradients,weighted_min_gradient
from tools.replay_first_harmful_crossings import require_equal


def test_adam_recurrence_and_decomposition():
    p=torch.nn.Parameter(torch.tensor([1.,-2.,3.],dtype=torch.float64))
    opt=torch.optim.AdamW([p],lr=.03,weight_decay=.1)
    p.grad=torch.tensor([.4,-.6,.2],dtype=p.dtype);opt.step()
    before={'x':p.detach().clone()};old=deepcopy(opt.state_dict())
    g=np.array([-.8,.3,.1]);p.grad=torch.tensor(g);opt.step()
    proposed={'x':p.detach().clone()};new=deepcopy(opt.state_dict())
    result=adam_parts(before,proposed,old,new,g)
    assert result['reconstruction_max_error']<1e-15
    np.testing.assert_allclose(sum(result[k] for k in ('history','current','decay','rounding')),flat(proposed)-flat(before),atol=1e-15)
    new['state'][0]['step']+=1
    with pytest.raises(ValueError,match='CROSS_OPT_STEP'):adam_parts(before,proposed,old,new,g)


def test_first_harmful_detection_and_scope():
    a=np.array([[[0.,2.,1.],[2.,0.,1.],[0.,1.,2.]]]*2)
    b=np.array([[[2.,0.,1.],[0.,2.,1.],[2.,1.,0.]]]*2)
    y=np.array([[0.,1.,-1.],[0.,1.,-1.],[0.,1.,2.]])
    assert harmful_rows(a,b,y,[0,1])==[0]
    assert harmful_rows(a,b,y,[0,1,2])==[0,2]


def test_batch_duplicates_and_projection_attribution_without_held_gradients():
    head=RoleHeads(torch.nn.Sequential(torch.nn.Linear(4,3),torch.nn.SiLU(),torch.nn.Linear(3,3)).double(),1,17)
    z=np.random.default_rng(8).normal(size=(2,4,4));y=np.random.default_rng(9).normal(size=(4,3))
    batch=np.array([0,0,1]);fit=np.array([0,1,2]);opt=torch.optim.AdamW(head.parameters(),lr=.001)
    def step():
        opt.zero_grad();loss=cost_objective(head(torch.tensor(z[:,batch])).mean(2),torch.tensor(y[batch]),'cost',.7)
        loss.backward();g=torch.cat([p.grad.flatten() for p in head.parameters()]).numpy().copy();opt.step();return g
    step();before=deepcopy(head.state_dict());old=deepcopy(opt.state_dict());g=step()
    proposed=deepcopy(head.state_dict());new=deepcopy(opt.state_dict());accepted=deepcopy(proposed)
    action=native_forward(before,z)[0].min(0).argmax(1)[0];other=(action+1)%3
    for t in (0,1):accepted[f'twins.{t}.0.2.bias'][other]+=10
    result=attribute(before,proposed,accepted,old,new,g,z,y,fit,batch,np.array([0]),.7)
    event=result['events'][0];assert event['batch_multiplicity']==2
    assert event['after_action']==other and event['margin_after']<0
    coeff=np.zeros_like(y);coeff[0,action]=1;coeff[0,other]=-1
    j=weighted_min_gradient(before,z,coeff)[0];row_g=cost_gradients(before,z[:,:1],y[:1],.7)[0][0]
    assert event['own_batch_descent']==pytest.approx(-j@row_g*2/3)
    assert event['own_batch_descent']+event['others_batch_descent']==pytest.approx(event['actual_cuda_batch_descent'])
    assert event['finite_adam']+event['finite_repair']==pytest.approx(event['margin_after']-event['margin_before'])
    other_y=y.copy();other_y[3]+=1e5
    assert attribute(before,proposed,accepted,old,new,g,z,other_y,fit,batch,np.array([0]),.7)==result
    with pytest.raises(ValueError,match='CROSS_HELD_GRADIENT'):
        attribute(before,proposed,accepted,old,new,g,z,y,fit,np.array([3]),np.array([0]),.7)


def test_replay_rejects_tampered_receipt_or_weights():
    require_equal({'x':torch.tensor([1.])},{'x':torch.tensor([1.])},'TEST')
    with pytest.raises(ValueError,match='CROSS_REPLAY_TEST'):require_equal({'offset':1},{'offset':2},'TEST')
