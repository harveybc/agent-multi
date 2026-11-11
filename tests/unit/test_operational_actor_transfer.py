import numpy as np
import pytest
import torch
from app.operational_actor_transfer import actor_loss, fit_indices, check_lineage
from app.operational_opening_critic import BOUNDARY, END


def test_actor_loss_is_categorical_expectation_and_detaches_critic():
    z=torch.tensor([[.2,-.1,.8]],requires_grad=True)
    q=torch.tensor([[1.,2.,3.]],requires_grad=True)
    alpha=torch.tensor(.6,requires_grad=True)
    loss=actor_loss(z,q,alpha)
    p=z.softmax(-1)
    assert torch.allclose(loss,(p*(.6*z.log_softmax(-1)-q.detach())).sum())
    loss.backward()
    assert z.grad.abs().sum()>0 and q.grad is None and alpha.grad is None


def test_equal_action_values_have_only_entropy_gradient():
    z=torch.zeros(2,3,requires_grad=True)
    actor_loss(z,torch.ones_like(z)*5,torch.tensor(1.)).backward()
    torch.testing.assert_close(z.grad,torch.zeros_like(z))


def test_nonfinite_or_wrong_action_count_refused():
    with pytest.raises(ValueError):actor_loss(torch.zeros(1,2),torch.zeros(1,2),torch.tensor(1.))
    with pytest.raises(ValueError):actor_loss(torch.zeros(1,3),torch.full((1,3),float('nan')),torch.tensor(1.))


def test_fit_excludes_monitor_and_forbidden_states():
    def row(role,forbidden):
        end=BOUNDARY if role=='fit' else END
        return dict(role=role,forbidden=forbidden,timestamp=end-100,terminal=end,
                    window_start=BOUNDARY if role=='monitor' else 0,position=0,
                    suffixes=[dict(actions=[dict(soft_return=.1,first_effect=dict(entries=0 if forbidden else 1))]*3)]*3)
    r=[row('fit',False),row('fit',True),row('monitor',False)]
    assert fit_indices(r).tolist()==[0]
    r[0]['terminal']=END
    with pytest.raises(ValueError):fit_indices(r)


def test_lineage_checks_source_and_control():
    c=dict(identity='control',source_sha256='weights',support_sha256='rows')
    r=dict(source_sha256='weights',control='control')
    check_lineage('weights','rows',c,r)
    with pytest.raises(ValueError):check_lineage('other','rows',c,r)
    with pytest.raises(ValueError):check_lineage('weights','other',c,r)
    with pytest.raises(ValueError):check_lineage('weights','rows',c,dict(r,control='other'))
