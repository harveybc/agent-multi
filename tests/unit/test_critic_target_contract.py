import copy

import numpy as np
import pytest
import torch

from app.critic_target_contract import visitation_probabilities,contract_loss


def rows():
    return [dict(role='fit',year=2020,kind=k,outcomes=[99.]) for k in
            ('flat_both_allowed','flat_both_allowed','open_long','open_short')]


def test_visitation_weights_match_stratum_mass_and_ignore_targets():
    r=rows();counts={'2020:flat_both_allowed':80,'2020:open_long':10,'2020:open_short':10}
    p=visitation_probabilities(r,[0,1,2,3],counts)
    np.testing.assert_allclose(p,[.4,.4,.1,.1])
    changed=copy.deepcopy(r);changed[0]['outcomes']=[-999.]
    np.testing.assert_array_equal(visitation_probabilities(changed,[0,1,2,3],counts),p)


def test_weights_reject_missing_stratum_monitor_and_bad_year_counts():
    r=rows();counts={'2020:flat_both_allowed':80,'2020:open_long':10,'2020:open_short':10}
    with pytest.raises(ValueError,match='COVERAGE'):visitation_probabilities(r,[0,1,2],counts)
    with pytest.raises(ValueError,match='COUNTS'):visitation_probabilities(r,[0,1,2,3],counts|{'2023:open_long':1})
    r[0]['role']='monitor'
    with pytest.raises(ValueError,match='FIT_YEAR'):visitation_probabilities(r,[0,1,2,3],counts)


def test_losses_have_distinct_expected_values_and_gradients():
    q=torch.full((2,2,3),2.,requires_grad=True);y=torch.zeros(2,3)
    assert contract_loss(q,y,'smooth_l1').item()==pytest.approx(1.5)
    assert contract_loss(q,y,'mse').item()==pytest.approx(4.)
    contract_loss(q,y,'mse').backward()
    assert torch.isfinite(q.grad).all() and (q.grad>0).all()
    with pytest.raises(ValueError):contract_loss(q,y,'unknown')
    with pytest.raises(ValueError):contract_loss(q[:1],y,'mse')
