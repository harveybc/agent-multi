import torch
import pytest
from tests.unit.test_cumulative_readmission import ToyActor,recipe
from app.cumulative_readmission import expand_actor
from app.selected_actor_export import transfer_expanded_actor


def test_swap_preserves_weights_and_masked_function():
    full=recipe(['a','b','c','d'],{'g':[0,1,2],'extra':[3]})
    old=recipe(['a','b'],{'g':[0,1]})
    selected=recipe(['a','c'],{'g':[0,1]})
    expanded,info=expand_actor(ToyActor(old),old,full,[(ToyActor(full),full,'old')])
    dest=ToyActor(selected)
    result=transfer_expanded_actor(expanded,dest,full,selected,info['group_order'])
    x=torch.randn(7,13,4);x[:,:,1]=0;x[:,:,3]=0
    torch.testing.assert_close(dest.surface(x[:,:,[0,2]]),expanded.surface(x),atol=1e-6,rtol=1e-5)
    assert result['physical_input_width']==2
    assert torch.equal(dest.head.weight,expanded.head.weight)


def test_removing_nonzero_original_group_rejected():
    full=recipe(['a','b'],{'g':[0],'other':[1]})
    selected=recipe(['a'],{'g':[0]})
    with pytest.raises(ValueError,match='NONZERO_REMOVED'):
        transfer_expanded_actor(ToyActor(full),ToyActor(selected),full,selected,['g','other'])


def test_centered_roundoff_recorded_but_material_response_rejected():
    full=recipe(['a','b'],{'g':[0],'extra':[1]});old=recipe(['a'],{'g':[0]})
    expanded,info=expand_actor(ToyActor(old),old,full,[(ToyActor(full),full,'old')])
    with torch.no_grad():expanded.encoder.branches[1].neutral.add_(1e-7)
    result=transfer_expanded_actor(expanded,ToyActor(old),full,old,info['group_order'])
    assert 0<result['centered_neutral_residuals']['extra']<1e-5
    with torch.no_grad():expanded.encoder.branches[1].neutral.add_(1e-3)
    with pytest.raises(ValueError,match='NONZERO_REMOVED'):
        transfer_expanded_actor(expanded,ToyActor(old),full,old,info['group_order'])
