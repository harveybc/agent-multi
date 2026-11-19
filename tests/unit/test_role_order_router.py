import pytest

from app.role_order_router import greedy_command,role_mode


@pytest.mark.parametrize('values,action', [((2,1),1),((1,2),2),((2,2),0),((-1,-2),0),((0,-1),0)])
def test_flat_arbitrates_without_directional_priority(values,action):
    assert greedy_command(0,{'flat':values})==action


@pytest.mark.parametrize('direction', [-1,1])
def test_nothing_but_active_closer_can_close(direction):
    for signal,expect in [(1,3),(0,0),(-1,0)]:
        assert greedy_command(direction,{role_mode(direction):[signal]},entries_allowed=False)==expect
    assert greedy_command(direction,{},force_flat=True)==3


def test_calendar_and_bad_logits():
    assert greedy_command(0,{},entries_allowed=False)==0
    with pytest.raises(ValueError):greedy_command(0,{'flat':[float('nan'),1]})
