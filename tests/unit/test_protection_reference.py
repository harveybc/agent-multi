import numpy as np
import pytest


def test_stop_before_target_and_gap_fill_have_literal_expectations():
    from app.protection_reference import protection_exit
    # Long: both barriers touched, conservative SL, then a gap case.
    assert protection_exit(1, 12., 8., [10.], [13.], [7.], [11.], 0.) == (0,8.,'SL')
    assert protection_exit(1, 12., 8., [7.], [13.], [6.], [11.], 0.) == (0,7.,'SL')
    assert protection_exit(-1,8.,12.,[13.],[14.],[7.],[9.],0.) == (0,13.,'SL')


def test_early_close_expected_before_stop_but_not_before_target():
    from app.protection_reference import protection_exit, prefer_close
    bad=protection_exit(1,12.,8.,[10.,9.],[10.5,9.5],[9.5,7.],[10.,8.],0.)
    good=protection_exit(1,12.,8.,[10.,11.],[10.5,13.],[9.5,10.],[10.,12.],0.)
    assert prefer_close(close_net=0.,hold_net=bad[1]-10.,minimum_advantage=0.)
    assert not prefer_close(close_net=0.,hold_net=good[1]-10.,minimum_advantage=0.)
    assert not prefer_close(0.,0.,0.)


def test_risk_filter_and_competing_directions_are_not_first_buy_wins():
    from app.protection_reference import entry_command
    assert entry_command([.2,.4],[1.,1.],minimum_gain_risk=1.)==0
    assert entry_command([2.,3.],[1.,1.],minimum_gain_risk=1.)==2
    assert entry_command([3.,2.],[1.,1.],minimum_gain_risk=1.)==1
    assert entry_command([2.,2.],[1.,1.],minimum_gain_risk=1.)==0
    assert entry_command([-.1,-.2],[1.,1.],minimum_gain_risk=0.)==0


def test_reflected_paths_have_swapped_barriers_and_same_exit_profit():
    from app.protection_reference import protection_exit
    o=np.array([10.,11.]);h=np.array([11.,13.]);l=np.array([9.,10.]);c=np.array([10.5,12.])
    a=protection_exit(1,12.,8.,o,h,l,c,.1)
    b=protection_exit(-1,8.,12.,20-o,20-l,20-h,20-c,.1)
    assert a[0]==b[0] and a[2]==b[2]
    assert a[1]-10.==10.-b[1]


@pytest.mark.parametrize('gain,risk', [([float('nan'),1.],[1.,1.]),([1.,1.],[0.,1.])])
def test_invalid_risk_estimates_cannot_open(gain,risk):
    from app.protection_reference import entry_command
    with pytest.raises(ValueError):entry_command(gain,risk,1.)
