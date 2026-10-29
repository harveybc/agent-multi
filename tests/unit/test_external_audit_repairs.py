"""Synthetic, independently specified boundary witnesses from the external audits."""
from dataclasses import replace

import numpy as np
import pytest

from app.protection_reference import protection_exit
from app.close_hold_controller import CloseController, value_paths
from tests.unit.test_close_hold_controller import position, report


@pytest.mark.parametrize('direction', [1, -1])
def test_exact_native_through_boundary(direction):
    tp=1.1; through=.00002; entry=tp-direction*.001
    extreme=tp+direction*through
    o=np.array([entry]); h=np.array([max(entry,extreme)]); l=np.array([min(entry,extreme)])
    assert protection_exit(direction,tp,entry-direction*.01,o,h,l,o,through)[2]=='TP'


@pytest.mark.parametrize('direction', [1, -1])
def test_deadline_quote_uses_terminal_spread(direction):
    p=position(direction,100-direction*5,100+direction*5)
    r=replace(report([100.,100.]),terminal_spreads=np.array([2.]))
    values=value_paths(p,r,np.zeros(2),np.zeros(2))
    assert values[0,0]==0.
    assert values[0,-1]==-1.


def test_self_consistent_truncated_deadline_rejected():
    r=report([100.,98.])
    with pytest.raises(ValueError,match='EXTERNAL_DEADLINE'):
        CloseController().decide(position(),r,10,np.zeros(2),np.zeros(2),expected_deadline=13)


@pytest.mark.parametrize('clock', [9,11])
def test_external_clock_rejects_stale_or_future_report(clock):
    r=report()
    with pytest.raises(ValueError,match='STALE_OR_FUTURE'):
        CloseController().decide(position(),r,clock,np.zeros(3),np.zeros(3),expected_deadline=13)


def test_cached_values_cannot_drop_clock_contract():
    r=report(tag='synthetic_ideal')
    with pytest.raises(ValueError,match='STALE_OR_FUTURE'):
        CloseController(allow_synthetic=True).decide_synthetic_values(
            position().identity,11,1,np.zeros((1,4)),r.weights,2.,
            report=r,expected_deadline=13)
