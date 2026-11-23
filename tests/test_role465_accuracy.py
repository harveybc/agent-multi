import pytest
from tools.role465_accuracy_refit import guarded_hit

REF=dict(orders=100,buy=60,wins=45,ret=20,dd=5,sharpe=2)


def test_feasible_winner():
    assert guarded_hit([dict(REF,wins=48)],[REF])==.48


@pytest.mark.parametrize('updates',[{'ret':19.9},{'dd':5.1},{'sharpe':1.9},
                                    {'buy':53},{'buy':65}])
def test_win_rate_cannot_override_guards(updates):
    assert guarded_hit([dict(REF,**updates,wins=70)],[REF])<0
