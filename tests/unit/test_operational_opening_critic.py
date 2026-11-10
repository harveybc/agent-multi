import copy
import pytest
from app.operational_opening_critic import validate_labels,BOUNDARY,END


def records():
    return [dict(role=role,timestamp=end-100,terminal=end,window_start=start,position=0,forbidden=True,
        suffixes=[dict(actions=[dict(soft_return=.1)]*3) for _ in range(3)])
        for role,start,end in [('fit',0,BOUNDARY),('monitor',BOUNDARY,END)]]


def test_fit_return_tail_never_reaches_monitor():
    r=records();assert validate_labels(r)
    r[0]['terminal']=END
    with pytest.raises(ValueError,match='SUPPORT'):validate_labels(r)


def test_equivalent_actions_and_real_position_gates():
    r=records();r[0]['position']=1
    with pytest.raises(ValueError):validate_labels(r)
    r=copy.deepcopy(records());r[0]['suffixes'][0]['actions']=[dict(soft_return=.1),dict(soft_return=.2),dict(soft_return=.1)]
    with pytest.raises(ValueError,match='EQUIVALENT'):validate_labels(r)
