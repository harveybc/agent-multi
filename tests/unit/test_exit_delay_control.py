import copy

import numpy as np
import pytest

from app.exit_delay_control import fit_delay, evaluate_delay
from app.optimal_exit import outcome
from tools.optimal_exit_study import summarize
from tools.exit_learning_repair import qualifies


def cases(peaks=(1, 1)):
    rows=[]
    for d in (1,-1):
        for i,peak in enumerate(peaks):
            net=[0.,1.,2.,-1.];net[peak]=10.
            rows.append(dict(id=f'{d}_{i}',direction=d,close_net=net,terminal_net=-5.,risk_usd=10.))
    return rows


def test_train_delay_is_frozen_not_refitted_on_confirmation():
    control=fit_delay(cases())
    assert control['delay']==1 and control['selection_split']=='TRAIN'
    report=evaluate_delay(cases((2,2)),control)
    assert report['mean_regret_risk']>0
    assert fit_delay(cases((2,2)))['delay']==2
    mutated=copy.deepcopy(control);mutated['delay']=2
    with pytest.raises(ValueError,match='DELAY_CONTROL_IDENTITY'):
        evaluate_delay(cases(),mutated)


def test_delay_after_automatic_terminal_keeps_real_terminal():
    control=fit_delay(cases((2,2)))
    shorter=[dict(c,close_net=[0.]) for c in cases()]
    report=evaluate_delay(shorter,control)
    assert report['early_close_share']==0.
    assert report['mean_net_usd']==-5.


def test_fixed_clock_is_rejected_but_a_better_conditional_policy_can_pass():
    for peaks,expected in (((1,1),False),((1,2),True)):
        source=cases(peaks);control=fit_delay(source)
        rows=[]
        for c in source:
            actions=np.zeros(len(c['close_net']));actions[np.argmax(c['close_net'])]=1
            rows.append(outcome(c,actions))
        report=dict(deterministic=dict(summary=summarize(rows)),fixed_delay_control=control,
                    constant_hazard={str(d):dict(regret_risk=.5) for d in (1,-1)})
        assert qualifies(report,source)==expected
        del report['fixed_delay_control']
        with pytest.raises(ValueError,match='FIXED_DELAY_CONTROL_REQUIRED'):
            qualifies(report,source)
