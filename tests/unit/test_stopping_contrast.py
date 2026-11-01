import copy
import numpy as np
import pytest

from app.three_signal_learning import Patience
from app.stopping_contrast import ShadowRule, make_rules, observe_rules


def report(train,val,eligible=True):
    return dict(score=(train+val)/2,eligible=eligible,
                internal_validation=[{'return_per_week':val}]*3)


@pytest.mark.parametrize('patience',[3,5,10])
def test_control_matches_existing_stopping(patience):
    old=Patience(100,patience,.001)
    new=ShadowRule('control',patience,100,.001,False)
    for step,value in enumerate([0,1,1,2]+[-1]*20,1):
        eligible=step%3!=0
        expected=old.observe(value,eligible,step*50)
        actual=new.observe(report(value,value,eligible),step*50)
        assert actual['improved']==expected['improved']
        assert new.bad==expected['bad']
        assert new.stopped==expected['stop']
        if new.stopped: break


def test_guard_rejects_train_compensation():
    r=ShadowRule('guard',3,100,.001,True)
    assert r.observe(report(0,0),50)['improved']
    assert not r.observe(report(10,-1),100)['improved']
    assert r.selected_step==50
    assert r.observe(report(2,.1),150)['improved']
    assert r.selected_step==150


def test_no_eligible_and_frozen_rule():
    r=ShadowRule('guard',2,100,.001,True)
    r.observe(report(0,0,False),100)
    r.observe(report(0,0,False),150)
    assert r.stopped and r.reason=='no_eligible_checkpoint' and r.selected_step is None
    before=copy.deepcopy(r.state())
    r.observe(report(100,100),200)
    assert r.state()==before


def test_six_rules_and_rng_independence():
    rules=make_rules(dict(minimum_steps=100,min_delta=.001))
    assert len(rules)==6
    before=np.random.get_state()
    result=observe_rules(rules,report(1,1),100)
    assert not result['stop']
    after=np.random.get_state()
    assert all(np.array_equal(a,b) for a,b in zip(before,after))


def test_guard_nonfinite_rejected():
    r=ShadowRule('guard',3,100,.001,True)
    with pytest.raises(ValueError,match='NONFINITE'):
        r.observe(report(0,float('nan')),100)


def test_stopped_checkpoint_cannot_be_replaced_by_future_success():
    r=ShadowRule('original',1,100,.001,False)
    r.observe(report(1,1),50)
    r.observe(report(0,0),100)
    assert r.stopped and r.selected_step==50
    before=copy.deepcopy(r.state())
    r.observe(report(100,100),150)
    assert r.state()==before


def test_receipt_tamper_rejected(tmp_path):
    from tools.run_stopping_contrast import seal, verified
    for name in ('HISTORY.json','TRAINING.json','RULES.json','last.pt','INITIAL_PROBE.json'):
        (tmp_path/name).write_text('{}')
    rules=make_rules(dict(minimum_steps=100,min_delta=.001))
    result=dict(seed=7,rules={r.name:r.state() for r in rules},accounts={})
    seal(tmp_path,dict(identity='unit-test'),result)
    assert verified(tmp_path/'RESULT.json','unit-test')
    (tmp_path/'last.pt').write_text('changed')
    with pytest.raises(ValueError,match='ARTIFACT_CHANGED'):
        verified(tmp_path/'RESULT.json','unit-test')
