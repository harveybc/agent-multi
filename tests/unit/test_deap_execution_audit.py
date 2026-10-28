import copy

import pytest

from tools.audit_deap_output_calibration import validate_state


def fixture():
    cfg=dict(population=20,generations=100,patience=10,tolerance=1e-8)
    state=dict(generation=0,population=[[.5]*6]*20,scores=[2.1]*20,best=[.5]*6,best_score=2.1,
        anchor=2.1,patience=0,evaluations=20,stop_reason=None,
        history=[dict(generation=0,best_score=2.1,patience=0,evaluations=20)])
    record=dict(components=dict(sharpe=2.,return_fraction=.1,orders_per_day=.5),
                report=dict(eligible=True,reasons=[]))
    return cfg,state,lambda genes:record


def test_checkpoint_scores_bound_to_actual_objective():
    cfg,state,lookup=fixture()
    validate_state(state,'sharpe_return',cfg,lookup)
    with pytest.raises(ValueError,match='OBJECTIVE_SCORE'):
        validate_state(state,'sharpe',cfg,lookup)


def test_false_patience_and_population_rejected():
    cfg,state,lookup=fixture()
    for name,value,code in [('patience',1,'PATIENCE'),('population',[[.5]*6],'POPULATION'),
                            ('best_score',3,'BEST_SCORE')]:
        s=copy.deepcopy(state);s[name]=value
        with pytest.raises(ValueError,match=code):validate_state(s,'sharpe_return',cfg,lookup)
