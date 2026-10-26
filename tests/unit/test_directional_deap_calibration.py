import pytest

from app.directional_risk_calibration import DirectionalRisk, fitness_components, objective


def test_domains_and_direction():
    p=DirectionalRisk((0,1,0,1,0,1))
    assert p.for_command(1)==(0,1,0)
    assert p.for_command(2)==(1,0,1)
    for values in ((0,)*5,(float('nan'),)*6,(-.01,)*6,(1.01,)*6):
        with pytest.raises(ValueError):DirectionalRisk(values)


def test_fitness_components_and_constraint_priority():
    waves=[]
    for weeks in (8,6):
        for mode in ('ideal','inverted','nop'):
            waves.append(dict(mode=mode,weeks=weeks,metrics=dict(sharpe_daily_sqrt365=2,
                net_profit_usd=1000,trades=weeks*5)))
    c=fitness_components(waves,10000)
    assert c==dict(sharpe=2.,return_fraction=.1,orders_per_day=1.)
    assert objective('sharpe_return_activity',c,True,[])==3.1
    assert objective('sharpe_return',c,True,[])==2.1
    assert objective('sharpe',c,True,[])==2.
    assert objective('sharpe',c,False,['bad'])<0


def test_nonfinite_fitness_rejected():
    with pytest.raises(ValueError):
        objective('sharpe',dict(sharpe=float('nan'),return_fraction=0,orders_per_day=0),True,[])


def test_deap_resume_and_complete_adoption(tmp_path):
    from tools.deap_output_calibration import evolve
    calls=[]
    def score(x):
        calls.append(tuple(x))
        return -sum((v-.3)**2 for v in x)
    cfg=dict(population=20,generations=4,patience=10,tolerance=1e-8)
    a=evolve(tmp_path/'a',7,score,cfg,{'test':1},stop_after=2)
    assert a['generation']==2
    b=evolve(tmp_path/'a',7,score,cfg,{'test':1})
    c=evolve(tmp_path/'b',7,score,cfg,{'test':1})
    assert b==c
    n=len(calls)
    assert evolve(tmp_path/'a',7,score,cfg,{'test':1})==b
    assert len(calls)==n
