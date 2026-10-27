from app.optimal_exit_search import panel,score,search


def cases():
    return [dict(close_net=[0.,10.,30.,-5.],terminal_net=-40.,risk_usd=50.,direction=d)
            for d in (1,-1)]


def test_ideal_closed_correctly_throughout_declared_bias_bounds():
    rows=panel(cases(),7)
    for genes in ([0,0],[-2,2],[2,-2]):
        assert score(genes,rows)['max_ideal_regret']==0.


def test_search_is_reproducible_and_has_two_real_close_parameters():
    rows=panel(cases(),19)
    a=search(rows,'negative_exit_regret',7,population=4,generations=2,patience=2)
    b=search(rows,'negative_exit_regret',7,population=4,generations=2,patience=2)
    assert a==b and len(a['genes'])==2
    assert all(-2<=v<=2 for v in a['genes'])
    assert a['calibration']['max_ideal_regret']==0.
