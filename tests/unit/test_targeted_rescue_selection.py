from app.targeted_rescue_selection import best_rescue


def test_rejected_is_never_selected():
    bad=dict(accepted=False,regret=0.,cost=0.,row=0)
    good=dict(accepted=True,regret=2.,cost=1.,row=3)
    assert best_rescue([bad]) is None
    assert best_rescue([bad,good])==good


def test_declared_order_and_permutation_stability():
    rows=[dict(accepted=True,regret=r,cost=c,row=i) for r,c,i in [(2.,1.,0),(1.,2.,1),(1.,1.,3),(1.,1.,2)]]
    assert best_rescue(rows)['row']==2
    assert best_rescue(rows[::-1])['row']==2
