from docs.work_plan.generalization.short_target381 import summarize

def test_head_error_vs_changed_target_and_tie():
    rows=[dict(q='20Q1',risk=2.,scalar=False,table=True,A=-2.,B=-2.,C=-4.),
          dict(q='20Q2',risk=2.,scalar=False,table=True,A=2.,B=-3.,C=-2.),
          dict(q='20Q3',risk=2.,scalar=False,table=True,A=0.,B=0.,C=0.)]
    s=summarize(rows)
    assert s['flips']['AB']==1 and s['flips']['BC']==0
    assert s['targets']['A']['ties']==1
    assert s['targets']['C']['table_regret_R']==1
    assert s['table_C_errors_by_A']=={'wrong':1,'right':1,'tie':0}
