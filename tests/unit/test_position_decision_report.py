import numpy as np

from tools.report_position_decision_loss import conflict_metrics


def test_conflicts_partition_whole_banks_without_changing_rows():
    records=[dict(role='fit',gap=y,local_delta=y,effective_close=True) for y in (2.,-1.,1.,3.)]
    records.append(dict(role='monitor',gap=-5.,local_delta=-5.,effective_close=True))
    pred=np.ones(5);banks=[[0,1],[2,3],[4]]
    conflict=conflict_metrics(pred,records,banks,'fit',True)
    other=conflict_metrics(pred,records,banks,'fit',False)
    assert conflict['n']==2 and conflict['banks']==1 and conflict['bank_regret']==.5
    assert conflict['harmful_effective_closes']==1 and other['bank_regret']==0
    assert 'gap_rmse' not in conflict
    assert conflict_metrics(pred,records,banks,'monitor',True) is None
