import pytest
from tools.render_sac_reliability import termination


def test_fixed_updates_are_not_epochs_or_patience():
    arm=dict(updates=512,samples=65536,replay_rows=24025,distinct_rows_sampled=22435,
        history=[dict(step=s,metrics={}) for s in (0,64,256,512)])
    result=termination(arm)
    assert result['optimizer_updates']==512 and result['online_training_epochs']==0
    assert result['early_stopping_enabled'] is False and result['patience_used'] is None
    assert result['batch_size']==128 and result['account_evaluation_updates']==[0,64,256,512]
    arm['updates']=511
    with pytest.raises(ValueError,match='INCOMPLETE'):termination(arm)
