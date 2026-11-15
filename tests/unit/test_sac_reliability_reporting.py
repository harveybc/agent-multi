import pytest
import json
from tools.render_sac_reliability import termination
from app.training_diagnostics import render_diagnostics


def test_fixed_updates_are_not_epochs_or_patience():
    arm=dict(updates=512,samples=65536,replay_rows=24025,distinct_rows_sampled=22435,
        history=[dict(step=s,metrics={}) for s in (0,64,256,512)])
    result=termination(arm)
    assert result['optimizer_updates']==512 and result['online_training_epochs']==0
    assert result['early_stopping_enabled'] is False and result['patience_used'] is None
    assert result['batch_size']==128 and result['account_evaluation_updates']==[0,64,256,512]
    arm['updates']=511
    with pytest.raises(ValueError,match='INCOMPLETE'):termination(arm)


def test_epoch_zero_checkpoint_remains_in_exported_history(tmp_path):
    rows=[dict(epoch=e,metrics={'TRAIN':{'return_pct':1},'VALIDATION':{'return_pct':1-e}},
        monitor={'score':1-e,'best':1,'bad':e,'improved':e==0}) for e in (0,1)]
    result=render_diagnostics(rows,tmp_path,{},best_epoch=0)
    assert result['best_epoch']==0
    assert result['epochs']==1 and result['evaluation_points']==2
    assert json.loads((tmp_path/'DIAGNOSTICS.json').read_text())['best_epoch']==0
    assert (tmp_path/'EPOCHS.csv').read_text().splitlines()[1].startswith('0,')
