import csv
import hashlib
import json
import random

import numpy as np
import pytest
import torch

from app.training_diagnostics import render_diagnostics


def history():
    return [dict(epoch=i, steps=100*i, updates=25*i, training_seconds=2,
                 evaluation_seconds=1, metrics={
                     'TRAIN': dict(return_pct=100*i, return_per_week=i, sharpe=2,
                                   max_drawdown_pct=5, orders_per_week=10, win_rate=.6),
                     'VALIDATION': dict(return_pct=3-i, return_per_week=(3-i)/54,
                                        sharpe=-1, max_drawdown_pct=10, orders_per_week=4)},
                 monitor=dict(score=(3-i)/54, best=2/54, bad=i-1, improved=i==1))
            for i in (1, 2, 3)]


def test_artifacts_are_exact_bounded_and_do_not_mutate_training(tmp_path):
    rows=history(); rows[-1]['metrics']['VALIDATION']['win_rate']=.5
    original=json.dumps(rows, sort_keys=True)
    nr=np.random.get_state(); pr=random.getstate(); tr=torch.get_rng_state().clone()
    result=render_diagnostics(rows,tmp_path,dict(monitor_start_epoch=20,patience=20),best_epoch=1)
    assert result['best_epoch']==1 and result['epochs']==3
    assert json.dumps(rows,sort_keys=True)==original
    assert np.array_equal(nr[1],np.random.get_state()[1])
    assert pr==random.getstate() and torch.equal(tr,torch.get_rng_state())
    records=list(csv.DictReader((tmp_path/'EPOCHS.csv').open()))
    assert float(records[0]['TRAIN.return_pct'])==100
    assert float(records[-1]['VALIDATION.return_pct'])==0
    assert records[0]['VALIDATION.win_rate']==''
    for name in ('TRAIN_VALIDATION.png','TRAINING_DIAGNOSTICS.png'):
        assert (tmp_path/name).read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()}
    render_diagnostics(rows,tmp_path,dict(monitor_start_epoch=20,patience=20),best_epoch=1)
    assert hashes=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()}


def test_single_epoch_missing_fields_and_nonfinite(tmp_path):
    rows=history()[:1]; rows[0]['metrics']['TRAIN']['sharpe']=float('nan')
    render_diagnostics(rows,tmp_path,{},best_epoch=None)
    assert (tmp_path/'EPOCHS.csv').exists()
    with pytest.raises(ValueError,match='EPOCH_ORDER'):
        render_diagnostics(rows+rows,tmp_path,{})
