import numpy as np
import pytest

from tests.unit.test_discrete_curve_native import curve_env
from tests.unit.test_double_tcn_probe import probe_recipe
from app.three_signal_learning import synthetic_env,evaluate
from tools.run_discrete_curve_probe import read,write
from tools.attribute_ranking_improvement import attribute_pair


def test_full_pair_attribution_reconciles_native_accounts(curve_env,tmp_path):
    def make():
        return synthetic_env(probe_recipe(),curve_env.risk_config,17,1,.99)[0]
    env=make();n=len(env.inputs.valid)
    a=np.tile([3.,0.,0.],(n,1));b=a.copy();b[::7]=[0.,3.,0.]
    evaluate(make(),a,save=tmp_path/'a');evaluate(make(),b,save=tmp_path/'b')
    result=attribute_pair(make,env.inputs.row_ts_ns,tmp_path/'a',tmp_path/'b',lambda **kw:None)
    assert result['reconciliation']['error_usd']==pytest.approx(0.,abs=1e-7)
    assert result['candidate_trades']>0 and result['events']
    assert sum(x['contribution_usd'] for x in result['events'])==pytest.approx(result['delta_usd'],abs=1e-7)
    # Matching the diagnostic continuation to the control tape must reproduce
    # isolated returns without changing any cash contribution.
    extended=attribute_pair(make,env.inputs.row_ts_ns,tmp_path/'a',tmp_path/'b',
        lambda **kw:None,calibration_signals=np.zeros(n,dtype=np.int64))
    assert extended['delta_usd']==result['delta_usd']
    for before,after in zip(result['events'],extended['events']):
        assert after['contribution_usd']==before['contribution_usd']
        assert after['calibration_old']==before['isolated_old']
        assert after['calibration_new']==before['isolated_new']
    with pytest.raises(ValueError,match='CALIBRATION_TAPE'):
        attribute_pair(make,env.inputs.row_ts_ns,tmp_path/'a',tmp_path/'b',
            lambda **kw:None,calibration_signals=np.zeros(n-1,dtype=np.int64))
    trace=read(tmp_path/'b/ACTIONS.json');trace[0]['timestamp']+=1
    write(tmp_path/'b/ACTIONS.json',trace)
    with pytest.raises(ValueError,match='SUPPORT'):
        attribute_pair(make,env.inputs.row_ts_ns,tmp_path/'a',tmp_path/'b',lambda **kw:None)
