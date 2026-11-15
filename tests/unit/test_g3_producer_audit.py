import numpy as np
import pandas as pd
import pytest

from app.g3_producer_audit import dependency_inventory, parity, support_report


def frame():
    return pd.DataFrame({'ATR':np.arange(40,dtype=float)},
                        index=pd.date_range('2020-01-01',periods=40,freq='h',tz='UTC'))


def test_activity_transitive_dependencies_not_just_prices():
    inv = dependency_inventory(['ATR','act.p_hit','fac.fx_f1.ret_24'],[
        'dly.dax.ret_5d@48','dly.spx.ret_1d@48','cal.utc_dow_cos@120'])
    activity = inv['columns'][1]
    assert set(activity['primitive_dependencies']) == {
        'DAX_RELEASES','SPX_RELEASES','FROZEN_ACTIVITY_OOF',
        'ADMITTED_TIMESTAMPS','CLOSURES','TZ_DATABASE'}
    assert 'USDCHF_CLOSE' in inv['columns'][2]['primitive_dependencies']
    assert inv['activity_predictors'][-1]['lag_rows'] == 120


@pytest.mark.parametrize('columns,activity', [(['new.column'],[]),(['ATR','ATR'],[]),
    (['ATR@clip'],[]),(['act.p_hit'],[]),(['act.p_hit'],['new.column@1']),
    (['act.p_hit'],['dow_cos@-1'])])
def test_unreviewed_inputs_rejected(columns,activity):
    with pytest.raises(ValueError,match='G3_'):
        dependency_inventory(columns,activity)


def test_window_support_keeps_gaps_and_stride():
    df = frame()
    df.iloc[26] = np.nan
    out = support_report(df,['ATR'],list(range(-24,1,2)))
    assert out['finite_rows']['count'] == 39
    assert out['finite_windows']['count'] == 9
    assert out['by_column']['ATR']['missing'] == 1
    assert np.isnan(df.iloc[26,0])


@pytest.mark.parametrize('offsets', [[0,-2],[-2,1],[-2,-2,0],[-2],[],[-2.,0]])
def test_invalid_window_rejected(offsets):
    with pytest.raises(ValueError,match='WINDOW'):
        support_report(frame(),['ATR'],offsets)


def test_missing_reordered_and_duplicate_axes_rejected():
    df = frame()
    with pytest.raises(ValueError,match='MISSING_COLUMN'):
        support_report(df,['missing'],[0])
    with pytest.raises(ValueError,match='GRID'):
        parity(df.iloc[::-1],df)
    with pytest.raises(ValueError,match='GRID'):
        support_report(pd.concat([df,df]),['ATR'],[0])
    with pytest.raises(ValueError,match='AXES'):
        parity(df.iloc[1:],df.iloc[:-1])


def test_parity_detects_values_masks_and_all_nan():
    expected = frame()
    actual = expected.copy()
    actual.iloc[10] = np.nan
    assert not parity(actual,expected)['ATR']['passed']
    actual = expected+1
    assert not parity(actual,expected)['ATR']['passed']
    assert parity(expected,expected)['ATR']['exact']
    actual[:] = np.nan
    assert not parity(actual,actual)['ATR']['passed']


def test_atomic_resume_rejects_tampered_report(tmp_path,monkeypatch):
    import tools.audit_g3_producers as tool
    from app.g3_producer_audit import identity
    monkeypatch.setattr(tool,'authenticate',lambda spec:None)
    spec = {'test':True}
    from pathlib import Path
    binding = dict(spec_sha256=identity(spec),code={str(p):tool.sha(p) for p in (
        Path(tool.__file__).resolve(),Path(tool.__file__).resolve().parents[1]/'app/g3_producer_audit.py',
        Path(tool.__file__).resolve().parent/'recover_stl_price_warmup.py')})
    tool.write(tmp_path/'BINDING.json',binding)
    tool.write(tmp_path/'RESULT.json',{'status':'test'})
    tool.write(tmp_path/'DONE.json',dict(binding=binding,files={
        'RESULT.json':tool.sha(tmp_path/'RESULT.json')}))
    assert tool.run(spec,tmp_path)['status'] == 'test'
    tool.write(tmp_path/'RESULT.json',{'status':'changed'})
    with pytest.raises(ValueError,match='TERMINAL_DIGEST'):
        tool.run(spec,tmp_path)


def test_external_perturbation_handles_integer_cot_without_mutating_sources():
    from tools.audit_g3_producers import perturb_external_after
    cutoff = pd.Timestamp('2020-01-02',tz='UTC')
    fx = pd.Series([1.,2.,3.],index=pd.date_range('2020-01-01',periods=3,tz='UTC'))
    daily = fx.copy()
    daily.index = daily.index.tz_localize(None)
    cot = pd.DataFrame({'Date':daily.index,'EUR_Long':[4,5,6],
                       'EUR_Short':[2,1,3],'EUR_Net_Long':[2,4,3]})
    original = dict(fx={'test':fx},daily={'test':daily},cot=cot)
    changed = perturb_external_after(original,cutoff)
    assert list(changed['cot'].EUR_Net_Long) == pytest.approx([2,4.8,3.6])
    assert list(cot.EUR_Net_Long) == [2,4,3]
    assert list(fx) == [1,2,3]
    assert changed['fx']['test'].iloc[0] == fx.iloc[0]
    assert changed['daily']['test'].iloc[-1] == pytest.approx(3.6)


def test_joint_history_does_not_fill_or_count_training():
    from tools.audit_g3_history import summarize
    df = frame()
    df['other'] = 1.
    df.iloc[3,1] = np.nan
    report = summarize(df,df.index[10])
    assert report['candidate_rows'] == 10
    assert report['finite_rows'] == 9
    assert report['missing_by_column']['other'] == 1
    assert report['last'] == str(df.index[9])
