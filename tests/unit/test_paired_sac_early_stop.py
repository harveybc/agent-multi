import csv
import math
from pathlib import Path

import pytest

from agent_plugins.paired_early_stopping import PairedMonitor, daily_growth


def trace(tmp_path, rows):
    path = tmp_path / 'returns.csv'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['timestamp', 'equity'])
        writer.writeheader()
        writer.writerows(dict(timestamp=t, equity=e) for t, e in rows)
    return path


def test_daily_growth_uses_day_close_and_initial_equity(tmp_path):
    p = trace(tmp_path, [('2023-01-02T15:00:00+00:00', 105),
                         ('2023-01-02T20:00:00+00:00', 110),
                         ('2023-01-03T20:00:00+00:00', 121)])
    result = daily_growth(p, 100, 3, 'America/New_York')
    assert result['days'] == 2
    assert result['mean_daily_log_return'] == pytest.approx(math.log(1.1))


@pytest.mark.parametrize('equity', [0, -1, float('nan'), float('inf')])
def test_bad_equity_not_a_valid_monitor(tmp_path, equity):
    p = trace(tmp_path, [('2023-01-02T20:00:00+00:00', equity)])
    with pytest.raises(ValueError, match='PAIRED_ES_'):
        daily_growth(p, 100, 1, 'America/New_York')


def test_partial_support_refused(tmp_path):
    p = trace(tmp_path, [('2023-01-02T20:00:00+00:00', 101)])
    with pytest.raises(ValueError, match='PAIRED_ES_SUPPORT'):
        daily_growth(p, 100, 2, 'America/New_York')


def test_patience_and_best_are_not_reset_by_subthreshold_change():
    m = PairedMonitor(patience=10, min_delta=1e-8)
    assert m.observe(5000, .01, .01)['improved']
    for step in range(10000, 55000, 5000):
        result = m.observe(step, .01 + 1e-10, .01)
        assert not result['stop']
        assert not result['improved']
    result = m.observe(55000, .01, .01)
    assert result['stop'] and result['patience_used'] == 10
    assert m.best_step == 5000


def test_improvement_resets_patience_and_minimizes_negative_mean():
    m = PairedMonitor(patience=10, min_delta=1e-8)
    m.observe(5000, .01, -.02)
    m.observe(10000, .001, -.02)
    result = m.observe(15000, .01, .02)
    assert result['score'] == pytest.approx(-.015)
    assert result['improved'] and result['patience_used'] == 0
    assert m.best_step == 15000


def test_different_duration_growth_is_comparable(tmp_path):
    train = trace(tmp_path, [('2023-01-02T20:00:00+00:00', 110),
                             ('2023-01-03T20:00:00+00:00', 121)])
    a = daily_growth(train, 100, 2, 'America/New_York')
    val = trace(tmp_path, [('2024-01-02T20:00:00+00:00', 110)])
    b = daily_growth(val, 100, 1, 'America/New_York')
    assert a['mean_daily_log_return'] == pytest.approx(b['mean_daily_log_return'])


def test_nonfinite_score_and_repeated_step_refused():
    m = PairedMonitor(patience=10, min_delta=1e-8)
    m.observe(5000, .01, .01)
    with pytest.raises(ValueError, match='PAIRED_ES_'):
        m.observe(5000, .01, .01)
    with pytest.raises(ValueError, match='PAIRED_ES_'):
        m.observe(10000, float('nan'), .01)


def test_pair_reports_are_published_once_per_role_with_integrity(tmp_path, monkeypatch):
    import json
    from tools import selected_rl_comparison as selected
    from tools import curated_rl_campaign as queue
    directory=tmp_path/'task'/'attempt-1'/'paired-monitor'/'step-005000'
    directory.mkdir(parents=True)
    reports=[]
    for role in ('train','validation'):
        file=directory/(role+'.json'); file.write_text('{}')
        reports.append(dict(arm='SAC_FLAT',seed=7,split=role,
            summary=str(file),artifacts=dict(summary=queue.digest(file))))
    pair=directory/'PAIR.json'
    pair.write_text(json.dumps(dict(schema='paired_sac_checkpoint.v1',reports=reports)))
    calls=[]
    monkeypatch.setattr(queue,'_enqueue',lambda body,state,task,box:calls.append(task))
    body=dict(common_config=dict(paired_early_stopping=True),seeds=[7])
    assert selected.publish_monitors(body,tmp_path,{})==2
    assert [t['split'] for t in calls]==['train','validation']
    Path(reports[0]['summary']).write_text('changed')
    with pytest.raises(ValueError,match='MONITOR_ARTIFACT_CHANGED'):
        selected.publish_monitors(body,tmp_path,{})
