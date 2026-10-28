import math
import json
import re

import pytest

from tools.paired_checkpoint_diagnostics import compare_curve, trace_metrics
from tools.paired_checkpoint_diagnostics import analyze, digest, report_key


def point(step, train, validation):
    return dict(step=step, train_growth=train, validation_growth=validation)


def test_governed_report_key_shape_and_identity():
    result = dict(contract_sha256='a' * 64, analyzer_sha256='b' * 64)
    assert re.fullmatch(r'[A-Za-z0-9._:-]{1,128}', report_key(result, 5000))
    assert report_key(result, 5000) != report_key(result, 10000)
    assert report_key(result, 5000) != report_key({**result, 'analyzer_sha256':'c' * 64}, 5000)


def test_mean_resets_while_validation_worsens_against_incumbent():
    result = compare_curve([point(5000, .01, .02), point(10000, .05, .01)])
    assert result['resets_worse_than_incumbent'] == 1
    assert result['selected_step'] == 10000
    assert result['rows'][1]['previous_selected_step'] == 5000
    assert result['rows'][1]['validation_delta_incumbent'] == pytest.approx(-.01)


def test_distinguishes_previous_checkpoint_from_selected_incumbent():
    result = compare_curve([point(5000, .01, .02), point(10000, -.1, .04),
                            point(15000, .02, .03)])
    assert result['resets_worse_than_previous'] == 1
    assert result['resets_worse_than_incumbent'] == 0
    assert result['rows'][2]['previous_selected_step'] == 5000


def test_decline_does_not_select_validation_champion():
    result = compare_curve([point(5000, 0., .02), point(10000, .1, -.01)])
    assert result['selected_step'] == 10000
    assert result['positive_validation_checkpoints'] == 1


def test_exact_threshold_is_not_improvement():
    result = compare_curve([point(5000, 0., 0.), point(10000, 2e-8, 0.)])
    assert result['selected_step'] == 5000
    assert result['rows'][1]['patience_used'] == 1


@pytest.mark.parametrize('rows', [[], [point(5000, math.nan, 0)],
    [point(5000, 0, 0), point(5000, 0, 0)]])
def test_bad_curve_rejected(rows):
    with pytest.raises(ValueError):
        compare_curve(rows)


def test_trace_uses_all_observed_dates_and_timestamp_support(tmp_path):
    path = tmp_path / 'trace.csv'
    path.write_text('timestamp,equity\n2024-01-01T12:00:00Z,100\n'
                    '2024-01-02T12:00:00Z,110\n')
    a = trace_metrics(path, 100, 'America/New_York')
    assert a['growth'] == pytest.approx(math.log(1.1) / 2)
    assert a['return_pct'] == pytest.approx(10)
    assert a['rows'] == 2
    path.write_text(path.read_text().replace(',110', ',120'))
    b = trace_metrics(path, 100, 'America/New_York')
    assert a['support_sha256'] == b['support_sha256']
    path.write_text(path.read_text().replace('2024-01-02', '2024-01-03'))
    assert a['support_sha256'] != trace_metrics(path, 100, 'America/New_York')['support_sha256']


@pytest.mark.parametrize('body', ['', 'timestamp,equity\n2024-01-01,100\n',
    'timestamp,equity\n2024-01-01T12:00:00Z,0\n',
    'timestamp,equity\n2024-01-02T12:00:00Z,100\n2024-01-01T12:00:00Z,100\n'])
def test_invalid_trace_rejected(tmp_path, body):
    path = tmp_path / 'trace.csv'
    path.write_text(body)
    with pytest.raises(ValueError):
        trace_metrics(path, 100, 'America/New_York')


@pytest.fixture
def campaign(tmp_path):
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    root = tmp_path / 'source'
    save(root / 'CONTRACT.json', dict(common_config=dict(paired_early_stopping=dict(
        min_delta=1e-8, patience=10)), splits={r: dict(curated_input_manifest_sha256=r)
                                            for r in ('train', 'validation')}))
    points = []
    for step, end in ((5000, 110), (10000, 120)):
        directory = root / 'unit' / 'attempt-1' / 'paired-monitor' / f'step-{step:06d}'
        directory.mkdir(parents=True)
        checkpoint = directory / 'policy.zip'
        checkpoint.write_bytes(b'synthetic-not-a-model')
        reports, growth = [], {}
        for role, year in (('train', 2023), ('validation', 2024)):
            child = directory / role
            child.mkdir()
            trace = child / 'returns.csv'
            trace.write_text(f'timestamp,equity\n{year}-01-01T12:00:00Z,100\n'
                             f'{year}-01-02T12:00:00Z,{end}\n')
            save(child / 'config.json', dict(mode='inference', initial_cash=100,
                entry_day_timezone='America/New_York', window_size=1, observation_stride_bars=2))
            save(child / 'effective_config.json', {})
            save(child / 'summary.json', dict(curated_input_identity=dict(manifest_sha256=role,
                split=role, row_count=3), evaluation_truncated_by_cap=False,
                static_policy=dict(before='weights', after='weights'), total_return=end / 100 - 1))
            task = dict(split=role, config=str(child / 'config.json'), summary=str(child / 'summary.json'),
                effective_config=str(child / 'effective_config.json'), return_trace_file=str(trace),
                checkpoint=str(checkpoint))
            task['artifacts'] = {k: digest(v) for k, v in task.items() if k != 'split'}
            reports.append(task)
            growth[role] = dict(mean_daily_log_return=math.log(end / 100) / 2, days=2, rows=2)
        points.append(point(step, growth['train']['mean_daily_log_return'], growth['validation']['mean_daily_log_return']))
        save(directory / 'PAIR.json', dict(step=step, policy_tensor_sha256='weights',
             reports=reports, growth=growth, decision=compare_curve(points)['rows'][-1]))
    return root


def test_analyzer_full_trace_replay_does_not_write_source(campaign):
    before = {str(p): digest(p) for p in campaign.rglob('*') if p.is_file()}
    result, _, _ = analyze(campaign)
    assert result['selected_step'] == 10000
    assert result['resets_worse_than_incumbent'] == 0
    assert before == {str(p): digest(p) for p in campaign.rglob('*') if p.is_file()}


def test_corrupt_summary_rejected(campaign):
    next(campaign.rglob('summary.json')).write_text('{}')
    with pytest.raises(ValueError, match='ARTIFACT_HASH_MISMATCH'):
        analyze(campaign)


def test_forged_patience_rejected(campaign):
    path = next(campaign.rglob('PAIR.json'))
    value = json.loads(path.read_text())
    value['decision']['patience_used'] = 9
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='DECISION_REPLAY_MISMATCH'):
        analyze(campaign)


def test_resigned_different_timestamp_support_rejected(campaign):
    path = sorted(campaign.rglob('PAIR.json'))[-1]
    pair = json.loads(path.read_text())
    task = next(t for t in pair['reports'] if t['split'] == 'validation')
    from pathlib import Path
    trace = Path(task['return_trace_file'])
    trace.write_text(trace.read_text().replace('2024-01-02', '2024-01-03'))
    task['artifacts']['return_trace_file'] = digest(trace)
    path.write_text(json.dumps(pair))
    with pytest.raises(ValueError, match='SUPPORT_CHANGED_BETWEEN_CHECKPOINTS'):
        analyze(campaign)
