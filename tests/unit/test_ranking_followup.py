from types import SimpleNamespace

import pytest

from tools import run_ranking_followup as followup


def test_paired_summary_preserves_all_endpoints_and_negative_results(monkeypatch):
    def rows(value):
        return [dict(arm=arm, actor_updates=point, period=period,
                     net_profit_usd=100 + delta, return_pct=1 + delta / 100,
                     sharpe=delta / 20, max_drawdown_pct=5 - delta / 100, eligible=False)
                for arm, delta in [('control', value), ('paired_greedy', value - 10)]
                for point in (1, 8, 32) for period in ('TRAIN', 'VALIDATION')]
    monkeypatch.setattr(followup, 'trading_rows', rows)
    result = followup.paired_rows(dict(r19=19, r43=43))
    assert len(result) == 24
    assert {r['actor_updates'] for r in result} == {1, 8, 32}
    assert all(r['delta_net_usd'] == -10 for r in result if r['arm'] == 'paired_greedy')
    assert not any(r['eligible'] for r in result)


def test_sequential_attribution_replicas_reports_and_resume(tmp_path, monkeypatch):
    order = []
    args = SimpleNamespace(**{k: tmp_path / k for k in
        ('source', 'source_runtime', 'probe', 'attribution', 'calibration', 'ranking', 'output')},
        worker_runtime='runtime', worker_host='worker', worker_python='python', gpu_id='gpu')
    def invoke(command, **kwargs):
        if 'tools.run_action_ranking_contrast' in command:
            order.append(command[command.index('--replica') + 1])
        elif command[0] == 'ssh':
            order.append('attribution')
    monkeypatch.setattr(followup.subprocess, 'run', invoke)
    monkeypatch.setattr(followup, 'verify_collection', lambda path: {})
    monkeypatch.setattr(followup, 'export', lambda path, output: order.append('report_' + path.name))
    monkeypatch.setattr(followup, 'summarize', lambda path: order.append('combined'))
    followup.run(args)
    assert order == ['attribution', 'r19', 'report_r19', 'r43', 'report_r43', 'combined']
    followup.run(args)
    assert len(order) == 6
    config = followup.read(args.output / 'CONFIG.json'); config['replicas'] = ['changed']
    followup.write(args.output / 'CONFIG.json', config)
    with pytest.raises(ValueError, match='CONFIG_CHANGED'):
        followup.run(args)
