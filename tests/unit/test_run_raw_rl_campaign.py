import json
import sqlite3

from tools import run_raw_rl_campaign as runner


def test_restart_adopts_existing_journal_without_new_campaign(tmp_path, monkeypatch):
    (tmp_path/'STATUS.json').write_text('{}')
    seen = []
    monkeypatch.setattr(runner.queue, 'run', lambda path, root, **kw: seen.append(kw) or 0)
    assert runner.run(tmp_path) == 0
    assert seen == [{'resume': True}]


def test_status_has_no_eta_before_measured_training(tmp_path):
    (tmp_path/'STATUS.json').write_text(json.dumps({'schema':'curated_rl_queue_status.v1',
        'state':'RUNNING', 'tasks':[{'state':'RUNNING', 'arm':'NO_TRADE', 'split':'train'}]}))
    value = runner.read_status(tmp_path)
    assert value['current_training_eta_seconds'] is None
    assert value['completed_tasks'] == 0


def test_results_keep_train_validation_and_null_risk_distinct(tmp_path):
    summary = tmp_path/'summary.json'
    summary.write_text(json.dumps({'net_profit': 0, 'sharpe_ratio': None, 'trades_closed': 0}))
    (tmp_path/'STATUS.json').write_text(json.dumps({'schema':'curated_rl_queue_status.v1',
        'state':'COMPLETED_OUTBOX_PENDING', 'tasks':[{'state':'ENQUEUED', 'arm':'NO_TRADE',
            'seed':7, 'split':'validation', 'summary':str(summary)}]}))
    rows = runner.results(tmp_path)
    assert rows[0]['split'] == 'validation'
    assert rows[0]['net_profit'] == 0 and rows[0]['sharpe_ratio'] is None


def test_close_requires_readback_then_keeps_closed_physical_snapshot(tmp_path, monkeypatch):
    (tmp_path/'CONTRACT.json').write_text('{}')
    (tmp_path/'STATUS.json').write_text(json.dumps({'schema':'curated_rl_queue_status.v1',
        'state':'COMPLETED_OUTBOX_PENDING', 'tasks':[{'state':'ENQUEUED'}]}))
    connection = sqlite3.connect(tmp_path/'warehouse-outbox.sqlite')
    connection.execute('create table example(value)')
    connection.execute('insert into example values (7)')
    connection.commit()
    connection.close()
    monkeypatch.setattr(runner.queue, 'pump_warehouse', lambda *a: 2)
    assert runner.publish_and_close(tmp_path) == 2
    assert not (tmp_path/'CLOSE.json').exists()
    monkeypatch.setattr(runner.queue, 'pump_warehouse', lambda *a: 0)
    (tmp_path/'WAREHOUSE_STATUS.json').write_text(json.dumps({'warehouse':{
        'expected':1, 'delivered':1, 'pending':0, 'mismatch':0}}))
    assert runner.publish_and_close(tmp_path) == 0
    close = json.loads((tmp_path/'CLOSE.json').read_text())
    assert close['state'] == 'COMPLETED_AND_WAREHOUSE_READBACK_VERIFIED'
    connection = sqlite3.connect(tmp_path/close['physical_copy'])
    assert connection.execute('select value from example').fetchone() == (7,)
    connection.close()
