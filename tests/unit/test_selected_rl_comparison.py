"""Selected-information transport only; synthetic inputs, no model learning."""
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

import pytest

from tools import selected_rl_comparison as selected
from tools import curated_rl_campaign as queue


ROOT = Path(__file__).resolve().parents[2]
COMPLETE = 'WITHIN_DECLARED_ADMITTED_POPULATION'
COLUMNS = ['causal_return_3', 'causal_range_5']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, body):
    path.write_text(json.dumps(body, sort_keys=True))
    return digest(path)


def columns_digest(columns):
    return hashlib.sha256(json.dumps(columns, separators=(',', ':')).encode()).hexdigest()


def fixture(tmp_path, parquet=False):
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    flags = dict(scope='SELECTED_ADMITTED_POPULATION', feature_selection_complete=True,
                 selection_complete=COMPLETE, full_inventory_complete=False)
    selection = dict(schema='selected_rl_selection.v1', **flags,
        feature_columns=COLUMNS, population_sha256='a' * 64,
        causal_transforms_verified=True, validation_used_for_selection=False,
        fit_period_start='2020-01-01T00:00:00+00:00',
        fit_period_end='2024-01-01T00:00:00+00:00',
        availability_column='SELECTED_AVAILABLE_AT',
        availability_semantics='row_max_selected_available_at',
        normalization=dict(mode='rolling_zscore', window_bars=256,
                           fit_scope='PAST_ONLY', binary_columns=[]))
    selection_pin = write(bundle / 'SELECTION.json', selection)
    close_pin = write(bundle / 'CLOSE.json', dict(
        state='FILTER_SELECTION_OLAP_AND_TWO_PHYSICAL_COPIES_VERIFIED',
        selection_sha256=selection_pin, selected_count=len(COLUMNS),
        pending=0, mismatch=0, verified_files=3, full_inventory_complete=False))
    calendar_pin = write(bundle / 'closures.json', [])
    receipt = dict(schema='selected_rl_bundle.v1', status='ADMITTED', **flags,
        selection_file='SELECTION.json', selection_sha256=selection_pin,
        selection_close_file='CLOSE.json', selection_close_sha256=close_pin,
        feature_columns=COLUMNS, feature_columns_sha256=columns_digest(COLUMNS),
        population_sha256='a' * 64, calendar_sha256=calendar_pin,
        closure_intervals_file='closures.json', closure_intervals_sha256=calendar_pin,
        splits={})
    for role, year, end in [('train', 2020, 2024), ('validation', 2024, 2025)]:
        stamps = [datetime(year, 1, 6, tzinfo=timezone.utc) + timedelta(hours=i) for i in range(40)]
        rows = [dict(DATE_TIME=s.isoformat(), OPEN=1.1, HIGH=1.2, LOW=1., CLOSE=1.15,
                     ASSUMED_SPREAD=0.0001, SELECTED_AVAILABLE_AT=s.isoformat(),
                     causal_return_3=i / 1000, causal_range_5=(i + 1) / 2000)
                for i, s in enumerate(stamps)]
        data = bundle / (role + ('.parquet' if parquet else '.csv'))
        if parquet:
            import pandas as pd
            pd.DataFrame(rows).to_parquet(data, index=False)
        else:
            with data.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        manifest = dict(schema='curated_rl_input.v1', status='ADMITTED', **flags,
            availability_verified=True, open_session_coverage_verified=True,
            availability_column='SELECTED_AVAILABLE_AT',
            input_sha256=digest(data), split=role, feature_columns=COLUMNS,
            selection_sha256=selection_pin, inventory_sha256='a' * 64,
            calendar_sha256=calendar_pin, timestamp_semantics='completed_bar_available_at',
            period_start=f'{year}-01-01T00:00:00+00:00',
            period_end=f'{end}-01-01T00:00:00+00:00', row_count=len(rows),
            expected_timestamps_sha256=hashlib.sha256(
                ''.join(s.isoformat() + '\n' for s in stamps).encode()).hexdigest())
        manifest_path = bundle / (role + '.json')
        receipt['splits'][role] = dict(input_file=data.name,
            manifest_file=manifest_path.name, manifest_sha256=write(manifest_path, manifest))
    request = dict(schema='selected_rl_comparison_request.v1',
        base_design_file=str(ROOT / 'examples/config/rl_raw_eurusd_static.v1.json'),
        base_design_sha256=digest(ROOT / 'examples/config/rl_raw_eurusd_static.v1.json'),
        receipt_file=str(bundle / 'RECEIPT.json'), receipt_sha256=write(bundle / 'RECEIPT.json', receipt),
        selection_sha256=selection_pin, population_sha256='a' * 64,
        feature_columns_sha256=columns_digest(COLUMNS),
        execution=dict(device='cpu', gym_root=str(tmp_path / 'gym'), gym_code_commit='a' * 40,
            task_timeout_seconds=dict(train=14400, validation=3600), max_attempts_per_task=2),
        publication=dict(mode='WAREHOUSE_REPORTS', predictor_root=str(tmp_path / 'publication-runtime'),
            predictor_code_commit='b' * 40, code_commit='c' * 40, actor='synthetic-selected',
            lake='synthetic-selected', warehouse_url='http://localhost:5057',
            token_env='SYNTHETIC_LAKE_TOKEN', experiment_set_key='synthetic-selected-comparison',
            resources=dict(train='selected/train.csv', validation='selected/validation.csv')))
    path = tmp_path / 'request.json'
    write(path, request)
    return path, request, receipt


def repin(path, request, receipt):
    bundle = Path(request['receipt_file']).parent
    for item in receipt['splits'].values():
        item['manifest_sha256'] = digest(bundle / item['manifest_file'])
    request['receipt_sha256'] = write(bundle / 'RECEIPT.json', receipt)
    write(path, request)


def test_balance_budget_request_changes_only_order_allocation(tmp_path):
    path, request, _ = fixture(tmp_path)
    request.update(schema='selected_rl_comparison_request.v2', order_budget=dict(
        rel_volume=.1, min_order_volume=1, max_order_volume=1e12, volume_step=1))
    write(path, request)
    body, _ = selected._build(path, tmp_path / 'out')
    config = body['common_config']
    assert config['execution_envelope']['sizing_mode'] == 'balance_budget'
    assert config['execution_envelope']['rel_volume'] == .1
    assert config['execution_envelope']['leverage'] == config['leverage'] == 1
    assert config['window_size'] == 13 and config['observation_stride_bars'] == 2
    assert config['total_timesteps'] == 100000
    assert config['execution_envelope']['atr_sl_mult'] == 2
    assert config['execution_envelope']['atr_tp_mult'] == 4


@pytest.mark.parametrize('leverage', [50, 100])
def test_explicit_leverage_binds_broker_and_envelope_together(tmp_path, leverage):
    path, request, _ = fixture(tmp_path)
    request.update(schema='selected_rl_comparison_request.v3', leverage=leverage,
        order_budget=dict(rel_volume=.1, min_order_volume=1,
                          max_order_volume=1e12, volume_step=1))
    write(path, request)
    body, _ = selected._build(path, tmp_path / 'out')
    config = body['common_config']
    assert config['leverage'] == leverage
    assert config['execution_envelope']['leverage'] == leverage
    assert config['execution_envelope']['leverage_cap'] == leverage
    assert config['execution_envelope']['rel_volume'] == .1
    assert config['window_size'] == 13 and config['observation_stride_bars'] == 2
    assert config['total_timesteps'] == 100000


@pytest.mark.parametrize('leverage', [0, -50, 101, True, float('inf'), '50'])
def test_invalid_explicit_leverage_refused_before_dispatch(tmp_path, leverage):
    path, request, _ = fixture(tmp_path)
    request.update(schema='selected_rl_comparison_request.v3', leverage=leverage,
        order_budget=dict(rel_volume=.1, min_order_volume=1,
                          max_order_volume=1e12, volume_step=1))
    write(path, request)
    with pytest.raises(ValueError, match='SELECTED_LEVERAGE'):
        selected._build(path, tmp_path / 'out')


@pytest.mark.parametrize('key,value', [('rel_volume', 0), ('rel_volume', 1.01),
    ('min_order_volume', -1), ('volume_step', 0), ('max_order_volume', .5),
    ('rel_volume', float('nan'))])
def test_bad_order_budget_refused_before_dispatch(tmp_path, key, value):
    path, request, _ = fixture(tmp_path)
    request.update(schema='selected_rl_comparison_request.v2', order_budget=dict(
        rel_volume=.1, min_order_volume=1, max_order_volume=1e12, volume_step=1))
    request['order_budget'][key] = value
    write(path, request)
    with pytest.raises(ValueError, match='SELECTED_ORDER_BUDGET'):
        selected._build(path, tmp_path / 'out')


@pytest.mark.parametrize('parquet', [False, True])
def test_selected_csv_and_parquet_use_existing_queue_and_same_information(tmp_path, parquet):
    path, request, receipt = fixture(tmp_path, parquet)
    output = tmp_path / 'comparison'
    contract = selected.materialize(path, output)
    body = queue.load_contract(contract)
    design = json.loads(Path(request['base_design_file']).read_text())
    assert body['arms'] == design['arms']
    assert body['arms'][:2] == ['SAC_FLAT', 'SAC_PER_FEATURE_CONV1D']
    assert body['seeds'] == [7]
    assert body['feature_selection_complete'] is True
    assert body['selection_complete'] == COMPLETE
    assert body['full_inventory_complete'] is False
    assert body['selected_input_binding']['selection_sha256'] == request['selection_sha256']
    assert body['selected_input_binding']['feature_columns'] == COLUMNS
    assert body['selected_input_binding']['selection_close_sha256'] == receipt['selection_close_sha256']
    assert body['selected_input_binding']['selection_close_file'] == 'CLOSE.json'
    reference = design['common_config']
    assert all(body['common_config'][key] == value for key, value in reference.items()
               if key != 'session_exposure_policy')
    assert {k: v for k, v in body['common_config']['session_exposure_policy'].items()
            if k != 'calendar_identity'} == reference['session_exposure_policy']
    configs = []
    for arm in ('SAC_FLAT', 'SAC_PER_FEATURE_CONV1D'):
        directory = tmp_path / arm
        directory.mkdir()
        cfg = queue._task_config(body, dict(arm=arm, split='train', seed=7), directory,
                                 directory / 'policy.zip')
        assert cfg['feature_columns'] == COLUMNS
        assert cfg['selection_sha256'] == request['selection_sha256']
        assert cfg['total_timesteps'] == cfg['training_transition_cap'] == 100000
        assert cfg['window_size'] == 13 and cfg['observation_stride_bars'] == 2
        assert cfg['feature_scaling'] == 'rolling_zscore'
        configs.append(cfg)
    assert [c['observation_encoder'] for c in configs] == ['flat', 'per_feature_conv1d']
    for key in ('feature_columns', 'input_data_file', 'curated_input_manifest_sha256',
                'learning_rate', 'buffer_size', 'commission', 'slippage_perc',
                'financing_annual_rate', 'feature_scaling_window'):
        assert configs[0][key] == configs[1][key]
    before = contract.read_bytes()
    assert selected.materialize(path, output).read_bytes() == before
    assert selected.validate(output) == body


@pytest.mark.parametrize('mutation', [
    'raw', 'incomplete', 'broad_completion', 'scope_missing', 'column_order', 'column_hash',
    'selection_hash', 'stale_subset', 'population', 'future', 'nonfinite', 'missing_column',
    'duplicate_header', 'support', 'outside_period', 'sealed_period', 'normalization',
    'validation_selection', 'selection_fit_end', 'path_escape', 'budget', 'override'])
def test_fail_closed_before_outputs_or_dispatch(tmp_path, mutation):
    path, request, receipt = fixture(tmp_path)
    bundle = Path(request['receipt_file']).parent
    item = receipt['splits']['validation']
    manifest_path = bundle / item['manifest_file']
    manifest = json.loads(manifest_path.read_text())
    selection_path = bundle / receipt['selection_file']
    selection = json.loads(selection_path.read_text())
    if mutation == 'raw': receipt['scope'] = 'RAW_BASELINE'
    if mutation == 'incomplete': receipt['feature_selection_complete'] = False
    if mutation == 'broad_completion': receipt['full_inventory_complete'] = True
    if mutation == 'scope_missing': receipt.pop('selection_complete')
    if mutation == 'column_order': manifest['feature_columns'] = list(reversed(COLUMNS))
    if mutation == 'column_hash': request['feature_columns_sha256'] = '0' * 64
    if mutation == 'selection_hash': manifest['selection_sha256'] = '0' * 64
    if mutation == 'stale_subset': request['selection_sha256'] = '0' * 64
    if mutation == 'population': manifest['inventory_sha256'] = '0' * 64
    if mutation == 'support': manifest['expected_timestamps_sha256'] = '0' * 64
    if mutation == 'sealed_period': manifest['period_end'] = '2026-01-01T00:00:00+00:00'
    if mutation == 'normalization': selection['normalization']['fit_scope'] = 'ALL_ROWS'
    if mutation == 'validation_selection': selection['validation_used_for_selection'] = True
    if mutation == 'selection_fit_end': selection['fit_period_end'] = '2024-02-01T00:00:00+00:00'
    if mutation in ('normalization', 'validation_selection', 'selection_fit_end'):
        request['selection_sha256'] = receipt['selection_sha256'] = write(selection_path, selection)
        close_path = bundle / receipt['selection_close_file']
        close = json.loads(close_path.read_text())
        close['selection_sha256'] = request['selection_sha256']
        receipt['selection_close_sha256'] = write(close_path, close)
        for split in receipt['splits'].values():
            p = bundle / split['manifest_file']
            value = json.loads(p.read_text())
            value['selection_sha256'] = request['selection_sha256']
            write(p, value)
        manifest['selection_sha256'] = request['selection_sha256']
    if mutation in ('future', 'nonfinite', 'missing_column', 'duplicate_header', 'outside_period'):
        data = bundle / item['input_file']
        with data.open(newline='') as stream:
            reader = csv.DictReader(stream)
            fields, rows = reader.fieldnames, list(reader)
        if mutation == 'future': rows[-1]['SELECTED_AVAILABLE_AT'] = '2024-12-01T00:00:00+00:00'
        if mutation == 'nonfinite': rows[-1][COLUMNS[0]] = 'nan'
        if mutation == 'missing_column':
            fields.remove(COLUMNS[0])
            for row in rows: row.pop(COLUMNS[0])
        if mutation == 'duplicate_header': fields.append(COLUMNS[0])
        if mutation == 'outside_period': rows[-1]['DATE_TIME'] = '2025-01-01T00:00:00+00:00'
        with data.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        manifest['input_sha256'] = digest(data)
    if mutation == 'path_escape': item['input_file'] = '../outside.csv'
    if mutation == 'budget':
        design = json.loads(Path(request['base_design_file']).read_text())
        design['common_config']['total_timesteps'] = 200000
        request['base_design_file'] = str(tmp_path / 'bad-design.json')
        request['base_design_sha256'] = write(Path(request['base_design_file']), design)
    if mutation == 'override': request['common_config'] = {'commission': 0}
    write(manifest_path, manifest)
    repin(path, request, receipt)
    output = tmp_path / 'comparison'
    with pytest.raises(ValueError):
        selected.materialize(path, output)
    assert not output.exists()


@pytest.mark.parametrize('member', ['receipt', 'selection', 'data', 'manifest', 'selection_close'])
def test_source_tamper_on_resume_refused_before_queue(tmp_path, monkeypatch, member):
    path, request, receipt = fixture(tmp_path)
    output = tmp_path / 'comparison'
    selected.materialize(path, output)
    bundle = Path(request['receipt_file']).parent
    target = {'receipt': Path(request['receipt_file']), 'selection': bundle / 'SELECTION.json',
              'data': bundle / 'train.csv', 'manifest': bundle / 'train.json',
              'selection_close': bundle / 'CLOSE.json'}[member]
    target.write_bytes(target.read_bytes() + b' ')
    monkeypatch.setattr(queue, 'run', lambda *a, **k: pytest.fail('dispatched'))
    with pytest.raises(ValueError):
        selected.run(output)


def test_finite_automation_delegates_without_a_new_learning_loop(tmp_path, monkeypatch):
    path, _, _ = fixture(tmp_path)
    root = tmp_path / 'comparison'
    contract = selected.materialize(path, root)
    seen = []
    monkeypatch.setattr(queue, 'run', lambda *a, **kw: seen.append((a, kw)) or 2)
    assert selected.run(root) == 2
    assert seen == [((contract, root), {'resume': False})]
    write(root / 'STATUS.json', dict(schema='curated_rl_queue_status.v1', state='FAILED_TIMEOUT',
                                    contract_sha256=digest(contract), tasks=[]))
    assert selected.run(root) == 2
    assert seen[-1][1] == {'resume': True}
    status = selected.read_status(root)
    assert status['selection_complete'] == COMPLETE
    assert status['full_inventory_complete'] is False
    assert status['feature_selection_complete'] is True


def test_publication_requires_readback_and_backups_are_pinned(tmp_path, monkeypatch):
    path, _, _ = fixture(tmp_path)
    root = tmp_path / 'comparison'
    contract = selected.materialize(path, root)
    body = json.loads(contract.read_text())
    tasks = [dict(arm=arm, seed=7, split=role, state='ENQUEUED', artifacts={})
             for arm in body['arms'] for role in ('train', 'validation')]
    write(root / 'STATUS.json', dict(schema='curated_rl_queue_status.v1', state='COMPLETED_OUTBOX_PENDING',
                                    contract_sha256=digest(contract), tasks=tasks))
    db = sqlite3.connect(root / 'warehouse-outbox.sqlite')
    db.execute('CREATE TABLE synthetic (value TEXT)')
    db.execute("INSERT INTO synthetic VALUES ('transport only')")
    db.commit()
    db.close()
    warehouse = dict(expected=12, delivered=12, pending=0, mismatch=0)
    def pump(*args):
        write(root / 'WAREHOUSE_STATUS.json', dict(warehouse=warehouse))
        return 0
    monkeypatch.setattr(queue, 'pump_warehouse', pump)
    warehouse['delivered'] = 11
    assert selected.publish_and_close(root) == 2
    assert not (root / 'CLOSE.json').exists()
    warehouse['delivered'] = 12
    assert selected.publish_and_close(root) == 0
    close = json.loads((root / 'CLOSE.json').read_text())
    assert close['selection_complete'] == COMPLETE and close['full_inventory_complete'] is False
    assert close['physical_sha256'] == digest(root / close['physical_copy'])
    assert close['selected_input_binding']['selection_sha256'] == body['selection_sha256']
    before = (root / 'CLOSE.json').read_bytes()
    assert selected.publish_and_close(root) == 0
    assert (root / 'CLOSE.json').read_bytes() == before
    (root / close['physical_copy']).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='COPY_CHANGED'):
        selected.publish_and_close(root)


def test_selection_contract_drift_is_not_silently_replaced(tmp_path):
    path, request, receipt = fixture(tmp_path)
    root = tmp_path / 'comparison'
    selected.materialize(path, root)
    original = (root / 'CONTRACT.json').read_bytes()
    request['publication']['experiment_set_key'] = 'different'
    write(path, request)
    with pytest.raises(ValueError):
        selected.materialize(path, root)
    assert (root / 'CONTRACT.json').read_bytes() == original


@pytest.mark.parametrize('fault', [
    'missing_link', 'missing_pin', 'missing_file', 'tamper', 'selection_mismatch',
    'nonclosed', 'unmirrored', 'pending', 'mismatch', 'count', 'count_bool',
    'verified_files_bool', 'verified_files_negative', 'broad_completion',
    'pending_bool', 'mismatch_bool', 'missing_field', 'absolute_path',
    'outside_root', 'symlink_escape', 'nested_close'])
def test_selection_close_gate_refuses_before_source_reads_or_outputs(tmp_path, monkeypatch, fault):
    path, request, receipt = fixture(tmp_path)
    bundle = Path(request['receipt_file']).parent
    close_path = bundle / receipt['selection_close_file']
    close = json.loads(close_path.read_text())
    updates = {
        'selection_mismatch': {'selection_sha256': '0' * 64},
        'nonclosed': {'state': 'FILTER_SELECTION_SEALED_REMOTE_PUBLICATION_PENDING'},
        'unmirrored': {'verified_files': 0}, 'pending': {'pending': 1},
        'mismatch': {'mismatch': 1}, 'count': {'selected_count': len(COLUMNS) + 1},
        'count_bool': {'selected_count': True}, 'verified_files_bool': {'verified_files': True},
        'verified_files_negative': {'verified_files': -1},
        'broad_completion': {'full_inventory_complete': True},
        'pending_bool': {'pending': False}, 'mismatch_bool': {'mismatch': False}}
    if fault in updates:
        close.update(updates[fault])
        receipt['selection_close_sha256'] = write(close_path, close)
    if fault == 'missing_field':
        close.pop('verified_files')
        receipt['selection_close_sha256'] = write(close_path, close)
    if fault == 'missing_link': receipt.pop('selection_close_file')
    if fault == 'missing_pin': receipt.pop('selection_close_sha256')
    if fault == 'missing_file': close_path.unlink()
    if fault == 'tamper': close_path.write_bytes(close_path.read_bytes() + b' ')
    if fault == 'absolute_path': receipt['selection_close_file'] = str(close_path)
    if fault in ('outside_root', 'symlink_escape'):
        external = tmp_path / 'external-close.json'
        external.write_bytes(close_path.read_bytes())
        if fault == 'outside_root':
            receipt['selection_close_file'] = '../external-close.json'
        else:
            close_path.unlink()
            close_path.symlink_to(external)
    if fault == 'nested_close':
        nested = bundle / 'nested'
        nested.mkdir()
        (nested / 'CLOSE.json').write_bytes(close_path.read_bytes())
        receipt['selection_close_file'] = 'nested/CLOSE.json'
    repin(path, request, receipt)
    monkeypatch.setattr(selected, '_source', lambda *a: pytest.fail('read financial rows before close gate'))
    output = tmp_path / 'comparison'
    with pytest.raises(ValueError):
        selected.materialize(path, output)
    assert not output.exists()


@pytest.mark.parametrize('resume', [False, True])
def test_selection_close_rechecked_immediately_before_dispatch(tmp_path, monkeypatch, resume):
    path, request, _ = fixture(tmp_path)
    root = tmp_path / 'comparison'
    contract = selected.materialize(path, root)
    if resume:
        write(root / 'STATUS.json', dict(schema='curated_rl_queue_status.v1',
            state='FAILED_TIMEOUT', contract_sha256=digest(contract), tasks=[]))
    close_path = Path(request['receipt_file']).parent / 'CLOSE.json'
    close = json.loads(close_path.read_text())
    close['pending'] = 1
    write(close_path, close)
    monkeypatch.setattr(queue, 'run', lambda *a, **kw: pytest.fail('dispatched without closed selection'))
    with pytest.raises(ValueError):
        selected.run(root)


def test_deployment_receipt_filename_is_supported(tmp_path):
    path, request, receipt = fixture(tmp_path)
    deployment = Path(request['receipt_file']).with_name('DEPLOYMENT_RECEIPT.json')
    request['receipt_file'] = str(deployment)
    request['receipt_sha256'] = write(deployment, receipt)
    write(path, request)
    root = tmp_path / 'comparison'
    contract = selected.materialize(path, root)
    binding = json.loads(contract.read_text())['selected_input_binding']
    assert binding['selection_close_sha256'] == receipt['selection_close_sha256']
