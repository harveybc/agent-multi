"""Bind owner-selected inputs to the existing finite static RL queue.

No selection, feature generation, admission issuance or alternate learning loop.
The operator supplies pinned local objects; nothing is discovered from campaigns.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile

from tools import curated_rl_campaign as queue
from tools import run_raw_rl_campaign as shared_status
from tools.collect_closed_raw_rl import live_monitor_state
from tools.verify_rl_static_engineering import save_status


COMPLETE = 'WITHIN_DECLARED_ADMITTED_POPULATION'
SCOPE = 'SELECTED_ADMITTED_POPULATION'
PERIODS = {'train': ('2020-01-01T00:00:00+00:00', '2024-01-01T00:00:00+00:00'),
           'validation': ('2024-01-01T00:00:00+00:00', '2025-01-01T00:00:00+00:00')}
FLAGS = dict(feature_selection_complete=True, selection_complete=COMPLETE,
             full_inventory_complete=False)
SELECTION_CLOSED = 'FILTER_SELECTION_OLAP_AND_TWO_PHYSICAL_COPIES_VERIFIED'
REQUEST_KEYS = {'schema', 'base_design_file', 'base_design_sha256', 'receipt_file',
                'receipt_sha256', 'selection_sha256', 'population_sha256',
                'feature_columns_sha256', 'execution', 'publication'}


def _fail(reason):
    raise ValueError('SELECTED_' + reason)


def _json(path):
    try:
        body = json.loads(Path(path).read_text())
        if not isinstance(body, dict):
            _fail('OBJECT_REQUIRED')
        return body
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError('SELECTED_OBJECT_UNAVAILABLE') from exc


def _pinned(path, pin):
    if not queue._pin(pin) or not Path(path).is_file() or queue.digest(path) != pin:
        _fail('CONTENT_PIN_MISMATCH')
    return Path(path)


def _member(root, name):
    if not isinstance(name, str) or Path(name).is_absolute():
        _fail('RELATIVE_BUNDLE_MEMBER_REQUIRED')
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        _fail('BUNDLE_MEMBER_OUTSIDE_ROOT')
    return path


def _time(value):
    stamp = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if stamp.tzinfo is None:
        _fail('TIMEZONE_REQUIRED')
    return stamp.astimezone(timezone.utc)


def _scoped(body):
    if (body.get('scope') != SCOPE
            or body.get('feature_selection_complete') is not True
            or body.get('selection_complete') != COMPLETE
            or body.get('full_inventory_complete') is not False):
        _fail('SCOPED_COMPLETED_SELECTION_REQUIRED')


def _period(body, role, start_key='period_start', end_key='period_end'):
    if tuple(_time(body[k]) for k in (start_key, end_key)) != tuple(map(_time, PERIODS[role])):
        _fail('TRAIN2020_23_VALID2024_REQUIRED')


def _selection_close(root, receipt, selection_pin, selected_count):
    path = _member(root, receipt['selection_close_file'])
    if path.parent != root.resolve():
        _fail('SELECTION_CLOSE_SAME_ROOT_REQUIRED')
    pin = receipt['selection_close_sha256']
    close = _json(_pinned(path, pin))
    if (close.get('state') != SELECTION_CLOSED
            or close.get('selection_sha256') != selection_pin
            or type(close.get('selected_count')) is not int
            or close['selected_count'] != selected_count
            or type(close.get('pending')) is not int or close['pending'] != 0
            or type(close.get('mismatch')) is not int or close['mismatch'] != 0
            or type(close.get('verified_files')) is not int or close['verified_files'] <= 0
            or close.get('full_inventory_complete') is not False):
        _fail('SELECTION_OLAP_AND_TWO_COPIES_CLOSE_REQUIRED')
    return dict(selection_close_file=path.name, selection_close_sha256=pin)


def _source(path):
    """Return exact CSV text, or an unscaled lossless parquet transport export."""
    if path.suffix.lower() == '.csv':
        return path.read_text()
    if path.suffix.lower() not in ('.parquet', '.pq'):
        _fail('CSV_OR_PARQUET_REQUIRED')
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    if len(set(table.column_names)) != len(table.column_names):
        _fail('DUPLICATE_COLUMNS')
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=table.column_names)
    writer.writeheader()
    for row in table.to_pylist():
        writer.writerow({k: v.isoformat() if isinstance(v, datetime) else v
                         for k, v in row.items()})
    return stream.getvalue()


def _check_rows(text, manifest, columns, availability_column, common, binary):
    reader = csv.DictReader(io.StringIO(text, newline=''))
    fields = reader.fieldnames or []
    required = set(columns) | {common['date_column'], 'OPEN', 'HIGH', 'LOW', 'CLOSE',
                               common['session_spread_column'], availability_column}
    if len(set(fields)) != len(fields) or not required.issubset(fields):
        _fail('COLUMNS_MISSING_OR_DUPLICATED')
    numbers = required - {common['date_column'], availability_column}
    if 'VOLUME' in fields:
        numbers.add('VOLUME')
    start, end = _time(manifest['period_start']), _time(manifest['period_end'])
    stamp_hash, previous, count = hashlib.sha256(), None, 0
    for row in reader:
        if None in row or any(v is None for v in row.values()):
            _fail('MALFORMED_ROW')
        stamp, available = _time(row[common['date_column']]), _time(row[availability_column])
        if not start <= stamp < end or (previous is not None and stamp <= previous):
            _fail('TIMESTAMP_SUPPORT_MISMATCH')
        if available > stamp:
            _fail('FUTURE_SELECTED_AVAILABILITY')
        values = {k: float(row[k]) for k in numbers}
        if not all(math.isfinite(v) for v in values.values()):
            _fail('NONFINITE_INPUT')
        op, hi, lo, cl = (values[k] for k in ('OPEN', 'HIGH', 'LOW', 'CLOSE'))
        if not 0 < lo <= min(op, cl) <= max(op, cl) <= hi:
            _fail('INVALID_OHLC')
        if values[common['session_spread_column']] < 0:
            _fail('NEGATIVE_SPREAD')
        if any(values[k] not in (-1, 0, 1) for k in binary):
            _fail('INVALID_BINARY_SEMANTICS')
        stamp_hash.update((stamp.isoformat() + '\n').encode())
        previous, count = stamp, count + 1
    if (type(manifest['row_count']) is not int or count != manifest['row_count']
            or count < 26 or stamp_hash.hexdigest() != manifest['expected_timestamps_sha256']):
        _fail('TIMESTAMP_SUPPORT_MISMATCH')


def _build(request_path, output_root):
    request = _json(request_path)
    es_mode = request.get('schema') == 'selected_rl_comparison_request.v4'
    leverage_mode = request.get('schema') in ('selected_rl_comparison_request.v3',
                                            'selected_rl_comparison_request.v4')
    budget_mode = request.get('schema') in ('selected_rl_comparison_request.v2',
                                          'selected_rl_comparison_request.v3',
                                          'selected_rl_comparison_request.v4')
    keys = REQUEST_KEYS | {'order_budget'} if budget_mode else REQUEST_KEYS
    if leverage_mode:
        keys = keys | {'leverage'}
    if es_mode:
        keys = keys | {'early_stopping'}
    if (request.get('schema') not in ('selected_rl_comparison_request.v1',
                                     'selected_rl_comparison_request.v2',
                                     'selected_rl_comparison_request.v3',
                                     'selected_rl_comparison_request.v4')
            or set(request) != keys):
        _fail('REQUEST_SCHEMA_OR_OVERRIDES_REFUSED')
    if es_mode:
        from agent_plugins.paired_early_stopping import validate_settings
        validate_settings(request['early_stopping'])
    if leverage_mode and (type(request['leverage']) not in (int, float)
                          or not math.isfinite(request['leverage'])
                          or not 1 <= request['leverage'] <= 100):
        _fail('LEVERAGE_INVALID')
    if budget_mode:
        budget = request['order_budget']
        if (not isinstance(budget, dict) or set(budget) != {
                'rel_volume', 'min_order_volume', 'max_order_volume', 'volume_step'}
                or any(type(v) not in (int, float) or not math.isfinite(v)
                       for v in budget.values())
                or not 0 < budget['rel_volume'] <= 1
                or not 0 <= budget['min_order_volume'] <= budget['max_order_volume']
                or budget['max_order_volume'] <= 0 or budget['volume_step'] <= 0):
            _fail('ORDER_BUDGET_INVALID')
    base_path = _pinned(request['base_design_file'], request['base_design_sha256'])
    design = _json(base_path)
    reference = _json(queue.ROOT / 'examples/config/rl_raw_eurusd_static.v1.json')
    # Only the input binding changes. No new economic/optimizer/compute policy.
    if (design != reference or design.get('refit_cadence') != 'NONE'
            or design.get('seeds') != [7]
            or design['common_config'].get('total_timesteps') != 100000):
        _fail('UNCHANGED_STATIC_DESIGN_REQUIRED')
    receipt_path = _pinned(request['receipt_file'], request['receipt_sha256']).resolve()
    root, receipt = receipt_path.parent, _json(receipt_path)
    _scoped(receipt)
    if receipt.get('schema') != 'selected_rl_bundle.v1' or receipt.get('status') != 'ADMITTED':
        _fail('OWNER_ADMITTED_BUNDLE_REQUIRED')
    selection_path = _member(root, receipt['selection_file'])
    _pinned(selection_path, request['selection_sha256'])
    if receipt['selection_sha256'] != request['selection_sha256']:
        _fail('STALE_SELECTED_SUBSET')
    selection = _json(selection_path)
    _scoped(selection)
    if (selection.get('schema') != 'selected_rl_selection.v1'
            or selection.get('causal_transforms_verified') is not True
            or selection.get('validation_used_for_selection') is not False):
        _fail('TRAIN_ONLY_CAUSAL_SELECTION_REQUIRED')
    _period(selection, 'train', 'fit_period_start', 'fit_period_end')
    columns = selection['feature_columns']
    if (not isinstance(columns, list) or not columns
            or any(not isinstance(c, str) or not c.strip() for c in columns)
            or len(set(columns)) != len(columns)):
        _fail('ORDERED_UNIQUE_SELECTED_NAMES_REQUIRED')
    col_pin = hashlib.sha256(json.dumps(columns, separators=(',', ':')).encode()).hexdigest()
    if (receipt['feature_columns'] != columns or receipt['feature_columns_sha256'] != col_pin
            or request['feature_columns_sha256'] != col_pin):
        _fail('SELECTED_COLUMN_HASH_OR_ORDER_MISMATCH')
    selection_close = _selection_close(root, receipt, request['selection_sha256'], len(columns))
    population = request['population_sha256']
    if (not queue._pin(population) or selection['population_sha256'] != population
            or receipt['population_sha256'] != population):
        _fail('DECLARED_POPULATION_MISMATCH')
    availability = selection['availability_column']
    if (not isinstance(availability, str) or not availability
            or availability in set(columns) | {'DATE_TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE'}
            or selection.get('availability_semantics') != 'row_max_selected_available_at'):
        _fail('SELECTED_AVAILABILITY_EVIDENCE_REQUIRED')
    norm = selection['normalization']
    binary = norm.get('binary_columns')
    if (set(norm) != {'mode', 'window_bars', 'fit_scope', 'binary_columns'}
            or norm['mode'] != 'rolling_zscore' or norm['window_bars'] != 256
            or norm['fit_scope'] != 'PAST_ONLY' or not isinstance(binary, list)
            or any(c not in columns for c in binary) or len(set(binary)) != len(binary)):
        _fail('CAUSAL_ROLLING_NORMALIZATION_REQUIRED')
    common = copy.deepcopy(design['common_config'])
    if es_mode:
        common['paired_early_stopping'] = request['early_stopping']
    if leverage_mode:
        common['leverage'] = request['leverage']
        common['execution_envelope']['leverage_cap'] = request['leverage']
    if budget_mode:
        common['execution_envelope'].update(request['order_budget'],
            sizing_mode='balance_budget', leverage=common['leverage'])
    common.update(FLAGS, selection_sha256=request['selection_sha256'],
                  selected_population_sha256=population, feature_binary_columns=binary)
    closures = _pinned(_member(root, receipt['closure_intervals_file']),
                       receipt['closure_intervals_sha256'])
    intervals = json.loads(closures.read_text())
    if not isinstance(intervals, list) or not queue._pin(receipt['calendar_sha256']):
        _fail('ADMITTED_CALENDAR_REQUIRED')
    common['session_calendar_intervals'] = intervals
    common['session_exposure_policy']['calendar_identity'] = receipt['calendar_sha256']
    common['session_account_fingerprint'] = hashlib.sha256(
        b'selected-eurusd-analytical-simulator-v1').hexdigest()
    splits, exports, source_pins = {}, {}, {}
    if set(receipt['splits']) != set(PERIODS):
        _fail('TRAIN_VALIDATION_ONLY')
    for role, item in receipt['splits'].items():
        manifest_path = _pinned(_member(root, item['manifest_file']), item['manifest_sha256'])
        manifest = _json(manifest_path)
        _scoped(manifest)
        _period(manifest, role)
        if (manifest.get('schema') != 'curated_rl_input.v1'
                or manifest.get('status') != 'ADMITTED' or manifest.get('split') != role
                or manifest.get('availability_verified') is not True
                or manifest.get('open_session_coverage_verified') is not True
                or manifest.get('timestamp_semantics') != 'completed_bar_available_at'
                or manifest.get('feature_columns') != columns
                or manifest.get('selection_sha256') != request['selection_sha256']
                or manifest.get('inventory_sha256') != population
                or manifest.get('calendar_sha256') != receipt['calendar_sha256']
                or manifest.get('availability_column') != availability):
            _fail('SPLIT_SELECTED_IDENTITY_MISMATCH')
        data = _pinned(_member(root, item['input_file']), manifest['input_sha256'])
        text = _source(data)
        _check_rows(text, manifest, columns, availability, common, binary)
        source_pins[role] = dict(input_sha256=manifest['input_sha256'],
                                 manifest_sha256=item['manifest_sha256'])
        pin = item['manifest_sha256']
        if data.suffix.lower() != '.csv':
            csv_bytes = text.encode()
            manifest = {**manifest, 'input_sha256': hashlib.sha256(csv_bytes).hexdigest(),
                        'source_parquet_sha256': manifest['input_sha256'],
                        'source_manifest_sha256': pin}
            raw_manifest = (json.dumps(manifest, sort_keys=True, indent=2) + '\n').encode()
            data, manifest_path = output_root / (role + '.csv'), output_root / (role + '.json')
            exports[data.name], exports[manifest_path.name] = csv_bytes, raw_manifest
            pin = hashlib.sha256(raw_manifest).hexdigest()
        splits[role] = dict(input_data_file=str(data), curated_input_manifest=str(manifest_path),
            curated_input_manifest_sha256=pin, curated_input_split=role, feature_columns=columns)
    binding = dict(selection_sha256=request['selection_sha256'], feature_columns=columns,
        feature_columns_sha256=col_pin, population_sha256=population, **FLAGS,
        source_receipt_sha256=request['receipt_sha256'], source_splits=source_pins,
        **selection_close)
    body = dict(schema='curated_rl_campaign.v1', classification='NON_GOVERNING',
        scientific_scope='REAL_DATA_DEVELOPMENT', execution_backend='local_subprocess',
        seeds=[7], arms=['SAC_FLAT'] if es_mode else list(design['arms']), refit_cadence='NONE',
        common_config=common, splits=splits, **FLAGS,
        selection_sha256=request['selection_sha256'], selected_input_binding=binding,
        input_policy=dict(kind=SCOPE, **binding), cost_scope=design['cost_scope'],
        design_sha256=request['base_design_sha256'], input_receipt_sha256=request['receipt_sha256'],
        selected_adapter=dict(request_file=str(request_path), request_sha256=queue.digest(request_path)),
        execution=request['execution'], publication=request['publication'])
    return body, exports


def _prepare(request_path, output_root):
    try:
        return _build(request_path, output_root)
    except (KeyError, TypeError, OSError, OverflowError) as exc:
        raise ValueError('SELECTED_INVALID_OR_UNAVAILABLE_INPUT') from exc


def _output_root(root):
    root = Path(root).resolve()
    live = (Path.home() / '.local/share/agent-multi', Path.home() / '.local/state/agent-multi')
    if root.is_relative_to(queue.ROOT) or any(root.is_relative_to(p) for p in live):
        _fail('PRIVATE_NONCAMPAIGN_OUTPUT_REQUIRED')
    return root


def _save_export(path, data):
    # Only new adapter-owned files; never overwrite a changed transport artifact.
    if path.exists():
        if path.read_bytes() != data:
            _fail('EXISTING_TRANSPORT_DIFFERS')
        return
    with path.open('xb') as stream:
        stream.write(data)


def materialize(request_path, output_root):
    request_path, output_root = Path(request_path).resolve(), _output_root(output_root)
    body, exports = _prepare(request_path, output_root)
    receipt_root = Path(_json(request_path)['receipt_file']).resolve().parent
    if (output_root.is_relative_to(receipt_root) or receipt_root.is_relative_to(output_root)
            or request_path.is_relative_to(output_root)):
        _fail('OUTPUT_MUST_NOT_OVERLAP_INPUTS')
    contract = output_root / 'CONTRACT.json'
    if contract.exists() and _json(contract) != body:
        _fail('EXISTING_CONTRACT_DIFFERS')
    # Exercise the real queue admission path before creating campaign outputs.
    with tempfile.TemporaryDirectory(prefix='selected-rl-check-') as temp:
        scratch = Path(temp)
        check_body, check_exports = _prepare(request_path, scratch)
        for name, data in check_exports.items():
            _save_export(scratch / name, data)
        save_status(scratch / 'CONTRACT.json', check_body)
        queue.load_contract(scratch / 'CONTRACT.json')
    output_root.mkdir(parents=True, exist_ok=True)
    for name, data in exports.items():
        _save_export(output_root / name, data)
    if not contract.exists():
        save_status(contract, body)
    validate(output_root)
    return contract


def validate(root):
    root = _output_root(root)
    body = _json(root / 'CONTRACT.json')
    adapter = body.get('selected_adapter', {})
    request = _pinned(adapter.get('request_file', ''), adapter.get('request_sha256'))
    expected, exports = _prepare(request.resolve(), root)
    if expected != body:
        _fail('CONTRACT_OR_CURRENT_SELECTION_DRIFT')
    for name, data in exports.items():
        if not (root / name).is_file() or (root / name).read_bytes() != data:
            _fail('TRANSPORT_DRIFT')
    return queue.load_contract(root / 'CONTRACT.json')


def run(root):
    root = _output_root(root)
    validate(root)
    return queue.run(root / 'CONTRACT.json', root, resume=(root / 'STATUS.json').exists())


def read_status(root):
    root = Path(root)
    body = _json(root / 'CONTRACT.json')
    binding = body['selected_input_binding']
    _scoped(dict(scope=SCOPE, **binding))
    result = shared_status.read_status(root)
    state = queue.status(root)
    if state.get('contract_sha256', queue.digest(root / 'CONTRACT.json')) != queue.digest(root / 'CONTRACT.json'):
        _fail('STATUS_CONTRACT_DRIFT')
    if body['common_config'].get('paired_early_stopping'):
        if state.get('state') == 'UNAVAILABLE':
            result['declared_tasks'] = len(body['seeds']) * len(body['arms']) * 2
            dispatch = root / 'DISPATCH_STATUS.json'
            if dispatch.is_file():
                result['dispatch'] = _json(dispatch)
                result['state'] = result['dispatch']['state']
        current = next((t for t in state.get('tasks', []) if t['state'] == 'RUNNING'), None)
        if current and current.get('config'):
            monitor = Path(current['config']).parent / 'paired-monitor' / 'STATUS.json'
            if monitor.exists():
                detail = _json(monitor)
                result['paired_monitor'] = detail
                if detail.get('state') == 'EVALUATING':
                    result['current_training_eta_seconds'] = None
                    progress = Path(detail['progress_file'])
                    if progress.is_file():
                        result['paired_monitor_progress'] = _json(progress)
    return {**result, **FLAGS, 'selected_input_binding': binding,
            'input_integrity': 'NOT_RECHECKED_READ_ONLY_STATUS'}


def publish_monitors(body, root, state):
    if not body['common_config'].get('paired_early_stopping'):
        return 0
    count = 0
    for pair in sorted(root.glob('*/*/paired-monitor/step-*/PAIR.json')):
        record = _json(pair)
        if (record.get('schema') != 'paired_sac_checkpoint.v1'
                or [t['split'] for t in record['reports']] != ['train', 'validation']):
            _fail('MONITOR_PAIR_INVALID')
        for original in record['reports']:
            task = copy.deepcopy(original)
            if task['arm'] != 'SAC_FLAT' or task['seed'] not in body['seeds']:
                _fail('MONITOR_TASK_INVALID')
            for key, pin in task['artifacts'].items():
                path = Path(task[key]).resolve()
                if not path.is_relative_to(root.resolve()) or queue.digest(path) != pin:
                    _fail('MONITOR_ARTIFACT_CHANGED')
            queue._enqueue(body, state, task, None, warehouse_root=root)
            count += 1
    return count


def publish_and_close(root):
    """Reuse queue publication/readback; keep closure truthfully selection-scoped.

    The raw publisher's fixed scope makes its close routine inapplicable. This
    small closure adapter uses the same SQLite backup protocol, not a new queue.
    """
    root = _output_root(root)
    body = validate(root)
    with (root / '.publish.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 2
        close_path = root / 'CLOSE.json'
        pin = queue.digest(root / 'CONTRACT.json')
        if close_path.exists():
            close = _json(close_path)
            if (close.get('contract_sha256') != pin
                    or close.get('selected_input_binding') != body['selected_input_binding']):
                _fail('CLOSE_IDENTITY_DRIFT')
            if queue.digest(_member(root, close['physical_copy'])) != close['physical_sha256']:
                _fail('CLOSED_PHYSICAL_COPY_CHANGED')
            for name, value in close.get('monitor_files', {}).items():
                path = _member(root, name)
                if body['common_config'].get('paired_early_stopping') and live_monitor_state(name):
                    continue
                if queue.digest(path) != value:
                    _fail('CLOSED_MONITOR_CHANGED')
            return 0
        state = queue.status(root)
        monitor_reports = publish_monitors(body, root, state)
        if queue.pump_warehouse(body, root) != 0:
            return 2
        state = queue.status(root)
        tasks = state.get('tasks', [])
        expected = [(arm, seed, role) for seed in body['seeds'] for arm in body['arms']
                    for role in PERIODS]
        warehouse = _json(root / 'WAREHOUSE_STATUS.json')['warehouse']
        if (state.get('contract_sha256') != pin
                or [(t['arm'], t['seed'], t['split']) for t in tasks] != expected
                or any(t['state'] != 'ENQUEUED' for t in tasks)
                or warehouse['expected'] != len(expected) + monitor_reports
                or warehouse['delivered'] != len(expected) + monitor_reports
                or warehouse['pending'] or warehouse['mismatch']):
            return 2
        queue._verify_artifacts(state)
        partial, target = root / 'metrics-backup.partial', root / 'metrics-backup.sqlite'
        source = sqlite3.connect(f'file:{root / "warehouse-outbox.sqlite"}?mode=ro', uri=True)
        destination = sqlite3.connect(partial)
        try:
            source.backup(destination)
        finally:
            destination.close()
            source.close()
        with partial.open('rb') as stream:
            os.fsync(stream.fileno())
        os.replace(partial, target)
        monitor_files = {}
        if body['common_config'].get('paired_early_stopping'):
            if monitor_reports < 2:
                _fail('MONITOR_EVIDENCE_REQUIRED')
            for directory in root.glob('*/*/paired-monitor'):
                for file in directory.rglob('*'):
                    if file.is_file():
                        if file.is_symlink() or not file.resolve().is_relative_to(root.resolve()):
                            _fail('MONITOR_PATH_ESCAPE')
                        if live_monitor_state(str(file.relative_to(root))):
                            continue
                        monitor_files[str(file.relative_to(root))] = queue.digest(file)
        save_status(close_path, dict(schema='selected_rl_close.v1',
            state='COMPLETED_AND_WAREHOUSE_READBACK_VERIFIED', contract_sha256=pin,
            completed_tasks=len(tasks), physical_copy=target.name, physical_sha256=queue.digest(target),
            warehouse=warehouse, scientific_scope=body['scientific_scope'], **FLAGS,
            selected_input_binding=body['selected_input_binding'],
            **(dict(monitor_reports=monitor_reports, monitor_files=monitor_files)
               if monitor_files else {})))
        return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('materialize', 'check', 'run', 'status', 'results', 'publish'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--request', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.action == 'materialize':
            if args.request is None:
                parser.error('--request is required for materialize')
            contract = materialize(args.request, args.root)
            print(json.dumps(dict(contract=str(contract), sha256=queue.digest(contract))))
            return 0
        if args.action == 'run':
            code = run(args.root)
            if code == 0:
                return publish_and_close(args.root)
            return 78 if queue.status(args.root)['state'] == 'ATTEMPTS_EXHAUSTED' else code
        if args.action == 'publish':
            return publish_and_close(args.root)
        if args.action == 'check':
            validate(args.root)
            payload = dict(state='INPUTS_VERIFIED_NO_DISPATCH', **FLAGS)
        elif args.action == 'status':
            payload = read_status(args.root)
        else:
            payload = shared_status.results(args.root)
        print(json.dumps(payload, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError):
        # No paths, selected names, credentials or private values in refusal output.
        print(json.dumps(dict(state='SELECTED_CONTRACT_OR_INTEGRITY_REFUSAL')))
        return 78


if __name__ == '__main__':
    raise SystemExit(main())
