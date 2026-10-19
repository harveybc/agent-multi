"""Bounded CPU acceptance checks, with durable status; never a financial launcher."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

GYM_TESTS = [
    'test_financing_debit', 'test_next_open_transition', 'test_next_open_repeated_accounting',
    'test_submitted_order_cancellation', 'test_completed_bar_clock',
    'test_submitted_oco_bracket_regression',
    'test_daily_entry_quota',
    'test_strided_observation', 'test_feature_window_preprocessor',
    'test_protected_order_execution', 'test_shared_execution_envelope',
    'test_trading_metrics', 'test_session_exposure_env',
    'test_trade_reconciliation_stream', 'test_solvency_modes',
    'test_env_foundations', 'test_execution_cost_context',
]
AGENT_TESTS = [
    'unit/test_rl_selection_metric_integrity', 'unit/test_rl_pipeline_with_validation_splits',
    'unit/test_lexicographic_selection', 'unit/test_curated_rl_input',
    'unit/test_temporal_sac_features', 'integration/test_curated_temporal_gym_bridge',
]


def save_status(path, state):
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(state, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(gym_root, agent_root, output_root, timeout):
    gym_root, agent_root, output_root = map(Path.resolve, (gym_root, agent_root, output_root))
    if timeout <= 0 or any(output_root.is_relative_to(p) for p in (gym_root, agent_root)):
        raise ValueError('positive timeout and output outside both repositories required')
    # A distinct run directory prevents accidental overwrites or concurrent writers.
    output_root.mkdir(parents=True, exist_ok=False)
    state = {'schema': 'rl_static_engineering_checks.v1', 'state': 'RUNNING',
             'financial_fits': 0, 'scientific_admission': False, 'stages': []}
    status = output_root / 'STATUS.json'
    save_status(status, state)
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1',
           'OPENBLAS_NUM_THREADS': '1', 'GYM_FX_CHECKOUT': str(gym_root)}
    for name, root, tests in [('gym', gym_root, GYM_TESTS), ('agent', agent_root, AGENT_TESTS)]:
        stage = {'name': name, 'state': 'RUNNING', 'started_at_unix': time.time()}
        state['stages'].append(stage)
        save_status(status, state)
        report = output_root / f'{name}.xml'
        command = [sys.executable, '-m', 'pytest',
                   *[f'tests/{t}.py' for t in tests], '-q', '--tb=short', f'--junitxml={report}']
        try:
            with (output_root / f'{name}.log').open('w') as log:
                result = subprocess.run(command, cwd=root, env=env, stdout=log,
                                        stderr=subprocess.STDOUT, timeout=timeout, check=False)
            stage['returncode'] = result.returncode
            suites = ET.parse(report).getroot().iter('testsuite')
            totals = dict.fromkeys(('tests', 'failures', 'errors', 'skipped'), 0)
            for suite in suites:
                for key in totals:
                    totals[key] += int(suite.get(key, '0'))
            stage['counts'] = totals
            stage['state'] = ('PASSED' if result.returncode == 0 and totals['tests'] > 0
                              and totals['failures'] == totals['errors'] == 0 else 'FAILED')
        except subprocess.TimeoutExpired:
            stage['state'] = 'FAILED_TIMEOUT'
        except (OSError, ET.ParseError, ValueError) as exc:
            stage.update(state='FAILED', error_type=type(exc).__name__)
        stage['elapsed_seconds'] = time.time() - stage['started_at_unix']
        if stage['state'] != 'PASSED':
            state['state'] = stage['state']
            save_status(status, state)
            return 2
        save_status(status, state)
    state['state'] = 'ENGINEERING_CHECKS_PASSED'
    save_status(status, state)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gym-root', type=Path)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--timeout-seconds', type=int, default=120)
    parser.add_argument('--status', action='store_true')
    args = parser.parse_args()
    if args.status:
        state = json.loads((args.output_root / 'STATUS.json').read_text())
        print(json.dumps(state, indent=2))
        return 0 if state['state'] == 'ENGINEERING_CHECKS_PASSED' else 2
    if args.gym_root is None:
        parser.error('--gym-root is required when running checks')
    return run(args.gym_root, Path(__file__).resolve().parents[1],
               args.output_root, args.timeout_seconds)


if __name__ == '__main__':
    raise SystemExit(main())
