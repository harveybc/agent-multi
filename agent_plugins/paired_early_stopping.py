"""Paired full-support monitoring around one uninterrupted SB3 SAC learn call."""
from __future__ import annotations

import copy
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from zoneinfo import ZoneInfo

PROTOCOL = 'paired_full_daily_log_growth.v1'
SETTINGS = dict(protocol=PROTOCOL, eval_interval=5000, patience=10, min_delta=1e-8)


def validate_settings(value):
    from agent_plugins.fixed_budget_monitor import SETTINGS as FIXED_SETTINGS
    expected=FIXED_SETTINGS if isinstance(value,dict) and value.get('protocol')==FIXED_SETTINGS['protocol'] else SETTINGS
    if value != expected or any(type(value[k]) is not type(v) for k, v in expected.items()):
        raise ValueError('PAIRED_ES_APPROVED_SETTINGS_REQUIRED')


def daily_growth(path, initial_cash, expected_rows, timezone):
    days, previous, count = {}, None, 0
    zone = ZoneInfo(timezone)
    if not math.isfinite(initial_cash) or initial_cash <= 0:
        raise ValueError('PAIRED_ES_BAD_INITIAL_EQUITY')
    with Path(path).open(newline='') as stream:
        for row in csv.DictReader(stream):
            stamp = datetime.fromisoformat(row['timestamp'].replace('Z', '+00:00'))
            equity = float(row['equity'])
            if (stamp.tzinfo is None or (previous is not None and stamp <= previous)
                    or not math.isfinite(equity) or equity <= 0):
                raise ValueError('PAIRED_ES_BAD_TRACE')
            days[stamp.astimezone(zone).date()] = equity
            previous, count = stamp, count + 1
    if count != expected_rows or not days:
        raise ValueError('PAIRED_ES_SUPPORT_MISMATCH')
    # The telescoping log-growth equals the average over observed day closes.
    growth = math.log(next(reversed(days.values())) / initial_cash) / len(days)
    return dict(mean_daily_log_return=growth, days=len(days), rows=count)


class PairedMonitor:
    def __init__(self, patience, min_delta):
        if type(patience) is not int or patience <= 0 or not math.isfinite(min_delta) or min_delta < 0:
            raise ValueError('PAIRED_ES_BAD_PATIENCE')
        self.patience, self.min_delta = patience, min_delta
        self.best_score, self.best_step, self.last_step = None, None, 0
        self.patience_used = 0

    def observe(self, step, train_growth, validation_growth):
        if step <= self.last_step or not all(math.isfinite(v) for v in (train_growth, validation_growth)):
            raise ValueError('PAIRED_ES_BAD_SCORE_OR_STEP')
        self.last_step = step
        score = -.5 * (train_growth + validation_growth)
        improved = self.best_score is None or self.best_score - score > self.min_delta
        if improved:
            self.best_score, self.best_step, self.patience_used = score, step, 0
        else:
            self.patience_used += 1
        return dict(step=step, score=score, train_daily_log_growth=train_growth,
            validation_daily_log_growth=validation_growth,
            generalization_gap=train_growth - validation_growth, improved=improved,
            best_step=self.best_step, best_score=self.best_score,
            patience_used=self.patience_used, patience_max=self.patience,
            stop=self.patience_used >= self.patience)


def monitor_pair(config, model, step):
    """Existing inference tasks in fresh processes; no network or parent RNG use."""
    from tools import curated_rl_campaign as queue
    from tools.verify_rl_static_engineering import save_status
    from agent_plugins.sac_agent import _policy_tensor_hash
    root = Path(config['paired_campaign_root'])
    body = queue.load_contract(root / 'CONTRACT.json')
    if queue.digest(root / 'CONTRACT.json') != config['paired_campaign_sha256']:
        raise ValueError('PAIRED_ES_CONTRACT_DRIFT')
    arm = config.get('queue_arm', 'SAC_FLAT')
    if body['arms'] != [arm] or arm not in ('SAC_FLAT', 'SAC_PER_FEATURE_CONV1D'):
        raise ValueError('PAIRED_ES_ARM_BINDING_MISMATCH')
    monitor_root = Path(config['save_model']).parent / 'paired-monitor'
    directory = monitor_root / f'step-{step:06d}'
    directory.mkdir(parents=True, exist_ok=False)
    checkpoint = directory / 'policy.zip'
    before = _policy_tensor_hash(model.policy)
    model.save(checkpoint)
    saved_pin = queue.digest(checkpoint)
    growth, reports = {}, []
    started = time.monotonic()
    for role in ('train', 'validation'):
        child = directory / role
        child.mkdir()
        task = dict(arm=arm, split=role, seed=config['train_seed'], attempt=1,
            id=f"{config['paired_source_unit']}:{step}:{role}",
            started_at=queue._now(), policy_tensor_sha256=before)
        child_body = copy.deepcopy(body)
        child_body['common_config'].pop('paired_early_stopping', None)
        cfg = queue._task_config(child_body, task, child, checkpoint)
        cfg.update(mode='inference', save_model=None, load_model=str(checkpoint))
        for key in ('paired_early_stopping', 'paired_campaign_root',
                    'paired_campaign_sha256', 'paired_source_unit'):
            cfg.pop(key, None)
        save_status(child / 'config.json', cfg)
        save_status(monitor_root / 'STATUS.json', dict(schema='paired_sac_monitor_status.v1',
            state='EVALUATING', step=step, split=role, updated_at=queue._now(),
            progress_file=cfg['progress_file']))
        start_role = time.monotonic()
        code = queue._subprocess(cfg, child, body['execution']['task_timeout_seconds']['validation'])
        if code != 0:
            raise ValueError('PAIRED_ES_INFERENCE_FAILED')
        summary = json.loads(Path(cfg['results_file']).read_text())
        static = summary.get('static_policy', {})
        if (static.get('before') != before or static.get('after') != before
                or queue.digest(checkpoint) != saved_pin
                or summary.get('evaluation_truncated_by_cap') is not False):
            raise ValueError('PAIRED_ES_EVALUATION_INTEGRITY')
        manifest = json.loads(Path(cfg['curated_input_manifest']).read_text())
        expected = manifest['row_count'] - ((cfg['window_size'] - 1) * cfg['observation_stride_bars'] + 1)
        growth[role] = daily_growth(summary['return_trace_file'], cfg['initial_cash'],
                                   expected, cfg['entry_day_timezone'])
        task.update(config=str(child / 'config.json'), summary=cfg['results_file'],
            checkpoint=str(checkpoint), effective_config=cfg['save_config'],
            wall_seconds=time.monotonic() - start_role, finished_at=queue._now())
        for key in ('return_trace_file', 'return_trace_metadata_file'):
            task[key] = summary[key]
        summary.update(mean_daily_log_return=growth[role]['mean_daily_log_return'],
                       observed_days=growth[role]['days'], monitor_step=step)
        save_status(Path(cfg['results_file']), summary)
        task['artifacts'] = {k: queue.digest(task[k]) for k in
                            ('config', 'summary', 'checkpoint', 'effective_config',
                             'return_trace_file', 'return_trace_metadata_file')}
        reports.append(task)
    if _policy_tensor_hash(model.policy) != before:
        raise ValueError('PAIRED_ES_PARENT_POLICY_CHANGED')
    return dict(schema='paired_sac_checkpoint.v1', step=step,
        checkpoint=str(checkpoint), checkpoint_sha256=saved_pin, policy_tensor_sha256=before,
        growth=growth, reports=reports, wall_seconds=time.monotonic() - started,
        directory=str(directory), monitor_root=str(monitor_root))


def make_callback(config):
    from agent_plugins.fixed_budget_monitor import SETTINGS as FIXED_SETTINGS, make_callback as fixed_callback
    if config['paired_early_stopping'].get('protocol')==FIXED_SETTINGS['protocol']:
        return fixed_callback(config)
    from stable_baselines3.common.callbacks import BaseCallback
    from tools.verify_rl_static_engineering import save_status
    settings = config['paired_early_stopping']
    validate_settings(settings)

    class Callback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.monitor = PairedMonitor(settings['patience'], settings['min_delta'])
            self.records = []
            self.best = None

        def _on_step(self):
            step = self.model.num_timesteps
            if step % settings['eval_interval']:
                return True
            record = monitor_pair(config, self.model, step)
            decision = self.monitor.observe(step,
                record['growth']['train']['mean_daily_log_return'],
                record['growth']['validation']['mean_daily_log_return'])
            record['decision'] = decision
            from tools.curated_rl_campaign import digest
            for task in record['reports']:
                summary_path = Path(task['summary'])
                summary = json.loads(summary_path.read_text())
                summary.update(monitor_score=decision['score'],
                    monitor_generalization_gap=decision['generalization_gap'],
                    monitor_patience_used=decision['patience_used'],
                    monitor_best_step=decision['best_step'])
                save_status(summary_path, summary)
                task['artifacts']['summary'] = digest(summary_path)
            save_status(Path(record['directory']) / 'PAIR.json', record)
            self.records.append(record)
            if decision['improved']:
                self.best = record
            save_status(Path(record['monitor_root']) / 'STATUS.json', dict(
                schema='paired_sac_monitor_status.v1', state='TRAINING' if not decision['stop'] else 'STOPPING',
                updated_at=datetime.now(timezone.utc).isoformat(),
                **decision))
            return not decision['stop']

        def _on_training_end(self):
            from stable_baselines3 import SAC
            from agent_plugins.sac_agent import _policy_tensor_hash
            from tools.curated_rl_campaign import digest
            if not self.best or digest(self.best['checkpoint']) != self.best['checkpoint_sha256']:
                raise ValueError('PAIRED_ES_NO_VERIFIED_BEST')
            best = SAC.load(self.best['checkpoint'], device=self.model.device)
            self.model.policy.load_state_dict(best.policy.state_dict(), strict=True)
            for component in ('actor', 'critic'):
                getattr(self.model, component).optimizer.load_state_dict(
                    copy.deepcopy(getattr(best, component).optimizer.state_dict()))
            if self.model.log_ent_coef is not None:
                self.model.log_ent_coef.data.copy_(best.log_ent_coef.data)
                self.model.ent_coef_optimizer.load_state_dict(copy.deepcopy(best.ent_coef_optimizer.state_dict()))
            if _policy_tensor_hash(self.model.policy) != self.best['policy_tensor_sha256']:
                raise ValueError('PAIRED_ES_RESTORE_FAILED')
            evidence = dict(protocol=PROTOCOL, evaluations=len(self.records),
                best_step=self.monitor.best_step, best_score=self.monitor.best_score,
                actual_training_steps=self.model.num_timesteps,
                selected_policy_tensor_sha256=self.best['policy_tensor_sha256'],
                stop_reason='PATIENCE' if self.monitor.patience_used >= self.monitor.patience else 'TRANSITION_CAP',
                patience_used=self.monitor.patience_used, settings=settings)
            self.model.paired_early_stopping_evidence = evidence
            save_status(Path(self.best['monitor_root']) / 'FINAL.json', evidence)
            save_status(Path(self.best['monitor_root']) / 'STATUS.json', dict(
                state='BEST_CHECKPOINT_RESTORED', **evidence))
    return Callback()
