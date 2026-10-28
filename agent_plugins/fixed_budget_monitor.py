"""Sparse, full-support observations without stopping or checkpoint tuning."""
import math
from pathlib import Path

from agent_plugins.paired_early_stopping import monitor_pair

PROTOCOL='fixed_budget_full_daily_log_growth.v1'
STEPS=(25000,50000,75000,100000)
SETTINGS=dict(protocol=PROTOCOL,eval_interval=25000,early_stopping=False,
              checkpoint_selection='FINAL_100000')


def decision(step, train_growth, validation_growth):
    if step not in STEPS or not all(math.isfinite(v) for v in (train_growth,validation_growth)):
        raise ValueError('FIXED_BUDGET_INVALID_STEP_OR_SCORE')
    return dict(step=step,score=-.5*(train_growth+validation_growth),
        train_daily_log_growth=train_growth,validation_daily_log_growth=validation_growth,
        generalization_gap=train_growth-validation_growth,stop=False,
        selected_step=100000 if step==100000 else None,checkpoint_selection='FINAL_100000')


def make_callback(config):
    from stable_baselines3.common.callbacks import BaseCallback
    from tools.verify_rl_static_engineering import save_status
    from tools.curated_rl_campaign import digest
    from agent_plugins.sac_agent import _policy_tensor_hash
    from agent_plugins.paired_early_stopping import validate_settings
    validate_settings(config['paired_early_stopping'])
    if config['paired_early_stopping']!=SETTINGS or config.get('total_timesteps')!=100000:
        raise ValueError('FIXED_BUDGET_EXACT_PROTOCOL_REQUIRED')
    class Callback(BaseCallback):
        def __init__(self):
            super().__init__();self.records=[]

        def _observe(self):
            step=self.model.num_timesteps
            expected=STEPS[len(self.records)] if len(self.records)<len(STEPS) else None
            if step!=expected:raise ValueError('FIXED_BUDGET_STEP_SEQUENCE')
            record=monitor_pair(config,self.model,step)
            observed=decision(step,record['growth']['train']['mean_daily_log_return'],
                              record['growth']['validation']['mean_daily_log_return'])
            record.update(decision=observed,monitor_protocol=PROTOCOL)
            record['gradient_updates_at_checkpoint']=self.model._n_updates
            for task in record['reports']:
                import json
                path=Path(task['summary']);summary=json.loads(path.read_text())
                summary.update(monitor_score=observed['score'],
                    monitor_generalization_gap=observed['generalization_gap'],
                    checkpoint_selection='FINAL_100000',early_stopping_enabled=False)
                save_status(path,summary);task['artifacts']['summary']=digest(path)
            save_status(Path(record['directory'])/'PAIR.json',record)
            self.records.append(record)
            save_status(Path(record['monitor_root'])/'STATUS.json',dict(
                state='TRAINING_FIXED_BUDGET',protocol=PROTOCOL,**observed))
            return True

        def _on_step(self):
            # SB3 calls this before that transition's gradient update. Observe
            # the terminal policy only after its final update, at training end.
            if self.model.num_timesteps in STEPS[:-1]:return self._observe()
            return True

        def _on_training_end(self):
            if (self.model.num_timesteps!=100000 or [r['step'] for r in self.records]!=list(STEPS[:-1])):
                raise ValueError('FIXED_BUDGET_INCOMPLETE_NOT_A_FINAL_RESULT')
            self._observe()
            last=self.records[-1]
            if (digest(last['checkpoint'])!=last['checkpoint_sha256']
                    or _policy_tensor_hash(self.model.policy)!=last['policy_tensor_sha256']):
                raise ValueError('FIXED_BUDGET_FINAL_POLICY_CHANGED')
            evidence=dict(protocol=PROTOCOL,evaluations=4,selected_step=100000,
                actual_training_steps=100000,selected_policy_tensor_sha256=last['policy_tensor_sha256'],
                stop_reason='FIXED_BUDGET',checkpoint_selection='FINAL_100000',settings=SETTINGS)
            self.model.paired_early_stopping_evidence=evidence
            save_status(Path(last['monitor_root'])/'FINAL.json',evidence)
            save_status(Path(last['monitor_root'])/'STATUS.json',dict(state='FINAL_CHECKPOINT_RETAINED',**evidence))
    return Callback()
