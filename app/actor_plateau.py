"""Actor-only LR scheduling, with explicit resumed-run observation budget."""
from copy import deepcopy
import math

import torch


class ActorPlateau:
    def __init__(self, optimizer, best, patience, origin_epoch, config):
        expected = {'factor', 'threshold', 'cooldown', 'min_actor_lr', 'post_reduction_epochs'}
        if set(config) != expected or len(optimizer.param_groups) != 3 or not math.isfinite(best):
            raise ValueError('ACTOR_PLATEAU_CONFIG')
        if type(patience) is not int or patience < 1 or type(origin_epoch) is not int or origin_epoch < 0:
            raise ValueError('ACTOR_PLATEAU_COUNTER')
        if not 0 < config['factor'] < 1 or not 0 < config['min_actor_lr'] <= optimizer.param_groups[0]['lr']:
            raise ValueError('ACTOR_PLATEAU_RATE')
        if (config['threshold'] < 0 or not math.isfinite(config['threshold'])
                or type(config['cooldown']) is not int or config['cooldown'] < 0
                or type(config['post_reduction_epochs']) is not int or config['post_reduction_epochs'] < 1):
            raise ValueError('ACTOR_PLATEAU_BUDGET')
        self.optimizer = optimizer
        self.config = deepcopy(config)
        self.fixed_rates = [float(g['lr']) for g in optimizer.param_groups[1:]]
        self.last_reduction = None
        self.origin_epoch = origin_epoch
        self.last_epoch = origin_epoch
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
            factor=config['factor'], patience=math.ceil(patience/3), threshold=config['threshold'],
            threshold_mode='abs', cooldown=config['cooldown'],
            min_lr=[config['min_actor_lr'], *self.fixed_rates], eps=0.)
        self.scheduler.best = best

    def observe(self, score, eligible, epoch):
        if not math.isfinite(score) or type(eligible) is not bool or epoch != self.last_epoch+1:
            raise ValueError('ACTOR_PLATEAU_OBSERVATION')
        before = [float(g['lr']) for g in self.optimizer.param_groups]
        if before[1:] != self.fixed_rates:
            raise ValueError('ACTOR_PLATEAU_OTHER_ROLE_CHANGED')
        self.scheduler.step(score if eligible else -math.inf)
        after = [float(g['lr']) for g in self.optimizer.param_groups]
        if after[1:] != self.fixed_rates:
            raise ValueError('ACTOR_PLATEAU_MUTATED_OTHER_ROLE')
        reduced = after[0] < before[0]
        if reduced:
            self.last_reduction = epoch
        self.last_epoch = epoch
        return dict(epoch=epoch, before=before, after=after, reduced=reduced,
            best=self.scheduler.best, bad=self.scheduler.num_bad_epochs,
            cooldown=self.scheduler.cooldown_counter, eligible=eligible,
            earliest_patience_stop=self.earliest_stop())

    def earliest_stop(self):
        anchor = self.origin_epoch if self.last_reduction is None else self.last_reduction
        return anchor + self.config['post_reduction_epochs']

    def state_dict(self):
        scheduler = deepcopy(self.scheduler.state_dict())
        # Native torch uses -inf as a mode sentinel; persistent JSON forbids NaN/Inf.
        scheduler['mode_worse'] = '-inf'
        return dict(config=deepcopy(self.config), fixed_rates=self.fixed_rates.copy(),
            origin_epoch=self.origin_epoch, last_epoch=self.last_epoch,
            last_reduction=self.last_reduction, scheduler=scheduler)

    def load_state_dict(self, saved):
        if saved['config'] != self.config or saved['fixed_rates'] != self.fixed_rates or saved['origin_epoch'] != self.origin_epoch:
            raise ValueError('ACTOR_PLATEAU_RESUME_CONTRACT')
        state = deepcopy(saved['scheduler'])
        if state['mode_worse'] != '-inf':
            raise ValueError('ACTOR_PLATEAU_SENTINEL')
        state['mode_worse'] = -math.inf
        self.scheduler.load_state_dict(state)
        if [float(g['lr']) for g in self.optimizer.param_groups] != self.scheduler.get_last_lr():
            raise ValueError('ACTOR_PLATEAU_RESUME_RATES')
        self.last_epoch = saved['last_epoch']
        self.last_reduction = saved['last_reduction']
