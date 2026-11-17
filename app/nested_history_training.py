"""Explicit gradient-budget treatment for full-history learning curves."""
from copy import deepcopy


def updates_at_decision(decision, decisions, quota):
    if (any(type(v) is not int for v in (decision, decisions, quota))
            or decisions < 1 or not 1 <= decision <= decisions or quota < 0):
        raise ValueError('UPDATE_SCHEDULE')
    return (decision*quota)//decisions - ((decision-1)*quota)//decisions


def training_design(parent):
    design=deepcopy(parent)
    design.pop('actor_plateau',None)
    design.update(head_only=False,initialize_epoch_zero=True,allow_ineligible_epoch_zero=True,
        warmup=0,warmup_policy=True,update_every=4,matched_updates_per_epoch=6006,
        matched_collection_epochs=1,max_epochs=101,minimum_epochs=101,matched_budget=True,
        monitor_mode='balanced_min',shared_monitor_decision=True,
        monitor='balanced_min_full_accounts_return_per_week',monitor_start_epoch=20,
        select_during_warmup=True,require_activity=True,patience=20,min_delta=1e-8,
        diagnostics=True,replay_backend='memmap')
    return design
