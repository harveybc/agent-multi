import copy
import json

import pytest
from tests.unit.test_paired_checkpoint_diagnostics import campaign

from tests.unit.test_selected_rl_capacity_sequence import paired_request
from tests.unit.test_selected_rl_comparison import write


def test_explicit_fixed_request_changes_only_monitor(tmp_path):
    from tools import selected_rl_comparison as selected
    from tools.selected_rl_temporal_minimal import ARCH
    from agent_plugins.fixed_budget_monitor import SETTINGS
    path,request=paired_request(tmp_path)
    request.update(schema='selected_rl_comparison_request.v7',temporal_encoder=ARCH)
    write(path,request);old,_=selected._build(path,tmp_path/'old')
    request.update(schema='selected_rl_comparison_request.v8',early_stopping=SETTINGS)
    write(path,request);body,_=selected._build(path,tmp_path/'new')
    expected=copy.deepcopy(old['common_config']);expected['paired_early_stopping']=SETTINGS
    assert body['common_config']==expected
    assert body['arms']==old['arms'] and body['selected_input_binding']==old['selected_input_binding']
    request['schema']='selected_rl_comparison_request.v7';write(path,request)
    with pytest.raises(ValueError):selected._build(path,tmp_path/'bad')


def test_fixed_decisions_do_not_stop_or_select_intermediate_winner():
    from agent_plugins.fixed_budget_monitor import decision
    rows=[decision(s,t,v) for s,t,v in [(25000,.1,.1),(50000,.2,0),(75000,.3,-.1),(100000,.4,-.2)]]
    assert all(r['stop'] is False for r in rows)
    assert [r['selected_step'] for r in rows]==[None,None,None,100000]
    assert all('patience_used' not in r and 'best_step' not in r for r in rows)
    with pytest.raises(ValueError):decision(5000,0,0)


def test_fixed_curve_replays_final_not_minimum_score():
    from tools.paired_checkpoint_diagnostics import compare_fixed_curve
    points=[dict(step=s,train_growth=t,validation_growth=v) for s,t,v in
        [(25000,.1,.1),(50000,.05,.05),(75000,0,0),(100000,-.1,-.1)]]
    result=compare_fixed_curve(points)
    assert result['selected_step']==100000
    assert compare_fixed_curve(points[:2])['selected_step'] is None
    with pytest.raises(ValueError):compare_fixed_curve(points[1:])


def test_fixed_units_have_distinct_identity_and_publisher(tmp_path):
    from tools import selected_rl_temporal_minimal as temporal
    path,_=paired_request(tmp_path);root=tmp_path/'fixed'
    spec=temporal.prepare(path,root,tmp_path/'old','f'*64,fixed_budget=True)
    assert spec['monitoring']['checkpoint_selection']=='FINAL_100000'
    paths=temporal.render(spec,root/'units',tmp_path/'private.env',root/'SEQUENCE.json')
    assert len(paths)==4
    assert (root/'units/rl-selected-temporal-fixed-conv1d2x4-n8.service').exists()
    assert 'selected_rl_temporal_minimal' in (root/'units/rl-selected-temporal-fixed-publication.service').read_text()


def test_callback_retains_terminal_post_update_weights_and_never_stops(tmp_path,monkeypatch):
    import types
    import torch
    from agent_plugins import fixed_budget_monitor as module
    from agent_plugins.sac_agent import _policy_tensor_hash
    from tools.paired_checkpoint_diagnostics import digest
    model=types.SimpleNamespace(num_timesteps=0,_n_updates=0,policy=torch.nn.Linear(1,1))
    root=tmp_path/'monitor';calls=[]
    def pair(config,actual,step):
        directory=root/f'step-{step:06d}';directory.mkdir(parents=True)
        path=directory/'policy.zip';path.write_bytes(_policy_tensor_hash(actual.policy).encode())
        calls.append((step,actual._n_updates,_policy_tensor_hash(actual.policy)))
        return dict(step=step,directory=str(directory),monitor_root=str(root),checkpoint=str(path),
            checkpoint_sha256=digest(path),policy_tensor_sha256=_policy_tensor_hash(actual.policy),reports=[],
            growth={r:dict(mean_daily_log_return=-step/100000) for r in ('train','validation')})
    monkeypatch.setattr(module,'monitor_pair',pair)
    callback=module.make_callback(dict(paired_early_stopping=module.SETTINGS,total_timesteps=100000))
    callback.model=model
    for step in (25000,50000,75000):
        model.num_timesteps=step;assert callback._on_step() is True
    model.num_timesteps=100000
    assert callback._on_step() is True and len(calls)==3
    with torch.no_grad():model.policy.weight.add_(1)
    model._n_updates=123
    final_hash=_policy_tensor_hash(model.policy)
    callback._on_training_end()
    assert calls[-1]==(100000,123,final_hash)
    assert _policy_tensor_hash(model.policy)==final_hash
    evidence=json.loads((root/'FINAL.json').read_text())
    assert evidence['selected_step']==100000 and evidence['stop_reason']=='FIXED_BUDGET'
    assert 'best_step' not in evidence


def test_incomplete_callback_cannot_close(tmp_path):
    import types
    from agent_plugins.fixed_budget_monitor import make_callback,SETTINGS
    callback=make_callback(dict(paired_early_stopping=SETTINGS,total_timesteps=100000))
    callback.model=types.SimpleNamespace(num_timesteps=75000)
    with pytest.raises(ValueError,match='INCOMPLETE'):callback._on_training_end()


def test_real_artifact_fixed_plots_idempotence_and_forged_selection(campaign):
    import shutil
    from pathlib import Path
    from tools.paired_checkpoint_diagnostics import read,digest,analyze
    from tools.fixed_budget_diagnostics import refresh
    from agent_plugins.fixed_budget_monitor import SETTINGS,decision
    body=read(campaign/'CONTRACT.json');body['common_config']['paired_early_stopping']=SETTINGS
    write(campaign/'CONTRACT.json',body)
    old=sorted(campaign.glob('*/*/paired-monitor/step-*/PAIR.json'))
    for path,step in zip(old,(25000,50000)):
        before=path.parent;after=before.parent/f'step-{step:06d}';before.rename(after)
        pair=read(after/'PAIR.json');pair['step']=step
        for task in pair['reports']:
            for key in ('config','summary','effective_config','return_trace_file','checkpoint'):
                task[key]=task[key].replace(str(before),str(after))
            summary=read(task['summary']);summary.update(net_profit=10,max_drawdown_pct=1,
                sharpe_ratio=.1,trades_closed=2,win_rate_pct=50,profit_factor=1.1)
            write(Path(task['summary']),summary)
            task['artifacts']={k:digest(task[k]) for k in task['artifacts']}
        pair['decision']=decision(step,pair['growth']['train']['mean_daily_log_return'],pair['growth']['validation']['mean_daily_log_return'])
        write(after/'PAIR.json',pair)
    source=after
    for step in (75000,100000):
        target=source.parent/f'step-{step:06d}';shutil.copytree(source,target)
        pair=read(target/'PAIR.json');pair['step']=step
        for task in pair['reports']:
            for key in task['artifacts']:task[key]=task[key].replace(str(source),str(target))
            task['artifacts']={k:digest(task[k]) for k in task['artifacts']}
        pair['decision']=decision(step,pair['growth']['train']['mean_daily_log_return'],pair['growth']['validation']['mean_daily_log_return'])
        write(target/'PAIR.json',pair)
    snapshot={p:digest(p) for p in campaign.rglob('*') if p.is_file()}
    assert refresh(campaign)['pairs']==4
    assert all(digest(p)==pin for p,pin in snapshot.items())
    output=source.parent/'diagnostics';png=output/'CURVES.png'
    assert png.stat().st_size>10000
    original=digest(png)
    assert refresh(campaign)['state']=='UNCHANGED_AUTHENTICATED_PAIRS' and digest(png)==original
    assert analyze(campaign)[0]['selected_step']==100000
    forged=target/'PAIR.json';pair=read(forged);pair['decision']['selected_step']=25000;write(forged,pair)
    with pytest.raises(ValueError,match='DECISION_REPLAY_MISMATCH'):refresh(campaign)
    assert digest(png)==original
