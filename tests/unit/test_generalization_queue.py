import json

import pytest

from tools.run_generalization_queue import matched_reference,pilot_gate,coverage,resource_failure,coordinator,memory_guard_overlap


def metrics(score,activity=3):
    return {p:dict(return_per_week=score,orders_per_week=activity) for p in ('TRAIN','VALIDATION')}


def test_pilot_activity_is_not_optional_and_tolerance_is_explicit():
    assert pilot_gate(metrics(2),metrics(1))
    assert not pilot_gate(metrics(2,0),metrics(1))
    assert not pilot_gate(metrics(2,2),metrics(1))
    assert not pilot_gate(metrics(1+1e-9),metrics(1))


def test_matched_monitor_requires_enough_history_and_uses_early_peak():
    d=dict(patience=2,min_delta=1e-8,monitor_mode='validation',monitor_start_epoch=2,select_during_warmup=True)
    h=[dict(epoch=i,metrics=metrics(s)) for i,s in enumerate([5,0,0,0],1)]
    result=matched_reference(h,d)
    assert result['selected']['epoch']==1 and result['stop_epoch']==4
    with pytest.raises(ValueError,match='DOES_NOT_REACH'):
        matched_reference(h[:3],d)
    with pytest.raises(ValueError,match='SEQUENCE'):
        matched_reference(h[1:],d)


def test_joint_coverage_cannot_silently_admit_price_only_generator(tmp_path):
    rec=tmp_path/'recipe.json'
    rec.write_text(json.dumps(dict(identity='r',list=dict(columns=['ATR','cot.eur.net_share','cal.utc_dow_sin']))))
    result=coverage(rec,tmp_path/'audit.json')
    assert result['status']=='NO_ADMISSIBLE_JOINT_PRODUCER'
    assert len(result['columns'])==3 and not result['training_launched']
    assert not any(r['validated_joint_producer'] for r in result['columns'])


def test_only_resource_failures_trigger_fallback():
    assert resource_failure(dict(Result='oom-kill',ExecMainStatus='9'))
    assert resource_failure(dict(Result='start-limit-hit',ExecMainStatus='137'))
    assert resource_failure(dict(Result='success',ExecMainCode='2',ExecMainStatus='15'))
    assert not resource_failure(dict(Result='exit-code',ExecMainStatus='1'))


def test_memguard_success_exit_requires_actual_overlapping_event():
    s=dict(ExecMainStartTimestampMonotonic='50000000',ExecMainExitTimestampMonotonic='60000000')
    event=dict(__MONOTONIC_TIMESTAMP='59900000',MESSAGE='memguard: MemAvailable low; SIGTERM batch')
    assert memory_guard_overlap(s,[event])
    assert not memory_guard_overlap(s,[dict(event,__MONOTONIC_TIMESTAMP='10000000')])
    assert not memory_guard_overlap(s,[dict(event,MESSAGE='normal service termination')])


def test_coordinator_completes_paired_queues_and_resumes_without_recollecting(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import tools.run_generalization_queue as queue
    recipe=tmp_path/'recipe.json';recipe.write_text(json.dumps(dict(identity='r',list=dict(columns=['ATR']))))
    def m(value):
        return {p:dict(return_per_week=value,orders_per_week=3,return_pct=value,sharpe=.1,
                       trades=120,win_rate=.5,max_drawdown_pct=5) for p in ('TRAIN','VALIDATION')}
    reference=tmp_path/'reference.json'
    reference.write_text(json.dumps(dict(matched_monitor=dict(selected=dict(epoch=1,metrics=m(1))))))
    def job(key,host,seed): return dict(key=key,host=host,unit='sac-generalization-'+key,seed=seed)
    jobs=[job('G2-19','primary',19),job('R0-19','secondary',19),job('G2-31','primary',31),job('R0-31','secondary',31)]
    config=tmp_path/'config.json';out=tmp_path/'out'
    config.write_text(json.dumps(dict(output=str(out),recipe=str(recipe),reference=str(reference),
        pilot=job('G2-7','primary',7),replications=jobs,poll_seconds=0)))
    monkeypatch.setattr(queue,'progress',lambda j:{'TASK.json':{'status':'COMPLETE'}})
    calls=[]
    def fetch(j,o):
        calls.append(j['key']);r=dict(selected_epoch=1,result=dict(seed=j['seed'],metrics=m(2)))
        queue.write(o/'receipts'/(j['key']+'.json'),r);return r
    monkeypatch.setattr(queue,'fetch_receipt',fetch)
    coordinator(SimpleNamespace(config=config))
    assert calls==['G2-7','G2-19','R0-19','G2-31','R0-31']
    assert json.loads((out/'QUEUE.json').read_text())['status']=='COMPLETE'
    coordinator(SimpleNamespace(config=config))
    assert len(calls)==5
