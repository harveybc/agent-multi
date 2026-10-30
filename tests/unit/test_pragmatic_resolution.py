import json
import os
from pathlib import Path

import pytest


def test_dependency_wait_failure_and_authentication(tmp_path):
    from tools.run_pragmatic_resolution import dependency_ready,verify_dependency_hash
    from tools.run_crossed_account_queue import save
    assert dependency_ready(tmp_path,'id','source',False) is None
    save(tmp_path/'MANIFEST.json','id',dict(source_identity='source',smoke=False))
    save(tmp_path/'COMPLETE.json','id',dict(artifacts={}))
    assert dependency_ready(tmp_path,'id','source',False)
    verify_dependency_hash('same','same')
    for observed in (None,'changed'):
        with pytest.raises(ValueError,match='CHANGED'): verify_dependency_hash(observed,'same')
    with pytest.raises(ValueError,match='BINDING'): dependency_ready(tmp_path,'id','other',False)
    with pytest.raises(ValueError): dependency_ready(tmp_path,'wrong','source',False)
    (tmp_path/'COMPLETE.json').unlink()
    (tmp_path/'STATE.json').write_text(json.dumps({'status':'FAILED'}))
    with pytest.raises(ValueError,match='FAILED'): dependency_ready(tmp_path,'id','source',False)


def test_exact_pairs_and_fresh_banks():
    from tools.run_pragmatic_resolution import paired_configs,fresh_specs
    root=Path(__file__).resolve().parents[2]
    source=json.loads((root/'docs/audits/evidence/pragmatic_contrast_launch_20261030/MANIFEST.json').read_text())
    selected=json.loads((root/'docs/audits/evidence/pragmatic_contrast_full_20261030/SELECTION.json').read_text())['data']
    config=json.loads((root/'configs/pragmatic_resolution_contrast.json').read_text())
    configs=paired_configs(source,selected,config)
    assert len(configs)==6 and len({r['key'] for r in configs})==6
    for new in (r for r in configs if r['arm']=='sequential_1h'):
        old=next(r for r in configs if r['key']==new['parent'])
        assert [k for k in new['config'] if new['config'][k]!=old['config'][k]]==['close_offsets']
        assert new['config']['close_offsets']==[1,2,3,6,12,24]
    a=fresh_specs(source,'calibration',config,False)
    b=fresh_specs(source,'confirmation',config,False)
    assert len(a)==len(b)==36
    assert {r['seed'] for r in a}=={960000}
    assert {r['seed'] for r in b}=={970000}
    assert not {r['seed'] for r in a+b}&{r['seed'] for r in source['confirmation']+source['calibration']}


def test_historical_missing_short_hold_witness():
    native=os.environ.get('STRATEGY_ENGINE_ROOT')
    if not native: pytest.skip('native engine required')
    from app.zigzag_opportunities import engine
    from app.pragmatic_strategy import (PragmaticWorld,ControllerConfig,PragmaticController,
        rollout_plan,select_plan,run_account)
    engine(Path(native));root=Path(__file__).resolve().parents[2]
    source=json.loads((root/'docs/audits/evidence/pragmatic_contrast_launch_20261030/MANIFEST.json').read_text())
    risk=json.loads((root/'docs/audits/evidence/shared_risk_rules_20261028/CONFIG.json').read_text())
    world=PragmaticWorld(source['confirmation'][0],risk,'ideal')
    cfg=ControllerConfig('sequential',alpha=1,uncertainty=0,persistence=2)
    observations=[]
    class Witness(PragmaticController):
        def decide(self,w,e,equity,horizon):
            decision=super().decide(w,e,equity,horizon)
            if e.next_decision in (50,62):
                original=decision['plan'];r=w.report(e.next_decision)
                seconds=list(dict.fromkeys(x['plan'][2] for x in original['plans']))
                plans=[('HOLD',None,'NOP')]+[('HOLD',delay,s) for delay in (1,2) for s in seconds]
                choice=select_plan(plans,[rollout_plan(w,e,equity,r,p) for p in plans],r.weights,cfg)
                assert decision['action']=='CLOSE'
                assert choice['score']>original['score']+.1
                assert choice['worst_dd_pct']==pytest.approx(original['worst_dd_pct'])
                observations.append((e.next_decision,choice['plan'][1]))
            return decision
    run_account(world,cfg,controller=Witness(cfg))
    assert observations==[(50,1),(62,2)]
